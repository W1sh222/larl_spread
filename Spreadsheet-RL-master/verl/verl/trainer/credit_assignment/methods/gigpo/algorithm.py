# GiGPO 核心算法 (Eq.3/5/6/7/8). 独立计算模块: 只用输入数据算 advantage,
# 不触碰 rollout / tool / ray / PPO optimizer / actor forward / reward manager.

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl.trainer.credit_assignment.anchor_fingerprint import INVALID_FINGERPRINT
from verl.trainer.credit_assignment.interface import CreditAssignmentInput, CreditAssignmentMethod, CreditAssignmentOutput
from verl.trainer.credit_assignment.methods.gigpo.config import GiGPOConfig
from verl.trainer.credit_assignment.methods.gigpo.grouping import build_step_groups
from verl.trainer.credit_assignment.registry import register_credit_assignment
from verl.trainer.credit_assignment.normalization import normalize_within_groups
from verl.trainer.credit_assignment.state_meta import (
    ANCHOR_MODE_STATE,
    StepMeta,
    build_step_meta,
    clamp_response_span,
    meta_diagnostics,
)


@register_credit_assignment("gigpo")
class GiGPO(CreditAssignmentMethod):
    name = "gigpo"
    # 静态硬需求: step 级 span 与 observation. anchor='state' 时额外需要 structured
    # transitions (见 required_metadata), 但 observation anchor 基线不需要 -> 不写入静态 requires.
    requires = frozenset({"step_spans", "step_observations"})

    @classmethod
    def required_metadata(cls, params: dict[str, Any]) -> frozenset:
        """anchor='state' (默认) 额外依赖 structured transitions; anchor='observation' 不依赖.

        initial_states 仅为 step-0 缺 pre 时的可选 fallback (build_step_meta 容忍 None),
        因此不强制声明为必需; final_states / terminal_trusted GiGPO 不消费, 不声明.
        """
        req = set(cls.requires)
        if str((params or {}).get("anchor", "state")) == ANCHOR_MODE_STATE:
            req.add("step_transitions")
        return frozenset(req)

    def compute(self, inputs: CreditAssignmentInput, params: dict[str, Any]) -> CreditAssignmentOutput:
        cfg = GiGPOConfig.from_params(params)
        if inputs.step_spans is None or inputs.step_observations is None:
            raise ValueError("GiGPO requires step_spans and step_observations in CreditAssignmentInput")
        transitions = inputs.extra.get("step_transitions")
        if cfg.anchor == ANCHOR_MODE_STATE and transitions is None:
            # 不静默退化为 observation anchor: 那样会得出"看起来是 state 实验、实际是
            # observation 实验"的结果. 要么开采集, 要么显式 anchor=observation.
            raise ValueError(
                "GiGPO anchor='state' requires rollout metadata 'ca_step_transitions' "
                "(set algorithm.credit_assignment.collect_step_states=true, or use "
                "params.anchor='observation' for the observation-based baseline)."
            )
        bs = inputs.batch_size
        response_length = inputs.response_length
        device = inputs.assistant_mask.device

        traj_rewards = np.asarray(inputs.trajectory_rewards, dtype=np.float64)
        uids = np.asarray(inputs.episode_group_ids, dtype=object)

        # --- Eq.3: episode-level advantage, 分组单位是 trajectory 而非 step row. ---
        # 组内归一化 -> step 数多的 trajectory 不会获得更大权重.
        episode_adv = normalize_within_groups(
            traj_rewards,
            group_keys=[uids[i] for i in range(bs)],
            remove_std=cfg.remove_std,
            epsilon=cfg.epsilon,
            singleton_as_zero=False,
        )  # (bs,)

        # 展平所有 step entry: (row, step_index_in_row). anchor 三态: INITIAL/OBSERVED/INVALID.
        # anchor 键与 step 有效性判定共用 state_meta (单一分类来源, 避免两侧分叉).
        step_row: list[int] = []
        step_idx: list[int] = []
        step_anchor_fp: list[str] = []
        step_truncated: list[bool] = []
        meta_rows: list[list[StepMeta]] = []
        initial_states = inputs.extra.get("initial_states") or [None] * bs
        terminal_reasons = inputs.extra.get("terminal_reasons") or [None] * bs
        for row in range(bs):
            spans = inputs.step_spans[row] or []
            obs = inputs.step_observations[row] or []
            trans_row = transitions[row] if transitions is not None else None
            metas = build_step_meta(
                spans=spans,
                response_length=response_length,
                anchor_mode=cfg.anchor,
                transitions_row=trans_row,
                observations_row=obs,
                initial_state=initial_states[row] if row < len(initial_states) else None,
                terminal_reason=terminal_reasons[row] if row < len(terminal_reasons) else None,
            )
            meta_rows.append(metas)
            for meta in metas:
                step_row.append(row)
                step_idx.append(meta.index)
                step_anchor_fp.append(meta.anchor_fp)
                step_truncated.append(meta.truncated)

        num_steps = len(step_row)

        # --- Eq.5: step discounted return G[k] = r[k] + gamma * G[k+1]. ---
        # terminal outcome reward 只放在最后一个 step, 不重复分配; 中间 step 默认 0.
        step_return = np.zeros(num_steps, dtype=np.float64)
        row_step_positions: dict[int, list[int]] = defaultdict(list)
        for pos in range(num_steps):
            row_step_positions[step_row[pos]].append(pos)

        for row, positions in row_step_positions.items():
            positions = sorted(positions, key=lambda p: step_idx[p])  # 按 step_index 升序
            n = len(positions)
            if inputs.step_rewards is not None:
                immediate = np.asarray(inputs.step_rewards[row], dtype=np.float64)
                rewards = [float(immediate[k]) if k < len(immediate) else 0.0 for k in range(n)]
                # 若无任何中间 reward, 用 terminal outcome 兜底放最后一步.
                if all(r == 0.0 for r in rewards):
                    rewards[-1] = traj_rewards[row]
            else:
                rewards = [0.0] * (n - 1) + [float(traj_rewards[row])]
            running = 0.0
            for offset in reversed(range(n)):
                running = rewards[offset] + cfg.gamma * running
                step_return[positions[offset]] = running

        # --- Eq.6 + Eq.7: anchor grouping 内对 discounted return 归一化. ---
        # INVALID anchor: 不与其他 step 分组, step advantage 直接为 0 (仅保留 episode 部分);
        # 避免把"观测缺失"当作同一状态而误归组 (也不当作初始状态).
        # include_truncated_in_baseline=False 时, 被截断 step 也不参与分组 (其 A_S=0);
        # 默认 True, 保持与现有实现完全一致的 baseline 数值.
        step_adv = np.zeros(num_steps, dtype=np.float64)
        valid = [
            p
            for p in range(num_steps)
            if step_anchor_fp[p] != INVALID_FINGERPRINT
            and (cfg.include_truncated_in_baseline or not step_truncated[p])
        ]
        if valid:
            group_keys_map = build_step_groups(uids, [(step_row[p], step_anchor_fp[p]) for p in valid])
            group_keys = [group_keys_map[local] for local, p in enumerate(valid)]
            valid_returns = np.array([step_return[p] for p in valid], dtype=np.float64)
            # singleton 组 / 组内 variance 0 -> step advantage 0 (显式策略, 见 grouping 说明).
            normalized = normalize_within_groups(
                valid_returns,
                group_keys=group_keys,
                remove_std=cfg.remove_std,
                epsilon=cfg.epsilon,
                singleton_as_zero=True,
            )
            for local, p in enumerate(valid):
                step_adv[p] = normalized[local]

        # --- Eq.8: A_final = A_episode + step_advantage_w * A_step, 广播到各 step 的 assistant span. ---
        advantages = torch.zeros((bs, response_length), dtype=torch.float32, device=device)
        for pos in range(num_steps):
            row = step_row[pos]
            span = inputs.step_spans[row][step_idx[pos]]
            start, end = clamp_response_span(span, response_length)
            if end <= start:
                continue  # 截断后无有效 span
            value = float(episode_adv[row] + cfg.step_advantage_w * step_adv[pos])
            row_mask = inputs.assistant_mask[row, start:end] > 0
            advantages[row, start:end] = torch.where(
                row_mask, torch.tensor(value, dtype=torch.float32, device=device), torch.zeros_like(row_mask, dtype=torch.float32)
            )

        # tool observation token / padding 已在 assistant_mask 处被置 0.
        advantages = advantages * (inputs.assistant_mask > 0).to(advantages.dtype)

        metrics = meta_diagnostics(meta_rows)
        metrics["baseline_excluded_truncated"] = 0.0 if cfg.include_truncated_in_baseline else 1.0
        if num_steps:
            metrics["step_adv_nonzero_frac"] = float(np.count_nonzero(step_adv)) / num_steps
            metrics["anchor_group_count"] = float(len({step_anchor_fp[p] for p in valid}))
        return CreditAssignmentOutput(advantages=advantages, returns=advantages, metrics=metrics)
