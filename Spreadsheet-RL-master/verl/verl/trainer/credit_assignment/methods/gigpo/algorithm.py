# GiGPO 核心算法 (Eq.3/5/6/7/8). 独立计算模块: 只用输入数据算 advantage,
# 不触碰 rollout / tool / ray / PPO optimizer / actor forward / reward manager.

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl.trainer.credit_assignment.interface import CreditAssignmentInput, CreditAssignmentMethod, CreditAssignmentOutput
from verl.trainer.credit_assignment.methods.gigpo.config import GiGPOConfig
from verl.trainer.credit_assignment.methods.gigpo.grouping import INVALID_FINGERPRINT, build_step_groups, fingerprint_anchor
from verl.trainer.credit_assignment.registry import register_credit_assignment


def _normalize_within_groups(
    values: np.ndarray,
    group_keys: list[Any],
    remove_std: bool,
    epsilon: float,
    singleton_as_zero: bool,
) -> np.ndarray:
    """组内 mean/std 归一化 (对齐原 GiGPO episode_norm/step_norm 行为).

    singleton_as_zero=True: 单元素组的 baseline 取自身 -> advantage 0 (step-level 策略).
    singleton_as_zero=False: 单元素组 baseline 取 0 -> advantage 保留原值 (episode-level 策略).
    """
    out = np.array(values, dtype=np.float64, copy=True)
    key_to_positions: dict[Any, list[int]] = defaultdict(list)
    for pos, k in enumerate(group_keys):
        key_to_positions[k].append(pos)

    for k, positions in key_to_positions.items():
        arr = values[positions]
        if len(positions) == 1:
            mean = float(arr[0]) if singleton_as_zero else 0.0
            std = 1.0
        else:
            mean = float(np.mean(arr))
            # 与原实现一致: torch.std 为无偏 (sample) std.
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
        for pos in positions:
            if remove_std:
                out[pos] = values[pos] - mean
            else:
                out[pos] = (values[pos] - mean) / (std + epsilon)
    return out


@register_credit_assignment("gigpo")
class GiGPO(CreditAssignmentMethod):
    name = "gigpo"
    requires = frozenset({"step_spans", "step_observations"})

    def compute(self, inputs: CreditAssignmentInput, params: dict[str, Any]) -> CreditAssignmentOutput:
        cfg = GiGPOConfig.from_params(params)
        if inputs.step_spans is None or inputs.step_observations is None:
            raise ValueError("GiGPO requires step_spans and step_observations in CreditAssignmentInput")
        bs = inputs.batch_size
        response_length = inputs.response_length
        device = inputs.assistant_mask.device

        traj_rewards = np.asarray(inputs.trajectory_rewards, dtype=np.float64)
        uids = np.asarray(inputs.episode_group_ids, dtype=object)

        # --- Eq.3: episode-level advantage, 分组单位是 trajectory 而非 step row. ---
        # 组内归一化 -> step 数多的 trajectory 不会获得更大权重.
        episode_adv = _normalize_within_groups(
            traj_rewards,
            group_keys=[uids[i] for i in range(bs)],
            remove_std=cfg.remove_std,
            epsilon=cfg.epsilon,
            singleton_as_zero=False,
        )  # (bs,)

        # 展平所有 step entry: (row, step_index_in_row). anchor 三态: INITIAL/OBSERVED/INVALID.
        step_row: list[int] = []
        step_idx: list[int] = []
        step_anchor_fp: list[str] = []
        for row in range(bs):
            spans = inputs.step_spans[row] or []
            obs = inputs.step_observations[row] or []
            n_steps = len(spans)
            for k in range(n_steps):
                step_row.append(row)
                step_idx.append(k)
                raw = obs[k] if k < len(obs) else None
                step_anchor_fp.append(fingerprint_anchor(raw))

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
        step_adv = np.zeros(num_steps, dtype=np.float64)
        valid = [p for p in range(num_steps) if step_anchor_fp[p] != INVALID_FINGERPRINT]
        if valid:
            group_keys_map = build_step_groups(uids, [(step_row[p], step_anchor_fp[p]) for p in valid])
            group_keys = [group_keys_map[local] for local, p in enumerate(valid)]
            valid_returns = np.array([step_return[p] for p in valid], dtype=np.float64)
            # singleton 组 / 组内 variance 0 -> step advantage 0 (显式策略, 见 grouping 说明).
            normalized = _normalize_within_groups(
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
            start, end = int(span[0]), int(span[1])
            start = max(0, start)
            end = min(response_length, end)
            if end <= start:
                continue  # 截断后无有效 span
            value = float(episode_adv[row] + cfg.step_advantage_w * step_adv[pos])
            row_mask = inputs.assistant_mask[row, start:end] > 0
            advantages[row, start:end] = torch.where(
                row_mask, torch.tensor(value, dtype=torch.float32, device=device), torch.zeros_like(row_mask, dtype=torch.float32)
            )

        # tool observation token / padding 已在 assistant_mask 处被置 0.
        advantages = advantages * (inputs.assistant_mask > 0).to(advantages.dtype)

        return CreditAssignmentOutput(advantages=advantages, returns=advantages)
