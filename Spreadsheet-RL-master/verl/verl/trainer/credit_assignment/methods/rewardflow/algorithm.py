# RewardFlow 核心算法 (论文 Eq.1-6; 官方 core_rewardflow.compute_rewardflow_outcome_advantage).
#
# 与 GiGPO 的本质区别: step-level 信号不是 "anchor-group 内的 discounted return 归一化",
# 而是 "uid 组内 workbook state graph 的 potential difference" r_t = Phi(s_{t+1}) - Phi(s_t).
# 独立计算模块: 只用输入数据算 advantage, 不触碰 rollout / tool / PPO optimizer / reward manager.
#
# Terminal decision 策略 (对齐官方 last-step fallback 的意图, 实现不同):
#   final_answer / truncated / not-executed step 不构造 graph edge, action advantage = 0,
#   仅保留 episode-level advantage; 避免把 "不存在的 transition" 当作 potential 0 参与分组.
# Self-loop 策略 (与官方 clean_trajectory discard 等价): edge 不入图 (graph.py 拒绝),
#   但 r̃=0 作为样本参与 (uid, pre-state) normalization 分组 — "无状态变化" 是有效比较信号.

import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl.trainer.credit_assignment.interface import CreditAssignmentInput, CreditAssignmentMethod, CreditAssignmentOutput
from verl.trainer.credit_assignment.methods.rewardflow.config import RewardFlowConfig
from verl.trainer.credit_assignment.methods.rewardflow.graph import StateGraph
from verl.trainer.credit_assignment.methods.rewardflow.state import (
    is_graph_edge,
    normalize_state_id,
    normalize_transition,
)
from verl.trainer.credit_assignment.registry import register_credit_assignment

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_UNKNOWN_INITIAL = "<state_unknown>"


def _normalize_within_groups(
    values: np.ndarray,
    group_keys: list[Any],
    remove_std: bool,
    epsilon: float,
    singleton_as_zero: bool,
) -> np.ndarray:
    """组内 mean(/std) 归一化. 行为对齐官方 episode_norm_reward / step_norm_reward:
    episode: singleton baseline 0 (advantage=raw); step: singleton baseline 自身 (advantage=0)."""
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
            std = float(np.std(arr, ddof=1))
        for pos in positions:
            if remove_std:
                out[pos] = values[pos] - mean
            else:
                out[pos] = (values[pos] - mean) / (std + epsilon)
    return out


@register_credit_assignment("rewardflow")
class RewardFlow(CreditAssignmentMethod):
    name = "rewardflow"
    requires = frozenset({"step_spans", "step_observations", "step_transitions", "terminal_trusted"})

    def compute(self, inputs: CreditAssignmentInput, params: dict[str, Any]) -> CreditAssignmentOutput:
        cfg = RewardFlowConfig.from_params(params)
        if inputs.step_spans is None:
            raise ValueError("RewardFlow requires step_spans in CreditAssignmentInput")
        raw_transitions = inputs.extra.get("step_transitions")
        if raw_transitions is None:
            raise ValueError(
                "RewardFlow requires structured state transitions (ca_step_transitions). "
                "Enable algorithm.credit_assignment.collect_step_states (auto-on for method=rewardflow) "
                "and ensure rollout uses default_agent_loop=tool_agent with sheet_arena data."
            )

        bs = inputs.batch_size
        response_length = inputs.response_length
        device = inputs.assistant_mask.device

        traj_rewards = np.asarray(inputs.trajectory_rewards, dtype=np.float64)
        if not np.all(np.isfinite(traj_rewards)):
            raise ValueError("RewardFlow got non-finite trajectory rewards")
        uids = np.asarray(inputs.episode_group_ids, dtype=object)

        # --- 行级 metadata 对齐校验 (silent degradation 禁止: 行级缺失 -> raise). ---
        # 正常采集的 "空 step list" ([] with 无 span) 与 "缺失" (None) 在 adapter 层已严格区分.
        transitions: list[list[dict[str, Any]]] = []
        for row in range(bs):
            spans = inputs.step_spans[row] or []
            raw = raw_transitions[row] if row < len(raw_transitions) else None
            if raw is None:
                raise ValueError(
                    f"RewardFlow row {row}: transition metadata row is missing (None). "
                    "Possible causes: rollout batch mixed agent loops that do not emit "
                    "ca_step_transitions, or collect_step_states was disabled for part of the data. "
                    "Episode-only degradation is intentionally not allowed."
                )
            if len(raw) != len(spans):
                raise ValueError(
                    f"RewardFlow row {row}: ca_step_transitions has {len(raw)} entries but "
                    f"ca_step_spans has {len(spans)}; metadata misalignment is a structural error."
                )
            transitions.append([normalize_transition(t) for t in raw])

        self._warn_if_not_binary(traj_rewards, cfg)

        # --- Episode-level advantage (官方 episode_norm_reward: uid 组内 mean_norm). ---
        episode_adv = _normalize_within_groups(
            traj_rewards,
            group_keys=[str(uids[i]) for i in range(bs)],
            remove_std=cfg.remove_std,
            epsilon=cfg.epsilon,
            singleton_as_zero=False,
        )  # (bs,)

        # --- uid (+initial state 防碰撞) 分组构建 state graph. ---
        initials = inputs.extra.get("initial_states") or [None] * bs
        finals = inputs.extra.get("final_states") or [None] * bs
        row_gkey: list[tuple[str, str]] = []
        for row in range(bs):
            init = normalize_state_id(initials[row]) if row < len(initials) else None
            # 同 uid 但初始 state 不同 (理论不该发生) -> 自动拆开, 不共享拓扑.
            row_gkey.append((str(uids[row]), init if init is not None else _UNKNOWN_INITIAL))

        graphs: dict[tuple[str, str], StateGraph] = defaultdict(lambda: StateGraph(cfg.gamma))
        edges_added = 0
        steps_graph_eligible = 0
        steps_unknown_state = 0
        steps_invalid = 0
        for row in range(bs):
            graph = graphs[row_gkey[row]]
            for tr in transitions[row]:
                if tr["pre"] is None or tr["post"] is None:
                    if tr["executed"]:
                        steps_unknown_state += 1
                    continue
                if not is_graph_edge(tr):
                    steps_invalid += 1
                    continue
                steps_graph_eligible += 1
                if graph.add_edge(tr["pre"], tr["post"]):
                    edges_added += 1

        # Success anchor (论文 Eq.1): 仅当同时满足 (1) 成功 reward (2) terminal fingerprint known
        # (3) terminal trusted 时, 以 terminal workbook state 为 Phi=1 锚点.
        # graph-valid edge 存在与否 NOT 是锚定前置条件: 无 incoming edge 的孤立 anchor 仍合法,
        # 只是不向其他 state 传播 (归入 connectivity 诊断, 不凭空补馈 potential).
        # 严禁: 用 last eligible post 替代 anchor / 回退锚到更早中间 state (可能非成功 state) /
        # 把 excluded transition 恢复为通向 success 的 edge.
        # known 与 trusted 分离 (rollout 侧判定): fingerprint 存在不代表外部 writeback 已完成.
        # reward 0 不作 negative anchor (可能为评估基础设施失败).
        terminal_trusted_rows = inputs.extra.get("terminal_trusted")
        anchors = 0
        anchors_missing = 0      # terminal fingerprint unknown (采集失败): 不回退
        anchors_untrusted = 0    # fingerprint known 但外部 writeback 完成不可证: 不回退
        for row in range(bs):
            if traj_rewards[row] < cfg.success_threshold - cfg.success_tolerance:
                continue
            graph = graphs[row_gkey[row]]
            final = normalize_state_id(finals[row]) if row < len(finals) else None
            if final is None:
                anchors_missing += 1
                continue
            trusted = bool(terminal_trusted_rows[row]) if (
                terminal_trusted_rows is not None and row < len(terminal_trusted_rows)
            ) else False
            if not trusted:
                anchors_untrusted += 1
                continue
            if graph.add_success_anchor(final):
                anchors += 1

        potentials: dict[tuple[str, str], dict[str, float]] = {
            gkey: graph.compute_potentials() for gkey, graph in graphs.items()
        }

        # --- Action reward r_t = Phi(s_{t+1}) - Phi(s_t) (论文 Eq.3). ---
        # 仅 graph-eligible step 参与分组, invalid/unknown 不污染正常组的 mean/std;
        # 排除 step 的 action advantage 固定 0 (episode 分量照常保留).
        flat_steps: list[tuple[int, int]] = []  # (row, k)
        flat_rtilde: list[float] = []
        flat_norm_keys: list[tuple] = []
        step_advantage: dict[tuple[int, int], float] = {}

        for row in range(bs):
            pot = potentials[row_gkey[row]]
            for k, tr in enumerate(transitions[row]):
                if not is_graph_edge(tr) or tr["pre"] not in pot or tr["post"] not in pot:
                    step_advantage[(row, k)] = 0.0
                    continue
                flat_steps.append((row, k))
                flat_rtilde.append(pot[tr["post"]] - pot[tr["pre"]])
                flat_norm_keys.append((row_gkey[row], tr["pre"]))  # 论文 Eq.4: 同 pre-state 分组

        if flat_rtilde:
            normalized = _normalize_within_groups(
                np.asarray(flat_rtilde, dtype=np.float64),
                group_keys=flat_norm_keys,
                remove_std=cfg.remove_std,
                epsilon=cfg.epsilon,
                singleton_as_zero=True,  # singleton 组 -> advantage 0
            )
            for (row, k), r_norm in zip(flat_steps, normalized):
                step_advantage[(row, k)] = float(r_norm)

        # --- Eq.6: A = alpha_traj * A_traj + alpha_act * A_act; broadcast 到 assistant span. ---
        advantages = torch.zeros((bs, response_length), dtype=torch.float32, device=device)
        zero_action_steps = 0
        total_steps = 0
        for row in range(bs):
            spans = inputs.step_spans[row] or []
            for k, span in enumerate(spans):
                total_steps += 1
                start, end = int(span[0]), int(span[1])
                start = max(0, start)
                end = min(response_length, end)
                if end <= start:
                    continue  # 截断后无有效 span
                a_act = step_advantage.get((row, k), 0.0)
                if a_act == 0.0:
                    zero_action_steps += 1
                value = float(cfg.alpha_traj * episode_adv[row] + cfg.alpha_act * a_act)
                if not np.isfinite(value):
                    value = 0.0
                row_mask = inputs.assistant_mask[row, start:end] > 0
                advantages[row, start:end] = torch.where(
                    row_mask,
                    torch.tensor(value, dtype=torch.float32, device=device),
                    torch.zeros_like(row_mask, dtype=torch.float32),
                )
        advantages = advantages * (inputs.assistant_mask > 0).to(advantages.dtype)

        self._log_summary(
            bs=bs,
            groups=len(graphs),
            nodes=sum(g.stats()["nodes"] for g in graphs.values()),
            edges=sum(g.stats()["edges"] for g in graphs.values()),
            self_loops=sum(g.stats()["self_loops"] for g in graphs.values()),
            shared=sum(g.stats()["shared_in_nodes"] for g in graphs.values()),
            anchors=anchors,
            anchors_missing=anchors_missing,
            anchors_untrusted=anchors_untrusted,
            edges_added=edges_added,
            eligible=steps_graph_eligible,
            unknown_state=steps_unknown_state,
            invalid=steps_invalid,
            anchors_no_incoming=sum(g.stats()["anchors_without_valid_incoming_edge"] for g in graphs.values()),
            unreachable=sum(g.stats()["unreachable_valid_states"] for g in graphs.values()),
            total_steps=total_steps,
            zero_action_steps=zero_action_steps,
            advantages=advantages,
        )

        return CreditAssignmentOutput(advantages=advantages, returns=advantages)

    @staticmethod
    def _warn_if_not_binary(traj_rewards: np.ndarray, cfg: RewardFlowConfig) -> None:
        unique = np.unique(traj_rewards)
        non_binary = [float(v) for v in unique if not (abs(v) < 1e-6 or abs(v - 1.0) < 1e-6)]
        if non_binary:
            logger.warning(
                "RewardFlow expects binary outcome reward for success anchoring; got non-binary values %s "
                "(success_threshold=%.3f still applies, but verify reward pipeline semantics)",
                non_binary[:5],
                cfg.success_threshold,
            )

    @staticmethod
    def _log_summary(**s: Any) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        adv = s["advantages"]
        logger.info(
            "RewardFlow: rows=%d groups=%d nodes=%d edges=%d self_loops=%d shared=%d anchors=%d "
            "anchors_missing=%d anchors_untrusted=%d anchors_no_incoming=%d unreachable_states=%d "
            "eligible_steps=%d unknown_state_steps=%d invalid_steps=%d "
            "zero_action_steps=%d/%d adv_mean=%.5f adv_std=%.5f",
            s["bs"], s["groups"], s["nodes"], s["edges"], s["self_loops"], s["shared"], s["anchors"],
            s["anchors_missing"], s["anchors_untrusted"], s["anchors_no_incoming"], s["unreachable"],
            s["eligible"], s["unknown_state"], s["invalid"],
            s["zero_action_steps"], s["total_steps"], float(adv.mean()), float(adv.std()),
        )
