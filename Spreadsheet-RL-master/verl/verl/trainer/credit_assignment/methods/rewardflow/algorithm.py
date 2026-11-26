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
from collections import Counter, defaultdict
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
from verl.trainer.credit_assignment.normalization import normalize_within_groups
from verl.trainer.credit_assignment.registry import register_credit_assignment
from verl.trainer.credit_assignment.state_meta import clamp_response_span

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_UNKNOWN_INITIAL = "<state_unknown>"


@register_credit_assignment("rewardflow")
class RewardFlow(CreditAssignmentMethod):
    name = "rewardflow"
    # RewardFlow 不消费 step_observations (仅用 pre/post transition 指纹), 因此不声明它.
    # "eval_fingerprint" 是 SOFT requirement: success anchor 要求实际提交 bytes 的 canonical
    # fingerprint 与 terminal fingerprint 一致. rollout 侧据此对每个样本写入 emit 标记,
    # reward 侧按需计算; adapter 不将其作为 hard-missing 强制 (允许逐行 None -> 仅该
    # anchor fail-closed 排除, 整体 reward 继续计算).
    requires = frozenset({"step_spans", "step_transitions", "terminal_trusted", "eval_fingerprint"})

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
        # raw environment outcome (token_level_scores.sum(-1)): success anchor 与 binary 诊断必须
        # 用它, 而非 KL 塑形后的训练 reward (use_kl_in_reward=True 时二者不同). 不回退到塑形分.
        if inputs.raw_trajectory_rewards is None:
            raise ValueError(
                "RewardFlow requires raw environment outcomes (raw_trajectory_rewards from "
                "token_level_scores) to select success anchors, but they are absent. Ensure the "
                "trainer forwards batch['token_level_scores'] through credit_assignment.compute. "
                "Silent fallback to KL-shaped rewards is intentionally not allowed."
            )
        raw_rewards = np.asarray(inputs.raw_trajectory_rewards, dtype=np.float64)
        if raw_rewards.shape != traj_rewards.shape:
            raise ValueError(
                f"RewardFlow raw_trajectory_rewards shape {raw_rewards.shape} != "
                f"trajectory_rewards shape {traj_rewards.shape}"
            )
        if not np.all(np.isfinite(raw_rewards)):
            raise ValueError("RewardFlow got non-finite raw trajectory rewards")
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

        self._warn_if_not_binary(raw_rewards, cfg)

        # --- Episode-level advantage (官方 episode_norm_reward: uid 组内 mean_norm). ---
        # 训练用 trajectory reward (可能 KL 塑形) 决定 trajectory-level advantage; 与
        # success anchor 用的 raw outcome 区分 (见下方 anchor 段).
        episode_adv = normalize_within_groups(
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

        # Success anchor (论文 Eq.1): 同时满足 (1) raw 环境 outcome 达 success 阈值
        # (2) terminal fingerprint known (3) terminal trusted (rollout 侧: 无未证 writeback 且
        # 实时 state 与交付 artifact 一致) (4) 实际评估 bytes 的 canonical fingerprint 存在且有效
        # (5) 评估 fingerprint == terminal fingerprint 时, 以 terminal state 为 Phi=1 锚点.
        # anchor 使用 raw outcome (非 KL 塑形训练分), 保证 use_kl_in_reward 开启时仍正确选锚.
        # graph-valid edge 存在与否 NOT 是锚定前置条件: 无 incoming edge 的孤立 anchor 仍合法,
        # 只是不向其他 state 传播 (归入 connectivity 诊断, 不凭空补馈 potential).
        # 严禁: 用 last eligible post 替代 anchor / 回退锚到更早中间 state / 把 excluded transition
        # 恢复为通向 success 的 edge / 将 eval fingerprint 作为新 graph node 插入.
        # (4)(5) 是 TOCTOU 防护: 路径/rollout 时点一致 != 被评估 bytes 一致; 缺少评估指纹或与之
        # 不符 -> 仅排除该 anchor, trajectory advantage 保留, 不回退锚到 rollout state.
        # reward 0 不作 negative anchor (可能为评估基础设施失败).
        terminal_trusted_rows = inputs.extra.get("terminal_trusted")
        eval_fps = inputs.extra.get("eval_fingerprints")
        anchors = 0
        anchors_missing = 0      # terminal fingerprint unknown (采集失败): 不回退
        anchors_untrusted = 0    # fingerprint known 但 rollout trust 不可证: 不回退
        anchors_eval_missing = 0    # 评估 bytes fingerprint 缺失/计算失败 (含未启用发射开关)
        anchors_eval_mismatch = 0   # 评估 fingerprint 与 terminal fingerprint 不一致 (TOCTOU)
        for row in range(bs):
            if raw_rewards[row] < cfg.success_threshold - cfg.success_tolerance:
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
            eval_fp = (
                normalize_state_id(eval_fps[row])
                if (eval_fps is not None and row < len(eval_fps))
                else None
            )
            if eval_fp is None:
                anchors_eval_missing += 1
                continue
            if eval_fp != final:
                anchors_eval_mismatch += 1
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
            normalized = normalize_within_groups(
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
                start, end = clamp_response_span(span, response_length)
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

        # --- 诊断量 (复用已算的 graph 统计 + 现有 counters; 不重重新计算整个 graph). ---
        per_graph_stats = [g.stats() for g in graphs.values()]
        _sum = lambda key: sum(st[key] for st in per_graph_stats)  # noqa: E731
        node_count = _sum("nodes")
        edge_count = _sum("edges")
        self_loop_count = _sum("self_loops")
        anchors_no_incoming = _sum("anchors_without_valid_incoming_edge")
        unreachable = _sum("unreachable_valid_states")
        shared = _sum("shared_in_nodes")
        group_count = len(graphs)

        # action group 分组统计 (singleton / nonzero) 复用已构建的 flat_.
        group_sizes = Counter(flat_norm_keys)
        num_action_groups = len(group_sizes)
        singleton_groups = sum(1 for sz in group_sizes.values() if sz == 1)
        num_actions = len(flat_steps)
        nonzero_action_adv = sum(1 for (row, k) in flat_steps if step_advantage.get((row, k), 0.0) != 0.0)
        nonzero_traj_adv = int(np.count_nonzero(episode_adv))
        # raw outcome 中非 binary 的个数 (用 raw environment outcome 而非 KL 塑形分).
        raw_fractional = int(np.count_nonzero((raw_rewards > 1e-6) & (np.abs(raw_rewards - 1.0) > 1e-6)))
        # rollout 侧新增诊断标志 (可能缺失 -> 默认 0).
        artifact_rows = inputs.extra.get("terminal_artifact_consistent")
        artifact_mismatch = 0
        if artifact_rows is not None:
            artifact_mismatch = sum(1 for r in artifact_rows[:bs] if r is not None and not bool(r))
        wb_rows = inputs.extra.get("unresolved_writeback")
        unresolved_writeback = 0
        if wb_rows is not None:
            unresolved_writeback = sum(1 for r in wb_rows[:bs] if bool(r))
        # 本 batch 内带有效评估 fingerprint 的 row 数 (判断发射开关是否生效).
        eval_present = 0
        if eval_fps is not None:
            eval_present = sum(1 for r in eval_fps[:bs] if normalize_state_id(r) is not None)

        denom_steps = max(1, total_steps)
        denom_actions = max(1, num_actions)
        denom_groups = max(1, num_action_groups)
        denom_bs = max(1, bs)
        denom_nodes = max(1, node_count)
        metrics = {
            "graph_count": float(group_count),
            "node_count": float(node_count),
            "edge_count": float(edge_count),
            "self_loop_count": float(self_loop_count),
            "shared_in_nodes": float(shared),
            "anchor_count": float(anchors),
            "anchors_missing": float(anchors_missing),
            "anchors_untrusted": float(anchors_untrusted),
            "anchors_eval_missing": float(anchors_eval_missing),
            "anchors_eval_mismatch": float(anchors_eval_mismatch),
            "anchors_no_incoming": float(anchors_no_incoming),
            "unknown_state_step_count": float(steps_unknown_state),
            "graph_valid_action_count": float(steps_graph_eligible),
            "invalid_action_count": float(steps_invalid),
            "unreachable_node_frac": float(unreachable) / denom_nodes,
            "singleton_action_group_frac": float(singleton_groups) / denom_groups,
            "nonzero_action_adv_frac": float(nonzero_action_adv) / denom_actions,
            "nonzero_traj_adv_frac": float(nonzero_traj_adv) / denom_bs,
            "nonzero_token_adv_frac": float(torch.count_nonzero(advantages).item()) / max(1, bs * response_length),
            "raw_fractional_frac": float(raw_fractional) / denom_bs,
            "terminal_artifact_mismatch_count": float(artifact_mismatch),
            "unresolved_writeback_count": float(unresolved_writeback),
            # 评估指纹发射开关是否生效 (整 batch 都无 eval fingerprint -> 0): 区分
            # "内容不一致" 与 "根本没回传评估 snapshot". 分母 = bs.
            "eval_fingerprint_present_frac": float(eval_present) / denom_bs,
            "zero_action_step_frac": float(zero_action_steps) / denom_steps,
        }

        self._log_summary(
            bs=bs,
            groups=group_count,
            nodes=node_count,
            edges=edge_count,
            self_loops=self_loop_count,
            shared=shared,
            anchors=anchors,
            anchors_missing=anchors_missing,
            anchors_untrusted=anchors_untrusted,
            edges_added=edges_added,
            eligible=steps_graph_eligible,
            unknown_state=steps_unknown_state,
            invalid=steps_invalid,
            anchors_no_incoming=anchors_no_incoming,
            unreachable=unreachable,
            total_steps=total_steps,
            zero_action_steps=zero_action_steps,
            advantages=advantages,
        )

        return CreditAssignmentOutput(advantages=advantages, returns=advantages, metrics=metrics)

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
