# verl batch/non_tensor_batch <-> CreditAssignmentInput 之间的薄 adapter.
# 只有本模块感知 verl 数据结构; 各 method 算法本身不依赖 DataProto.

from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.credit_assignment.interface import (
    CreditAssignmentInput,
    CreditAssignmentOutput,
    EVAL_FINGERPRINT_REQ,
)
from verl.trainer.credit_assignment.registry import get_credit_assignment_class
from verl.trainer.credit_assignment.registry import run as _registry_run

# rollout 层 (tool_agent) 写入 extra_fields -> non_tensor_batch 的算法中立 key.
STEP_SPANS_KEY = "ca_step_spans"
STEP_OBS_KEY = "ca_step_observations"
# structured workbook-state transition metadata (仅 collect_step_states 启用时存在).
STEP_TRANSITIONS_KEY = "ca_step_transitions"
INITIAL_STATE_KEY = "ca_initial_state"
FINAL_STATE_KEY = "ca_final_state"
TERMINAL_REASON_KEY = "ca_terminal_reason"
STATE_STATS_KEY = "ca_state_stats"
TERMINAL_TRUSTED_KEY = "ca_terminal_trusted"
# terminal 一致性诊断 (仅用于 metrics): 终端 fingerprint 与被评估 artifact 是否一致 /
# 本 trajectory 是否含未解决的 external writeback. rollout 侧写入, 不影响采集开关.
TERMINAL_ARTIFACT_CONSISTENT_KEY = "ca_terminal_artifact_consistent"
UNRESOLVED_WRITEBACK_KEY = "ca_unresolved_writeback"
# 评估侧回传的 "实际提交 bytes 的 canonical fingerprint" (reward 路径的 generic
# reward_extra_info -> non_tensor_batch 落入; 与 sheet_arena.EVAL_FINGERPRINT_KEY 同名).
# 可选: 缺失/行内 None -> RewardFlow 不锁定 terminal anchor (不 raise).
EVAL_FINGERPRINT_KEY = "eval_workbook_fingerprint"

# CreditAssignmentInput 字段 -> non_tensor_batch key (仅 step 级可选字段需要映射).
_FIELD_TO_KEY = {"step_spans": STEP_SPANS_KEY, "step_observations": STEP_OBS_KEY}
# 经 inputs.extra 透传的 requires 字段 (method 自行解释 schema).
_EXTRA_FIELD_TO_KEY = {"step_transitions": STEP_TRANSITIONS_KEY, "terminal_trusted": TERMINAL_TRUSTED_KEY}


def _row_list(obj: Any, row: int, length: int) -> Optional[list]:
    val = obj[row] if obj is not None and len(obj) == length else None
    if val is None:
        return None
    # object array 元素可能是 numpy list / 原生 list.
    return list(val)


def build_inputs(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    non_tensor_batch: dict,
    requires: frozenset = frozenset(),
    method: str = "",
    raw_token_level_scores: Optional[torch.Tensor] = None,
) -> CreditAssignmentInput:
    """抽取通用字段; step 级字段按需抽取, 仅对 method.requires 声明的字段强校验.

    新 method 若无 step 需求 (requires 为空), rollout 未提供 metadata 也能运行;
    需要的字段缺失则明确报错, 不静默退化为错误的 advantage.
    """
    bs = token_level_rewards.shape[0]

    spans_arr = non_tensor_batch.get(STEP_SPANS_KEY)
    obs_arr = non_tensor_batch.get(STEP_OBS_KEY)

    step_spans = None if spans_arr is None else [_row_list(spans_arr, i, bs) or [] for i in range(bs)]
    step_observations = None if obs_arr is None else [_row_list(obs_arr, i, bs) or [] for i in range(bs)]

    # structured state metadata: 保留 None 行 (与 "空列表" 区分 -> 缺失可被感知).
    trans_arr = non_tensor_batch.get(STEP_TRANSITIONS_KEY)
    step_transitions = None if trans_arr is None else [_row_list(trans_arr, i, bs) for i in range(bs)]

    def _scalar_rows(arr: Any) -> Optional[list]:
        if arr is None or len(arr) != bs:
            return None
        return [arr[i] if arr[i] is not None else None for i in range(bs)]

    extra: dict[str, Any] = {}
    if step_transitions is not None:
        extra["step_transitions"] = step_transitions
    for extra_name, key_name in (
        ("initial_states", INITIAL_STATE_KEY),
        ("final_states", FINAL_STATE_KEY),
        ("terminal_trusted", TERMINAL_TRUSTED_KEY),
        ("terminal_reasons", TERMINAL_REASON_KEY),
        ("state_stats", STATE_STATS_KEY),
        ("terminal_artifact_consistent", TERMINAL_ARTIFACT_CONSISTENT_KEY),
        ("unresolved_writeback", UNRESOLVED_WRITEBACK_KEY),
        ("eval_fingerprints", EVAL_FINGERPRINT_KEY),
    ):
        rows = _scalar_rows(non_tensor_batch.get(key_name))
        if rows is not None:
            extra[extra_name] = rows

    # 按 method 声明的 requires 校验: 缺失则明确报错 (不静默退化为错误 advantage).
    field_values = {
        "step_spans": step_spans,
        "step_observations": step_observations,
        "step_transitions": step_transitions,
        "terminal_trusted": extra.get("terminal_trusted"),
    }
    requires_map = {**_FIELD_TO_KEY, **_EXTRA_FIELD_TO_KEY}
    # §2: 区分 "supported-but-missing" 与 "unsupported requirement name".
    # 算法声明了 adapter 无法提供的 requirement 名 (如拼写错误) -> 明确报错, 不静默跳过.
    # eval_fingerprint 是已识别的 SOFT requirement: 通过 extra["eval_fingerprints"] surfacing,
    # 但刻意不纳入下方 hard-missing 校验 (requires_map ∩ requires) ->
    # 逐行 fingerprint 缺失/None 不 raise, 由 RewardFlow 侧 fail-closed 排除 anchor.
    supported = set(requires_map) | {EVAL_FINGERPRINT_REQ}
    unknown_req = [f for f in sorted(requires) if f not in supported]
    if unknown_req:
        raise ValueError(
            f"credit assignment method '{method}' declares unsupported metadata requirement(s) "
            f"{unknown_req}; adapter-supported requirement names are {sorted(supported)}. "
            "Unknown/typo'd requirement names are rejected rather than silently ignored. To consume "
            "rollout metadata, reuse an existing supported key (via the generic non_tensor_batch "
            "plumbing); do not introduce algorithm-specific requirement names."
        )
    missing = [requires_map[f] for f in sorted(requires) if f in requires_map and field_values.get(f) is None]
    if missing:
        raise ValueError(
            f"credit assignment method '{method}' requires metadata {missing} in non_tensor_batch, "
            f"but it is absent. Ensure rollout uses default_agent_loop=tool_agent "
            "(which emits ca_step_* fields via extra_fields) and that "
            "algorithm.credit_assignment.collect_step_states is enabled when required."
        )

    if step_spans is not None and step_observations is not None:
        # 两者同时存在时校验 per-row 数量对齐, 避免 span/observation 错位.
        for i in range(bs):
            if len(step_spans[i]) != len(step_observations[i]):
                raise ValueError(
                    f"row {i}: {STEP_SPANS_KEY} ({len(step_spans[i])}) and {STEP_OBS_KEY} "
                    f"({len(step_observations[i])}) length mismatch"
                )

    trajectory_rewards = token_level_rewards.sum(dim=-1).detach().cpu().numpy()

    # raw environment outcome: 从 token_level_scores 派生, 仅在提供时计算; shape 必须与
    # token_level_rewards 一致 (否则行对齐会静默错位). 提供时验证, 不提供保持 None (由 method 自判).
    raw_trajectory_rewards = None
    if raw_token_level_scores is not None:
        if raw_token_level_scores.shape != token_level_rewards.shape:
            raise ValueError(
                f"credit assignment method '{method}': raw_token_level_scores shape "
                f"{tuple(raw_token_level_scores.shape)} != token_level_rewards shape "
                f"{tuple(token_level_rewards.shape)}"
            )
        raw_trajectory_rewards = raw_token_level_scores.sum(dim=-1).detach().cpu().numpy()

    return CreditAssignmentInput(
        episode_group_ids=np.asarray(index, dtype=object),
        trajectory_rewards=trajectory_rewards,
        assistant_mask=response_mask,
        step_spans=step_spans,
        step_observations=step_observations,
        step_rewards=None,
        raw_trajectory_rewards=raw_trajectory_rewards,
        extra=extra,
    )


def compute(
    method: str,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    non_tensor_batch: dict,
    params: dict[str, Any],
    raw_token_level_scores: Optional[torch.Tensor] = None,
) -> CreditAssignmentOutput:
    cls = get_credit_assignment_class(method)
    # 与 rollout 采集端使用同一动态需求解析器 (config-dependent), 避免静态校验漂移.
    inputs = build_inputs(
        token_level_rewards,
        response_mask,
        index,
        non_tensor_batch,
        requires=cls.required_metadata(params),
        method=method,
        raw_token_level_scores=raw_token_level_scores,
    )
    return _registry_run(method, inputs, params)
