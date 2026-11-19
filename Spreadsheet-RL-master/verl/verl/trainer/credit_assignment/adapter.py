# verl batch/non_tensor_batch <-> CreditAssignmentInput 之间的薄 adapter.
# 只有本模块感知 verl 数据结构; 各 method 算法本身不依赖 DataProto.

from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.credit_assignment.interface import CreditAssignmentInput, CreditAssignmentOutput
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

    return CreditAssignmentInput(
        episode_group_ids=np.asarray(index, dtype=object),
        trajectory_rewards=trajectory_rewards,
        assistant_mask=response_mask,
        step_spans=step_spans,
        step_observations=step_observations,
        step_rewards=None,
        extra=extra,
    )


def compute(
    method: str,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    non_tensor_batch: dict,
    params: dict[str, Any],
) -> CreditAssignmentOutput:
    cls = get_credit_assignment_class(method)
    inputs = build_inputs(
        token_level_rewards,
        response_mask,
        index,
        non_tensor_batch,
        requires=getattr(cls, "requires", frozenset()),
        method=method,
    )
    return _registry_run(method, inputs, params)
