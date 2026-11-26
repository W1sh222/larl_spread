# 算法中立的 rollout step-state metadata (`ca_step_transitions`) 读取与分类.
#
# 使用者是 credit assignment method: step anchor 分组、truncation/unexecuted 诊断,
# 以及 baseline 分组是否计入被截断的 step. policy loss 侧 **不** 依赖本模块
# (GiGPO 官方实现只用 token-level PPO, step advantage 由本模块的使用方广播到 token).
#
# Schema (emit 侧: verl/experimental/agent_loop/tool_agent_loop.py):
#   transitions[k] = {"pre", "post", "executed", "status", "graph_valid", "reason",
#                     "tool_status": [{"tool", "status"}, ...]}
# status 词表:
#   "pending"                 -> 仅写了 placeholder (只有 loop 异常中断时才可能残留)
#   "executed"                -> tool 已执行, post = fingerprint 或 None(采集失败)
#   "final_answer"            -> 无 tool call 的收尾回答; action 完整但不构成 state edge
#   "terminated_before_tools" -> action 已生成但 tool 未执行 (长度/轮次上限)

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

from verl.trainer.credit_assignment.methods.gigpo.grouping import (
    INITIAL_FINGERPRINT,
    INVALID_FINGERPRINT,
    fingerprint_anchor,
)

STEP_STATUS_PENDING = "pending"
STEP_STATUS_EXECUTED = "executed"
STEP_STATUS_FINAL_ANSWER = "final_answer"
STEP_STATUS_UNEXECUTED = "terminated_before_tools"

# 截断分类 (优先级: partial_action > unexecuted > none)
TRUNC_NONE = "none"
TRUNC_PARTIAL_ACTION = "partial_action"
TRUNC_UNEXECUTED = "unexecuted"

ANCHOR_MODE_STATE = "state"
ANCHOR_MODE_OBSERVATION = "observation"


@dataclass(frozen=True)
class StepMeta:
    """单个 step (=一次 assistant decision) 的状态/有效性判定结果.

    anchor_fp: 按 anchor_mode 选定的分组键 (见 resolve_anchor).
    state_fp / observation_fp: 两种 anchor 的原始值都保留, 便于同一份数据做
        anchor A/B 对照而无需重跑 rollout.
    truncated: assistant action 的 token span 是否被 response 上限截断.
        被截断的 action 不对应任何"被完整采样过的动作", 因此 Eq.7 baseline 分组
        可以选择排除它 (include_truncated_in_baseline, 默认 True = 仍计入).
    executed: tool 是否执行. 与"是否被截断"无关, 且不影响 anchor 分组.
    graph_valid: 该 step 能否作为可信 state edge (外部 writeback 时序不可证时为
        False). False **不等于** 该 decision 不该拿到 advantage: 可信度与分组是
        两个独立问题, 本模块只记录事实, 由 method 各自决定如何用.
    """

    index: int
    pre: Optional[str]
    post: Optional[str]
    status: str
    executed: bool
    graph_valid: bool
    reason: Optional[str]
    span: tuple[int, int]
    span_visible: tuple[int, int]
    truncated: bool
    anchor_fp: str
    anchor_kind: str  # "state" | "observation" | "initial" | "invalid"
    state_fp: Optional[str] = None
    observation_fp: str = INVALID_FINGERPRINT
    terminal_reason: Optional[str] = None

    @property
    def num_tokens(self) -> int:
        """截断后仍可见的 assistant token 数 (诊断用平均 step 长度)."""
        return max(0, self.span_visible[1] - self.span_visible[0])

    @property
    def is_anchor_valid(self) -> bool:
        return self.anchor_fp != INVALID_FINGERPRINT

    @property
    def span_clipped(self) -> bool:
        """recorded span 与 padded response 坐标系下的可见 span 是否不同."""
        return self.span != self.span_visible


def _as_pair(span: Any) -> tuple[int, int]:
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return 0, 0
    return start, end


def _clamp_span(span: tuple[int, int], response_length: int) -> tuple[int, int]:
    start = max(0, min(span[0], response_length))
    end = max(start, min(span[1], response_length))
    return start, end


def clamp_response_span(span: Any, response_length: int) -> tuple[int, int]:
    """公开工具: 将 [start, end) 解析并夹到响应轴合法范围 (不位移坐标).

    与内部 build_step_meta 使用同一套 [start,end)/clamp 语义; 供各 method 的 token
    broadcast 循环复用 (GiGPO / RewardFlow), 避免平行实现漂移. 返回的 (start, end)
    满足 0 <= start <= end <= response_length; end <= start 表示无可见 token (应跳过).
    malformed span (非两元/不可转 int) 归为 (0, 0) -> 自然跳过.
    """
    return _clamp_span(_as_pair(span), response_length)


def classify_truncation(span: tuple[int, int], status: str, response_length: int) -> str:
    """区分 "action 被截断" / "tool 未执行" / 正常 (两者可同时发生, 前者优先)."""
    if span[1] > response_length:
        return TRUNC_PARTIAL_ACTION
    if status == STEP_STATUS_UNEXECUTED:
        return TRUNC_UNEXECUTED
    return TRUNC_NONE


def resolve_anchor(meta: StepMeta, mode: str) -> tuple[str, str]:
    """按 anchor mode 返回 (anchor_fp, anchor_kind). 两种 mode 绝不混合.

    state anchor 的 `pre` 是"该 action 生成之前"的状态 (tool_agent_loop 写
    transition placeholder 时刻的 `_cur_state`), 因此是正确的 decision context.
    但状态未知时保持 INVALID: 拿"不知道的状态"去冒充初始状态或观测 fingerprint,
    会把不同真实环境状态混进同一 anchor 组 (under-split), 污染 state anchor 实验.

    - "state": 只用 pre-action workbook fingerprint; 未知 -> INVALID.
    - "observation": 只用 tool 观测 fingerprint (既有实现); INITIAL 哨兵仅在
      rollout 显式写了 step 0 marker 时出现.
    """
    if mode == ANCHOR_MODE_STATE:
        if meta.state_fp:
            return meta.state_fp, "state"
        return INVALID_FINGERPRINT, "invalid"
    if mode == ANCHOR_MODE_OBSERVATION:
        return meta.observation_fp, _observation_kind(meta.observation_fp)
    raise ValueError(f"unknown anchor mode '{mode}' (expected 'state' or 'observation')")


def _observation_kind(fp: str) -> str:
    if fp == INVALID_FINGERPRINT:
        return "invalid"
    if fp == INITIAL_FINGERPRINT:
        return "initial"
    return "observation"


def build_step_meta(
    *,
    spans: Sequence[Any],
    response_length: int,
    anchor_mode: str = ANCHOR_MODE_STATE,
    transitions_row: Optional[Sequence[dict]] = None,
    observations_row: Optional[Sequence[Any]] = None,
    initial_state: Optional[str] = None,
    terminal_reason: Optional[str] = None,
) -> list[StepMeta]:
    """为一条 trajectory 的所有 step 生成状态/有效性判定.

    state 与 observation 两种 fingerprint 都会算出来, 仅 anchor_fp 随 anchor_mode
    切换 -> 同一批 rollout 可低成本对比 anchor.
    """
    metas: list[StepMeta] = []
    for k in range(len(spans)):
        span = _as_pair(spans[k])
        visible = _clamp_span(span, response_length)
        tr = transitions_row[k] if transitions_row is not None and k < len(transitions_row) else None
        if isinstance(tr, dict):
            pre = tr.get("pre")
            post = tr.get("post")
            status = str(tr.get("status", STEP_STATUS_PENDING))
            executed = bool(tr.get("executed", False))
            graph_valid = bool(tr.get("graph_valid", False))
            reason = tr.get("reason")
        else:
            pre = post = None
            status = STEP_STATUS_PENDING
            executed = False
            graph_valid = False
            reason = "transition_missing"

        # step 0 缺 pre 时用 ca_initial_state 回填 (二者本应相同); 仍缺则保持 None.
        state_fp = str(pre) if pre else (str(initial_state) if (k == 0 and initial_state) else None)
        raw_obs = observations_row[k] if observations_row is not None and k < len(observations_row) else None
        observation_fp = fingerprint_anchor(raw_obs)
        truncation = classify_truncation(span, status, response_length)

        meta = StepMeta(
            index=k,
            pre=state_fp,
            post=post,
            status=status,
            executed=executed,
            graph_valid=graph_valid,
            reason=reason,
            span=span,
            span_visible=visible,
            truncated=truncation == TRUNC_PARTIAL_ACTION,
            anchor_fp=INVALID_FINGERPRINT,
            anchor_kind="invalid",
            state_fp=state_fp,
            observation_fp=observation_fp,
            terminal_reason=terminal_reason,
        )
        anchor_fp, anchor_kind = resolve_anchor(meta, anchor_mode)
        metas.append(replace(meta, anchor_fp=anchor_fp, anchor_kind=anchor_kind))
    return metas


def meta_diagnostics(rows: Sequence[Sequence[StepMeta]]) -> dict[str, float]:
    """anchor 类型 / step 有效性分布诊断 (只观测, 不影响任何训练数值)."""
    flat = [m for row in rows for m in row]
    n = len(flat)
    if n == 0:
        return {}

    def frac(pred) -> float:
        return sum(1 for m in flat if pred(m)) / n

    out = {
        "steps_per_traj": n / float(len(list(rows))),
        "anchor_state_frac": frac(lambda m: m.anchor_kind == "state"),
        "anchor_observation_frac": frac(lambda m: m.anchor_kind in ("observation", "initial")),
        "anchor_invalid_frac": frac(lambda m: m.anchor_kind == "invalid"),
        "truncated_step_frac": frac(lambda m: m.truncated),
        "unexecuted_step_frac": frac(lambda m: m.status == STEP_STATUS_UNEXECUTED),
        "final_answer_step_frac": frac(lambda m: m.status == STEP_STATUS_FINAL_ANSWER),
        "graph_invalid_step_frac": frac(lambda m: not m.graph_valid),
        "empty_span_step_frac": frac(lambda m: m.num_tokens == 0),
        "mean_step_tokens": sum(m.num_tokens for m in flat) / n,
        "n_anchor_distinct": float(len({m.anchor_fp for m in flat})),
    }
    for reason in sorted({str(m.reason) for m in flat if m.reason}):
        out[f"reason_{reason}"] = frac(lambda m, r=reason: m.reason == r)
    return out
