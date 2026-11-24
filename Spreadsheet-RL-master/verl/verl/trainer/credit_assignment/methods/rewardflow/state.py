# Transition metadata 规范化 (rollout dict -> 强类型) 与 state identity 判定.
# unknown state (None) 必须与 "potential 0 的已知 state" 严格区分, 且不得互相合并.

from typing import Any, Optional

_MAX_REASON_LEN = 128


def normalize_state_id(raw: Any) -> Optional[str]:
    if isinstance(raw, str) and raw.startswith("sha256:") and len(raw) > 7:
        return raw
    return None  # None/空/非法 -> unknown


def _as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    # object array 经 Ray/msgpack 后可能是 numpy bool / int.
    try:
        return bool(raw)
    except Exception:
        return False


def _as_reason(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    return str(raw)[:_MAX_REASON_LEN]


def normalize_transition(raw: Any) -> dict[str, Any]:
    """返回 {"pre","post","executed","status","graph_valid","reason","tool_status"}."""
    if not isinstance(raw, dict):
        return {
            "pre": None,
            "post": None,
            "executed": False,
            "status": "malformed",
            "graph_valid": False,
            "reason": "malformed_transition",
            "tool_status": None,
        }
    return {
        "pre": normalize_state_id(raw.get("pre")),
        "post": normalize_state_id(raw.get("post")),
        "executed": _as_bool(raw.get("executed")),
        "status": str(raw.get("status") or "unknown"),
        "graph_valid": _as_bool(raw.get("graph_valid")),
        "reason": _as_reason(raw.get("reason")),
        "tool_status": raw.get("tool_status"),
    }


def is_graph_edge(tr: dict[str, Any]) -> bool:
    """可入 graph 的 edge: graph_valid 且两端 state 均 known (unknown 拒绝合并/传播)."""
    return bool(tr["graph_valid"]) and tr["pre"] is not None and tr["post"] is not None
