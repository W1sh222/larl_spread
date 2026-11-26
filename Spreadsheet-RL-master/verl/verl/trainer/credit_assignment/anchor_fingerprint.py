# 算法中立的 anchor fingerprint (INITIAL / OBSERVED / INVALID 三态).
#
# 抽取自 methods/gigpo/grouping.py: state_meta 与各 method 都要用, 公共层不能
# 反向依赖某一具体 method 的实现.

import hashlib
import json
from typing import Any, Optional

from verl.trainer.credit_assignment.interface import INITIAL_STATE_MARKER

# 三态 anchor fingerprint:
# - INITIAL: rollout 显式记录的初始状态 (同 uid 内可安全分组, 环境保证初始 workbook 相同).
# - OBSERVED: 有效 pre-action 观测 -> sha256 fingerprint.
# - INVALID: 观测缺失/为空 -> 不与其他 step 分组, step advantage 置 0 (不当作初始状态).
INITIAL_FINGERPRINT = "ca::initial"
INVALID_FINGERPRINT = "ca::invalid"


def _strip_nondeterministic(obj: Any) -> Any:
    """递归剔除已知非确定性字段, 避免同一状态被判为不同 fingerprint.

    只做保守剔除: 时间戳/请求 id/临时 workspace id 等标识符.
    不删除能区分真实 workbook 状态的内容 (cell value / formula / sheet 名).
    """
    _VOLATILE_KEYS = {
        "timestamp", "time", "elapsed", "duration", "request_id", "req_id",
        "trace_id", "session_id", "workspace_id", "sandbox_workspace_id",
    }
    if isinstance(obj, dict):
        return {k: _strip_nondeterministic(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_nondeterministic(v) for v in obj]
    return obj


def _normalize_obs_text(text: str) -> str:
    """单条观测文本的确定性规范化: JSON -> 去 volatile key + key 排序 + 紧凑序列化; 否则折叠空白."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if parsed is not None:
        return json.dumps(_strip_nondeterministic(parsed), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return " ".join(text.split())


def _is_structured_obs(parsed: Any) -> bool:
    return (
        isinstance(parsed, list)
        and len(parsed) > 0
        and all(isinstance(it, dict) and "tool" in it and "text" in it for it in parsed)
    )


def fingerprint_anchor(raw: Optional[str]) -> str:
    """把算法中立的 pre-action 观测转成确定性 fingerprint (OBSERVED), 或返回 INITIAL/INVALID 哨兵.

    结构化观测 (rollout 写入的 [{"tool": ..., "text": ...}, ...] JSON): 逐条独立规范化后
    保留 tool 名与 call 顺序 (serial write 的顺序影响后继状态, 不得排序), 避免:
    - 拼接后不再是合法 JSON 导致规范化失效;
    - 不同 tool 结果拼接后文本碰巧相同;
    - 单条内嵌 JSON 的 key 顺序/空白/volatile 字段引起的误分裂.
    非结构化观测: 整体走 _normalize_obs_text 兼容路径.
    """
    if raw is None:
        return INVALID_FINGERPRINT  # 字段缺失: 不得静默当作初始状态
    if raw == INITIAL_STATE_MARKER:
        return INITIAL_FINGERPRINT

    text = raw.strip()
    if not text:
        return INVALID_FINGERPRINT  # 空观测: 无信息, 不参与分组

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None

    if _is_structured_obs(parsed):
        # 保留顺序; tool 名纳入 key, 区分 "同内容不同 tool" 的观测.
        normalized = [{"tool": str(it["tool"]), "text": _normalize_obs_text(str(it["text"]))} for it in parsed]
        key = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    else:
        key = _normalize_obs_text(text)

    return hashlib.sha256(key.encode("utf-8")).hexdigest()