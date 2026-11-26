# Exact anchor grouping (GiGPO Eq.6).
#
# fingerprint 构造本身在公共模块 `credit_assignment/anchor_fingerprint.py`
# (state_meta 也要用, 不能反向依赖本 method).

from collections import defaultdict
from typing import Any


def build_step_groups(
    episode_group_ids,
    step_entries: list[tuple[int, str]],
) -> dict[int, str]:
    """Exact anchor grouping (Eq.6): 在同一 episode(uid) 内, 对 fingerprint 相同的 step 归为一组.

    Args:
        episode_group_ids: (bs,) uid, 索引到 trajectory.
        step_entries: list of (row_index, anchor_fingerprint), 覆盖所有 step.

    Returns:
        dict: step_position (在 step_entries 中的下标) -> group_key.
        group_key = (uid, fingerprint), 保证不跨 task 分组.
    """
    grouping: dict[int, str] = {}
    # 先按 uid 聚合再按 fingerprint 聚类, 语义等价于 (uid, fp) 直接分组, 保持与原实现一致的 "组内" 概念.
    uid_to_fp_to_steps: dict[Any, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for pos, (row, fp) in enumerate(step_entries):
        uid = episode_group_ids[row]
        uid_to_fp_to_steps[uid][fp].append(pos)

    for uid, fp_to_steps in uid_to_fp_to_steps.items():
        for fp, positions in fp_to_steps.items():
            group_key = f"{uid}::{fp}"
            for pos in positions:
                grouping[pos] = group_key

    missing = [p for p in range(len(step_entries)) if p not in grouping]
    if missing:
        raise RuntimeError(f"step grouping failed for positions: {missing[:8]}")
    return grouping
