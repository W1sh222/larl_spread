# Credit assignment 组内归一化公用工具 (GiGPO / RewardFlow 共享).
#
# 抽取自两算法此前功能等价的 _normalize_within_groups 实现; 保留原有数值语义:
#   - float64 NumPy 计算
#   - 组内 mean (可选 std) 归一化
#   - sample std (ddof=1) + epsilon 防除零
#   - singleton 组两种策略 (step: baseline=自身 -> advantage 0; episode: baseline=0 -> 保留原值)
# 调用方各自的 grouping / filtering 政策不在此 (由 method 传入 values/group_keys 前自行处理).

from collections import defaultdict
from typing import Any

import numpy as np


def normalize_within_groups(
    values: np.ndarray,
    group_keys: list[Any],
    remove_std: bool,
    epsilon: float,
    singleton_as_zero: bool,
) -> np.ndarray:
    """组内 mean(/std) 归一化 (对齐原 GiGPO / RewardFlow episode_norm/step_norm 行为).

    values: (N,) float-like 数组.
    group_keys: 长度 N 的分组键 (与 values 位置一一对应).
    remove_std=True: 仅减 mean; 否则 (v - mean) / (std + epsilon).
    singleton_as_zero=True: 单元素组 baseline 取自身 -> advantage 0 (step-level 策略).
    singleton_as_zero=False: 单元素组 baseline 取 0 -> advantage 保留原值 (episode-level 策略).
    """
    values = np.asarray(values, dtype=np.float64)
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
            # 与原实现一致: 无偏 (sample) std, 与 torch.std 对齐.
            std = float(np.std(arr, ddof=1))
        for pos in positions:
            if remove_std:
                out[pos] = values[pos] - mean
            else:
                out[pos] = (values[pos] - mean) / (std + epsilon)
    return out
