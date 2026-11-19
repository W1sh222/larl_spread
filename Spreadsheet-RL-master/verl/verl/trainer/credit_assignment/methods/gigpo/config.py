# GiGPO 专用参数与默认值. 含义/默认值对齐原论文实现 (arXiv:2505.10978).

from dataclasses import dataclass
from typing import Any

# 支持的 grouping 模式. similarity/embedding 版本次不实现, 传入即报错.
_SUPPORTED_GROUPING = {"exact"}


@dataclass
class GiGPOConfig:
    gamma: float = 0.95  # step discounted return 折扣因子 (Eq.5)
    step_advantage_w: float = 1.0  # step advantage 组合权重 (Eq.8)
    mode: str = "mean_norm"  # "mean_norm" (去 std) 或 "mean_std_norm" (标准归一化)
    grouping: str = "exact"  # anchor 分组方式
    epsilon: float = 1e-6  # 防止除零

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "GiGPOConfig":
        params = dict(params or {})
        unknown = set(params) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"GiGPO received unknown params {sorted(unknown)}; "
                f"supported: {sorted(cls.__dataclass_fields__)}"
            )
        cfg = cls(**params)
        if cfg.mode not in ("mean_norm", "mean_std_norm"):
            raise ValueError(f"GiGPO unknown mode '{cfg.mode}' (expected 'mean_norm' or 'mean_std_norm')")
        if cfg.grouping not in _SUPPORTED_GROUPING:
            raise ValueError(
                f"GiGPO grouping '{cfg.grouping}' is not implemented in this version; "
                f"supported: {sorted(_SUPPORTED_GROUPING)}"
            )
        if not (0.0 < cfg.gamma <= 1.0):
            raise ValueError(f"GiGPO gamma must be in (0, 1]; got {cfg.gamma}")
        return cfg

    @property
    def remove_std(self) -> bool:
        # 与原实现一致: "mean_norm" -> remove_std=True.
        return self.mode == "mean_norm"
