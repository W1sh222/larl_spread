# RewardFlow 专用参数与默认值. 论文: arXiv:2603.18859; 官方实现: github.com/tmlr-group/RewardFlow.

import math
from dataclasses import dataclass
from typing import Any

_SUPPORTED_NORMALIZATION = {"mean_norm", "mean_std_norm"}


@dataclass
class RewardFlowConfig:
    gamma: float = 0.95  # potential 折扣因子, Phi = gamma^d (论文 Eq.2; 官方脚本用 0.9, 论文附录用 0.95)
    alpha_act: float = 1.0  # action advantage 权重 (论文 Eq.6; ablation 最优 (1,1))
    alpha_traj: float = 1.0  # trajectory (episode) advantage 权重 (论文 Eq.6)
    # 官方代码默认 mean_norm (step_norm_reward remove_std=True, 不除 std);
    # mean_std_norm 对应论文 Eq.4 的 sigma 归一化.
    normalization: str = "mean_norm"
    epsilon: float = 1e-6
    # 当前 reward 为 binary {0,1}: success := r >= success_threshold - tol.
    # 不做 0.5 之类的中间阈值 (0/1 之间不存在取值); reward 0 不视为确定失败, 仅作
    # "非 anchor" 处理, 不产生 negative anchor.
    success_threshold: float = 1.0
    success_tolerance: float = 1e-6

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "RewardFlowConfig":
        params = dict(params or {})
        unknown = set(params) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"RewardFlow received unknown params {sorted(unknown)}; "
                f"supported: {sorted(cls.__dataclass_fields__)}"
            )
        cfg = cls(**params)
        if not (0.0 < cfg.gamma <= 1.0):
            raise ValueError(f"RewardFlow gamma must be in (0, 1]; got {cfg.gamma}")
        if cfg.normalization not in _SUPPORTED_NORMALIZATION:
            raise ValueError(
                f"RewardFlow normalization '{cfg.normalization}' not supported; "
                f"supported: {sorted(_SUPPORTED_NORMALIZATION)}"
            )
        for name in ("alpha_act", "alpha_traj"):
            v = getattr(cfg, name)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"RewardFlow {name} must be finite and >= 0; got {v}")
        if not (0.0 < cfg.success_threshold <= 1.0):
            raise ValueError(f"RewardFlow success_threshold must be in (0, 1]; got {cfg.success_threshold}")
        if cfg.epsilon <= 0.0:
            raise ValueError(f"RewardFlow epsilon must be > 0; got {cfg.epsilon}")
        return cfg

    @property
    def remove_std(self) -> bool:
        return self.normalization == "mean_norm"
