# GRPO wrapper 参数. 只暴露现有 vanilla GRPO 实现真正支持的开关.

from dataclasses import dataclass
from typing import Any


@dataclass
class GRPOConfig:
    # 与 algorithm.norm_adv_by_std_in_grpo 同义 (True=原 GRPO, False=Dr.GRPO).
    norm_adv_by_std_in_grpo: bool = True
    epsilon: float = 1e-6

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "GRPOConfig":
        params = dict(params or {})
        unknown = set(params) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"GRPO received unknown params {sorted(unknown)}; supported: {sorted(cls.__dataclass_fields__)}"
            )
        flag = params.get("norm_adv_by_std_in_grpo", True)
        if not isinstance(flag, bool):
            raise ValueError(f"GRPO norm_adv_by_std_in_grpo must be a bool; got {flag!r}")
        eps = params.get("epsilon", 1e-6)
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise ValueError(f"GRPO epsilon must be a number; got {eps!r}")
        if not (0.0 < float(eps) <= 1e-1):
            raise ValueError(f"GRPO epsilon must be in (0, 0.1]; got {eps}")
        return cls(norm_adv_by_std_in_grpo=flag, epsilon=float(eps))
