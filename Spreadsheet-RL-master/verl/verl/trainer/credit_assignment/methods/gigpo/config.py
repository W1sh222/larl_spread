# GiGPO 专用参数与默认值. 含义/默认值对齐原论文实现 (arXiv:2505.10978).
#
# YAML 位置 (单一 parser 入口, 与本 dataclass 字段一一对应):
#   algorithm.credit_assignment.method = "gigpo"
#   algorithm.credit_assignment.params.<field> = ...   # adv_estimator=credit_assignment 时生效
# policy loss 侧开关不存在: 本模块只产出 token-level advantage, 优化仍由现有 vanilla PPO 负责.

from dataclasses import dataclass
from typing import Any

# 支持的 grouping 模式. similarity/embedding 版本次不实现, 传入即报错.
_SUPPORTED_GROUPING = {"exact"}
# 支持的 anchor 类型. "state" = pre-action workbook fingerprint (论文 §4.2 字面含义),
# "observation" = tool 观测文本 fingerprint (既有实现, 保留用于 A/B 对比).
_SUPPORTED_ANCHOR = {"state", "observation"}
# anchor 的状态投影范围. 仅实现 "full" (整本 workbook 内容指纹);
# "answer_range" / "observation_projection" 为后续实验选项, 传入即报错 (不静默降级).
_SUPPORTED_ANCHOR_SCOPE = {"full"}


@dataclass
class GiGPOConfig:
    gamma: float = 0.95  # step discounted return 折扣因子 (Eq.5)
    step_advantage_w: float = 1.0  # step advantage 组合权重 (Eq.8)
    mode: str = "mean_norm"  # "mean_norm" (去 std) 或 "mean_std_norm" (标准归一化)
    grouping: str = "exact"  # anchor 分组方式
    epsilon: float = 1e-6  # 防止除零
    anchor: str = "state"  # step anchor: "state" | "observation"
    anchor_scope: str = "full"  # state anchor 的状态投影范围 (当前仅实现 "full")
    # 是否把被截断的 step (assistant action 撞上 response 上限) 计入 Eq.7 baseline 分组.
    # 默认 True = 不擅自改变现有 Credit Assignment 计算.
    # policy loss 侧没有任何开关: 按 GiGPO 官方实现, step advantage 广播到 token 后
    # 直接交给现有 token-level vanilla PPO。
    include_truncated_in_baseline: bool = True

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
        if cfg.anchor not in _SUPPORTED_ANCHOR:
            raise ValueError(f"GiGPO anchor '{cfg.anchor}' must be one of {sorted(_SUPPORTED_ANCHOR)}")
        if cfg.anchor_scope not in _SUPPORTED_ANCHOR_SCOPE:
            raise ValueError(
                f"GiGPO anchor_scope '{cfg.anchor_scope}' is not implemented; "
                f"supported: {sorted(_SUPPORTED_ANCHOR_SCOPE)} "
                "('answer_range'/'observation_projection' are planned experiment options)"
            )
        if not (0.0 < cfg.gamma <= 1.0):
            raise ValueError(f"GiGPO gamma must be in (0, 1]; got {cfg.gamma}")
        return cfg

    @property
    def remove_std(self) -> bool:
        # 与原实现一致: "mean_norm" -> remove_std=True.
        return self.mode == "mean_norm"
