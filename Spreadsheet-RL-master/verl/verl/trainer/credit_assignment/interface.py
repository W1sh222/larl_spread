# Credit assignment 通用输入/输出接口 (schema 定义, 与 verl DataProto 解耦).

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

# 算法中立的 initial-state marker: rollout 在 step 0 (尚无任何 tool observation 前) 显式记录.
# 与 "观测缺失/为空"(invalid) 严格区分: 后者不得被当作初始状态.
INITIAL_STATE_MARKER = "<ca_initial_state>"


@dataclass
class CreditAssignmentInput:
    """算法中立的 rollout 输入. 由 adapter 从 verl batch/non_tensor_batch 抽取.

    1 trajectory = 1 row (Option B): step 维度落在 token axis 上, 不新增 row.
    通用必需仅 uid/reward/mask; step 级字段均为 optional, 由 method.requires 声明并校验.
    """

    # (bs,) episode(=task) 分组标识, 来自 uid. 同一 task 的多条 trajectory 共享.
    episode_group_ids: np.ndarray
    # (bs,) 每条 trajectory 的 outcome reward (token_level_rewards.sum(-1)).
    trajectory_rewards: np.ndarray
    # (bs, response_length) assistant token mask (1=assistant, 0=tool obs/padding).
    assistant_mask: torch.Tensor
    # 以下为可选 step 级 metadata; method 通过 requires 声明需求, adapter 按需求校验.
    # spans[i] = [[start, end), ...] 每条 trajectory 的 assistant token span.
    step_spans: Optional[list[list[list[int]]]] = None
    # observations[i] = 每个 step 的 pre-action 观测 (算法中立原文, 如结构化 tool response);
    # step 0 为 INITIAL_STATE_MARKER, 无有效观测时为 None/空串 (method 自行判 invalid).
    step_observations: Optional[list[list[Optional[str]]]] = None
    # 每条 trajectory 每个 step 的即时 reward; None 时按 terminal-only 处理.
    step_rewards: Optional[list[list[float]]] = None
    # 原始环境 outcome (token_level_scores.sum(-1)); 与训练用 trajectory_rewards 区分.
    # KL-in-reward 开启时, 训练 reward 会被 KL 塑形, 但 success anchor 必须依据未塑形的
    # raw outcome 判定. 未提供时为 None; 具体 method 自行决定是否强制要求 (见 RewardFlow).
    raw_trajectory_rewards: Optional[np.ndarray] = None
    # 算法专用 metadata 容器, 由具体 method 解释.
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.trajectory_rewards.shape[0])

    @property
    def response_length(self) -> int:
        return int(self.assistant_mask.shape[-1])


@dataclass
class CreditAssignmentOutput:
    """标准输出: PPO 可直接使用的 token-level advantage.

    metrics 为可选诊断量 (纯 float, 前缀由调用方决定); 不参与训练数值,
    仅用于区分 anchor 类型 / step 有效性等实验对照指标.
    """

    advantages: torch.Tensor  # (bs, response_length)
    returns: torch.Tensor  # (bs, response_length)
    metrics: dict[str, float] = field(default_factory=dict)


class CreditAssignmentMethod(ABC):
    """所有 credit assignment 算法的基类.

    子类只依赖 CreditAssignmentInput, 不得访问 rollout/trainer/reward manager.
    """

    name: str = ""
    # 本 method 计算所必需的输入字段名 (如 {"step_spans", "step_observations"});
    # adapter 据此按需抽取/校验, 未声明的字段缺失不报错 -> anchor-free method 也可运行.
    requires: frozenset = frozenset()

    @classmethod
    def required_metadata(cls, params: dict[str, Any]) -> frozenset:
        """给定 params 时本 method 实际需要的 metadata 字段集合.

        默认等于静态 requires (固定需求). 依赖 config 的 method (如 GiGPO 仅在
        anchor='state' 时才需要 structured transitions) 覆盖本 classmethod, 使
        rollout 采集开关与 adapter 校验都以算法真实依赖为准, 而非 method 名硬编码.
        仅读 class 属性 / 不实例化算法对象.
        """
        return cls.requires

    @abstractmethod
    def compute(self, inputs: CreditAssignmentInput, params: dict[str, Any]) -> CreditAssignmentOutput:
        raise NotImplementedError


# structured state transition 采集的触发键: method 需要它 <=> rollout 侧必须开启
# collect_step_states (ca_step_transitions 及其配套的 initial/final/trusted 采集).
STRUCTURED_STATE_KEY = "step_transitions"


def needs_structured_state(requirements: frozenset) -> bool:
    """requirement 集合是否要求 rollout 采集 structured workbook state."""
    return STRUCTURED_STATE_KEY in requirements


# evaluation-side fingerprint 需求键 (SOFT requirement): method 若要用实际提交 bytes 的
# canonical fingerprint 确认 success anchor 则声明 (如 RewardFlow). 不属于 adapter 的
# hard-missing 校验 (允许逐行 None -> fail-closed 排除 anchor); rollout 侧据此把
# emit_eval_fingerprint 标记写入该样本的 extra_fields, 由 reward 侧按样本开启采集
# (无进程级环境变量).
EVAL_FINGERPRINT_REQ = "eval_fingerprint"


def needs_eval_fingerprint(requirements: frozenset) -> bool:
    """requirement 集合是否要求 evaluation-side workbook fingerprint emission."""
    return EVAL_FINGERPRINT_REQ in requirements
