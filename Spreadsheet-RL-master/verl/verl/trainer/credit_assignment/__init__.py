"""Credit Assignment Library.

目标 API:
    advantages = credit_assignment.compute(method="gigpo", batch=batch, params={...})

分层原则:
- registry/interface 与 verl 解耦, 只处理标准输入输出 schema.
- adapter 是唯一感知 verl DataProto / tensor / non_tensor_batch 的地方.
- methods/ 下每个算法 (如 GiGPO) 是独立模块, 新增 method 无需改 trainer/PPO.
"""

from typing import Any

from verl.trainer.credit_assignment import methods as _methods  # noqa: F401  (触发注册)
from verl.trainer.credit_assignment.adapter import compute as _adapter_compute
from verl.trainer.credit_assignment.interface import (
    CreditAssignmentInput,
    CreditAssignmentMethod,
    CreditAssignmentOutput,
)
from verl.trainer.credit_assignment.registry import (
    available_methods,
    get_credit_assignment_class,
    register_credit_assignment,
)
from verl.trainer.credit_assignment.registry import run as run_on_inputs

__all__ = [
    "CreditAssignmentInput",
    "CreditAssignmentMethod",
    "CreditAssignmentOutput",
    "available_methods",
    "compute",
    "get_credit_assignment_class",
    "register_credit_assignment",
    "run_on_inputs",
]


def compute(
    method: str,
    token_level_rewards,
    response_mask,
    index,
    non_tensor_batch: dict,
    params: dict[str, Any] | None = None,
) -> CreditAssignmentOutput:
    """从 verl 张量/元数据直接计算 advantage (推荐入口)."""
    return _adapter_compute(
        method=method,
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        non_tensor_batch=non_tensor_batch,
        params=params or {},
    )
