# Credit assignment method registry: 按名字注册/查找算法实现.

from typing import Any, Callable

from verl.trainer.credit_assignment.interface import CreditAssignmentInput, CreditAssignmentMethod, CreditAssignmentOutput

_CREDIT_ASSIGNMENT_REGISTRY: dict[str, type[CreditAssignmentMethod]] = {}


def register_credit_assignment(name: str) -> Callable[[type[CreditAssignmentMethod]], type[CreditAssignmentMethod]]:
    """装饰器: 以唯一名字注册一个 credit assignment method."""

    def decorator(cls: type[CreditAssignmentMethod]) -> type[CreditAssignmentMethod]:
        if name in _CREDIT_ASSIGNMENT_REGISTRY and _CREDIT_ASSIGNMENT_REGISTRY[name] is not cls:
            raise ValueError(
                f"credit assignment method '{name}' already registered: "
                f"{_CREDIT_ASSIGNMENT_REGISTRY[name]} vs {cls}"
            )
        cls.name = name
        _CREDIT_ASSIGNMENT_REGISTRY[name] = cls
        return cls

    return decorator


def get_credit_assignment_class(name: str) -> type[CreditAssignmentMethod]:
    if name not in _CREDIT_ASSIGNMENT_REGISTRY:
        raise ValueError(
            f"Unknown credit assignment method '{name}'. "
            f"Registered: {sorted(_CREDIT_ASSIGNMENT_REGISTRY.keys())}"
        )
    return _CREDIT_ASSIGNMENT_REGISTRY[name]


def available_methods() -> list[str]:
    return sorted(_CREDIT_ASSIGNMENT_REGISTRY.keys())


def run(method: str, inputs: CreditAssignmentInput, params: dict[str, Any]) -> CreditAssignmentOutput:
    """查表实例化 method 并执行. method 各自负责校验自己的 params."""
    cls = get_credit_assignment_class(method)
    return cls().compute(inputs, dict(params or {}))
