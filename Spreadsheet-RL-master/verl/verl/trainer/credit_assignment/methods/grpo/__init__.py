# 由 self_improving 显式 import 以触发注册; 不改动 methods/__init__.py.

from verl.trainer.credit_assignment.methods.grpo.algorithm import GRPO  # noqa: F401

__all__ = ["GRPO"]
