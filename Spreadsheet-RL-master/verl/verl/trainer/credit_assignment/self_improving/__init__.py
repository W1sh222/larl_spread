# Self-Improving Credit Assignment 框架 (Slow Loop).
#
# 与 Tool Library 的分工: methods/ + registry 负责"怎么算 advantage"; 本包负责
# "什么时候用哪个算法、参数怎么调、经验怎么沉淀". ray_trainer 只调用 SelfImprovingLoop.

from verl.trainer.credit_assignment.self_improving.config import SelfImprovingConfig, load_self_improving_config
from verl.trainer.credit_assignment.self_improving.runtime import SkillConfig, SkillRuntime
from verl.trainer.credit_assignment.self_improving.slow_loop import SelfImprovingLoop

__all__ = [
    "SelfImprovingConfig",
    "SelfImprovingLoop",
    "SkillConfig",
    "SkillRuntime",
    "load_self_improving_config",
]
