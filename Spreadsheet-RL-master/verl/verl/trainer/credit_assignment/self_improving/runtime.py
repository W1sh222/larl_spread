"""Active Skill 的校验与运行时切换.

只改 driver 侧 config.algorithm.credit_assignment; 不重建 actor / optimizer / rollout worker,
因此新 Skill 从下一个 training step 生效.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.trainer.credit_assignment.interface import (
    EVAL_FINGERPRINT_REQ,
    STRUCTURED_STATE_KEY,
    needs_eval_fingerprint,
    needs_structured_state,
)
from verl.trainer.credit_assignment.registry import available_methods, get_credit_assignment_class
from verl.trainer.credit_assignment.self_improving.config import SelfImprovingConfig

# rollout 侧始终采集的算法中立 step 元数据 (default_agent_loop=tool_agent).
BASE_METADATA = frozenset({"step_spans", "step_observations"})


class SkillValidationError(ValueError):
    """LLM 决策 / 初始配置不合法: 调用方保持当前 Skill 并记录原因 (execution safety fallback)."""


@dataclass
class SkillConfig:
    method: str
    params: dict = field(default_factory=dict)

    def copy(self) -> "SkillConfig":
        return SkillConfig(method=self.method, params=dict(self.params))

    def to_dict(self) -> dict:
        return {"method": self.method, "params": dict(self.params)}


@dataclass
class ParamConstraint:
    """单个可校准参数的取值域.

    exposed=False 的参数仍可由 initial_skill 显式设置, 但不会出现在 LLM 的
    calibration space 里 (会改变算法语义 / 成功判定标准的参数默认不开放).
    """

    kind: str
    choices: tuple = ()
    lo: Optional[float] = None
    hi: Optional[float] = None
    exposed: bool = True
    note: str = ""

    def describe(self) -> str:
        if self.kind == "bool":
            return "bool"
        if self.kind == "choices":
            return "one of " + "/".join(repr(c) for c in self.choices)
        return f"number in [{self.lo}, {self.hi}]"

    def check(self, value: Any) -> Optional[str]:
        if self.kind == "bool":
            return None if isinstance(value, bool) else f"expected bool, got {value!r}"
        if self.kind == "choices":
            return None if value in self.choices else f"expected one of {list(self.choices)}, got {value!r}"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"expected number, got {value!r}"
        if not (self.lo <= float(value) <= self.hi):
            return f"expected a number in [{self.lo}, {self.hi}], got {value!r}"
        return None


# 初始 calibration space: 只列各 method config 里真实存在的字段.
CALIBRATION_SPACE: dict[str, dict[str, ParamConstraint]] = {
    "grpo": {
        "norm_adv_by_std_in_grpo": ParamConstraint(kind="bool"),
        # 数值稳定项, 不是性能旋钮 -> 不暴露给 LLM.
        "epsilon": ParamConstraint(kind="range", lo=1e-8, hi=1e-3, exposed=False),
    },
    "gigpo": {
        "gamma": ParamConstraint(kind="range", lo=0.5, hi=1.0),
        "step_advantage_w": ParamConstraint(kind="range", lo=0.0, hi=2.0),
        "mode": ParamConstraint(kind="choices", choices=("mean_norm", "mean_std_norm")),
        "grouping": ParamConstraint(kind="choices", choices=("exact",)),
        "anchor": ParamConstraint(kind="choices", choices=("state", "observation")),
        "anchor_scope": ParamConstraint(kind="choices", choices=("full",)),
        "epsilon": ParamConstraint(kind="range", lo=1e-8, hi=1e-3, exposed=False),
        # 改变 Eq.7 baseline 分组语义 -> opt-in.
        "include_truncated_in_baseline": ParamConstraint(kind="bool", exposed=False),
    },
    "rewardflow": {
        "gamma": ParamConstraint(kind="range", lo=0.5, hi=1.0),
        "alpha_act": ParamConstraint(kind="range", lo=0.0, hi=2.0),
        "alpha_traj": ParamConstraint(kind="range", lo=0.0, hi=2.0),
        "normalization": ParamConstraint(kind="choices", choices=("mean_norm", "mean_std_norm")),
        "epsilon": ParamConstraint(kind="range", lo=1e-8, hi=1e-3, exposed=False),
        # 成功判定标准, 不作为自由搜索项.
        "success_threshold": ParamConstraint(kind="range", lo=1e-6, hi=1.0, exposed=False),
        "success_tolerance": ParamConstraint(kind="range", lo=0.0, hi=1e-3, exposed=False),
    },
}


def ensure_methods_registered() -> None:
    """显式注册 GRPO (不改 methods/__init__.py). 重复 import 由 registry 幂等处理."""
    import importlib

    importlib.import_module("verl.trainer.credit_assignment.methods.grpo")


def _config_class(method: str):
    ensure_methods_registered()
    get_credit_assignment_class(method)  # unknown method -> raise
    if method == "grpo":
        from verl.trainer.credit_assignment.methods.grpo.config import GRPOConfig

        return GRPOConfig
    if method == "gigpo":
        from verl.trainer.credit_assignment.methods.gigpo.config import GiGPOConfig

        return GiGPOConfig
    if method == "rewardflow":
        from verl.trainer.credit_assignment.methods.rewardflow.config import RewardFlowConfig

        return RewardFlowConfig
    raise SkillValidationError(
        f"method '{method}' has no calibration config class mapping in self_improving.runtime; "
        "add it there (one entry) before using it as a candidate skill."
    )


def method_param_names(method: str) -> list:
    return sorted(_config_class(method).__dataclass_fields__)


def canonical_params(method: str, params: Optional[dict]) -> dict:
    """把部分参数补全成含默认值的 canonical 配置 (dataclass 默认值 -> from_params 校验后覆盖)."""
    cls = _config_class(method)
    cfg = cls.from_params({})
    full = {name: getattr(cfg, name) for name in cls.__dataclass_fields__}
    full.update(dict(params or {}))
    # 再走一次 method 自身校验, 保证组合合法性.
    validated = cls.from_params(full)
    return {name: getattr(validated, name) for name in cls.__dataclass_fields__}


def required_metadata(method: str, params: dict) -> frozenset:
    cls = get_credit_assignment_class(method)
    return frozenset(cls.required_metadata(dict(params or {})))


def _same_value(expected: Any, value: Any) -> bool:
    if isinstance(expected, bool) != isinstance(value, bool):
        return False
    return expected == value


def metadata_relevant_params(method: str) -> frozenset:
    """取值会改变 required_metadata 的参数 (例如 gigpo 的 anchor).

    这类参数属于 metadata bootstrap 语义 (决定 worker 采集范围), 因此允许出现在 launch
    params 里; 其余参数只影响 advantage 数值, 由 self_improving.initial_skill.params 负责.
    """
    cls = get_credit_assignment_class(method)
    defaults = canonical_params(method, {})
    base = set(cls.required_metadata(dict(defaults)))
    relevant: set = set()
    for name, constraint in CALIBRATION_SPACE.get(method, {}).items():
        if constraint.kind == "range":
            continue
        candidates = constraint.choices if constraint.kind == "choices" else (True, False)
        for value in candidates:
            try:
                params = canonical_params(method, {**defaults, name: value})
            except (ValueError, TypeError):
                continue
            if set(cls.required_metadata(params)) != base:
                relevant.add(name)
                break
    return frozenset(relevant)


def bootstrap_param_overrides(algorithm_cfg: Any) -> dict:
    """launch 期 algorithm.credit_assignment.params 中会被 initial_skill 覆盖的算法参数.

    Hydra 默认节点是 `params: {}`, 因此只有真正的 non-default override 会被列出;
    与 canonical 默认值完全相同的显式写法不算 override.
    """
    ca = algorithm_cfg.get("credit_assignment", None) if algorithm_cfg is not None else None
    if not ca:
        return {}
    params = dict(ca.get("params", None) or {})
    if not params:
        return {}
    method = ca.get("method", None)
    if not method:
        return params
    ensure_methods_registered()
    try:
        known = set(method_param_names(method))
        defaults = canonical_params(method, {})
        relevant = set(metadata_relevant_params(method))
    except (ValueError, SkillValidationError):
        return params
    return {
        k: v
        for k, v in params.items()
        if k not in known or (k not in relevant and not _same_value(defaults.get(k), v))
    }


def maximal_required_metadata(method: str) -> frozenset:
    """校准空间内所有取值组合可能需要的 metadata 上界 (例如 GiGPO anchor=state 追加 structured)."""
    ensure_methods_registered()
    method_cls = get_credit_assignment_class(method)
    space = CALIBRATION_SPACE.get(method, {})
    variants: list[dict] = [{}]
    for name, constraint in space.items():
        if constraint.kind != "choices":
            continue
        variants = [{**base, name: choice} for base in variants for choice in constraint.choices]
    req: set = set()
    for variant in variants:
        try:
            params = canonical_params(method, variant)
        except (ValueError, TypeError):
            continue
        req |= set(method_cls.required_metadata(params))
    return frozenset(req)


def collected_metadata(algorithm_cfg: Any) -> frozenset:
    """mirror of tool_agent_loop._resolve_collect_step_states: worker 初始化时冻结的采集范围.

    不 import rollout 模块 (重依赖), 只复用 interface 里的同一组判定原语, 避免语义漂移.
    """
    if algorithm_cfg is None or algorithm_cfg.get("adv_estimator", None) != "credit_assignment":
        return frozenset()
    ca = algorithm_cfg.get("credit_assignment", None)
    if not ca:
        return frozenset()
    collected = set(BASE_METADATA)
    method = ca.get("method", None)
    needs_structured = False
    needs_eval_fp = False
    if method:
        ensure_methods_registered()
        required = set(get_credit_assignment_class(method).required_metadata(dict(ca.get("params", None) or {})))
        needs_structured = needs_structured_state(frozenset(required))
        needs_eval_fp = needs_eval_fingerprint(frozenset(required))
    explicit = ca.get("collect_step_states", None)
    if explicit is not None:
        needs_structured = bool(explicit)
    if needs_structured:
        collected |= {STRUCTURED_STATE_KEY, "terminal_trusted"}
    if needs_eval_fp:
        collected.add(EVAL_FINGERPRINT_REQ)
    return frozenset(collected)


def bootstrap_metadata_signature(algorithm_cfg: Any) -> dict:
    ca = algorithm_cfg.get("credit_assignment", None) or {} if algorithm_cfg is not None else {}
    method = ca.get("method", None)
    relevant = set()
    if method:
        try:
            relevant = set(metadata_relevant_params(method))
        except (ValueError, SkillValidationError):
            relevant = set()
    launch_params = dict(ca.get("params", None) or {})
    return {
        "bootstrap_method": method,
        "collect_step_states": ca.get("collect_step_states", None),
        # 只记录真正决定采集范围的参数: 算法参数由 self_improving.initial_skill 负责.
        "bootstrap_params": {k: v for k, v in sorted(launch_params.items()) if k in relevant},
        "collected": sorted(collected_metadata(algorithm_cfg)),
    }


class SkillRuntime:
    """候选集合 / 参数域 / metadata 覆盖三重门禁下的 Skill 切换器."""

    def __init__(self, si_config: SelfImprovingConfig, algorithm_cfg: Any):
        self.config = si_config
        self.algorithm_cfg = algorithm_cfg
        self.collected = collected_metadata(algorithm_cfg)
        self._current: Optional[SkillConfig] = None
        self._metadata_gaps: dict[str, list] = {}
        self._preflight()

    # ---------- validation ----------
    def _preflight(self) -> None:
        ensure_methods_registered()
        if self.algorithm_cfg is None or self.algorithm_cfg.get("adv_estimator", None) != "credit_assignment":
            # adv_estimator 不是 credit_assignment 时, 换 method 不会改变 advantage -> 必须报错.
            raise SkillValidationError(
                f"self_improving requires algorithm.adv_estimator=credit_assignment (currently "
                f"{None if self.algorithm_cfg is None else self.algorithm_cfg.get('adv_estimator', None)!r}): "
                "runtime skill switching only takes effect through the credit assignment estimator. "
                "Launch with algorithm.adv_estimator=credit_assignment "
                "algorithm.credit_assignment.method=rewardflow (metadata bootstrap config)."
            )
        registered = set(available_methods())
        for method in list(self.config.candidate_skills) + [self.config.initial_skill.method]:
            if method not in registered:
                raise SkillValidationError(
                    f"candidate skill '{method}' is not registered; available {sorted(registered)}"
                )
        # §C-1: launch 节点只是 metadata bootstrap, 算法参数一律来自 initial_skill / decision.
        overrides = bootstrap_param_overrides(self.algorithm_cfg)
        if overrides:
            bootstrap_method = self.algorithm_cfg.get("credit_assignment", {}).get("method", None)
            raise SkillValidationError(
                f"algorithm.credit_assignment.params {overrides} are not metadata-bootstrap settings: with "
                f"self_improving.enabled=true the launch node only freezes what the rollout collects, and "
                f"apply_initial() replaces the whole node with self_improving.initial_skill "
                f"(bootstrap method '{bootstrap_method}'). Training parameters must not be silently dropped. "
                "Set them in ++self_improving.initial_skill.params (candidate_skills / controller may calibrate "
                "them afterwards), keep only metadata-relevant keys such as "
                f"{sorted(metadata_relevant_params(bootstrap_method)) if bootstrap_method in available_methods() else []}, "
                "or control collection explicitly with ++algorithm.credit_assignment.collect_step_states=true/false."
            )
        for method in self.config.candidate_skills:
            missing = sorted(maximal_required_metadata(method) - self.collected)
            if missing:
                self._metadata_gaps[method] = missing
        unreachable = [m for m, gap in self._metadata_gaps.items() if gap]
        if unreachable:
            raise SkillValidationError(
                f"launch-time metadata bootstrap (adv_estimator="
                f"{self.algorithm_cfg.get('adv_estimator', None) if self.algorithm_cfg is not None else None}, "
                f"collected={sorted(self.collected) or 'none'}) cannot satisfy candidate skill(s) {unreachable}: "
                f"missing {self._metadata_gaps}. Launch with algorithm.adv_estimator=credit_assignment and "
                "algorithm.credit_assignment.method=rewardflow (metadata bootstrap config) so structured "
                "states and evaluation fingerprints are collected for the whole run."
            )

    def metadata_gaps(self) -> dict:
        return {k: list(v) for k, v in self._metadata_gaps.items()}

    def candidate_metadata(self) -> dict:
        return {m: sorted(maximal_required_metadata(m)) for m in self.config.candidate_skills}

    def calibration_space(self, method: str) -> dict:
        space = CALIBRATION_SPACE.get(method, {})
        return {name: c for name, c in space.items() if c.exposed}

    def calibration_spec(self) -> dict:
        """给 controller prompt 用的可执行参数说明."""
        spec = {}
        for method in self.config.candidate_skills:
            defaults = canonical_params(method, {})
            entries = {}
            for name, constraint in sorted(self.calibration_space(method).items()):
                entries[name] = {"type": constraint.kind, "domain": constraint.describe(), "default": defaults[name]}
            spec[method] = {"defaults": defaults, "calibratable": entries}
        return spec

    def validate(self, skill: SkillConfig, from_llm: bool = False) -> SkillConfig:
        if not isinstance(skill, SkillConfig):
            raise SkillValidationError(f"expected SkillConfig, got {type(skill).__name__}")
        method = skill.method
        if not isinstance(method, str) or not method:
            raise SkillValidationError(f"invalid skill method name: {method!r}")
        ensure_methods_registered()
        if method not in set(available_methods()):
            raise SkillValidationError(f"skill '{method}' is not a registered credit assignment method")
        if method not in self.config.candidate_skills:
            raise SkillValidationError(
                f"skill '{method}' is not in self_improving.candidate_skills {self.config.candidate_skills}"
            )
        params = dict(skill.params or {})
        if from_llm:
            allowed = set(self.calibration_space(method))
            current = self._current
            defaults = canonical_params(method, {})
            hidden = set(method_param_names(method)) - allowed

            def unchanged(name: str) -> bool:
                value = params[name]
                if current is not None and current.method == method:
                    # 同一 skill: 必须与当前生效值一致, 连"改回默认值"也算越界.
                    return current.params.get(name, defaults[name]) == value
                return defaults.get(name) == value

            echoed = {name for name in hidden if name in params and unchanged(name)}
            rejected = sorted(set(params) - allowed - echoed)
            if rejected:
                raise SkillValidationError(
                    f"params {rejected} are not calibratable for skill '{method}'; "
                    f"allowed keys {sorted(allowed)}"
                )
            # 未暴露参数只能沿用当前生效值 / 默认值: 一次 LLM 决策不得重置 launch 期设定.
            params = {k: v for k, v in params.items() if k in allowed}
            if current is not None and current.method == method:
                for name in hidden:
                    if name in current.params:
                        params.setdefault(name, current.params[name])
        known = set(method_param_names(method))
        unknown = sorted(set(params) - known)
        if unknown:
            raise SkillValidationError(
                f"skill '{method}' received unsupported params {unknown}; supported {sorted(known)}"
            )
        space = CALIBRATION_SPACE.get(method, {})
        bad = []
        for name, value in params.items():
            constraint = space.get(name)
            if constraint is not None:
                problem = constraint.check(value)
                if problem:
                    bad.append(f"{name}: {problem}")
        if bad:
            raise SkillValidationError(f"skill '{method}' parameter validation failed -> " + "; ".join(bad))

        try:
            canonical = canonical_params(method, params)
        except (ValueError, TypeError) as e:
            raise SkillValidationError(f"skill '{method}' rejected its parameter combination: {e}") from e

        missing = sorted(required_metadata(method, canonical) - self.collected)
        if missing:
            raise SkillValidationError(
                f"skill '{method}' with params {canonical} needs metadata {missing}, "
                f"but the rollout only collects {sorted(self.collected)}"
            )
        return SkillConfig(method=method, params=canonical)

    # ---------- runtime application ----------
    def current(self) -> Optional[SkillConfig]:
        return self._current

    def read_active(self) -> SkillConfig:
        ca = self.algorithm_cfg.get("credit_assignment", None)
        if ca is None:
            raise SkillValidationError("algorithm.credit_assignment is absent from the driver config")
        return SkillConfig(method=str(ca.get("method")), params=dict(ca.get("params", None) or {}))

    def apply(self, skill: SkillConfig, from_llm: bool = False) -> SkillConfig:
        """整体替换 method + params. 校验失败时不留下半改状态."""
        validated = self.validate(skill, from_llm=from_llm)
        ca = self.algorithm_cfg.get("credit_assignment", None)
        if ca is None:
            raise SkillValidationError("algorithm.credit_assignment is absent from the driver config")
        before = (ca.get("method", None), dict(ca.get("params", None) or {}))
        try:
            # params 必须整体替换: 禁止 merge/update, 避免跨 method 残留键.
            ca["params"] = {k: v for k, v in validated.params.items()}
            ca["method"] = validated.method
            self._current = validated
        except Exception:
            # 写一半会让 method 与 params 来自不同 skill -> 回滚到本次 apply 之前的状态.
            ca["params"], ca["method"] = before[1], before[0]
            self._current = SkillConfig(method=str(before[0]), params=before[1]) if before[0] else None
            raise
        return validated

    def apply_initial(self, restore: Optional[SkillConfig] = None) -> SkillConfig:
        """启动时写入唯一合法的 Skill: 恢复的 snapshot 优先, 否则用 self_improving.initial_skill.

        绝不把 launch 期的 algorithm.credit_assignment.params merge / 继承进来.
        """
        initial = self.config.initial_skill
        target = restore if restore is not None else SkillConfig(method=initial.method, params=dict(initial.params))
        return self.apply(target, from_llm=False)
