# self_improving 配置 schema 与校验. 未知 key 一律报错, 不静默忽略.
#
# 权威来源: 本文件 + 各 method 的 config dataclass. skills/*.md 只是给 LLM 读的说明,
# 写错也不会扩大可执行参数空间.

from dataclasses import asdict, dataclass, field
from typing import Any, Optional
import warnings

CONFIG_VERSION = 1


class SelfImprovingConfigError(ValueError):
    """配置错误: 一律在训练开始前抛出, 不进入训练循环."""


def _as_plain_dict(node: Any, where: str) -> dict:
    if node is None:
        return {}
    if hasattr(node, "items"):
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(node):
                return OmegaConf.to_container(node, resolve=True) or {}
        except ImportError:  # pragma: no cover - omegaconf 总是存在
            pass
        return dict(node)
    raise SelfImprovingConfigError(f"{where}: expected a mapping, got {type(node).__name__}")


def _take(raw: dict, key: str, default: Any, expected_types, where: str) -> Any:
    if key not in raw:
        return default
    value = raw[key]
    if value is None:
        return default
    if expected_types is bool and not isinstance(value, bool):
        raise SelfImprovingConfigError(f"{where}.{key}: expected bool, got {value!r}")
    if expected_types in (int, float) and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise SelfImprovingConfigError(f"{where}.{key}: expected number, got {value!r}")
    if expected_types is str and not isinstance(value, str):
        raise SelfImprovingConfigError(f"{where}.{key}: expected str, got {value!r}")
    return value


def _reject_unknown(raw: dict, allowed: set, where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise SelfImprovingConfigError(f"{where}: unknown key(s) {sorted(unknown)}; supported {sorted(allowed)}")


@dataclass
class InitialSkillConfig:
    method: str = "grpo"
    params: dict = field(default_factory=dict)


@dataclass
class ContextLimits:
    max_trajectory_samples: int = 4
    max_steps_per_trajectory: int = 20
    max_text_chars_per_field: int = 1000
    max_context_chars: int = 40000


@dataclass
class LLMConfig:
    endpoint_env: str = "SPREADSHEET_RL_SI_LLM_URL"
    model_env: str = "SPREADSHEET_RL_SI_LLM_MODEL"
    api_key_env: str = "SPREADSHEET_RL_SI_LLM_API_KEY"
    timeout_seconds: float = 60.0
    max_retries: int = 1
    temperature: float = 0.0
    # mock: 不发起网络请求, 用于 CPU 集成测试与无 API 的 smoke run.
    client: str = "api"
    # JSONL 脚本文件, 每行一个 controller / knowledge 响应; 耗尽后回退 keep.
    mock_script: Optional[str] = None


@dataclass
class EvaluationConfig:
    use_training_metrics: bool = True
    use_existing_validation: bool = True
    validation_metric_key: Optional[str] = None
    min_window_steps: int = 5
    improvement_threshold: float = 0.01


@dataclass
class KnowledgeConfig:
    enabled: bool = True
    update_interval: int = 1
    max_prompt_chars: int = 20000


@dataclass
class PersistenceConfig:
    enabled: bool = True
    strict_resume: bool = True


_ALLOWED_LLM_CLIENTS = {"api", "mock"}
_TOP_KEYS = {
    "enabled",
    "interval_steps",
    "initial_skill",
    "candidate_skills",
    "context",
    "llm",
    "evaluation",
    "knowledge",
    "persistence",
}


@dataclass
class SelfImprovingConfig:
    enabled: bool = False
    interval_steps: int = 20
    initial_skill: InitialSkillConfig = field(default_factory=InitialSkillConfig)
    candidate_skills: list = field(default_factory=lambda: ["grpo", "gigpo", "rewardflow"])
    context: ContextLimits = field(default_factory=ContextLimits)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_self_improving_config(node: Any) -> SelfImprovingConfig:
    """从 config.self_improving 节点构造并校验配置."""
    raw = _as_plain_dict(node, "self_improving")
    _reject_unknown(raw, _TOP_KEYS, "self_improving")

    cfg = SelfImprovingConfig()
    cfg.enabled = bool(_take(raw, "enabled", False, bool, "self_improving"))
    cfg.interval_steps = int(_take(raw, "interval_steps", 20, int, "self_improving"))

    initial = _as_plain_dict(raw.get("initial_skill"), "self_improving.initial_skill")
    _reject_unknown(initial, {"method", "params"}, "self_improving.initial_skill")
    cfg.initial_skill = InitialSkillConfig(
        method=str(_take(initial, "method", "grpo", str, "self_improving.initial_skill")),
        params=_as_plain_dict(initial.get("params"), "self_improving.initial_skill.params"),
    )

    candidates = raw.get("candidate_skills", None)
    if candidates is not None:
        candidates = list(candidates)
        if not candidates or not all(isinstance(c, str) for c in candidates):
            raise SelfImprovingConfigError("self_improving.candidate_skills must be a non-empty list of strings")
        cfg.candidate_skills = [str(c) for c in candidates]

    cfg.context = _load_context(raw.get("context"))
    cfg.llm = _load_llm(raw.get("llm"))
    cfg.evaluation = _load_evaluation(raw.get("evaluation"))
    cfg.knowledge = _load_knowledge(raw.get("knowledge"))
    cfg.persistence = _load_persistence(raw.get("persistence"))

    validate_self_improving_config(cfg)
    return cfg


def _positive_int(raw, key, default, where):
    value = int(_take(raw, key, default, int, where))
    if value < 1:
        raise SelfImprovingConfigError(f"{where}.{key} must be >= 1; got {value}")
    return value


def _load_context(raw: Any) -> ContextLimits:
    d = _as_plain_dict(raw, "self_improving.context")
    _reject_unknown(d, {"max_trajectory_samples", "max_steps_per_trajectory", "max_text_chars_per_field", "max_context_chars"}, "self_improving.context")
    where = "self_improving.context"
    return ContextLimits(
        max_trajectory_samples=_positive_int(d, "max_trajectory_samples", 4, where),
        max_steps_per_trajectory=_positive_int(d, "max_steps_per_trajectory", 20, where),
        max_text_chars_per_field=_positive_int(d, "max_text_chars_per_field", 1000, where),
        max_context_chars=_positive_int(d, "max_context_chars", 40000, where),
    )


def _load_llm(raw: Any) -> LLMConfig:
    d = _as_plain_dict(raw, "self_improving.llm")
    _reject_unknown(
        d,
        {"endpoint_env", "model_env", "api_key_env", "timeout_seconds", "max_retries", "temperature", "client", "mock_script"},
        "self_improving.llm",
    )
    where = "self_improving.llm"
    timeout = float(_take(d, "timeout_seconds", 60.0, float, where))
    if timeout <= 0:
        raise SelfImprovingConfigError(f"{where}.timeout_seconds must be > 0")
    retries = int(_take(d, "max_retries", 1, int, where))
    if retries < 0:
        raise SelfImprovingConfigError(f"{where}.max_retries must be >= 0")
    temperature = float(_take(d, "temperature", 0.0, float, where))
    if not 0.0 <= temperature <= 2.0:
        raise SelfImprovingConfigError(f"{where}.temperature must be in [0, 2]")
    client = str(_take(d, "client", "api", str, where))
    if client not in _ALLOWED_LLM_CLIENTS:
        raise SelfImprovingConfigError(f"{where}.client must be one of {sorted(_ALLOWED_LLM_CLIENTS)}; got '{client}'")
    script = _take(d, "mock_script", None, str, where)
    return LLMConfig(
        endpoint_env=str(_take(d, "endpoint_env", LLMConfig.endpoint_env, str, where)),
        model_env=str(_take(d, "model_env", LLMConfig.model_env, str, where)),
        api_key_env=str(_take(d, "api_key_env", LLMConfig.api_key_env, str, where)),
        timeout_seconds=timeout,
        max_retries=retries,
        temperature=temperature,
        client=client,
        mock_script=script,
    )


def _load_evaluation(raw: Any) -> EvaluationConfig:
    d = _as_plain_dict(raw, "self_improving.evaluation")
    _reject_unknown(
        d,
        {"use_training_metrics", "use_existing_validation", "validation_metric_key", "min_window_steps", "improvement_threshold"},
        "self_improving.evaluation",
    )
    where = "self_improving.evaluation"
    threshold = float(_take(d, "improvement_threshold", 0.01, float, where))
    if threshold < 0:
        raise SelfImprovingConfigError(f"{where}.improvement_threshold must be >= 0")
    key = _take(d, "validation_metric_key", None, str, where)
    return EvaluationConfig(
        use_training_metrics=bool(_take(d, "use_training_metrics", True, bool, where)),
        use_existing_validation=bool(_take(d, "use_existing_validation", True, bool, where)),
        validation_metric_key=key,
        min_window_steps=_positive_int(d, "min_window_steps", 5, where),
        improvement_threshold=threshold,
    )


def _load_knowledge(raw: Any) -> KnowledgeConfig:
    d = _as_plain_dict(raw, "self_improving.knowledge")
    _reject_unknown(d, {"enabled", "update_interval", "max_prompt_chars"}, "self_improving.knowledge")
    where = "self_improving.knowledge"
    return KnowledgeConfig(
        enabled=bool(_take(d, "enabled", True, bool, where)),
        update_interval=_positive_int(d, "update_interval", 1, where),
        max_prompt_chars=_positive_int(d, "max_prompt_chars", 20000, where),
    )


def _load_persistence(raw: Any) -> PersistenceConfig:
    d = _as_plain_dict(raw, "self_improving.persistence")
    _reject_unknown(d, {"enabled", "strict_resume"}, "self_improving.persistence")
    where = "self_improving.persistence"
    return PersistenceConfig(
        enabled=bool(_take(d, "enabled", True, bool, where)),
        strict_resume=bool(_take(d, "strict_resume", True, bool, where)),
    )


def validate_self_improving_config(cfg: SelfImprovingConfig) -> None:
    """跨字段一致性检查. 与 method 注册表相关的检查在 SkillRuntime 里做."""
    if cfg.interval_steps < 1:
        raise SelfImprovingConfigError(f"self_improving.interval_steps must be >= 1; got {cfg.interval_steps}")
    if len(set(cfg.candidate_skills)) != len(cfg.candidate_skills):
        raise SelfImprovingConfigError(f"self_improving.candidate_skills contains duplicates: {cfg.candidate_skills}")
    if cfg.initial_skill.method not in cfg.candidate_skills:
        raise SelfImprovingConfigError(
            f"self_improving.initial_skill.method '{cfg.initial_skill.method}' must appear in "
            f"candidate_skills {cfg.candidate_skills}"
        )
    # §10 minor: interval < min_window 会让每个窗口都判 insufficient_evidence.
    # 这不是硬错误 (MVP 允许短 interval 做快速实验 / 失败重试), 但要明确警告.
    if cfg.enabled and cfg.interval_steps < cfg.evaluation.min_window_steps:
        warnings.warn(
            f"self_improving.interval_steps ({cfg.interval_steps}) is smaller than "
            f"self_improving.evaluation.min_window_steps ({cfg.evaluation.min_window_steps}): every "
            "intervention window will be shorter than the evaluator considers usable, so no decision "
            "will ever be backed by a positive/negative evaluation. Raise interval_steps or lower "
            "min_window_steps if you want evaluation-driven updates.",
            stacklevel=2,
        )
    if not cfg.enabled:
        return
    if cfg.llm.client == "api":
        import os

        missing = [name for name in (cfg.llm.endpoint_env, cfg.llm.model_env) if not os.environ.get(name)]
        if missing:
            raise SelfImprovingConfigError(
                f"self_improving.enabled=true but required environment variable(s) {missing} are not set. "
                "Set the LLM endpoint/model before launching, or use self_improving.llm.client=mock."
            )
        # API key 允许缺失 (部分自建服务不需要), 缺失时按无鉴权请求处理.
