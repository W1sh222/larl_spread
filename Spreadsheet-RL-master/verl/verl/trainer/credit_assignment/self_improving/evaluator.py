"""Training Feedback 评估: 比较"应用某 Skill 的 window"与"上一个 window".

MVP 只做 observational comparison: 没有 policy branching, 没有 counterfactual training,
所以性能差不能证明 credit skill 的因果效果. 所有数值比较都在 Python 侧完成, LLM 只读结果.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.trainer.credit_assignment.self_improving.config import EvaluationConfig

OUTCOME_POSITIVE = "positive"
OUTCOME_NEGATIVE = "negative"
OUTCOME_NEUTRAL = "neutral"
OUTCOME_INSUFFICIENT = "insufficient_evidence"
QUALITY_HIGH = "high"
QUALITY_MEDIUM = "medium"
QUALITY_LOW = "low"

# required_metadata() 的名字 -> context._metadata_health() 实际使用的键.
# 不存在的键不会出现在 health 里, 因此这里只列真实存在的字段 (§10 minor: 不存在 diagnostic key 的引用).
HEALTH_KEY_OF = {
    "step_spans": "step_spans",
    "step_observations": "step_observations",
    "step_transitions": "step_transitions",
    "terminal_trusted": "terminal_trusted",
    "eval_fingerprint": "eval_workbook_fingerprint",
}

# 比较用的 window 统计键 -> (aggregate 取值路径, 是否为"越大越好").
PRIMARY_METRICS = (
    ("success_rate", "success_rate", True),
    ("raw_reward_mean", ("raw_reward", "mean"), True),
)
SECONDARY_METRICS = (
    ("train_reward_mean", ("train_reward", "mean"), True),
    ("trajectory_length_mean", ("trajectory_length", "mean"), True),
    ("steps_per_trajectory_mean", ("steps_per_trajectory", "mean"), None),
    ("zero_advantage_ratio", ("advantage", "zero_ratio"), None),
    ("advantage_abs_mean", ("advantage", "abs_mean"), None),
    ("trusted_terminal_ratio", "trusted_terminal_ratio", True),
)
CORE_PRIORITY = ("success_rate", "raw_reward_mean")


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pick(node: dict, path) -> Optional[float]:
    if isinstance(path, str):
        return _num((node or {}).get(path))
    cursor = node or {}
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return _num(cursor)


def _validation_candidates(metrics: dict) -> list:
    keys = [k for k in (metrics or {}) if k.startswith("val-core/") and "/mean@" in k]
    return sorted(keys)


def _metadata_health_signals(health: dict, observed_requirements: Optional[list]):
    """返回 (hard_missing, hard_partial, fingerprint_coverage).

    hard_missing: 本窗口完全没采集的必需 metadata (health 中为 None 或缺失).
    hard_partial: 采集了但覆盖率 < 1.0 的必需 metadata -> {名称: 平均覆盖}.
    fingerprint_coverage: eval_workbook_fingerprint 的平均可用率 (仅在有要求时返回数值).
    """
    reqs = list(observed_requirements or [])
    hard_missing: dict = {}
    hard_partial: dict = {}
    fingerprint_coverage: Optional[float] = None
    if not reqs:
        # 没给 required_metadata 时, 退回旧的启发式: 只读 eval_workbook_fingerprint 是否有值.
        fp = health.get("eval_workbook_fingerprint")
        return {}, {}, _num(fp) if fp is not None else None
    for name in reqs:
        key = HEALTH_KEY_OF.get(name)
        if not key:
            continue
        value = _num(health.get(key))
        if key == "eval_workbook_fingerprint":
            fingerprint_coverage = value if value is not None else 0.0
        if name == "eval_fingerprint" and value is None:
            # RewardFlow: fingerprint 软要求. 完全缺失 = hard_missing, 部分 = hard_partial.
            hard_missing[name] = 0.0
            continue
        if value is None:
            hard_missing[name] = None
        elif value < 1.0:
            hard_partial[name] = value
    return hard_missing, hard_partial, fingerprint_coverage


@dataclass
class EvaluationResult:
    evaluation_id: str
    decision_id: str
    step: int

    previous_skill: str
    previous_parameters: dict
    applied_skill: str
    applied_parameters: dict
    action: str

    baseline_window: dict = field(default_factory=dict)
    observed_window: dict = field(default_factory=dict)
    metric_deltas: dict = field(default_factory=dict)
    validation_comparison: dict = field(default_factory=dict)

    metadata_health: dict = field(default_factory=dict)
    evidence_quality: str = QUALITY_LOW
    evidence_notes: list = field(default_factory=list)

    outcome: str = OUTCOME_INSUFFICIENT
    primary_metric: Optional[str] = None
    eligible_for_knowledge_update: bool = False
    evaluated_at_step: Optional[int] = None
    summary: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class DecisionEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def select_validation_key(self, metrics: dict) -> tuple[Optional[str], Optional[list]]:
        """返回 (选中的 key, 无法唯一决定时的候选列表). 绝不跨 validation dataset 平均."""
        if self.config.validation_metric_key:
            return self.config.validation_metric_key, None
        candidates = _validation_candidates(metrics)
        if len(candidates) == 1:
            return candidates[0], None
        if not candidates:
            return None, []
        return None, candidates

    def compare_validation(self, baseline: Optional[dict], observed: Optional[dict]) -> dict:
        if not self.config.use_existing_validation:
            return {"status": "disabled"}
        if not observed or not observed.get("metrics"):
            return {"status": "unavailable", "reason": "no validation run was recorded in this window"}
        if not baseline or not baseline.get("metrics"):
            return {"status": "unavailable", "reason": "no baseline validation snapshot"}
        if baseline.get("step") == observed.get("step"):
            # 同一 step 的 cached observation 不能当作新证据.
            return {"status": "stale", "step": observed.get("step")}
        key, candidates = self.select_validation_key(observed["metrics"])
        if key is None and candidates:
            return {
                "status": "ambiguous",
                "candidates": candidates[:8],
                "reason": "multiple validation metric keys; set self_improving.evaluation.validation_metric_key",
                "per_dataset": {c: _num(observed["metrics"].get(c)) for c in candidates[:8]},
            }
        if key is None:
            return {"status": "unavailable", "reason": "no val-core metric found"}
        before = _num(baseline["metrics"].get(key))
        after = _num(observed["metrics"].get(key))
        if before is None or after is None:
            return {"status": "unavailable", "key": key, "reason": "selected key missing in one of the snapshots"}
        return {
            "status": "compared",
            "key": key,
            "baseline_step": baseline.get("step"),
            "observed_step": observed.get("step"),
            "baseline_value": before,
            "observed_value": after,
            "delta": round(after - before, 6),
        }

    def evaluate(
        self,
        evaluation_id: str,
        decision: dict,
        baseline_window: dict,
        observed_window: dict,
        baseline_validation: Optional[dict],
        observed_validation: Optional[dict],
        step: int,
        observed_requirements: Optional[list] = None,
    ) -> EvaluationResult:
        result = EvaluationResult(
            evaluation_id=evaluation_id,
            decision_id=str(decision.get("decision_id")),
            step=int(step),
            previous_skill=str(decision.get("previous_skill", {}).get("method")),
            previous_parameters=dict(decision.get("previous_skill", {}).get("params", {})),
            applied_skill=str(decision.get("applied_skill", decision.get("skill"))),
            applied_parameters=dict(decision.get("applied_parameters", decision.get("params", {}))),
            action=str(decision.get("action")),
        )
        baseline_window = baseline_window or {}
        observed_window = observed_window or {}
        result.baseline_window = {
            "step_range": baseline_window.get("step_range"),
            "num_steps": baseline_window.get("num_steps"),
            "skills": baseline_window.get("skills"),
            "raw_reward": baseline_window.get("raw_reward"),
            "success_rate": _num(baseline_window.get("success_rate")),
            "trajectory_length": baseline_window.get("trajectory_length"),
            "advantage": baseline_window.get("advantage"),
        }
        result.observed_window = {
            "step_range": observed_window.get("step_range"),
            "num_steps": observed_window.get("num_steps"),
            "skills": observed_window.get("skills"),
            "raw_reward": observed_window.get("raw_reward"),
            "success_rate": _num(observed_window.get("success_rate")),
            "trajectory_length": observed_window.get("trajectory_length"),
            "advantage": observed_window.get("advantage"),
            "steps_per_trajectory": observed_window.get("steps_per_trajectory"),
        }
        result.metadata_health = {
            "baseline": baseline_window.get("metadata_health"),
            "observed": observed_window.get("metadata_health"),
        }

        deltas: dict[str, Optional[float]] = {}
        for name, path, _ in list(PRIMARY_METRICS) + list(SECONDARY_METRICS):
            before = _pick(baseline_window, path)
            after = _pick(observed_window, path)
            deltas[name] = round(after - before, 6) if before is not None and after is not None else None
        result.metric_deltas = deltas

        validation = self.compare_validation(baseline_validation, observed_validation)
        result.validation_comparison = validation

        notes, quality = self._assess_evidence(baseline_window, observed_window, deltas, validation, decision, observed_requirements)
        result.evidence_notes = notes
        result.evidence_quality = quality
        result.outcome, result.primary_metric = self._classify(deltas, validation, quality)
        result.eligible_for_knowledge_update = result.outcome != OUTCOME_INSUFFICIENT and quality != QUALITY_LOW
        result.summary = self._summarize(result)
        return result

    def _assess_evidence(self, baseline: dict, observed: dict, deltas: dict, validation: dict, decision: dict, observed_requirements: Optional[list] = None):
        notes: list[str] = []
        min_steps = self.config.min_window_steps
        observed_steps = _num(observed.get("num_steps")) or 0
        baseline_steps = _num(baseline.get("num_steps")) or 0
        if observed_steps < min_steps or baseline_steps < min_steps:
            notes.append(f"window shorter than min_window_steps={min_steps} (observed={int(observed_steps)}, baseline={int(baseline_steps)})")
        if len(observed.get("skills") or []) > 1:
            notes.append(f"observed window mixed multiple skill configurations: {observed.get('skills')}")
        if decision.get("status") == "fallback":
            notes.append("the window started from a failed/invalid controller decision")

        available = [name for name in CORE_PRIORITY if deltas.get(name) is not None]
        if not available and not self.config.use_training_metrics:
            notes.append("training metrics are disabled and no validation comparison exists")
        if not available:
            notes.append("no comparable core training metric (success_rate / raw_reward_mean) in either window")

        # §M-3: 区分"必需 metadata 完全未采集" vs "采集了但部分覆盖".
        # runtime 传入 observed_requirements 时逐项诊断; 为空时沿用旧的启发式 (兼容既有调用/测试).
        health = observed.get("metadata_health") or {}
        if observed_requirements:
            hard_missing, hard_partial, fingerprint_coverage = _metadata_health_signals(health, observed_requirements)
            if hard_missing:
                notes.append(
                    "required metadata completely absent in the observed window: "
                    + ", ".join(sorted(hard_missing))
                )
            if hard_partial:
                notes.append(
                    "required metadata only partially available: "
                    + ", ".join(f"{k}≈{v:.2f}" for k, v in sorted(hard_partial.items()))
                )
            if fingerprint_coverage is not None and fingerprint_coverage <= 0.0:
                notes.append("evaluation-side workbook fingerprints were not available in the observed window")
        else:
            if health.get("eval_fingerprint_any") is False:
                notes.append("evaluation-side workbook fingerprints were not available in the observed window")
            trans = _num(health.get("step_transitions"))
            if trans is not None and trans < 0.5:
                notes.append(f"structured state metadata coverage is low in the observed window ({trans:.2f})")

        if validation.get("status") == "stale":
            notes.append("validation observation is stale (same step as the baseline snapshot)")
        elif validation.get("status") == "ambiguous":
            notes.append("validation metric key is ambiguous across datasets; validation evidence not used")

        # 方向冲突: 两个核心训练指标, 或训练与 validation, 明显反向.
        signs = [
            1 if deltas.get(name, 0) > self.config.improvement_threshold else -1 if deltas.get(name, 0) < -self.config.improvement_threshold else 0
            for name in CORE_PRIORITY
            if deltas.get(name) is not None
        ]
        if validation.get("status") == "compared":
            v_delta = _num(validation.get("delta")) or 0.0
            if abs(v_delta) > self.config.improvement_threshold:
                signs.append(1 if v_delta > 0 else -1)
        if any(s > 0 for s in signs) and any(s < 0 for s in signs):
            notes.append("core performance indicators point in opposite directions")

        if notes:
            return notes, QUALITY_LOW
        # 算法零优势 / 零总优势 不一定为错; GRPO 全同组 / GiGPO singleton 都合理. 只在诊断中记录, 不作低证据理由.
        if validation.get("status") == "compared":
            return notes, QUALITY_HIGH
        return notes, QUALITY_MEDIUM

    def _classify(self, deltas: dict, validation: dict, quality: str) -> tuple:
        if quality == QUALITY_LOW:
            return OUTCOME_INSUFFICIENT, None
        threshold = self.config.improvement_threshold
        primary = None
        delta = None
        if validation.get("status") == "compared":
            primary = f"validation:{validation.get('key')}"
            delta = _num(validation.get("delta"))
        if delta is None:
            for name in CORE_PRIORITY:
                if deltas.get(name) is not None:
                    primary, delta = name, deltas[name]
                    break
        if delta is None:
            return OUTCOME_INSUFFICIENT, None
        if delta > threshold:
            return OUTCOME_POSITIVE, primary
        if delta < -threshold:
            return OUTCOME_NEGATIVE, primary
        return OUTCOME_NEUTRAL, primary

    def _summarize(self, result: EvaluationResult) -> str:
        delta = None
        if result.primary_metric and result.primary_metric.startswith("validation:"):
            delta = _num(result.validation_comparison.get("delta"))
        elif result.primary_metric:
            delta = result.metric_deltas.get(result.primary_metric)
        bits = [
            f"skill '{result.applied_skill}' observed over steps "
            f"{(result.observed_window.get('step_range') or [None, None])[0]}-{(result.observed_window.get('step_range') or [None, None])[1]}",
            f"outcome={result.outcome}",
        ]
        if result.primary_metric:
            bits.append(f"primary={result.primary_metric} delta={delta}")
        bits.append(f"evidence={result.evidence_quality}")
        bits.append("observational only (not a causal effect estimate)")
        return "; ".join(bits)
