"""Credit Skill Controller: 一次 LLM 调用同时完成 selection + calibration.

LLM 的输出只是"提案", 必须经 SkillRuntime 校验后才会写进 driver config. 任何解析/校验
失败都保持当前 Skill (execution safety fallback, 不是 performance rollback) 并记录原因.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.trainer.credit_assignment.self_improving.config import SelfImprovingConfig
from verl.trainer.credit_assignment.self_improving.context import jsonable
from verl.trainer.credit_assignment.self_improving.llm_client import (
    LLMError,
    LLMInvalidResponse,
    check_schema,
    parse_json_object,
)
from verl.trainer.credit_assignment.self_improving.runtime import SkillConfig, SkillRuntime, SkillValidationError

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
VALID_ACTIONS = ("keep", "switch", "calibrate")
MAX_REASONING_CHARS = 2000
MAX_EVIDENCE_ITEMS = 8
RECENT_HISTORY = 6

DECISION_SCHEMA = {
    "action": str,
    "selected_skill": str,
    "parameters": dict,
    "reasoning": (str, False),
    "evidence": (list, False),
    "expected_improvement": (str, False),
}


@dataclass
class ControllerDecision:
    decision_id: str
    step: int
    action: str
    skill: SkillConfig
    previous_skill: SkillConfig
    reasoning: str = ""
    evidence: list = field(default_factory=list)
    expected_improvement: Optional[str] = None
    status: str = "applied"
    error: Optional[str] = None
    knowledge_version: Optional[int] = None
    llm_latency_seconds: Optional[float] = None
    raw_response: Optional[dict] = None

    @property
    def changed(self) -> bool:
        return self.skill.method != self.previous_skill.method or self.skill.params != self.previous_skill.params

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "step": self.step,
            "action": self.action,
            "status": self.status,
            "error": self.error,
            "previous_skill": self.previous_skill.to_dict(),
            "applied_skill": self.skill.method,
            "applied_parameters": dict(self.skill.params),
            "changed": self.changed,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "expected_improvement": self.expected_improvement,
            "knowledge_version": self.knowledge_version,
            "llm_latency_seconds": self.llm_latency_seconds,
        }


class CreditSkillController:
    def __init__(
        self,
        config: SelfImprovingConfig,
        runtime: SkillRuntime,
        llm: Any,
        prompt_dir: str = PROMPT_DIR,
    ):
        self.config = config
        self.runtime = runtime
        self.llm = llm
        self._template = self._load_template(prompt_dir)

    @staticmethod
    def _load_template(prompt_dir: str) -> str:
        path = os.path.join(prompt_dir, "controller.md")
        with open(path, encoding="utf-8") as f:
            return f.read()

    # ---------- prompt ----------
    def build_messages(
        self,
        context: dict,
        knowledge_block: str,
        active: SkillConfig,
        previous_evaluation: Optional[dict] = None,
        recent_history: Optional[list] = None,
    ) -> list:
        spec = self.runtime.calibration_spec()
        rendered = (
            self._template.replace("{{candidate_skills}}", _json(self.config.candidate_skills))
            .replace("{{calibration_spec}}", _format_spec(spec, self.config.candidate_skills))
            .replace("{{knowledge_block}}", knowledge_block or "(no skill knowledge available)")
            .replace("{{current_skill}}", _json(active.to_dict()))
            .replace("{{previous_evaluation}}", _json(previous_evaluation))
            .replace("{{recent_history}}", _json(_trim_history(recent_history or [])))
            .replace("{{training_context}}", _json(_fit_context(context, self.config.context.max_context_chars)))
        )
        return [
            {"role": "system", "content": rendered},
            {
                "role": "user",
                "content": (
                    "Decide the credit assignment skill for the next training window. "
                    "Return only the JSON object described above."
                ),
            },
        ]

    # ---------- decision ----------
    def select_and_calibrate(
        self,
        context: dict,
        knowledge_block: str,
        decision_id: str,
        step: int,
        previous_evaluation: Optional[dict] = None,
        recent_history: Optional[list] = None,
        knowledge_version: Optional[int] = None,
    ) -> ControllerDecision:
        active = self.runtime.current() or self.runtime.read_active()
        messages = self.build_messages(context, knowledge_block, active, previous_evaluation, recent_history)
        try:
            raw = self.llm.chat_json(messages, schema=DECISION_SCHEMA)
        except LLMInvalidResponse as e:
            return self._fallback(decision_id, step, active, f"invalid llm payload: {e}", knowledge_version)
        except LLMError as e:
            return self._fallback(decision_id, step, active, f"llm call failed: {e}", knowledge_version)
        except Exception as e:  # pragma: no cover - 不让未知 LLM 异常打断 RL 训练
            return self._fallback(decision_id, step, active, f"llm call raised {type(e).__name__}: {e}", knowledge_version)

        latency = getattr(self.llm, "last_latency_seconds", None)
        payload = dict(raw)
        if isinstance(payload.get("parameters"), str):
            # 少数 provider 把嵌套对象序列化成字符串.
            try:
                payload["parameters"] = parse_json_object(payload["parameters"])
            except LLMError:
                return self._fallback(decision_id, step, active, "parameters is not a JSON object", knowledge_version, latency)
        return self._validate_payload(payload, decision_id, step, active, knowledge_version, latency, payload)

    def _validate_payload(
        self,
        payload: dict,
        decision_id: str,
        step: int,
        active: SkillConfig,
        knowledge_version: Optional[int],
        latency: Optional[float],
        raw: dict,
    ) -> ControllerDecision:
        action = str(payload.get("action", "")).strip().lower()
        if action not in VALID_ACTIONS:
            return self._fallback(decision_id, step, active, f"invalid action {action!r}", knowledge_version, latency, raw)
        method = payload.get("selected_skill")
        if not isinstance(method, str) or not method.strip():
            return self._fallback(decision_id, step, active, "missing/invalid selected_skill", knowledge_version, latency, raw)
        method = method.strip()
        if action in ("keep", "calibrate") and method != active.method:
            return self._fallback(
                decision_id, step, active, f"action '{action}' is inconsistent with selected_skill '{method}'", knowledge_version, latency, raw
            )
        if action == "switch" and method == active.method:
            return self._fallback(
                decision_id, step, active, f"action 'switch' selected the already active skill '{method}'", knowledge_version, latency, raw
            )

        # 只把 LLM 提交的参数交给 runtime; 未提交的参数沿用当前生效值 (runtime 负责).
        params = payload.get("parameters")
        params = dict(params) if isinstance(params, dict) else {}
        try:
            candidate = self.runtime.validate(SkillConfig(method=method, params=dict(params)), from_llm=True)
        except SkillValidationError as e:
            return self._fallback(decision_id, step, active, f"decision rejected by runtime: {e}", knowledge_version, latency, raw)

        if candidate.method == active.method and candidate.params == active.params:
            # LLM 声称调整但实际没变 -> 归一为 keep, 避免把下一个 window 当成切换效果.
            action = "keep"

        return ControllerDecision(
            decision_id=decision_id,
            step=int(step),
            action=action,
            skill=candidate,
            previous_skill=active.copy(),
            reasoning=_clip(payload.get("reasoning"), MAX_REASONING_CHARS),
            evidence=_clip_evidence(payload.get("evidence")),
            expected_improvement=_clip(payload.get("expected_improvement"), MAX_REASONING_CHARS),
            status="applied",
            knowledge_version=knowledge_version,
            llm_latency_seconds=latency,
            raw_response=raw,
        )

    def _fallback(
        self,
        decision_id: str,
        step: int,
        active: SkillConfig,
        error: str,
        knowledge_version: Optional[int],
        latency: Optional[float] = None,
        raw: Optional[dict] = None,
    ) -> ControllerDecision:
        return ControllerDecision(
            decision_id=decision_id,
            step=int(step),
            action="keep",
            skill=active.copy(),
            previous_skill=active.copy(),
            reasoning="(kept current skill)",
            status="fallback",
            error=error,
            knowledge_version=knowledge_version,
            llm_latency_seconds=latency,
            raw_response=raw if isinstance(raw, dict) else None,
        )


def _json(value: Any) -> str:
    if value is None:
        return "null"
    return json.dumps(jsonable(value), ensure_ascii=False, indent=2, default=str)


def _clip(value: Any, limit: int) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def _clip_evidence(value: Any) -> list:
    if not isinstance(value, list):
        return [] if value is None else [_clip(value, 300)]
    return [_clip(item, 300) for item in value[:MAX_EVIDENCE_ITEMS] if item is not None]


def _format_spec(spec: dict, methods: list) -> str:
    lines = []
    for method in methods:
        entry = spec.get(method) or {}
        calibratable = entry.get("calibratable") or {}
        body = (
            ", ".join(f"{k}: {v['domain']} (default {v['default']!r})" for k, v in sorted(calibratable.items()))
            or "(no parameters are calibratable)"
        )
        lines.append(f"- {method}\n  defaults: {json.dumps(entry.get('defaults'), default=str)}\n  calibratable: {body}")
    return "\n".join(lines)


def _trim_history(history: list) -> list:
    out = []
    for record in history[-RECENT_HISTORY:]:
        if not isinstance(record, dict):
            continue
        out.append(
            {
                k: record.get(k)
                for k in (
                    "record_type",
                    "decision_id",
                    "step",
                    "action",
                    "applied_skill",
                    "applied_parameters",
                    "outcome",
                    "metric_deltas",
                    "evidence_quality",
                    "error",
                )
                if record.get(k) is not None
            }
        )
    return out


def _fit_context(context: dict, max_chars: int) -> dict:
    """context 已由 ContextCollector 限过预算; 这里只兜底防止 prompt 过长."""
    if not isinstance(context, dict):
        return {}
    try:
        size = len(_json(context))
    except (TypeError, ValueError):
        size = max_chars + 1
    if size <= max_chars:
        return context
    trimmed = {k: v for k, v in context.items() if k != "trajectory_samples"}
    trimmed["trajectory_samples"] = context.get("trajectory_samples", [])[:1]
    trimmed["trajectory_samples_truncated"] = True
    return trimmed
