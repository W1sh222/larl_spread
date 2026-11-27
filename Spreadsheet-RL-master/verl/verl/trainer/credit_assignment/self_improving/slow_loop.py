"""Self-Improving Workflow 的单一 Orchestrator.

trainer 只接触四个入口: initialize / on_step_end / on_checkpoint_saved / finalize.
其余流程 (context -> evaluation -> knowledge update -> controller -> apply -> persist) 都在本文件.

Slow Loop 自身的失败不打断 RL 训练: 记录原因并保持当前 Skill (execution safety, 不是回滚),
失败的介入只消耗本次槽位, 到下一个预定 interval 再重试 (不会本 run 内永久停用).
状态不一致 / 反复落盘失败属于无法自证正确的情形 -> fail-fast 向上抛, 不藏在大范 exception 里.
"""

import json
import os
import time
import traceback
from typing import Any, Optional

from verl.trainer.credit_assignment.self_improving.config import (
    SelfImprovingConfig,
    SelfImprovingConfigError,
    load_self_improving_config,
)
from verl.trainer.credit_assignment.self_improving.context import ContextCollector, jsonable
from verl.trainer.credit_assignment.self_improving.controller import CreditSkillController
from verl.trainer.credit_assignment.self_improving.evaluator import (
    HEALTH_KEY_OF,
    DecisionEvaluator,
    EvaluationResult,
    OUTCOME_INSUFFICIENT,
    QUALITY_LOW,
)
from verl.trainer.credit_assignment.self_improving.knowledge import (
    KnowledgeError,
    KnowledgeStore,
    validate_skill_markdown,
)
from verl.trainer.credit_assignment.self_improving.llm_client import LLMError, build_llm_client
from verl.trainer.credit_assignment.self_improving.persistence import (
    RESEARCH_CRITICAL,
    SelfImprovingStateError,
    SelfImprovingStore,
    config_diff,
    experiment_identity,
    new_run_id,
)
from verl.trainer.credit_assignment.self_improving.runtime import (
    SkillConfig,
    SkillRuntime,
    SkillValidationError,
    bootstrap_metadata_signature,
    required_metadata,
)

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
SUBDIR = "self_improving"
# 连续落盘失败达到该上限视为不可恢复 -> fail-fast; Slow Loop 自身的应用逻辑失败不走这里.
PERSISTENCE_FAILURE_LIMIT = 3
VALIDATION_PREFIXES = ("val-core/", "val-aux/")
# 不能当成"Slow Loop 小问题"吞掉的错误: 状态不一致 / 配置非法 / 资源耗尽.
FATAL_ERRORS = (SelfImprovingStateError, SelfImprovingConfigError, MemoryError)
KNOWLEDGE_UPDATE_SCHEMA = {
    "update_required": bool,
    "target_skill": (str, False),
    "experience": (dict, False),
}


class SelfImprovingLoop:
    def __init__(self, llm: Any = None, prompt_dir: str = PROMPT_DIR):
        self.llm = llm
        self.prompt_dir = prompt_dir
        self.config: Optional[SelfImprovingConfig] = None
        self.trainer = None
        self.runtime: Optional[SkillRuntime] = None
        self.collector: Optional[ContextCollector] = None
        self.store: Optional[SelfImprovingStore] = None
        self.knowledge: Optional[KnowledgeStore] = None
        self.controller: Optional[CreditSkillController] = None
        self.evaluator: Optional[DecisionEvaluator] = None

        self.run_id = None
        self.active: Optional[SkillConfig] = None  # 下一步将使用的 Skill
        self.executed: Optional[SkillConfig] = None  # 当前 step 的 advantage 实际使用的 Skill
        self.window_start_step = None
        self.last_intervention_step = 0
        self.intervention_count = 0
        self.pending: Optional[dict] = None
        self.latest_validation: Optional[dict] = None
        self.validation_baseline: Optional[dict] = None
        self.carryover_window: Optional[dict] = None
        self.bootstrap: Optional[dict] = None
        self._pending_snapshot_step: Optional[int] = None
        self._windows_since_update = 0
        self._failure_streak = 0
        self._persistence_failures = 0
        self._last_persistence_error: Optional[str] = None
        self._finalized = False
        self._resume_config_changes: list = []
        self._restored_knowledge_version: Optional[int] = None
        self._restored_state: Optional[dict] = None
        self.state_misaligned = False
        self.history_max_step: Optional[int] = None
        self.resume_report: Optional[dict] = None
        self.last_context: Optional[dict] = None

    # ------------------------------------------------------------------ init
    def initialize(self, trainer) -> "SelfImprovingLoop":
        """配置错误 / Resume 状态不一致在这里直接 raise: 不允许带着未知状态开始训练."""
        self.trainer = trainer
        node = trainer.config.get("self_improving", None) if hasattr(trainer.config, "get") else None
        self.config = load_self_improving_config(node)
        if not self.config.enabled:
            raise SelfImprovingStateError("SelfImprovingLoop.initialize() called while self_improving.enabled=false")

        algorithm_cfg = trainer.config.algorithm
        # worker 的采集范围在初始化时冻结: 之后 Driver 换 Skill 也不会重新采集, 因此签名只能取 launch 时的配置.
        self.bootstrap = bootstrap_metadata_signature(algorithm_cfg)
        self.runtime = SkillRuntime(self.config, algorithm_cfg)
        self.collector = ContextCollector(self.config.context, tokenizer=getattr(trainer, "tokenizer", None))

        root = os.path.join(str(trainer.config.trainer.default_local_dir), SUBDIR)
        self.run_id = new_run_id()
        ckpt_step = int(getattr(trainer, "global_steps", 0) or 0)
        resuming = ckpt_step > 0
        self.store = SelfImprovingStore(
            root,
            strict_resume=self.config.persistence.strict_resume,
            enabled=self.config.persistence.enabled,
        )
        self.store.prepare(self.run_id, resuming)
        self.knowledge = KnowledgeStore(
            run_dir=root,
            candidate_skills=list(self.config.candidate_skills),
            max_prompt_chars=self.config.knowledge.max_prompt_chars,
            lineage=self.run_id,
        )
        if self.llm is None:
            self.llm = build_llm_client(self.config.llm)
        self.controller = CreditSkillController(self.config, self.runtime, self.llm, prompt_dir=self.prompt_dir)
        self.evaluator = DecisionEvaluator(self.config.evaluation)
        self._knowledge_template = self._load_knowledge_template()

        restored = self._restore_state(resuming, ckpt_step)
        # 先检查实验一致性, 再写任何产物: 不兼容时目录保持原样, 便于换回正确的 checkpoint 重试.
        self._check_resume_compatibility(restored, ckpt_step)
        self.store.write_config(
            self.config.to_dict(),
            experiment_identity(trainer.config),
            self.bootstrap,
        )
        # bootstrap / load_index 必须在恢复之后: 开新 lineage 时旧 knowledge 目录会被整体归档.
        self._check_knowledge_templates(ckpt_step)
        self._restore_knowledge(ckpt_step)
        target = self._restored_skill(restored)
        self.active = self.runtime.apply_initial(restore=target)
        self.executed = self.active.copy()
        self.window_start_step = ckpt_step + 1
        if isinstance(restored, dict) and restored.get("window_start_step") is not None:
            self.window_start_step = max(self.window_start_step, int(restored["window_start_step"]))
        self.collector.reset(self.window_start_step)

        self._history(
            {
                "record_type": "initialized",
                "step": ckpt_step,
                "run_id": self.run_id,
                "lineage": self.store.lineage,
                "resuming": resuming,
                "active_skill": self.active.to_dict(),
                "candidate_skills": list(self.config.candidate_skills),
                "metadata_bootstrap": self.bootstrap,
                "resume": self.resume_report,
                "interval_steps": self.config.interval_steps,
                "last_intervention_step": self.last_intervention_step,
            }
        )
        if self._resume_config_changes:
            self._history(
                {
                    "record_type": "config_change",
                    "step": ckpt_step,
                    "lineage": self.store.lineage,
                    "changes": jsonable(self._resume_config_changes),
                    "reason": "configuration of this session differs from the resumed self_improving state",
                }
            )
        print(
            f"[self_improving] enabled: run={self.run_id} initial/active skill="
            f"{self.active.method} params={self.active.params} interval={self.config.interval_steps} "
            f"collected_metadata={sorted(self.runtime.collected) or 'none'} "
            f"candidates={list(self.config.candidate_skills)} knowledge_version={self.knowledge.version}",
            flush=True,
        )
        if self.resume_report:
            print(f"[self_improving] resume: {json.dumps(self.resume_report, default=str)}", flush=True)
        return self

    def _load_knowledge_template(self) -> str:
        path = os.path.join(self.prompt_dir, "knowledge_update.md")
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""

    def _check_knowledge_templates(self, resume_step: int) -> None:
        """Knowledge 只是提示词素材: 缺章节只警告, 不改变可执行参数空间."""
        self.knowledge.bootstrap(int(resume_step))
        # resume 时不自动采纳 index 里的最新 (可能是未来) 版本.
        self.knowledge.load_index(not_after_step=resume_step if self.resume_report else None)
        for skill in self.config.candidate_skills:
            missing = validate_skill_markdown(self.knowledge.read(skill))
            if missing:
                print(f"[self_improving] warning: skill knowledge '{skill}' is missing sections {missing}", flush=True)

    def _restore_state(self, resuming: bool, ckpt_step: int) -> Optional[dict]:
        if not resuming:
            return None
        result = self.store.resume_state(ckpt_step)
        if not result:
            self.resume_report = {
                "resumed_from_step": ckpt_step,
                "state_source": None,
                "exact": False,
                "reason": (
                    "a policy checkpoint exists but no self_improving state does (the checkpoint was written "
                    "by a run without self_improving, or its directory was removed); starting a new window with "
                    "self_improving.initial_skill and last_intervention_step set to the resumed step"
                ),
            }
            self.last_intervention_step = ckpt_step
            return None
        state = result["state"] or None
        if state is None:
            # 盘上只剩比 checkpoint 更晚的状态: 从 initial template 开新 lineage, 与旧 knowledge / history 隔离.
            prior = self.store.start_new_lineage(
                f"no self_improving state at or before policy checkpoint step {ckpt_step}"
            )
            self.resume_report = {
                "resumed_from_step": ckpt_step,
                "state_source": None,
                "exact": False,
                "start_fresh": True,
                "prior_lineage": prior,
                "reason": result.get("reason"),
            }
            self.state_misaligned = True
            self.last_intervention_step = ckpt_step
            return None
        self.resume_report = {
            "resumed_from_step": ckpt_step,
            "state_source": result.get("source"),
            "exact": result.get("exact"),
            "missing_step": result.get("missing_step"),
            "restored_step": result.get("restored_step"),
            "reason": result.get("reason"),
            "state_lineage": state.get("lineage"),
        }
        # 恢复之后写的记录属于同一个 lineage, 但 session (run_id) 不同.
        if state.get("lineage") and self.store.lineage and state["lineage"] != self.store.lineage:
            self.resume_report["lineage_note"] = f"resumed state lineage '{state['lineage']}'"
        self.state_misaligned = not bool(result.get("exact"))
        # 同目录里可能有上一个 session 写到更晚 step 的记录, 它们对本次恢复来说是未来信息.
        self.history_max_step = ckpt_step
        legacy = state.get("slow_loop_disabled")
        if legacy:
            # 旧 schema 的永久停用标记: 本版本不再生效, 只记录不执行.
            print(f"[self_improving] note: restored state has slow_loop_disabled={legacy!r}; the slow loop no "
                  "longer disables itself permanently, so the flag is ignored (interventions keep retrying)", flush=True)
            self.resume_report["legacy_slow_loop_disabled"] = legacy
        self._restored_state = state
        self.last_intervention_step = int(state.get("last_intervention_step") or ckpt_step)
        self.intervention_count = int(state.get("intervention_count") or 0)
        self.pending = state.get("pending_decision") or None
        window = state.get("current_window_summary") or None
        # checkpoint 刚落在 intervention 之后时 summary 是空窗口, 不算"被 resume 打断".
        if window and not int(window.get("num_steps") or 0):
            window = None
        self.carryover_window = window
        self.latest_validation = state.get("latest_validation") or None
        self.validation_baseline = state.get("validation_baseline") or None
        self._windows_since_update = int(state.get("windows_since_knowledge_update") or 0)
        self._restored_knowledge_version = state.get("knowledge_version")
        if not result.get("exact"):
            print(
                f"[self_improving] restored state is NOT exactly aligned with policy checkpoint "
                f"step {ckpt_step}: {result.get('reason')}",
                flush=True,
            )
        return state

    def _restore_knowledge(self, ckpt_step: int) -> None:
        version = self._restored_knowledge_version
        if version is None or self.resume_report is None:
            return
        try:
            self.knowledge.adopt(int(version), ckpt_step)
        except KnowledgeError as e:
            if self.config.persistence.strict_resume:
                raise SelfImprovingStateError(f"cannot restore knowledge version {version}: {e}") from e
            self.knowledge.reset_to_templates()
            print(f"[self_improving] NON-STRICT RESUME: knowledge version {version} unavailable ({e}); "
                  "knowledge/current/ was reset to the read-only templates instead of keeping newer content",
                  flush=True)
            self.resume_report["knowledge_version_restored"] = False
            self.resume_report["knowledge_reset_to_templates"] = True
            self.state_misaligned = True
        else:
            self.resume_report["knowledge_version_restored"] = True

    def _check_resume_compatibility(self, state: Optional[dict], ckpt_step: int) -> None:
        """§M-6: resume 必须还是同一个实验. 运营设置可变 (只记录), 研究语义设置不一致 -> 报错."""
        self._resume_config_changes = []
        if not state:
            return
        current = experiment_identity(self.trainer.config)
        stored = state.get("experiment_identity")
        problems: list[str] = []
        drift: list[str] = []
        if not isinstance(stored, dict) or not stored:
            drift.append("the resumed state recorded no experiment_identity (older schema)")
        else:
            for key in RESEARCH_CRITICAL:
                if key in stored and stored[key] != current[key]:
                    problems.append(f"{key}: resumed run has {stored[key]!r}, this session has {current[key]!r}")
        stored_sig = state.get("metadata_collection_signature")
        if isinstance(stored_sig, dict) and stored_sig.get("collected") is not None:
            before = sorted(stored_sig.get("collected") or [])
            now = sorted(self.bootstrap.get("collected") or [])
            if before != now:
                problems.append(
                    "metadata_collection_signature: this session's rollout would collect "
                    f"{now} but the resumed run collected {before}; candidate skills would silently lose "
                    "required metadata"
                )
            elif stored_sig.get("bootstrap_method") != self.bootstrap.get("bootstrap_method"):
                drift.append(
                    f"metadata bootstrap method changed from {stored_sig.get('bootstrap_method')!r} to "
                    f"{self.bootstrap.get('bootstrap_method')!r} while collecting the same metadata"
                )
        else:
            drift.append("the resumed state recorded no metadata_collection_signature (older schema)")
        self._resume_config_changes = config_diff(state.get("config_summary"), self.config.to_dict())
        operational = [c["key"] for c in self._resume_config_changes if c["kind"] == "operational"]
        research = [c["key"] for c in self._resume_config_changes if c["kind"] == "research"]
        if not problems and not drift and not research and not operational:
            return
        detail = (
            f"incompatible: {problems}" if problems else ""
        ) + (f" unrecorded-then: {drift}" if drift else "") + (f" research config changed: {research}" if research else "")
        if problems and self.config.persistence.strict_resume:
            raise SelfImprovingStateError(
                f"resume at step {ckpt_step} does not match the experiment that produced the checkpoint{detail}. "
                "strict_resume=true refuses to continue because the active skill / candidate set / metadata "
                "collection of this session would contradict the restored policy. Resume the matching launch "
                "config, or set self_improving.persistence.strict_resume=false to accept the mismatch "
                "(subsequent windows are then not usable as knowledge evidence)."
            )
        print(f"[self_improving] resume compatibility check at step {ckpt_step}:{detail}"
              + (f"; operational settings changed (allowed, recorded): {operational}" if operational else ""),
              flush=True)
        if self.resume_report is not None:
            self.resume_report["identity_problems"] = problems
            self.resume_report["identity_drift"] = drift
            self.resume_report["research_config_changes"] = research
            self.resume_report["operational_config_changes"] = operational
        if problems:
            # 非严格模式下继续训练, 但不允许把不一致的窗口当成知识依据.
            self.state_misaligned = True

    def _restored_skill(self, state: Optional[dict]) -> Optional[SkillConfig]:
        if not state or not state.get("active_skill"):
            return None
        return SkillConfig(
            method=str(state["active_skill"]),
            params=dict(state.get("active_parameters") or {}),
        )

    # ------------------------------------------------------------- step end
    def on_step_end(
        self,
        batch,
        metrics: dict,
        timing_raw: Optional[dict] = None,
        global_step: int = 0,
        is_last_step: bool = False,
    ) -> None:
        started = time.monotonic()
        emitted: dict[str, float] = {}
        step = int(global_step)
        try:
            self._step(batch, metrics or {}, step, bool(is_last_step), emitted)
        except FATAL_ERRORS:
            # 状态不一致 / 反复落盘失败: 带着明确原因停下, 不让 SI 替它遮丑.
            raise
        except Exception as e:  # Slow Loop 自身故障不得终止 RL 训练
            self._record_failure("on_step_end", e, step, emitted)
        finally:
            emitted["self_improving/slow_loop_seconds"] = time.monotonic() - started
            emitted["self_improving/intervention_triggered"] = float(emitted.pop("_triggered", 0.0))
            # 失败计数只用于诊断: 不再作为停用条件.
            emitted["self_improving/consecutive_failures"] = float(self._failure_streak)
            emitted["self_improving/persistence_failures"] = float(self._persistence_failures)
            for key, value in emitted.items():
                if isinstance(value, bool):
                    value = float(value)
                if isinstance(value, (int, float)):
                    metrics[key] = value

    def _step(self, batch, metrics: dict, global_step: int, is_last_step: bool, emitted: dict) -> None:
        self.executed = self.runtime.read_active()
        self._capture_validation(metrics, global_step)
        summary = self.collector.observe(
            batch,
            metrics,
            global_step,
            skill=self.executed.method,
            params=self.executed.params,
        )
        emitted["self_improving/executed_skill_id"] = self._skill_code(self.executed.method)
        emitted["self_improving/next_skill_id"] = self._skill_code(self.executed.method)
        emitted["self_improving/intervention_count"] = float(self.intervention_count)
        emitted["self_improving/knowledge_version"] = float(self.knowledge.version)
        ratio = self._metadata_ratio(summary.get("metadata_health") or {}, self.executed)
        if ratio is not None:
            emitted["self_improving/metadata_available_ratio"] = ratio
        gaps = self._required_metadata_gaps(summary.get("metadata_health") or {}, self.executed)
        emitted["self_improving/metadata_missing_count"] = float(sum(1 for v in gaps.values() if v is None))
        emitted["self_improving/metadata_partial_count"] = float(sum(1 for v in gaps.values() if v is not None))

        if self._intervention_due(global_step):
            self._attempt_intervention(global_step, is_last_step, emitted)

        self._persist(global_step)

    def _attempt_intervention(self, global_step: int, is_last_step: bool, emitted: dict) -> None:
        # 失败的介入同样占用本轮 interval 槽位: 下一个预定周期再重试, 不在几个 step 内反复碰壁.
        try:
            self._intervene(global_step, is_last_step, emitted)
        except Exception:
            self.last_intervention_step = int(global_step)
            raise
        self._failure_streak = 0

    def _intervention_due(self, global_step: int) -> bool:
        if global_step <= self.last_intervention_step:
            return False  # 同一 step 不重复介入 (含 resume 后重复回放)
        return global_step - self.last_intervention_step >= self.config.interval_steps

    def _skill_code(self, method: str) -> float:
        skills = self.config.candidate_skills
        return float(skills.index(method)) if method in skills else -1.0

    def _metadata_ratio(self, health: dict, skill: SkillConfig) -> Optional[float]:
        values = []
        for name in required_metadata(skill.method, skill.params):
            key = HEALTH_KEY_OF.get(name)
            value = health.get(key) if key else None
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return sum(values) / len(values) if values else None

    def _required_metadata_gaps(self, health: dict, skill: SkillConfig) -> dict:
        """区分"没采集"与"采集了但覆盖不全", 两者对证据质量的含义不同."""
        gaps: dict[str, Any] = {}
        for name in sorted(required_metadata(skill.method, skill.params)):
            key = HEALTH_KEY_OF.get(name)
            if not key:
                continue
            value = health.get(key)
            if isinstance(value, bool):
                value = float(value)
            if value is None:
                gaps[name] = None
            elif isinstance(value, (int, float)) and value < 1.0:
                gaps[name] = round(float(value), 4)
        return gaps

    def _capture_validation(self, metrics: dict, step: int) -> None:
        snapshot = {}
        for key, value in metrics.items():
            if not str(key).startswith(VALIDATION_PREFIXES):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            snapshot[str(key)] = float(value)
        if snapshot:
            self.latest_validation = {"step": int(step), "metrics": snapshot}

    # ------------------------------------------------------------ intervene
    def _intervene(self, global_step: int, is_last_step: bool, emitted: dict) -> None:
        emitted["_triggered"] = 1.0
        observed_window = self.collector.aggregate()
        context = self.collector.build_context(
            global_step=global_step,
            skill=self.executed.method,
            params=self.executed.params,
            latest_validation=self.latest_validation,
            previous_evaluation=None,
            extra_notes=self._context_notes(global_step, observed_window, is_last_step),
        )
        self.last_context = context.to_dict()

        evaluation = self._evaluate(observed_window, global_step)
        context.previous_evaluation = evaluation.to_dict() if evaluation else None
        self.last_context["previous_evaluation"] = jsonable(context.previous_evaluation)
        emitted["self_improving/knowledge_updated"] = 0.0

        update = None
        if evaluation is not None:
            delta = (evaluation.metric_deltas or {}).get("raw_reward_mean")
            if isinstance(delta, (int, float)):
                emitted["self_improving/eval_delta_reward"] = float(delta)
            delta = (evaluation.metric_deltas or {}).get("success_rate")
            if isinstance(delta, (int, float)):
                emitted["self_improving/eval_delta_success_rate"] = float(delta)
            update = self._safe_knowledge_update(context, evaluation, global_step)
            emitted["self_improving/knowledge_updated"] = 1.0 if (update or {}).get("status") == "applied" else 0.0
        decision = None
        if is_last_step:
            self._history(
                {
                    "record_type": "selection_skipped",
                    "step": global_step,
                    "reason": "no further training step exists; running evaluation and persistence only",
                }
            )
        else:
            decision = self._select(context, global_step, emitted)

        # §M-7: apply 已经完成, 之后的记录失败都不能改写"实际生效的 Skill": runtime 是权威状态.
        self.active = self.runtime.current() or self.active
        emitted["self_improving/next_skill_id"] = self._skill_code(self.active.method)
        self._record_intervention(global_step, context, evaluation, decision, update, observed_window, emitted)
        self._start_next_window(global_step, observed_window, decision)
        emitted["self_improving/intervention_count"] = float(self.intervention_count)
        emitted["self_improving/knowledge_version"] = float(self.knowledge.version)

    def _context_notes(self, global_step: int, window: dict, is_last_step: bool) -> dict:
        notes = {
            "interval_steps": self.config.interval_steps,
            "steps_since_last_intervention": int(global_step - self.last_intervention_step),
            "observed_steps_in_window": int(self.collector.observed_steps()),
            "is_last_step": bool(is_last_step),
            "skill_codes": {str(m): self._skill_code(m) for m in self.config.candidate_skills},
            "metadata_bootstrap": self.bootstrap,
        }
        if self.executed is not None:
            notes["required_metadata"] = sorted(
                required_metadata(self.executed.method, self.executed.params)
            )
            coverage = self._required_metadata_gaps(window.get("metadata_health") or {}, self.executed)
            if coverage:
                notes["required_metadata_coverage"] = coverage
        if self.state_misaligned:
            notes["state_not_aligned_with_checkpoint"] = True
        if self.carryover_window:
            notes["window_partially_observed_after_resume"] = True
            notes["pre_resume_window_summary"] = jsonable(self.carryover_window)
        if self.runtime.metadata_gaps():
            notes["metadata_gaps"] = self.runtime.metadata_gaps()
        return notes

    def _evaluate(self, observed_window: dict, global_step: int) -> Optional[EvaluationResult]:
        if not self.pending:
            return None
        decision = self.pending.get("decision") or {}
        baseline = self.pending.get("baseline_window") or {"num_steps": 0}
        result = self.evaluator.evaluate(
            evaluation_id=f"eval-{self.intervention_count + 1:04d}-g{global_step}",
            decision=decision,
            baseline_window=baseline,
            observed_window=jsonable(observed_window),
            baseline_validation=self.pending.get("baseline_validation"),
            observed_validation=self.latest_validation,
            step=global_step,
            observed_requirements=(
                sorted(required_metadata(self.executed.method, self.executed.params)) if self.executed else None
            ),
        )
        result.evaluated_at_step = int(global_step)
        if self.carryover_window:
            result.evidence_notes.append("the observed window was split by a resume; only post-resume steps were re-collected")
        if self.state_misaligned:
            # baseline 来自与 policy checkpoint 不一致的 step (或知识链被重置), 窗口差值不可归因.
            result.evidence_notes.append("state after resume was not exactly aligned with the policy checkpoint")
            result.outcome = OUTCOME_INSUFFICIENT
            result.eligible_for_knowledge_update = False
        if not decision.get("changed", True):
            # keep: 性能变化不能记成新 Skill 变更的经验 (§17.5)
            result.evidence_notes.append("no skill or parameter actually changed in this window; deltas are policy-drift observations only")
            result.outcome = OUTCOME_INSUFFICIENT
            result.eligible_for_knowledge_update = False
        return result

    def _safe_knowledge_update(self, context, evaluation: EvaluationResult, global_step: int) -> Optional[dict]:
        """知识更新失败只丢掉这一条经验: 已存 knowledge / 当前 Skill 一律保持不动."""
        try:
            return self._maybe_update_knowledge(context, evaluation, global_step)
        except KnowledgeError as e:
            return {"status": "failed", "reason": f"knowledge update aborted: {type(e).__name__}: {e}"}
        except OSError as e:
            self._note_persistence_error("knowledge_update", e, global_step)
            return {"status": "failed", "reason": f"knowledge persistence failed: {e}"}

    def _maybe_update_knowledge(self, context, evaluation: EvaluationResult, global_step: int) -> dict:
        target = evaluation.applied_skill
        self._windows_since_update += 1
        if not self.config.knowledge.enabled:
            return {"status": "skipped", "reason": "self_improving.knowledge.enabled=false"}
        if not target or target not in self.config.candidate_skills:
            return {"status": "skipped", "reason": f"no knowledge file for target skill '{target}'"}
        if evaluation.evidence_quality == QUALITY_LOW or evaluation.outcome == OUTCOME_INSUFFICIENT:
            return {"status": "skipped", "reason": f"evidence not usable (quality={evaluation.evidence_quality}, outcome={evaluation.outcome})"}
        if self._windows_since_update < max(1, int(self.config.knowledge.update_interval)):
            return {"status": "deferred", "reason": f"knowledge update interval {self.config.knowledge.update_interval} not reached"}
        if not evaluation.eligible_for_knowledge_update:
            return {"status": "skipped", "reason": "evaluation is not eligible for a knowledge update"}
        payload = evaluation.to_dict()
        payload["evaluated_at_step"] = int(global_step)
        try:
            proposal = self.llm.chat_json(self._knowledge_messages(context, payload, target), schema=KNOWLEDGE_UPDATE_SCHEMA)
        except LLMError as e:
            return {"status": "failed", "reason": f"knowledge llm call failed: {e}", "latency_seconds": getattr(self.llm, "last_latency_seconds", None)}
        applied, entry, version = self.knowledge.apply_update(proposal, payload, global_step, target)
        if not applied:
            return {"status": "rejected", "reason": entry, "latency_seconds": getattr(self.llm, "last_latency_seconds", None)}
        self._windows_since_update = 0
        self._history(
            {
                "record_type": "knowledge_update",
                "step": global_step,
                "target_skill": target,
                "entry_id": entry,
                "knowledge_version": version,
                "decision_id": evaluation.decision_id,
                "evaluation_id": evaluation.evaluation_id,
            }
        )
        return {
            "status": "applied",
            "entry_id": entry,
            "knowledge_version": version,
            "latency_seconds": getattr(self.llm, "last_latency_seconds", None),
        }

    def _knowledge_messages(self, context, evaluation: dict, target: str) -> list:
        rendered = (
            self._knowledge_template.replace("{{evaluation}}", _json(evaluation))
            .replace("{{previous_decision}}", _json((self.pending or {}).get("decision")))
            .replace("{{training_context}}", _json(_clip_chars(jsonable(context.to_dict()), 6000)))
            .replace("{{knowledge}}", self.knowledge.read(target) or "(no knowledge)")
        )
        return [
            {"role": "system", "content": rendered},
            {"role": "user", "content": "Decide whether this observation is worth recording. Return only the JSON object described above."},
        ]

    def _select(self, context, global_step: int, emitted: dict):
        knowledge_block = self.knowledge.prompt_block()
        decision_id = f"decision-{self.intervention_count + 1:04d}-g{global_step}"
        began = time.monotonic()
        try:
            decision = self.controller.select_and_calibrate(
                context=context.to_dict(),
                knowledge_block=knowledge_block,
                decision_id=decision_id,
                step=global_step,
                previous_evaluation=context.previous_evaluation,
                # §M-5: resume 后上一个 session 可能写到更晚 step, 那些记录对本次恢复是未来信息.
                recent_history=self.store.read_history(6, max_step=self.history_max_step),
                knowledge_version=self.knowledge.version,
            )
        except SkillValidationError as e:
            # controller 内部已 fallback; 这里只兜住未预期情况, 保持当前 Skill
            self._history({"record_type": "decision_error", "step": global_step, "error": str(e)})
            return None
        emitted["self_improving/llm_call_seconds"] = time.monotonic() - began
        emitted["self_improving/decision_valid"] = 1.0 if decision.status == "applied" else 0.0
        if decision.status == "applied":
            try:
                self.active = self.runtime.apply(decision.skill, from_llm=True)
            except SkillValidationError as e:
                decision.status = "rejected"
                decision.error = f"runtime rejected the decision: {e}"
                # apply 失败已回滚, 以 runtime 为准, 不把未生效的 skill 记成 active.
                self.active = self.runtime.current() or self.runtime.read_active()
        else:
            self.active = self.runtime.current() or self.runtime.read_active()
        emitted["self_improving/next_skill_id"] = self._skill_code(self.active.method)
        return decision

    def _record_intervention(
        self, global_step: int, context, evaluation, decision, update, observed_window: Optional[dict] = None,
        emitted: Optional[dict] = None,
    ) -> None:
        # 所有记录都过 _stamp: 同一个目录里可能有上一个 run 的 append-only 历史.
        # §M-7: 这些写入都在 runtime.apply() 之后 -> 写失败只记诊断, 不得改变已生效的 Skill.
        if observed_window:
            self._safe_history(
                {
                    "record_type": "window_summary",
                    "step": int(global_step),
                    "decision_id": decision.decision_id if decision is not None else None,
                    "evaluation_id": evaluation.evaluation_id if evaluation is not None else None,
                    "skill": (self.executed.method if self.executed else None),
                    "window": _window_digest(observed_window),
                    "knowledge_version": int(self.knowledge.version),
                },
                global_step,
                emitted,
            )
        if evaluation is not None:
            payload = jsonable(evaluation.to_dict())
            payload["record_type"] = "evaluation"
            payload["step"] = int(global_step)
            self._safe_history(payload, global_step, emitted)
        if update is not None:
            payload = jsonable(update)
            payload.update({"record_type": "knowledge_attempt", "step": int(global_step)})
            self._safe_history(payload, global_step, emitted)
        if decision is not None:
            payload = jsonable(decision.to_dict())
            payload["record_type"] = "decision"
            payload["step"] = int(global_step)
            payload["knowledge_update"] = update
            payload["next_skill_after_apply"] = self.active.method if self.active else None
            self._safe_history(payload, global_step, emitted)
        try:
            self.store.append_context(
                self._stamp(
                    {
                        "record_type": "context",
                        "step": int(global_step),
                        "decision_id": decision.decision_id if decision is not None else None,
                        "evaluation_id": evaluation.evaluation_id if evaluation is not None else None,
                        "knowledge_version": self.knowledge.version,
                        "context": jsonable(context.to_dict()),
                    }
                )
            )
        except OSError as e:
            if emitted is not None:
                emitted["self_improving/record_error"] = 1.0
            self._note_persistence_error("append_context", e, global_step)

    def _start_next_window(self, global_step: int, observed_window: dict, decision) -> None:
        self.last_intervention_step = int(global_step)
        self.intervention_count += 1
        self.validation_baseline = dict(self.latest_validation) if self.latest_validation else None
        if decision is None:
            # 没有 decision 时不伪造 evaluation: 窗口摘要已经写进 history.jsonl.
            self.pending = None
        else:
            self.pending = {
                "decision": decision.to_dict(),
                "baseline_window": jsonable(observed_window),
                "baseline_validation": jsonable(self.validation_baseline),
            }
        self.carryover_window = None
        # 本次介入的窗口已经落盘: 之后的 window 与本 run 的新记录同源, 不再受 resume 影响.
        self.state_misaligned = False
        self.history_max_step = None
        self.window_start_step = int(global_step) + 1
        self.collector.reset(self.window_start_step)

    # -------------------------------------------------------------- persist
    def _persist(self, global_step: int) -> None:
        state = self._state(global_step)
        pending = self._pending_snapshot_step
        self._pending_snapshot_step = None
        try:
            self.store.save_state(state)
            if pending is not None:
                # 与 policy checkpoint 同步: 写入的是本 step 处理完 (含 intervention) 之后的状态.
                self.store.save_snapshot(int(pending), self._state(int(pending)))
        except OSError as e:
            self._pending_snapshot_step = pending
            self._note_persistence_error("persist", e, global_step)
            return
        if pending is not None:
            self._safe_history(
                {"record_type": "checkpoint_snapshot", "step": int(pending), "status": "written"},
                global_step,
                None,
            )

    def _state(self, global_step: int) -> dict:
        active = self.runtime.current() or self.active or self.runtime.read_active()
        return jsonable(
            {
                "completed_global_step": int(global_step),
                "active_skill": active.method,
                "active_parameters": dict(active.params),
                "executed_skill": self.executed.method if self.executed else None,
                "executed_parameters": dict(self.executed.params) if self.executed else {},
                "last_intervention_step": int(self.last_intervention_step),
                "intervention_count": int(self.intervention_count),
                "window_start_step": int(self.window_start_step) if self.window_start_step is not None else None,
                "current_window_summary": self.collector.aggregate() if self.collector else {},
                "pending_decision": self.pending,
                "latest_validation": self.latest_validation,
                "validation_baseline": self.validation_baseline,
                "knowledge_version": int(self.knowledge.version),
                "windows_since_knowledge_update": int(self._windows_since_update),
                "metadata_collection_signature": self.bootstrap,
                "experiment_identity": experiment_identity(self.trainer.config),
                "skill_id_codes": {str(m): self._skill_code(m) for m in self.config.candidate_skills},
                "config_summary": self.config.to_dict(),
                "consecutive_intervention_failures": int(self._failure_streak),
                "persistence_failures": int(self._persistence_failures),
                "state_misaligned": bool(self.state_misaligned),
                "last_persistence_error": self._last_persistence_error,
            }
        )

    def on_checkpoint_saved(self, global_step: int) -> None:
        """Policy checkpoint 已落盘: 本 step 的 Slow Loop 处理结果随后写成同 step sidecar snapshot."""
        self._pending_snapshot_step = int(global_step)

    def finalize(self) -> None:
        if self.store is None or self._finalized:
            return
        self._finalized = True
        step = int(getattr(self.trainer, "global_steps", 0) or 0)
        try:
            self._persist(max(self.last_intervention_step, step))
        except SelfImprovingStateError:
            raise
        except Exception as e:  # pragma: no cover
            print(f"[self_improving] finalize persistence failed: {e}", flush=True)
        active = (self.runtime.current() if self.runtime else None) or self.active
        self._history(
            {
                "record_type": "finalized",
                "step": step,
                "run_id": self.run_id,
                "lineage": self.store.lineage,
                "intervention_count": self.intervention_count,
                "knowledge_version": int(self.knowledge.version),
                "active_skill": active.to_dict() if active else None,
                "consecutive_intervention_failures": int(self._failure_streak),
                "persistence_failures": int(self._persistence_failures),
            }
        )
        print(
            f"[self_improving] finalized: {self.intervention_count} intervention(s), "
            f"knowledge version {self.knowledge.version}, active skill "
            f"{active.method if active else 'n/a'}",
            flush=True,
        )

    # ---------------------------------------------------------------- safety
    def _record_failure(self, where: str, error: Exception, global_step: int, emitted: dict) -> None:
        """可恢复的 Slow Loop 故障: 记录原因 + 保持当前 Skill, 下一个 interval 再试."""
        self._failure_streak += 1
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
        print(
            f"[self_improving] {where} failed at step {global_step} (failure #{self._failure_streak}, "
            f"retry at the next scheduled intervention): {type(error).__name__}: {detail[-800:]}",
            flush=True,
        )
        # 已经生效的 Skill 是权威状态: 失败后重新从 runtime 读, 不保留半途的 self.active.
        if self.runtime is not None:
            try:
                self.active = self.runtime.current() or self.runtime.read_active()
            except SkillValidationError:  # pragma: no cover
                pass
        try:
            self._history(
                {
                    "record_type": "error",
                    "step": int(global_step),
                    "where": where,
                    "consecutive_failures": int(self._failure_streak),
                    "active_skill": self.active.method if self.active else None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        except OSError as e:  # pragma: no cover
            self._note_persistence_error("record_failure", e, global_step)
        # 失败 step 也要有 state.json / snapshot, 否则 resume 找不到对齐状态.
        self._persist(int(global_step))
        if self.active is not None:
            emitted["self_improving/next_skill_id"] = self._skill_code(self.active.method)

    def _note_persistence_error(self, where: str, error: OSError, global_step: int) -> None:
        """落盘失败不能记成成功: 先计数重试, 连续多次则视为不可恢复 -> fail-fast."""
        self._persistence_failures += 1
        self._last_persistence_error = f"{where}: {type(error).__name__}: {error}"
        print(
            f"[self_improving] PERSISTENCE FAILURE at step {global_step} ({where}, "
            f"attempt {self._persistence_failures}): {type(error).__name__}: {error}",
            flush=True,
        )
        try:
            self._history(
                {
                    "record_type": "persistence_error",
                    "step": int(global_step),
                    "where": where,
                    "attempt": int(self._persistence_failures),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        except Exception:  # pragma: no cover - 记录通道本身坏了就不能再依赖它
            pass
        if self._persistence_failures >= PERSISTENCE_FAILURE_LIMIT:
            raise SelfImprovingStateError(
                f"self_improving persistence failed {self._persistence_failures} consecutive times "
                f"({where}: {error}); refusing to keep training with an unrecoverable Slow Loop state"
            )

    def _safe_history(self, record: dict, global_step: int, emitted: Optional[dict]) -> bool:
        """写入失败的记录不能推翻已经生效的状态: 记账后继续."""
        try:
            self._history(record)
            return True
        except OSError as e:
            if emitted is not None:
                emitted["self_improving/record_error"] = 1.0
            self._note_persistence_error(f"history:{record.get('record_type')}", e, global_step)
            return False

    def _stamp(self, record: dict) -> dict:
        record = dict(record)
        record.setdefault("run_id", self.run_id)
        record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return record

    def _history(self, record: dict) -> None:
        self.store.append_history(self._stamp(record))

    def observe_validation(self, val_metrics: dict, global_step: int) -> None:
        """初始验证 (训练开始前) 也可作为比较基线: trainer 侧可选的 thin hook."""
        if not val_metrics:
            return
        try:
            self._capture_validation(val_metrics, int(global_step))
            if self.latest_validation:
                self.validation_baseline = dict(self.latest_validation)
        except Exception as e:  # pragma: no cover
            print(f"[self_improving] validation capture failed: {e}", flush=True)


def _window_digest(window: dict) -> dict:
    """history 里的紧凑窗口摘要: 即使没有 decision 也保留核心指标与 metadata 健康度."""
    out = {
        "step_range": window.get("step_range"),
        "num_steps": window.get("num_steps"),
        "skills": window.get("skills"),
        "success_rate": (window.get("raw_reward") or {}).get("mean") and window.get("success_rate"),
        "raw_reward_mean": (window.get("raw_reward") or {}).get("mean"),
        "train_reward_mean": (window.get("train_reward") or {}).get("mean"),
        "advantage": window.get("advantage"),
        "trusted_terminal_ratio": window.get("trusted_terminal_ratio"),
        "metadata_health": window.get("metadata_health"),
        "credit_diagnostics": (window.get("credit_diagnostics") or {}).get("latest_step"),
    }
    out["success_rate"] = window.get("success_rate")
    return jsonable(out)


def _json(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, indent=2, default=str)


def _clip_chars(payload: Any, limit: int) -> Any:
    text = json.dumps(jsonable(payload), ensure_ascii=False, default=str)
    if len(text) <= limit:
        return payload
    return {"truncated": text[:limit] + "…"}
