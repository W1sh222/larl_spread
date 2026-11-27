# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Controller 决策与 Training Feedback 评估 (CPU).

覆盖 명령서 §21.6 (脚本化 LLM: keep / switch / calibrate / 非法 JSON / 非法 schema /
非法 Skill / 非法参数 / 超时 / API 失败 / 回退当前 Skill) 与 §21.7 (观测性评估:
正/负/中性变化, 证据不足, 缺 validation, stale validation, metadata 污染 window,
decision/window 归属, 负向评估不自动回滚).
"""

import copy
import unittest

import numpy as np
from omegaconf import OmegaConf

from verl.trainer.credit_assignment.self_improving.config import EvaluationConfig, load_self_improving_config
from verl.trainer.credit_assignment.self_improving.controller import MAX_REASONING_CHARS, CreditSkillController
from verl.trainer.credit_assignment.self_improving.evaluator import (
    OUTCOME_INSUFFICIENT,
    OUTCOME_NEUTRAL,
    OUTCOME_NEGATIVE,
    OUTCOME_POSITIVE,
    QUALITY_HIGH,
    QUALITY_LOW,
    QUALITY_MEDIUM,
    DecisionEvaluator,
)
from verl.trainer.credit_assignment.self_improving.llm_client import (
    LLMError,
    LLMHTTPError,
    LLMInvalidResponse,
    LLMNetworkError,
    LLMTimeout,
    OpenAICompatibleClient,
    build_llm_client,
    check_schema,
    parse_json_object,
)
from verl.trainer.credit_assignment.self_improving.runtime import SkillConfig, SkillRuntime


class ScriptedLLM:
    """명령서 §21.6 指定的 mock client."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.last_latency_seconds = 0.01

    def chat_json(self, messages, schema=None):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_setup(initial="grpo", bootstrap="rewardflow", candidates=("grpo", "gigpo", "rewardflow")):
    algorithm_cfg = OmegaConf.create(
        {"adv_estimator": "credit_assignment", "credit_assignment": {"method": bootstrap, "params": {}}}
    )
    config = load_self_improving_config(
        {
            "enabled": False,
            "initial_skill": {"method": initial},
            "candidate_skills": list(candidates),
            "llm": {"client": "mock"},
        }
    )
    runtime = SkillRuntime(config, algorithm_cfg)
    runtime.apply_initial()
    return config, algorithm_cfg, runtime


def fake_context(step=10):
    return {
        "global_step": step,
        "window_start_step": step - 9,
        "window_end_step": step,
        "active_skill": {"method": "grpo", "params": {"norm_adv_by_std_in_grpo": True}},
        "trajectory_statistics": {"num_steps": 10, "raw_reward": {"mean": 0.35}},
        "trajectory_samples": [{"kind": "success", "row": 0, "steps": [{"index": 0, "observation": "cell A1=1"}]}],
        "metadata_health": {"step_transitions": 1.0, "eval_fingerprint_any": True},
    }


class ControllerDecisionTest(unittest.TestCase):
    def setUp(self):
        self.config, self.algorithm_cfg, self.runtime = make_setup()
        self.active = self.runtime.current()

    def _controller(self, responses):
        return CreditSkillController(self.config, self.runtime, ScriptedLLM(responses))

    def _decide(self, responses, **kw):
        controller = self._controller(responses)
        decision = controller.select_and_calibrate(
            context=fake_context(),
            knowledge_block="### skill knowledge: grpo",
            decision_id="decision-0001-g10",
            step=10,
            knowledge_version=0,
            **kw,
        )
        return decision, controller

    def test_valid_keep_decision(self):
        decision, _ = self._decide([{"action": "keep", "selected_skill": "grpo", "parameters": dict(self.active.params)}])
        self.assertEqual(decision.status, "applied")
        self.assertEqual(decision.action, "keep")
        self.assertEqual(decision.skill.method, "grpo")
        self.assertFalse(decision.changed)
        self.assertEqual(decision.decision_id, "decision-0001-g10")
        # controller 只提案: 不自行改 driver config
        self.assertEqual(str(self.algorithm_cfg.credit_assignment.method), "grpo")

    def test_valid_switch_decision(self):
        decision, _ = self._decide([{"action": "switch", "selected_skill": "gigpo", "parameters": {"gamma": 0.9}}])
        self.assertEqual(decision.status, "applied")
        self.assertEqual(decision.skill.method, "gigpo")
        self.assertEqual(decision.skill.params["gamma"], 0.9)
        self.assertEqual(decision.previous_skill.method, "grpo")
        self.assertTrue(decision.changed)
        self.assertEqual(str(self.algorithm_cfg.credit_assignment.method), "grpo")
        # loop 侧 apply 之后才真正切换
        self.runtime.apply(decision.skill, from_llm=True)
        self.assertEqual(str(self.algorithm_cfg.credit_assignment.method), "gigpo")

    def test_valid_calibration_decision(self):
        decision, _ = self._decide(
            [{"action": "calibrate", "selected_skill": "grpo", "parameters": {"norm_adv_by_std_in_grpo": False}}]
        )
        self.assertEqual(decision.status, "applied")
        self.assertEqual(decision.action, "calibrate")
        self.assertEqual(decision.skill.params["norm_adv_by_std_in_grpo"], False)
        self.assertTrue(decision.changed)

    def test_no_op_calibration_is_normalized_to_keep(self):
        decision, _ = self._decide(
            [{"action": "calibrate", "selected_skill": "grpo", "parameters": dict(self.active.params)}]
        )
        self.assertEqual(decision.action, "keep")
        self.assertFalse(decision.changed)

    def test_invalid_json_response(self):
        decision, _ = self._decide([LLMInvalidResponse("no JSON object found in LLM response")])
        self.assertEqual(decision.status, "fallback")
        self.assertIn("invalid llm payload", decision.error)
        self.assertEqual(decision.skill.method, self.active.method)
        self.assertFalse(decision.changed)

    def test_invalid_schema_response(self):
        # 缺 selected_skill: ScriptedLLM 不做 schema 校验 -> controller 自己拒掉.
        decision, _ = self._decide([{"action": "switch"}])
        self.assertEqual(decision.status, "fallback")
        self.assertIn("selected_skill", decision.error)
        # 带 schema 的 client 会在 chat_json 里 raise -> 同样回退.
        from verl.trainer.credit_assignment.self_improving.llm_client import MockLLMClient

        controller = CreditSkillController(self.config, self.runtime, MockLLMClient([{"action": "keep"}]))
        decision = controller.select_and_calibrate(fake_context(), "", "d", 10)
        self.assertEqual(decision.status, "fallback")
        self.assertIn("missing required field 'selected_skill'", decision.error)

    def test_non_object_parameters_are_parsed_or_rejected(self):
        decision, _ = self._decide(
            [{"action": "switch", "selected_skill": "gigpo", "parameters": '{"gamma": 0.8}'}]
        )
        self.assertEqual(decision.status, "applied")
        self.assertEqual(decision.skill.params["gamma"], 0.8)
        decision, _ = self._decide([{"action": "switch", "selected_skill": "gigpo", "parameters": "not json"}])
        self.assertEqual(decision.status, "fallback")
        self.assertIn("parameters is not a JSON object", decision.error)

    def test_invalid_skill_rejected(self):
        for skill in ("ppo", "gae", "GRPO", "made_up"):
            decision, _ = self._decide([{"action": "switch", "selected_skill": skill, "parameters": {}}])
            self.assertEqual(decision.status, "fallback", skill)
            self.assertIn("rejected by runtime", decision.error)
            self.assertEqual(self.runtime.current().method, "grpo")

    def test_skill_outside_candidates_rejected(self):
        config, _, runtime = make_setup(candidates=("grpo", "rewardflow"))
        controller = CreditSkillController(config, runtime, ScriptedLLM([{"action": "switch", "selected_skill": "gigpo", "parameters": {}}]))
        decision = controller.select_and_calibrate(fake_context(), "", "d", 10)
        self.assertEqual(decision.status, "fallback")
        self.assertIn("candidate_skills", decision.error)

    def test_invalid_parameters_rejected(self):
        for params in ({"gamma": 1.7}, {"mode": "median"}, {"epsilon": 1e-5}, {"not_a_param": 1}):
            decision, _ = self._decide([{"action": "switch", "selected_skill": "gigpo", "parameters": params}])
            self.assertEqual(decision.status, "fallback", params)
            self.assertEqual(decision.skill.method, "grpo")

    def test_timeout_and_api_failure_fall_back(self):
        for error in (LLMTimeout("timed out"), LLMNetworkError("connection refused"), LLMError("boom"), RuntimeError("unexpected")):
            decision, _ = self._decide([error])
            self.assertEqual(decision.status, "fallback", type(error).__name__)
            self.assertEqual(decision.skill, self.active)
            self.assertFalse(decision.changed)
        self.assertEqual(str(self.algorithm_cfg.credit_assignment.method), "grpo")

    def test_action_skill_consistency_is_enforced(self):
        for payload in (
            {"action": "keep", "selected_skill": "gigpo", "parameters": {}},
            {"action": "calibrate", "selected_skill": "rewardflow", "parameters": {}},
            {"action": "switch", "selected_skill": "grpo", "parameters": {"epsilon": 1e-6}},
            {"action": "restart", "selected_skill": "grpo", "parameters": {}},
        ):
            decision, _ = self._decide([payload])
            self.assertEqual(decision.status, "fallback", payload)

    def test_reasoning_and_evidence_are_clipped(self):
        decision, _ = self._decide(
            [
                {
                    "action": "keep",
                    "selected_skill": "grpo",
                    "parameters": dict(self.active.params),
                    "reasoning": "长" * 5000,
                    "evidence": [f"item {i}" for i in range(30)],
                    "expected_improvement": "x" * 5000,
                }
            ]
        )
        self.assertEqual(len(decision.reasoning), MAX_REASONING_CHARS + 1)
        self.assertLessEqual(len(decision.evidence), 8)
        self.assertEqual(decision.knowledge_version, 0)
        self.assertEqual(decision.llm_latency_seconds, 0.01)
        payload = decision.to_dict()
        self.assertEqual(payload["decision_id"], "decision-0001-g10")
        self.assertEqual(payload["applied_skill"], "grpo")
        self.assertIn("previous_skill", payload)

    def test_prompt_contains_knowledge_and_calibration_space(self):
        controller = self._controller([])
        messages = controller.build_messages(
            fake_context(),
            "### skill knowledge: gigpo\n- exp-0001 · step 5",
            self.active,
            previous_evaluation={"outcome": "positive"},
            recent_history=[{"record_type": "decision", "step": 5, "applied_skill": "grpo"}],
        )
        system = messages[0]["content"]
        self.assertNotIn("{{", system)
        for needle in ("gigpo", "rewardflow", "norm_adv_by_std_in_grpo", "exp-0001", "positive", "cell A1=1", "calibratable"):
            self.assertIn(needle, system, needle)
        self.assertNotIn("epsilon", system.split("calibratable")[1][:400])
        self.assertEqual(messages[1]["role"], "user")

    def test_history_is_trimmed_to_recent_records(self):
        controller = self._controller([])
        history = [{"record_type": "decision", "step": i, "secret": "x"} for i in range(20)]
        messages = controller.build_messages(fake_context(), "", self.active, recent_history=history)
        self.assertNotIn("secret", messages[0]["content"])
        self.assertIn('"step": 19', messages[0]["content"])

    def test_oversized_context_is_deferred_not_dropped(self):
        controller = self._controller([])
        big = fake_context()
        big["trajectory_samples"] = [{"kind": "representative", "text": "y" * 4000} for _ in range(50)]
        messages = controller.build_messages(big, "", self.active)
        self.assertIn("trajectory_samples_truncated", messages[0]["content"])


class EvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = DecisionEvaluator(EvaluationConfig())

    def _decision(self, action="switch", previous="grpo", applied="gigpo", status="applied", params=None):
        return {
            "decision_id": "decision-0002-g20",
            "action": action,
            "status": status,
            "previous_skill": {"method": previous, "params": {"norm_adv_by_std_in_grpo": True}},
            "applied_skill": applied,
            "applied_parameters": params if params is not None else {"gamma": 0.9},
        }

    def _window(self, steps=8, start=13, skill="gigpo", raw=0.5, success=0.5, transitions=1.0, eval_fp=True):
        return {
            "num_steps": steps,
            "step_range": [start, start + steps - 1],
            "skills": [skill],
            "raw_reward": {"mean": raw, "std": 0.2, "min": 0.0, "max": 1.0},
            "train_reward": {"mean": raw},
            "success_rate": success,
            "trajectory_length": {"mean": 900.0},
            "steps_per_trajectory": {"mean": 3.0},
            "advantage": {"zero_ratio": 0.4, "abs_mean": 0.5, "mean": 0.0},
            "trusted_terminal_ratio": 1.0,
            "metadata_health": {"step_transitions": transitions, "eval_fingerprint_any": eval_fp},
        }

    def _eval(self, baseline, observed, validation=None, observed_validation=None, **kw):
        return self.evaluator.evaluate(
            evaluation_id="eval-0002-g20",
            decision=self._decision(**kw),
            baseline_window=baseline,
            observed_window=observed,
            baseline_validation=validation,
            observed_validation=observed_validation,
            step=20,
        )

    def test_positive_observed_change(self):
        result = self._eval(self._window(raw=0.3, success=0.3), self._window(raw=0.55, success=0.55))
        self.assertEqual(result.outcome, OUTCOME_POSITIVE)
        self.assertEqual(result.evidence_quality, QUALITY_MEDIUM)
        self.assertEqual(result.primary_metric, "success_rate")
        self.assertAlmostEqual(result.metric_deltas["success_rate"], 0.25, places=6)
        self.assertAlmostEqual(result.metric_deltas["raw_reward_mean"], 0.25, places=6)
        self.assertTrue(result.eligible_for_knowledge_update)
        self.assertIn("observational only", result.summary)

    def test_negative_observed_change(self):
        result = self._eval(self._window(raw=0.6, success=0.6), self._window(raw=0.35, success=0.35))
        self.assertEqual(result.outcome, OUTCOME_NEGATIVE)
        self.assertTrue(result.eligible_for_knowledge_update)
        self.assertLess(result.metric_deltas["success_rate"], 0)

    def test_neutral_change_within_threshold(self):
        result = self._eval(self._window(raw=0.5, success=0.5), self._window(raw=0.505, success=0.505))
        self.assertEqual(result.outcome, OUTCOME_NEUTRAL)
        self.assertEqual(result.primary_metric, "success_rate")

    def test_conflicting_directions_are_low_quality(self):
        observed = self._window(raw=0.7, success=0.4)
        result = self._eval(self._window(raw=0.5, success=0.6), observed)
        self.assertEqual(result.evidence_quality, QUALITY_LOW)
        self.assertEqual(result.outcome, OUTCOME_INSUFFICIENT)
        self.assertFalse(result.eligible_for_knowledge_update)
        self.assertIn("opposite directions", " ".join(result.evidence_notes))

    def test_insufficient_evidence_short_window(self):
        result = self._eval(self._window(steps=8), self._window(steps=2))
        self.assertEqual(result.outcome, OUTCOME_INSUFFICIENT)
        self.assertEqual(result.evidence_quality, QUALITY_LOW)
        self.assertIn("min_window_steps", " ".join(result.evidence_notes))

    def test_no_comparable_metric_is_insufficient(self):
        baseline = self._window()
        observed = self._window()
        for node in (baseline, observed):
            node.pop("success_rate")
            node["raw_reward"] = {"mean": None}
        result = self._eval(baseline, observed)
        self.assertEqual(result.outcome, OUTCOME_INSUFFICIENT)
        self.assertIn("no comparable core training metric", " ".join(result.evidence_notes))

    def test_missing_validation_is_reported_not_fabricated(self):
        result = self._eval(self._window(raw=0.5), self._window(raw=0.7), validation=None, observed_validation=None)
        self.assertEqual(result.validation_comparison["status"], "unavailable")
        self.assertEqual(result.evidence_quality, QUALITY_MEDIUM)
        self.assertIsNone(result.primary_metric if result.primary_metric.startswith("validation:") else None)

    def test_validation_comparison_uses_one_dataset_key(self):
        baseline = {"step": 12, "metrics": {"val-core/spreadsheet/acc/mean@4": 0.30}}
        observed = {"step": 20, "metrics": {"val-core/spreadsheet/acc/mean@4": 0.44}}
        result = self._eval(self._window(raw=0.5), self._window(raw=0.5), baseline, observed)
        self.assertEqual(result.validation_comparison["status"], "compared")
        self.assertAlmostEqual(result.validation_comparison["delta"], 0.14, places=6)
        self.assertEqual(result.outcome, OUTCOME_POSITIVE)
        self.assertEqual(result.evidence_quality, QUALITY_HIGH)
        self.assertEqual(result.primary_metric, "validation:val-core/spreadsheet/acc/mean@4")

    def test_stale_validation_is_not_reused_as_new_evidence(self):
        same = {"step": 20, "metrics": {"val-core/spreadsheet/acc/mean@4": 0.4}}
        result = self._eval(self._window(raw=0.5), self._window(raw=0.7), same, same)
        self.assertEqual(result.validation_comparison["status"], "stale")
        self.assertEqual(result.outcome, OUTCOME_INSUFFICIENT)
        self.assertIn("stale", " ".join(result.evidence_notes))

    def test_ambiguous_validation_key_is_not_averaged(self):
        observed = {"step": 20, "metrics": {"val-core/a/acc/mean@4": 0.4, "val-core/b/acc/mean@4": 0.9}}
        result = self._eval(self._window(raw=0.5), self._window(raw=0.7), {"step": 10, "metrics": observed["metrics"]}, observed)
        self.assertEqual(result.validation_comparison["status"], "ambiguous")
        self.assertEqual(result.outcome, OUTCOME_INSUFFICIENT)
        evaluator = DecisionEvaluator(EvaluationConfig(validation_metric_key="val-core/b/acc/mean@4"))
        result = evaluator.evaluate(
            "e", self._decision(), self._window(raw=0.5), self._window(raw=0.5),
            {"step": 10, "metrics": {"val-core/b/acc/mean@4": 0.95}}, observed, 20,
        )
        self.assertEqual(result.validation_comparison["status"], "compared")
        self.assertEqual(result.outcome, OUTCOME_NEGATIVE)

    def test_metadata_contaminated_window_is_not_eligible(self):
        result = self._eval(self._window(transitions=1.0), self._window(raw=0.8, transitions=0.2))
        self.assertEqual(result.evidence_quality, QUALITY_LOW)
        self.assertFalse(result.eligible_for_knowledge_update)
        self.assertIn("coverage is low", " ".join(result.evidence_notes))
        result = self._eval(self._window(), self._window(raw=0.8, eval_fp=False))
        self.assertIn("fingerprints were not available", " ".join(result.evidence_notes))

    def test_hard_required_metadata_missing_vs_partial_new_path(self):
        # §M-3: runtime 传入 observed_requirements 时逐项分类 hard-missing / hard-partial.
        base = self._window()
        # 完全未采集: step_transitions / eval fingerprint 均为 None.
        missing = self._window()
        missing["metadata_health"] = {"step_transitions": None, "eval_workbook_fingerprint": None}
        r1 = self.evaluator.evaluate(
            "eval-x", self._decision(), base, missing, None, None, 20,
            observed_requirements=["step_transitions", "eval_fingerprint"],
        )
        self.assertEqual(r1.evidence_quality, QUALITY_LOW)
        self.assertFalse(r1.eligible_for_knowledge_update)
        joined1 = " ".join(r1.evidence_notes)
        self.assertIn("required metadata completely absent", joined1)
        self.assertIn("eval_fingerprint", joined1)
        self.assertIn("not available", joined1)
        # 部分覆盖: 采集了但 < 1.0 -> 归为 partially available, 不是完全缺失.
        partial = self._window()
        partial["metadata_health"] = {"step_transitions": 0.6, "eval_workbook_fingerprint": 0.5}
        r2 = self.evaluator.evaluate(
            "eval-y", self._decision(), base, partial, None, None, 20,
            observed_requirements=["step_transitions", "eval_fingerprint"],
        )
        self.assertEqual(r2.evidence_quality, QUALITY_LOW)
        joined2 = " ".join(r2.evidence_notes)
        self.assertIn("only partially available", joined2)
        self.assertIn("step_transitions", joined2)
        self.assertNotIn("completely absent", joined2)

    def test_no_actual_configuration_change_is_not_attributed(self):
        # §10 minor: `changed` 字段缺失时不伪造 (loop 层的 keep 降级见 LoopSafetyTest).
        # decision 里没有 changed 字段时仍按正常窗口评估.
        legacy = self._decision()
        legacy.pop("changed", None)
        result = self.evaluator.evaluate(
            "eval-legacy", legacy, self._window(raw=0.3, success=0.3), self._window(raw=0.7, success=0.7), None, None, 20
        )
        self.assertEqual(result.outcome, OUTCOME_POSITIVE)
        # 明确标记 changed=False 时, evaluator 本身不改口径: 归属判断在 slow loop 完成.
        same = self._decision()
        same["changed"] = False
        result = self.evaluator.evaluate(
            "eval-same", same, self._window(raw=0.3, success=0.3), self._window(raw=0.7, success=0.7), None, None, 20
        )
        self.assertEqual(result.outcome, OUTCOME_POSITIVE)

    def test_mixed_skill_window_is_flagged(self):
        observed = self._window(raw=0.8)
        observed["skills"] = ["gigpo", "rewardflow"]
        result = self._eval(self._window(), observed)
        self.assertEqual(result.outcome, OUTCOME_INSUFFICIENT)
        self.assertIn("mixed multiple skill", " ".join(result.evidence_notes))

    def test_fallback_decision_window_is_flagged(self):
        result = self._eval(self._window(), self._window(raw=0.6), status="fallback")
        self.assertIn("failed/invalid controller decision", " ".join(result.evidence_notes))

    def test_decision_and_window_attribution(self):
        baseline = self._window(steps=10, start=1, skill="grpo", raw=0.4, success=0.4)
        observed = self._window(steps=8, start=13, skill="gigpo", raw=0.6, success=0.6)
        result = self._eval(baseline, observed)
        self.assertEqual(result.decision_id, "decision-0002-g20")
        self.assertEqual(result.evaluation_id, "eval-0002-g20")
        self.assertEqual(result.step, 20)
        self.assertEqual(result.previous_skill, "grpo")
        self.assertEqual(result.applied_skill, "gigpo")
        self.assertEqual(result.applied_parameters, {"gamma": 0.9})
        self.assertEqual(result.baseline_window["step_range"], [1, 10])
        self.assertEqual(result.observed_window["step_range"], [13, 20])
        self.assertEqual(result.observed_window["skills"], ["gigpo"])
        self.assertEqual(result.previous_parameters, {"norm_adv_by_std_in_grpo": True})
        self.assertIn("13-20", result.summary)
        jsonable = {k: v for k, v in result.to_dict().items()}
        self.assertEqual(set(jsonable), set(result.__dataclass_fields__))

    def test_negative_evaluation_does_not_roll_back(self):
        config, algorithm_cfg, runtime = make_setup()
        runtime.apply(SkillConfig("gigpo", {"gamma": 0.9}), from_llm=True)
        decision = self._decision(applied="gigpo")
        snapshot = copy.deepcopy(decision)
        result = self._eval(self._window(raw=0.7, success=0.7), self._window(raw=0.2, success=0.2))
        self.assertEqual(result.outcome, OUTCOME_NEGATIVE)
        self.assertEqual(decision, snapshot)
        self.assertEqual(str(algorithm_cfg.credit_assignment.method), "gigpo")
        self.assertEqual(runtime.current().method, "gigpo")
        self.assertFalse(any("rollback" in name for name in dir(result)))
        self.assertFalse(any("rollback" in name for name in dir(self.evaluator)))


class LLMResponseLayerTest(unittest.TestCase):
    def test_parse_json_object_tolerates_fences_and_prose(self):
        self.assertEqual(parse_json_object('{"action": "keep"}'), {"action": "keep"})
        self.assertEqual(parse_json_object('```json\n{"action": "keep"}\n```'), {"action": "keep"})
        self.assertEqual(parse_json_object('Sure! {"action": "keep"} hope that helps'), {"action": "keep"})
        for bad in ("", "not json at all", "[1, 2, 3]", None, '{"unterminated": '):
            with self.assertRaises(LLMInvalidResponse):
                parse_json_object(bad)

    def test_check_schema_rules(self):
        schema = {"action": str, "parameters": dict, "reasoning": (str, False)}
        self.assertEqual(check_schema({"action": "keep", "parameters": {}}, schema)["action"], "keep")
        self.assertEqual(check_schema({"action": "keep", "parameters": {}, "reasoning": None}, schema)["reasoning"], None)
        for payload in (None, [], {"parameters": {}}, {"action": 1, "parameters": {}}, {"action": "keep", "parameters": "x"}):
            with self.assertRaises(LLMInvalidResponse):
                check_schema(payload, schema)

    def test_client_reports_timeout_http_and_network_errors(self):
        config = type("C", (), {"endpoint_env": "X", "model_env": "Y", "api_key_env": "Z", "temperature": 0.0, "max_retries": 1, "timeout_seconds": 0.5})()
        import os

        os.environ["X"] = "http://example.invalid/v1"
        os.environ["Y"] = "test-model"
        try:
            client = OpenAICompatibleClient(config, poster=lambda h, b, t: (_ for _ in ()).throw(LLMTimeout("timed out")))
            with self.assertRaises(LLMTimeout):
                client.chat_json([{"role": "user", "content": "x"}])
            self.assertEqual(client.call_count, 2)  # 1 + max_retries
            client = OpenAICompatibleClient(config, poster=lambda h, b, t: '{"choices": [{"message": {"content": "{\\"action\\": \\"keep\\"}"}}], "usage": {"total_tokens": 7}}')
            self.assertEqual(client.chat_json([{"role": "user", "content": "x"}], schema={"action": str})["action"], "keep")
            self.assertEqual(client.last_usage, {"total_tokens": 7})
            client = OpenAICompatibleClient(config, poster=lambda h, b, t: '{"choices": []}')
            with self.assertRaises(LLMInvalidResponse):
                client.chat_json([{"role": "user", "content": "x"}])
            self.assertTrue(client.failure_count >= 1)
        finally:
            os.environ.pop("X", None)
            os.environ.pop("Y", None)

    def test_missing_endpoint_env_is_an_llm_error(self):
        import os

        os.environ.pop("SPREADSHEET_RL_SI_LLM_URL", None)
        client = build_llm_client(load_self_improving_config({"llm": {"client": "api"}, "initial_skill": {"method": "grpo"}}).llm)
        with self.assertRaises(LLMError):
            client.chat_json([{"role": "user", "content": "x"}])

    def test_mock_client_from_script_file(self):
        import json
        import os
        import tempfile

        from verl.trainer.credit_assignment.self_improving.llm_client import MockLLMClient

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "script.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("# comment\n")
                f.write(json.dumps({"response": {"action": "keep", "selected_skill": "grpo", "parameters": {}}}) + "\n")
                f.write('{"broken": \n')
                f.write(json.dumps({"action": "switch", "selected_skill": "gigpo", "parameters": {}}) + "\n")
            client = MockLLMClient(script_path=path)
            self.assertEqual(client.chat_json([])["action"], "keep")
            self.assertEqual(client.chat_json([])["action"], "switch")
            with self.assertRaises(LLMInvalidResponse):
                client.chat_json([])


if __name__ == "__main__":
    unittest.main()
