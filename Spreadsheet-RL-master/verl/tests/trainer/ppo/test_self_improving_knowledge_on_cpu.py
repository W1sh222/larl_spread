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

"""Skill Knowledge 演化与 run-local 持久化 (CPU).

覆盖 명령서 §21.8 (初始加载 / run 隔离 / 合法与非法经验更新 / 重复经验 / provenance /
版本递增 / 原始 markdown 不变 / prompt 预算) 与 §21.9 的 store 层 (state 存取,
checkpoint 对齐恢复, 严格/非严格 resume, NaN 序列化, run 目录隔离).
"""

import hashlib
import json
import os
import tempfile
import unittest

from omegaconf import OmegaConf

from verl.trainer.credit_assignment.self_improving.knowledge import (
    EXPERIENCE_HEADING,
    GUIDELINE_HEADING,
    NO_EXPERIENCE_TEXT,
    KnowledgeError,
    KnowledgeStore,
    validate_skill_markdown,
)
from verl.trainer.credit_assignment.self_improving.llm_client import LLMError, MockLLMClient
from verl.trainer.credit_assignment.self_improving.persistence import (
    SelfImprovingStateError,
    SelfImprovingStore,
    experiment_identity,
    new_run_id,
)

SKILLS = ["grpo", "gigpo", "rewardflow"]


def evaluation(**over):
    payload = {
        "evaluation_id": "eval-0001-g20",
        "decision_id": "decision-0001-g20",
        "step": 20,
        "previous_skill": "grpo",
        "previous_parameters": {"norm_adv_by_std_in_grpo": True},
        "applied_skill": "gigpo",
        "applied_parameters": {"gamma": 0.9, "step_advantage_w": 1.0},
        "action": "switch",
        "outcome": "positive",
        "evidence_quality": "medium",
        "eligible_for_knowledge_update": True,
        "evaluated_at_step": 20,
        "metric_deltas": {"success_rate": 0.12, "raw_reward_mean": 0.15, "trajectory_length_mean": None},
    }
    payload.update(over)
    return payload


def proposal(**over):
    payload = {
        "update_required": True,
        "target_skill": "gigpo",
        "experience": {
            "decision_id": "decision-0001-g20",
            "context": "spreadsheet multi-turn RL, 8 steps per trajectory",
            "observation": "success rate rose after the switch",
            "interpretation": "step-level credit matched the tool-use structure",
            "limitations": "single window, observational only",
        },
    }
    payload.update(over)
    return payload


class KnowledgeStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.store = KnowledgeStore(run_dir=self.root, candidate_skills=list(SKILLS), lineage="run-a")
        self.store.bootstrap(0)

    def _apply(self, store=None, prop=None, eval_payload=None, skill="gigpo", step=20):
        store = store or self.store
        return store.apply_update(prop or proposal(), eval_payload or evaluation(), step, skill)

    def test_initial_knowledge_is_loaded_and_complete(self):
        for skill in SKILLS:
            text = self.store.read(skill)
            self.assertTrue(text, skill)
            self.assertEqual(validate_skill_markdown(text), [], skill)
            self.assertIn(NO_EXPERIENCE_TEXT, text, skill)
        self.assertEqual(self.store.version, 0)
        self.assertEqual(self.store.skills_with_knowledge(), SKILLS)
        index = json.load(open(os.path.join(self.root, "knowledge", "index.json"), encoding="utf-8"))
        self.assertEqual([v["version"] for v in index["versions"]], [0])

    def test_missing_template_for_unknown_candidate_is_fatal(self):
        store = KnowledgeStore(run_dir=os.path.join(self.root, "other"), candidate_skills=["imaginary"])
        with self.assertRaises(KnowledgeError):
            store.bootstrap(0)

    def test_skill_names_are_confined_to_the_run_directory(self):
        with self.assertRaises(KnowledgeError):
            self.store.path_of("../escape")
        with self.assertRaises(KnowledgeError):
            self.store.path_of("a b")

    def test_run_local_copies_isolate_runs(self):
        other = KnowledgeStore(run_dir=os.path.join(self.root, "run-b"), candidate_skills=list(SKILLS), lineage="run-b")
        other.bootstrap(0)
        ok, entry, version = self._apply()
        self.assertTrue(ok)
        self.assertIn("exp-0001", self.store.read("gigpo"))
        self.assertNotIn("exp-0001", other.read("gigpo"))
        self.assertEqual(other.version, 0)

    def test_valid_experience_update(self):
        ok, entry_id, version = self._apply()
        self.assertTrue(ok, entry_id)
        self.assertEqual(entry_id, "exp-0001")
        self.assertEqual(version, 1)
        self.assertEqual(self.store.version, 1)
        text = self.store.read("gigpo")
        self.assertIn("### exp-0001 · step 20 · decision decision-0001-g20", text)
        self.assertNotIn(NO_EXPERIENCE_TEXT, text)
        self.assertIn(EXPERIENCE_HEADING, text)
        self.assertIn('"success_rate": 0.12', text)
        self.assertIn("single window, observational only", text)
        self.assertEqual(self.store.experience_entries("gigpo"), ["exp-0001"])

    def test_missing_limitations_falls_back_to_the_non_causal_disclaimer(self):
        prop = proposal(experience={"decision_id": "decision-0001-g20", "limitations": ""})
        self.assertTrue(self._apply(prop=prop)[0])
        self.assertIn("Observational evidence only; not a causal effect estimate.", self.store.read("gigpo"))

    def test_guideline_proposal_is_recorded_as_hypothesis(self):
        prop = proposal(guideline_proposal={"text": "prefer gamma near 0.9", "applies_when": "trajectories longer than 6 steps"})
        self._apply(prop=prop)
        text = self.store.read("gigpo")
        self.assertIn(GUIDELINE_HEADING, text)
        self.assertIn("[hypothesis · from exp-0001 · unverified] prefer gamma near 0.9", text)
        self.assertIn("applies when: trajectories longer than 6 steps", text)

    def test_second_entry_gets_the_next_id(self):
        self._apply()
        second_eval = evaluation(decision_id="decision-0002-g26", evaluation_id="eval-0002-g26", evaluated_at_step=26)
        second_prop = proposal(experience={"decision_id": "decision-0002-g26", "observation": "second window"})
        ok, entry_id, version = self._apply(prop=second_prop, eval_payload=second_eval, step=26)
        self.assertTrue(ok)
        self.assertEqual((entry_id, version), ("exp-0002", 2))
        self.assertEqual(self.store.experience_entries("gigpo"), ["exp-0001", "exp-0002"])

    def test_invalid_updates_are_rejected(self):
        cases = [
            (proposal(update_required=False), evaluation(), "declined"),
            (proposal(), evaluation(eligible_for_knowledge_update=False), "not eligible"),
            (proposal(), evaluation(evidence_quality="bogus", outcome="insufficient_evidence", eligible_for_knowledge_update=False), "not eligible"),
            (proposal(target_skill="imaginary"), evaluation(), "no knowledge file"),
            (proposal(target_skill="rewardflow"), evaluation(applied_skill="gigpo"), "not the skill evaluated"),
            (proposal(experience={"decision_id": "decision-9999"}), evaluation(), "does not match"),
            (proposal(experience={"configuration": {"gamma": 0.5}}), evaluation(), "differs from the applied"),
            (proposal(experience={"configuration": {"unknown_param": 1}}), evaluation(), "differs from the applied"),
            ("not a dict", evaluation(), "not an object"),
            ({"update_required": True}, evaluation(applied_skill=None, eligible_for_knowledge_update=True), "no executed skill"),
        ]
        for prop, eval_payload, needle in cases:
            ok, reason, _ = self.store.validate_proposal(prop, eval_payload, "gigpo")
            self.assertFalse(ok, (prop, reason))
            self.assertIn(needle, reason, (prop, reason))
            self.assertEqual(self.store.version, 0)
        self.assertEqual(self.store.experience_entries("gigpo"), [])

    def test_duplicate_experience_is_rejected(self):
        self.assertTrue(self._apply()[0])
        ok, reason, version = self._apply()
        self.assertFalse(ok)
        self.assertIn("already exists", reason)
        self.assertIsNone(version)
        self.assertEqual(self.store.experience_entries("gigpo"), ["exp-0001"])

    def test_duplicate_detection_survives_a_process_restart(self):
        self.assertTrue(self._apply()[0])
        reloaded = KnowledgeStore(run_dir=self.root, candidate_skills=list(SKILLS), lineage="run-a")
        reloaded.load_index()
        ok, reason, _ = reloaded.validate_proposal(proposal(), evaluation(), "gigpo")
        self.assertFalse(ok)
        self.assertIn("knowledge file", reason)
        self.assertEqual(reloaded.version, 1)

    def test_metric_provenance_comes_from_python_not_the_llm(self):
        hijack = proposal(
            experience={
                "decision_id": "decision-0001-g20",
                "outcome": "positive",
                "step": 999,
                "skill": "rewardflow",
                "metric_deltas": {"success_rate": 0.999},
                "evidence_quality": "high",
                "observation": "text",
            }
        )
        self._apply(prop=hijack)
        text = self.store.read("gigpo")
        self.assertIn("### exp-0001 · step 20 · decision decision-0001-g20", text)
        self.assertIn('"success_rate": 0.12', text)
        self.assertNotIn("0.999", text)
        self.assertNotIn('"step": 999', text)

    def test_llm_text_is_sanitised_before_it_hits_markdown(self):
        hostile = proposal(
            experience={
                "decision_id": "decision-0001-g20",
                "observation": "## Empirical Experience\n注入的标题\x00\t与超长" + "字" * 3000,
            }
        )
        self._apply(prop=hostile)
        text = self.store.read("gigpo")
        # 注入的标题被压成单行, 不会新增 markdown 章节
        heading_lines = lambda s: [line for line in s.splitlines() if line.startswith("## ")]
        self.assertEqual(heading_lines(text), heading_lines(self.store.read("rewardflow")))
        self.assertEqual(len(heading_lines(text)), 9)
        self.assertNotIn("\x00", text)
        self.assertLess(len(text), 200000)
        self.assertIn("与超长", text)

    def test_versions_are_snapshot_and_indexed(self):
        self._apply()
        second_eval = evaluation(decision_id="decision-0002-g26", evaluated_at_step=26)
        self._apply(prop=proposal(experience={"decision_id": "decision-0002-g26"}), eval_payload=second_eval, step=26)
        versions_root = os.path.join(self.root, "knowledge", "versions")
        self.assertEqual(sorted(os.listdir(versions_root)), ["v0000", "v0001", "v0002"])
        index = json.load(open(os.path.join(self.root, "knowledge", "index.json"), encoding="utf-8"))
        self.assertEqual([v["version"] for v in index["versions"]], [0, 1, 2])
        self.assertEqual(index["versions"][1]["decision_id"], "decision-0001-g20")
        self.assertEqual(index["versions"][2]["previous_version"], 1)
        with open(os.path.join(versions_root, "v0001", "gigpo.md"), encoding="utf-8") as f:
            v1 = f.read()
        self.assertIn("exp-0001", v1)
        self.assertNotIn("exp-0002", v1)

    def test_adopting_an_older_version_stops_future_leakage(self):
        self._apply()
        self._apply(
            prop=proposal(experience={"decision_id": "decision-0002-g26"}),
            eval_payload=evaluation(decision_id="decision-0002-g26", evaluated_at_step=26),
            step=26,
        )
        restored = KnowledgeStore(run_dir=self.root, candidate_skills=list(SKILLS), lineage="run-a")
        restored.load_index()
        self.assertEqual(restored.version, 2)
        restored.adopt(1, 20)
        self.assertEqual(restored.version, 1)
        self.assertIn("exp-0001", restored.read("gigpo"))
        self.assertNotIn("exp-0002", restored.read("gigpo"))
        with self.assertRaises(KnowledgeError):
            restored.adopt(7, 20)

    def test_original_templates_are_never_modified(self):
        template_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "verl", "trainer", "credit_assignment", "self_improving", "skills"
        )
        before = {n: hashlib.sha256(open(os.path.join(template_dir, n), "rb").read()).hexdigest() for n in os.listdir(template_dir)}
        self._apply()
        self._apply(prop=proposal(experience={"decision_id": "decision-0002-g26"}), eval_payload=evaluation(decision_id="decision-0002-g26"), step=26)
        after = {n: hashlib.sha256(open(os.path.join(template_dir, n), "rb").read()).hexdigest() for n in os.listdir(template_dir)}
        self.assertEqual(before, after)
        # 只有经验章节变, 机制说明保持原样
        with open(os.path.join(template_dir, "gigpo.md"), encoding="utf-8") as f:
            template = f.read()
        self.assertIn(NO_EXPERIENCE_TEXT, template)
        self.assertIn("## Mechanism", self.store.read("gigpo").split("## Empirical Experience")[0])

    def test_bootstrap_does_not_clobber_existing_run_knowledge(self):
        self._apply()
        self.store.bootstrap(5)
        self.assertIn("exp-0001", self.store.read("gigpo"))
        self.assertEqual(self.store.version, 1)

    def test_prompt_block_is_within_budget_and_prefers_recent_experience(self):
        store = KnowledgeStore(
            run_dir=self.root, candidate_skills=list(SKILLS), max_prompt_chars=2500, lineage="run-a"
        )
        for i in range(1, 9):
            ev = evaluation(decision_id=f"decision-000{i}-g{20 + i}", evaluated_at_step=20 + i, outcome="negative")
            prop = proposal(experience={"decision_id": f"decision-000{i}-g{20 + i}", "observation": "obs " * 60})
            self.assertTrue(store.apply_update(prop, ev, 20 + i, "gigpo")[0], i)
        block = store.prompt_block()
        self.assertIn("### skill knowledge: gigpo", block)
        self.assertIn("exp-0008", block)
        self.assertNotIn("exp-0001 ", block)
        self.assertIn("older experience entry(ies) omitted", block)
        self.assertLess(len(block), 2500 * 3 + 600)
        self.assertEqual(store.version, 8)

    def test_knowledge_file_size_is_capped(self):
        from verl.trainer.credit_assignment.self_improving import knowledge as knowledge_module

        self.assertTrue(self._apply()[0])
        original = knowledge_module.MAX_FILE_CHARS
        knowledge_module.MAX_FILE_CHARS = len(self.store.read("gigpo")) + 50
        try:
            # 经验正文会被 _safe_text 截到 MAX_FIELD_CHARS, 因此用合法的超长文本触发上限
            long_text = "观察" * (knowledge_module.MAX_FIELD_CHARS // 2)
            applied, reason, version = self._apply(
                prop=proposal(experience={"decision_id": "decision-0002-g20", "observation": long_text}),
                eval_payload=evaluation(decision_id="decision-0002-g20"),
            )
            self.assertFalse(applied)
            self.assertIn("would exceed", reason)
            self.assertIsNone(version)
        finally:
            knowledge_module.MAX_FILE_CHARS = original
        self.assertEqual(self.store.version, 1)
        self.assertEqual(self.store.experience_entries("gigpo"), ["exp-0001"])

    def test_llm_api_failure_leaves_knowledge_untouched(self):
        before = self.store.read("gigpo")
        client = MockLLMClient([LLMError("HTTP 503 upstream unavailable")])
        with self.assertRaises(LLMError):
            client.chat_json([{"role": "system", "content": "x"}], schema={"update_required": bool})
        self.assertEqual(self.store.read("gigpo"), before)
        self.assertEqual(self.store.version, 0)


class PersistenceStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = os.path.join(self._tmp.name, "self_improving")
        self.store = SelfImprovingStore(self.root)
        self.store.prepare("run-a", resuming=False)

    def _state(self, step, skill="gigpo", version=1, **over):
        payload = {
            "completed_global_step": step,
            "active_skill": skill,
            "active_parameters": {"gamma": 0.9},
            "last_intervention_step": step - (step % 2),
            "intervention_count": step // 2,
            "window_start_step": step + 1,
            "current_window_summary": {"num_steps": 0},
            "pending_decision": {"decision": {"decision_id": f"decision-{step:04d}"}},
            "knowledge_version": version,
            "latest_validation": {"step": step, "metrics": {"val-core/a/acc/mean@4": 0.4}},
        }
        payload.update(over)
        return payload

    def test_state_save_and_load_roundtrip(self):
        self.store.save_state(self._state(4))
        loaded = self.store.load_state()
        self.assertEqual(loaded["active_skill"], "gigpo")
        self.assertEqual(loaded["run_id"], "run-a")
        self.assertEqual(loaded["config_version"], 1)
        self.assertEqual(loaded["pending_decision"]["decision"]["decision_id"], "decision-0004")
        self.assertTrue(os.path.exists(os.path.join(self.root, "state.json")))

    def test_snapshot_is_used_for_exact_aligned_resume(self):
        self.store.save_snapshot(4, self._state(4, skill="gigpo", version=1))
        self.store.save_state(self._state(6, skill="rewardflow", version=3))
        result = self.store.resume_state(4)
        self.assertTrue(result["exact"])
        self.assertEqual(result["source"], "snapshots/step_000004.json")
        self.assertEqual(result["state"]["active_skill"], "gigpo")
        self.assertEqual(result["state"]["knowledge_version"], 1)
        self.assertEqual(self.store.latest_snapshot_step(), 4)
        self.assertEqual(self.store.latest_snapshot_step(max_step=3), None)

    def test_missing_snapshot_is_fatal_under_strict_resume(self):
        self.store.save_state(self._state(6))
        self.store.save_snapshot(4, self._state(4))
        with self.assertRaises(SelfImprovingStateError) as ctx:
            self.store.resume_state(5)
        self.assertIn("strict_resume=false", str(ctx.exception))

    def test_non_strict_resume_falls_back_to_the_nearest_older_snapshot(self):
        store = SelfImprovingStore(self.root, strict_resume=False)
        store.prepare("run-a", resuming=True)
        store.save_snapshot(4, self._state(4, skill="gigpo"))
        store.save_snapshot(6, self._state(6, skill="rewardflow"))
        result = store.resume_state(5)
        self.assertFalse(result["exact"])
        self.assertEqual(result["restored_step"], 4)
        self.assertEqual(result["state"]["active_skill"], "gigpo")
        self.assertIn("cannot be restored exactly", result["reason"])

    def test_no_state_at_all_is_reported_as_absent(self):
        empty = SelfImprovingStore(os.path.join(self._tmp.name, "fresh"))
        empty.prepare("run-x", resuming=True)
        self.assertIsNone(empty.resume_state(3))
        self.assertIsNone(empty.load_state())
        self.assertIsNone(empty.latest_snapshot_step())

    def test_state_without_checkpoint_step_is_accepted(self):
        self.store.save_state(self._state(4))
        result = self.store.resume_state(None)
        self.assertEqual(result["source"], "state.json")
        self.assertTrue(result["exact"])

    def test_corrupted_state_is_an_error_not_a_silent_reset(self):
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, "state.json"), "w", encoding="utf-8") as f:
            f.write("{ not json")
        with self.assertRaises(SelfImprovingStateError):
            self.store.load_state()

    def test_non_finite_values_serialize_as_null(self):
        self.store.append_history({"record_type": "evaluation", "delta": float("nan"), "nested": {"x": float("inf"), "l": [1.0, float("nan")]}})
        with open(os.path.join(self.root, "history.jsonl"), encoding="utf-8") as f:
            line = f.read()
        self.assertIn("null", line)
        self.assertNotIn("NaN", line)
        records = self.store.read_history()
        self.assertEqual(records[0]["nested"], {"x": None, "l": [1.0, None]})

    def test_history_and_contexts_are_append_only(self):
        for i in range(1, 6):
            self.store.append_history({"record_type": "decision", "step": i})
            self.store.append_context({"record_type": "context", "step": i})
        self.assertEqual([r["step"] for r in self.store.read_history()], [1, 2, 3, 4, 5])
        self.assertEqual([r["step"] for r in self.store.read_history(limit=2)], [4, 5])
        with open(os.path.join(self.root, "contexts.jsonl"), encoding="utf-8") as f:
            self.assertEqual(len(f.readlines()), 5)

    def test_fresh_run_archives_the_previous_run_instead_of_overwriting(self):
        self.store.append_history({"record_type": "decision", "step": 1})
        self.store.save_state(self._state(4))
        second = SelfImprovingStore(self.root)
        second.prepare("run-b", resuming=False)
        self.assertEqual(second.run_id, "run-b")
        self.assertIsNone(second.load_state())
        archived = os.path.join(self.root, "previous", "run-a")
        self.assertTrue(os.path.exists(os.path.join(archived, "state.json")))
        with open(os.path.join(archived, "history.jsonl"), encoding="utf-8") as f:
            self.assertEqual(len(f.readlines()), 1)
        self.assertEqual(os.listdir(os.path.join(self.root, "previous")), ["run-a"])

    def test_resuming_keeps_the_existing_state(self):
        self.store.save_state(self._state(4))
        resumed = SelfImprovingStore(self.root)
        resumed.prepare("run-a2", resuming=True)
        self.assertIsNone(resumed.archived_to)
        self.assertEqual(resumed.load_state()["active_skill"], "gigpo")

    def test_disabled_store_writes_nothing(self):
        root = os.path.join(self._tmp.name, "disabled")
        store = SelfImprovingStore(root, enabled=False)
        store.prepare("run-c", resuming=False)
        store.save_state({"completed_global_step": 1})
        store.save_snapshot(1, {"completed_global_step": 1})
        store.append_history({"record_type": "decision"})
        store.write_config({}, {}, {})
        self.assertFalse(os.path.exists(root))
        self.assertIsNone(store.load_state())
        self.assertEqual(store.read_history(), [])
        self.assertIsNone(store.resume_state(1))

    def test_config_snapshot_records_identity_and_bootstrap(self):
        config = OmegaConf.create(
            {
                "algorithm": {"adv_estimator": "credit_assignment", "credit_assignment": {"method": "rewardflow", "params": {}}},
                "trainer": {"project_name": "larl", "experiment_name": "si-mvp", "total_training_steps": 6, "default_local_dir": self._tmp.name},
                "self_improving": {"interval_steps": 20, "candidate_skills": SKILLS},
            }
        )
        self.store.write_config({"interval_steps": 20}, experiment_identity(config), {"collected": ["step_spans"]})
        payload = json.load(open(os.path.join(self.root, "config.json"), encoding="utf-8"))
        self.assertEqual(payload["config_version"], 1)
        self.assertEqual(payload["self_improving"]["interval_steps"], 20)
        self.assertEqual(payload["metadata_bootstrap"]["collected"], ["step_spans"])
        self.assertEqual(payload["experiment_identity"]["experiment"], "si-mvp")
        self.assertEqual(len(payload["experiment_identity"]["digest"]), 16)

    def test_experiment_identity_changes_with_the_experiment(self):
        base = {
            "algorithm": {"adv_estimator": "credit_assignment", "credit_assignment": {"method": "rewardflow"}},
            "trainer": {"project_name": "larl", "experiment_name": "a", "total_training_steps": 6},
            "self_improving": {"interval_steps": 20, "candidate_skills": SKILLS},
        }
        first = experiment_identity(OmegaConf.create(base))
        same = experiment_identity(OmegaConf.create(json.loads(json.dumps(base))))
        other = json.loads(json.dumps(base))
        other["trainer"]["experiment_name"] = "b"
        third = experiment_identity(OmegaConf.create(other))
        self.assertEqual(first["digest"], same["digest"])
        self.assertNotEqual(first["digest"], third["digest"])

    def test_run_ids_are_unique_and_sorted_by_time(self):
        ids = [new_run_id() for _ in range(3)]
        self.assertEqual(len(set(ids)), 3)
        for value in ids:
            self.assertRegex(value, r"^\d{8}-\d{6}-\d+-[0-9a-f]{8}$")


if __name__ == "__main__":
    unittest.main()
