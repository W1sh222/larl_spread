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

"""Self-Improving Slow Loop 端到端 CPU 集成.

覆盖 명령서 §21.10 (GRPO -> GiGPO -> RewardFlow 六步全流程, 接真实 registry / driver config /
持久化 / knowledge), §21.9 的 loop 层恢复 (pending evaluation / window / knowledge 版本 /
不泄露未来知识) 与 §21.11 (enabled=false 时什么都不发生).
"""

import ast
import copy
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.credit_assignment import available_methods, compute
from verl.trainer.credit_assignment.self_improving.llm_client import LLMError, LLMTimeout
from verl.trainer.credit_assignment.self_improving.persistence import SelfImprovingStateError
from verl.trainer.credit_assignment.self_improving.runtime import method_param_names
from verl.trainer.credit_assignment.self_improving.slow_loop import SelfImprovingLoop

RAY_TRAINER = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "verl", "trainer", "ppo", "ray_trainer.py"
    )
)
BS, LENGTH = 4, 12
CONTROLLER_MARKER = "## Executable parameter specification (authoritative)"
KNOWLEDGE_MARKER = "## Evaluation record (authoritative numbers)"

PROFILES = {
    "low": {"scores": [0.0, 1.0, 0.0, 0.0], "shaped": [0.0, 0.8, 0.0, 0.0], "acc": [0.0, 1.0, 0.0, 0.0]},
    "mid": {"scores": [1.0, 1.0, 0.0, 0.0], "shaped": [0.9, 0.9, 0.0, 0.0], "acc": [1.0, 1.0, 0.0, 0.0]},
    "down": {"scores": [0.5, 0.0, 0.0, 0.0], "shaped": [0.4, 0.0, 0.0, 0.0], "acc": [0.5, 0.0, 0.0, 0.0]},
}


def obj_array(rows):
    arr = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        arr[i] = row
    return arr


class FakeBatch:
    def __init__(self, tensors, non_tensor):
        self.batch = tensors
        self.non_tensor_batch = non_tensor


def make_batch(profile="mid"):
    p = PROFILES[profile] if isinstance(profile, str) else profile
    spans = [[[0, 4], [4, 8], [8, LENGTH]] for _ in range(BS)]
    scores = torch.zeros(BS, LENGTH)
    shaped = torch.zeros(BS, LENGTH)
    for i, value in enumerate(p["scores"]):
        scores[i, -1] = value
    for i, value in enumerate(p["shaped"]):
        shaped[i, -1] = value
    tensors = {
        "response_mask": torch.ones(BS, LENGTH),
        "responses": torch.arange(BS * LENGTH, dtype=torch.long).reshape(BS, LENGTH),
        "prompts": torch.zeros(BS, 4, dtype=torch.long),
        "token_level_scores": scores,
        "token_level_rewards": shaped,
    }
    non_tensor = {
        "uid": obj_array([f"g{i // 2}" for i in range(BS)]),
        "acc": obj_array(list(p["acc"])),
        "raw_prompt": obj_array([[{"role": "user", "content": f"fix workbook {i}"}] for i in range(BS)]),
        "ca_step_spans": obj_array(spans),
        "ca_step_observations": obj_array([["cell A1=1", "sheet2 opened", "formula written"] for _ in range(BS)]),
        "ca_step_transitions": obj_array([[{"pre": "sha256:a", "post": "sha256:b"}] * 3 for _ in range(BS)]),
        "ca_terminal_reason": obj_array(["final_answer", "max_turns", "final_answer", "max_turns"]),
        "ca_terminal_trusted": obj_array([True, True, False, True]),
        "ca_final_state": obj_array(["sha256:b"] * BS),
        "ca_state_stats": obj_array([{"parse_ms": 10.0} for _ in range(BS)]),
        "eval_workbook_fingerprint": obj_array(["sha256:eval"] * BS),
    }
    return FakeBatch(tensors, non_tensor)


class ScriptedLLM:
    """按顺序返回预设响应, 并记录每次调用属于 controller 还是 knowledge."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.last_latency_seconds = 0.01

    def extend(self, responses):
        self.responses.extend(responses)

    def chat_json(self, messages, schema=None):
        system = messages[0]["content"]
        kind = "controller" if CONTROLLER_MARKER in system else "knowledge" if KNOWLEDGE_MARKER in system else "other"
        self.calls.append({"kind": kind, "messages": messages, "system": system})
        if not self.responses:
            raise AssertionError(f"scripted LLM ran out of responses ({kind})")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def prompts(self, kind):
        return [c["system"] for c in self.calls if c["kind"] == kind]

    def count(self, kind):
        return len(self.prompts(kind))


def switch(skill, params):
    return {
        "action": "switch",
        "selected_skill": skill,
        "parameters": params,
        "reasoning": "step level credit should separate individual tool calls",
        "evidence": ["raw reward mean 0.25", "many zero-advantage assistant tokens"],
        "expected_improvement": "better per-step credit assignment",
        "risks": "depends on structured state metadata",
        "references": ["skills/" + skill + ".md"],
    }


def knowledge(target, decision_id, observation="the metric moved together with the switch", configuration=None):
    experience = {
        "decision_id": decision_id,
        "observation": observation,
        "interpretation": "consistent with the documented mechanism",
        "limitations": "single window, observational only",
    }
    if configuration is not None:
        experience["configuration"] = configuration
    return {"update_required": True, "target_skill": target, "experience": experience}


def si_node(local_dir, interval=2, enabled=True, **over):
    node = {
        "enabled": enabled,
        "interval_steps": interval,
        "initial_skill": {"method": "grpo", "params": {}},
        "candidate_skills": ["grpo", "gigpo", "rewardflow"],
        "llm": {"client": "mock"},
        # 测试窗口只有 2 步 (§21.10 允许在 CPU 测试里缩短 interval), 生产默认是 5 步.
        "evaluation": {"min_window_steps": 2},
        "persistence": {"enabled": True, "strict_resume": True},
    }
    node.update(over)
    return node


def trainer_config(local_dir, si_node_value):
    return {
        "algorithm": {"adv_estimator": "credit_assignment", "credit_assignment": {"method": "rewardflow", "params": {}}},
        "trainer": {
            "project_name": "larl",
            "experiment_name": "si-integration",
            "default_local_dir": local_dir,
            "total_training_steps": 6,
            "save_freq": 4,
        },
        "self_improving": si_node_value,
    }


class FakeTrainer:
    def __init__(self, local_dir, si_node_value, global_steps=0):
        self.config = OmegaConf.create(trainer_config(local_dir, si_node_value))
        self.global_steps = global_steps
        self.tokenizer = None
        self._si = None


class LoopTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.local_dir = os.path.join(self.root, "outputs")
        self.si_dir = os.path.join(self.local_dir, "self_improving")

    def _loop(self, responses, interval=2, global_steps=0, si_over=None):
        self.script = ScriptedLLM(responses)
        trainer = FakeTrainer(self.local_dir, si_node(self.local_dir, interval=interval, **(si_over or {})), global_steps)
        loop = SelfImprovingLoop(llm=self.script)
        loop.initialize(trainer)
        self.trainer = trainer
        return loop

    def _step(self, loop, step, profile, is_last=False, checkpoint=False):
        if checkpoint:
            loop.on_checkpoint_saved(step)
        metrics = {
            "actor/pg_loss": 0.25,
            "critic/score/mean": float(np.mean(PROFILES[profile]["scores"])),
            "response_length/mean": float(LENGTH),
            "credit_assignment/gigpo/anchor_count": 3.0,
        }
        loop.on_step_end(batch=make_batch(profile), metrics=metrics, global_step=step, is_last_step=is_last)
        return metrics

    def _run_six_steps(self, loop, checkpoints=(), skills_seen=None):
        sequence = [(1, "low"), (2, "low"), (3, "mid"), (4, "mid"), (5, "down"), (6, "down")]
        all_metrics = {}
        for step, profile in sequence:
            all_metrics[step] = self._step(loop, step, profile, is_last=(step == 6), checkpoint=(step in checkpoints))
            if skills_seen is not None:
                skills_seen[step] = loop.executed.method
        loop.finalize()
        return all_metrics

    def _history(self, loop, record_type=None):
        # history.jsonl 是 append-only 且跨 run 共用, 因此按 run_id 过滤.
        records = [r for r in loop.store.read_history() if r.get("run_id") == loop.run_id]
        return [r for r in records if record_type is None or r.get("record_type") == record_type]


class FullWorkflowTest(LoopTestCase):
    def setUp(self):
        super().setUp()
        self.responses = [
            switch("gigpo", {"gamma": 0.9, "step_advantage_w": 1.0}),
            knowledge("gigpo", "decision-0001-g2", configuration={"gamma": 0.9}),
            switch("rewardflow", {"gamma": 0.9, "alpha_act": 1.0}),
            knowledge("rewardflow", "decision-0002-g4", configuration={"gamma": 0.9, "alpha_act": 1.0}),
        ]

    def _run(self):
        self.skills_seen = {}
        self.loop = self._loop(self.responses)
        self.metrics = self._run_six_steps(self.loop, checkpoints=(4,), skills_seen=self.skills_seen)
        return self.loop

    def test_driver_config_follows_the_skill_trajectory(self):
        loop = self._run()
        cfg = self.trainer.config.algorithm.credit_assignment
        self.assertEqual(str(cfg.method), "rewardflow")
        self.assertEqual(OmegaConf.to_container(cfg.params, resolve=True), loop.active.params)
        self.assertEqual(
            sorted(loop.active.params), sorted(method_param_names("rewardflow"))
        )
        self.assertNotIn("norm_adv_by_std_in_grpo", loop.active.params)
        self.assertNotIn("step_advantage_w", loop.active.params)
        self.assertEqual(loop.active.params["gamma"], 0.9)
        self.assertEqual(loop.knowledge.version, 2)
        self.assertEqual(loop.intervention_count, 3)
        self.assertEqual(loop.last_intervention_step, 6)

    def test_decision_takes_effect_from_the_next_step(self):
        loop = self._run()
        self.assertEqual(
            self.skills_seen,
            {1: "grpo", 2: "grpo", 3: "gigpo", 4: "gigpo", 5: "rewardflow", 6: "rewardflow"},
        )
        codes = {step: self.metrics[step]["self_improving/executed_skill_id"] for step in self.skills_seen}
        self.assertEqual(codes, {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 2.0, 6: 2.0})
        self.assertEqual(self.metrics[2]["self_improving/next_skill_id"], 1.0)
        self.assertEqual(self.metrics[2]["self_improving/intervention_triggered"], 1.0)
        self.assertEqual(self.metrics[3]["self_improving/intervention_triggered"], 0.0)
        self.assertEqual(loop.executed.method, "rewardflow")

    def test_applied_config_runs_through_the_existing_registry(self):
        loop = self._loop(self.responses)
        fingerprints = {
            "grpo": "singleton_group_ratio",
            "gigpo": "step_adv_nonzero_frac",
            "rewardflow": "nonzero_action_adv_frac",
        }
        seen = []
        for step, profile in [(1, "low"), (2, "low"), (3, "mid"), (4, "mid"), (5, "down"), (6, "down")]:
            self._step(loop, step, profile, is_last=(step == 6))
            if step not in (1, 3, 5):
                continue
            algo = self.trainer.config.algorithm
            method = str(algo.credit_assignment.method)
            batch = make_batch(profile)
            out = compute(
                method=method,
                token_level_rewards=batch.batch["token_level_rewards"],
                response_mask=batch.batch["response_mask"],
                index=batch.non_tensor_batch["uid"],
                non_tensor_batch=batch.non_tensor_batch,
                params=dict(algo.credit_assignment.params or {}),
                raw_token_level_scores=batch.batch["token_level_scores"],
            )
            self.assertEqual(tuple(out.advantages.shape), (BS, LENGTH))
            self.assertTrue(np.isfinite(out.advantages.detach().numpy()).all())
            self.assertIn(fingerprints[method], out.metrics)
            self.assertEqual(method, loop.executed.method)
            seen.append((step, method))
        self.assertEqual(seen, [(1, "grpo"), (3, "gigpo"), (5, "rewardflow")])
        self.assertTrue({"grpo", "gigpo", "rewardflow"} <= set(available_methods()))

    def test_evaluations_are_attributed_to_the_right_windows(self):
        loop = self._run()
        records = self._history(loop, "evaluation")
        self.assertEqual(len(records), 2)
        first, second = records
        self.assertEqual(first["decision_id"], "decision-0001-g2")
        self.assertEqual(first["previous_skill"], "grpo")
        self.assertEqual(first["applied_skill"], "gigpo")
        self.assertEqual(first["baseline_window"]["step_range"], [1, 2])
        self.assertEqual(first["observed_window"]["step_range"], [3, 4])
        self.assertEqual(first["outcome"], "positive")
        self.assertAlmostEqual(first["metric_deltas"]["success_rate"], 0.25, places=6)
        self.assertEqual(second["decision_id"], "decision-0002-g4")
        self.assertEqual(second["previous_skill"], "gigpo")
        self.assertEqual(second["applied_skill"], "rewardflow")
        self.assertEqual(second["baseline_window"]["step_range"], [3, 4])
        self.assertEqual(second["observed_window"]["step_range"], [5, 6])
        self.assertEqual(second["outcome"], "negative")
        self.assertAlmostEqual(second["metric_deltas"]["success_rate"], -0.375, places=6)
        for record in records:
            self.assertIn("observational only (not a causal effect estimate)", record["summary"])
            self.assertEqual(record["evidence_quality"], "medium")
            self.assertEqual(record["evidence_notes"], [])
            self.assertTrue(record["eligible_for_knowledge_update"])

    def test_updated_knowledge_reaches_the_next_controller_input(self):
        loop = self._run()
        self.assertEqual(self.script.count("controller"), 2)
        self.assertEqual(self.script.count("knowledge"), 2)
        first_prompt = self.script.prompts("controller")[0]
        self.assertIn("_No verified experience recorded yet._", first_prompt)
        second_prompt = self.script.prompts("controller")[1]
        self.assertIn("exp-0001", second_prompt)
        self.assertIn("### skill knowledge: gigpo", second_prompt)
        self.assertNotIn("{{", second_prompt)
        self.assertIn("### exp-0001 · step 4 · decision decision-0001-g2", loop.knowledge.read("gigpo"))
        self.assertIn("### exp-0001 · step 6 · decision decision-0002-g4", loop.knowledge.read("rewardflow"))
        index = json.load(open(os.path.join(self.si_dir, "knowledge", "index.json"), encoding="utf-8"))
        self.assertEqual([v["version"] for v in index["versions"]], [0, 1, 2])
        self.assertEqual([v["target_skill"] for v in index["versions"][1:]], ["gigpo", "rewardflow"])
        self.assertEqual([v["decision_id"] for v in index["versions"][1:]], ["decision-0001-g2", "decision-0002-g4"])

    def test_negative_evaluation_does_not_roll_back(self):
        loop = self._run()
        cfg = self.trainer.config.algorithm.credit_assignment
        self.assertEqual(str(cfg.method), "rewardflow")
        self.assertEqual(loop.active.method, "rewardflow")
        self.assertEqual(loop.executed.method, "rewardflow")
        blob = json.dumps(loop.store.read_history(), default=str).lower()
        self.assertNotIn("rollback", blob)
        self.assertNotIn("rollback", json.dumps(loop._state(6), default=str).lower())
        self.assertEqual(self.script.count("controller"), 2)

    def test_last_step_skips_selection(self):
        loop = self._run()
        records = [r["record_type"] for r in self._history(loop)]
        self.assertEqual(records.count("selection_skipped"), 1)
        self.assertEqual(records.count("decision"), 2)
        self.assertEqual(records.count("knowledge_attempt"), 2)
        self.assertEqual(records.count("evaluation"), 2)
        skipped = self._history(loop, "selection_skipped")[0]
        self.assertEqual(skipped["step"], 6)
        self.assertEqual(self.script.count("controller"), 2)

    def test_history_state_and_sidecar_snapshot_are_written(self):
        loop = self._run()
        for name in ("state.json", "history.jsonl", "contexts.jsonl", "config.json"):
            self.assertTrue(os.path.exists(os.path.join(self.si_dir, name)), name)
        self.assertTrue(os.path.exists(os.path.join(self.si_dir, "snapshots", "step_000004.json")))
        state = json.load(open(os.path.join(self.si_dir, "state.json"), encoding="utf-8"))
        self.assertEqual(state["completed_global_step"], 6)
        self.assertEqual(state["active_skill"], "rewardflow")
        self.assertEqual(state["executed_skill"], "rewardflow")
        self.assertEqual(state["knowledge_version"], 2)
        self.assertEqual(state["intervention_count"], 3)
        self.assertEqual(state["window_start_step"], 7)
        self.assertIsNone(state["pending_decision"])
        snapshot = json.load(open(os.path.join(self.si_dir, "snapshots", "step_000004.json"), encoding="utf-8"))
        self.assertEqual(snapshot["snapshot_step"], 4)
        self.assertEqual(snapshot["active_skill"], "rewardflow")
        self.assertEqual(snapshot["knowledge_version"], 1)
        self.assertEqual(snapshot["pending_decision"]["decision"]["decision_id"], "decision-0002-g4")
        self.assertEqual(snapshot["window_start_step"], 5)
        self.assertEqual(snapshot["current_window_summary"], {"num_steps": 0})
        history = self._history(loop)
        self.assertTrue(all("run_id" in r and "ts" in r for r in history))
        finalized = [r for r in history if r["record_type"] == "finalized"][0]
        self.assertEqual(finalized["active_skill"]["method"], "rewardflow")
        self.assertEqual(finalized["knowledge_version"], 2)
        contexts = loop.store.read_contexts() if hasattr(loop.store, "read_contexts") else []
        if not contexts:
            with open(os.path.join(self.si_dir, "contexts.jsonl"), encoding="utf-8") as f:
                contexts = [json.loads(line) for line in f]
        self.assertEqual([c["step"] for c in contexts], [2, 4, 6])
        self.assertEqual([c["knowledge_version"] for c in contexts], [0, 1, 2])

    def test_metrics_are_numeric_and_window_scoped(self):
        loop = self._run()
        keys = set()
        for values in self.metrics.values():
            keys |= {k for k in values if k.startswith("self_improving/")}
        expected = {
            "self_improving/executed_skill_id",
            "self_improving/next_skill_id",
            "self_improving/intervention_count",
            "self_improving/knowledge_version",
            "self_improving/metadata_available_ratio",
            "self_improving/intervention_triggered",
            "self_improving/slow_loop_seconds",
        }
        self.assertTrue(expected <= keys, sorted(expected - keys))
        for step, values in self.metrics.items():
            for key, value in values.items():
                if not key.startswith("self_improving/"):
                    continue
                self.assertIsInstance(value, (int, float), (step, key, value))
                self.assertNotIsInstance(value, bool)
                self.assertTrue(np.isfinite(value), (step, key, value))
        self.assertEqual(self.metrics[2]["self_improving/knowledge_version"], 0.0)
        self.assertEqual(self.metrics[4]["self_improving/knowledge_version"], 1.0)
        self.assertEqual(self.metrics[4]["self_improving/eval_delta_success_rate"], 0.25)
        self.assertEqual(self.metrics[6]["self_improving/eval_delta_success_rate"], -0.375)
        self.assertEqual(self.metrics[6]["self_improving/knowledge_updated"], 1.0)
        self.assertEqual(self.metrics[3]["self_improving/metadata_available_ratio"], 1.0)
        self.assertNotIn("self_improving/metadata_available_ratio", self.metrics[1])
        self.assertNotIn("self_improving/llm_call_seconds", self.metrics[1])
        self.assertGreater(self.metrics[2]["self_improving/slow_loop_seconds"], 0.0)

    def test_controller_input_shape_matches_the_scenario(self):
        loop = self._loop(self.responses)
        self.assertEqual(
            loop.runtime.collected,
            frozenset({"step_spans", "step_observations", "step_transitions", "terminal_trusted", "eval_fingerprint"}),
        )
        self._step(loop, 1, "low")
        self._step(loop, 2, "low")
        context = loop.last_context
        self.assertEqual(context["global_step"], 2)
        self.assertEqual(context["window_start_step"], 1)
        self.assertEqual(context["window_end_step"], 2)
        self.assertEqual(context["active_skill"]["method"], "grpo")
        self.assertIsNone(context["previous_evaluation"])
        self.assertEqual(context["notes"]["interval_steps"], 2)
        self.assertEqual(context["notes"]["observed_steps_in_window"], 2)
        self.assertEqual(context["notes"]["skill_codes"], {"grpo": 0.0, "gigpo": 1.0, "rewardflow": 2.0})
        self.assertEqual(context["notes"]["metadata_bootstrap"]["collected"][0], "eval_fingerprint")
        self.assertEqual(context["trajectory_samples"][0]["steps"][0]["observation"], "cell A1=1")
        self.assertEqual(len(self.script.calls), 1)
        self.assertEqual(loop.knowledge.version, 0)


class LoopSafetyTest(LoopTestCase):
    def test_llm_failure_falls_back_without_touching_the_skill(self):
        loop = self._loop([LLMTimeout("connection timed out"), LLMTimeout("connection timed out")])
        self._step(loop, 1, "low")
        step_two = self._step(loop, 2, "low")
        self.assertEqual(loop.active.method, "grpo")
        self.assertEqual(loop.executed.method, "grpo")
        self.assertEqual(str(self.trainer.config.algorithm.credit_assignment.method), "grpo")
        decisions = self._history(loop, "decision")
        self.assertEqual(decisions[0]["status"], "fallback")
        self.assertEqual(step_two["self_improving/decision_valid"], 0.0)
        self.assertEqual(step_two["self_improving/next_skill_id"], 0.0)
        self._step(loop, 3, "mid")
        self._step(loop, 4, "mid")
        # §M-1: 没有永久停用标记, 仅保留诊断用的 streak.
        self.assertFalse(hasattr(loop, "disabled_reason"))
        self.assertEqual(loop.resume_report, None)
        self.assertEqual(loop.knowledge.version, 0)
        evaluations = self._history(loop, "evaluation")
        self.assertEqual(evaluations[0]["outcome"], "insufficient_evidence")
        self.assertIn("failed/invalid controller decision", json.dumps(evaluations[0]["evidence_notes"]))

    def test_unexpected_loop_error_does_not_stop_training(self):
        loop = self._loop([switch("gigpo", {"gamma": 0.9})] * 2)

        def boom(*args, **kwargs):
            raise RuntimeError("collector exploded")

        loop._intervene = boom
        for step in (1, 2, 3, 4):
            self._step(loop, step, "mid")
        # streak 只记录诊断, 不会让 Slow Loop 静默退出.
        self.assertGreaterEqual(loop._failure_streak, 2)
        self.assertFalse(hasattr(loop, "disabled_reason"))
        self.assertEqual(str(self.trainer.config.algorithm.credit_assignment.method), "grpo")
        history = self._history(loop)
        self.assertGreaterEqual(sum(1 for r in history if r["record_type"] == "error"), 2)
        self.assertEqual(sum(1 for r in history if r["record_type"] == "disabled"), 0)

    def test_repeated_recoverable_intervention_failures_retry_at_the_next_interval(self):
        """§M-1: 3 次以上失败不再永久停用, 下一个 interval 仍会重试."""
        loop = self._loop([], interval=1)
        attempts = {"count": 0}

        def boom(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] <= 4:
                raise RuntimeError(f"transient failure #{attempts['count']}")
            # 第 5 次起允许走一次完整路径: 交给 controller 返回 keep.
            return loop.__class__.__mro__[0].__dict__["_intervene"].__get__(loop)(*args, **kwargs)

        loop._intervene = boom
        seen = []
        for step in range(1, 6):
            seen.append(self._step(loop, step, "mid"))
        self.assertEqual(attempts["count"], 5)
        # 前 4 次失败只在诊断指标里累计 (不停用), 第 4 步 streak >= 4.
        self.assertGreaterEqual(seen[3]["self_improving/consecutive_failures"], 4.0)
        # 第 5 次重试成功 -> streak 归零: 这正是 M-1 的语义 (失败后仍按周期重试, 不永久停用).
        self.assertEqual(seen[4]["self_improving/consecutive_failures"], 0.0)
        self.assertNotIn("self_improving/slow_loop_disabled", seen[4])
        # 没有 "disabled" 记录: 只是反复 error.
        history = self._history(loop)
        self.assertGreaterEqual(sum(1 for r in history if r["record_type"] == "error"), 4)
        self.assertEqual(sum(1 for r in history if r["record_type"] == "disabled"), 0)
        # 当前 Skill 保持 grpo (apply 未发生, 仅记录原因).
        self.assertEqual(loop.active.method, "grpo")
        self.assertEqual(str(self.trainer.config.algorithm.credit_assignment.method), "grpo")

    def test_state_inconsistency_is_not_swallowed_by_the_generic_handler(self):
        """§M-2: SelfImprovingStateError 不被 except Exception 吞掉 -> fail-fast."""
        loop = self._loop([], interval=1)

        def bad_state(*args, **kwargs):
            from verl.trainer.credit_assignment.self_improving.persistence import SelfImprovingStateError

            raise SelfImprovingStateError("state schema disagreement")

        loop._intervene = bad_state
        with self.assertRaises(SelfImprovingStateError):
            self._step(loop, 1, "mid")
        # Fail-fast 不会产生 "error" 记录, 只记录到日志, 但不会静默.
        self.assertEqual(sum(1 for r in self._history(loop) if r["record_type"] == "error"), 0)

    def test_persistence_failures_eventually_escalate_to_state_error(self):
        """§M-7: 反复落盘失败 -> SelfImprovingStateError (累计达 PERSISTENCE_FAILURE_LIMIT)."""
        # interval=1: 每步都尝试介入, 每次介入的多条 history 写入都失败, 累计计数很快触顶.
        loop = self._loop([switch("gigpo", {"gamma": 0.9})], interval=1)
        # 让 history 写入一直失败 (state.json 仍能写, 只有记录通道坏).
        loop.store.append_history = lambda rec: (_ for _ in ()).throw(OSError("disk gone"))
        raised = False
        for step in range(1, 6):
            try:
                self._step(loop, step, "mid")
            except SelfImprovingStateError:
                raised = True
                break
        # 落盘失败不能记成成功: 计数触顶后 fail-fast, 而不是静默继续训练.
        self.assertTrue(raised)
        self.assertGreaterEqual(loop._persistence_failures, 3)
        # 失败前 state.json 仍按 step 落盘 (checkpoint 对齐的 sidecar 不会因 history 坏而丢).
        self.assertIsNotNone(loop.store.load_state())

    def test_rejected_knowledge_proposal_keeps_the_version_and_applies_the_decision(self):
        responses = [
            switch("gigpo", {"gamma": 0.9}),
            knowledge("rewardflow", "decision-0001-g2"),
            switch("rewardflow", {"gamma": 0.9}),
            {"update_required": False},
        ]
        loop = self._loop(responses)
        self._run_six_steps(loop)
        self.assertEqual(loop.knowledge.version, 0)
        self.assertEqual(loop.knowledge.experience_entries("gigpo"), [])
        attempts = self._history(loop, "knowledge_attempt")
        self.assertEqual([a["status"] for a in attempts], ["rejected", "rejected"])
        self.assertIn("not the skill evaluated", attempts[0]["reason"])
        self.assertIn("declined", attempts[1]["reason"])
        self.assertEqual(loop.active.method, "rewardflow")
        self.assertEqual(loop.knowledge.experience_entries("rewardflow"), [])

    def test_keep_decision_is_not_attributed_as_a_skill_change(self):
        """§10 minor: keep (无实际配置变更) 的窗口差异只能记为 policy drift, 不进知识."""
        keep = {
            "action": "keep",
            "selected_skill": "grpo",
            "parameters": {"norm_adv_by_std_in_grpo": True, "epsilon": 1e-06},
            "reasoning": "current credit signal is already the sparsest unbiased one",
            "evidence": ["raw reward mean stable"],
        }
        loop = self._loop([keep, keep, {"update_required": False}])
        self._step(loop, 1, "low")
        self._step(loop, 2, "mid")
        self._step(loop, 3, "mid")
        self._step(loop, 4, "mid", is_last=True)
        loop.finalize()
        decisions = self._history(loop, "decision")
        self.assertTrue(decisions)
        self.assertFalse(decisions[0]["changed"])
        evaluations = self._history(loop, "evaluation")
        self.assertTrue(evaluations)
        keep_eval = evaluations[-1]
        self.assertEqual(keep_eval["outcome"], "insufficient_evidence")
        self.assertFalse(keep_eval["eligible_for_knowledge_update"])
        self.assertIn("no skill or parameter actually changed", json.dumps(keep_eval["evidence_notes"]))
        # 窗口摘要仍保留 (不伪造 decision / evaluation, 也不丢诊断).
        self.assertTrue(self._history(loop, "window_summary"))
        self.assertEqual(loop.active.method, "grpo")
        self.assertEqual(loop.knowledge.version, 0)


class ResumeTest(LoopTestCase):
    """§21.9 loop 层: pending evaluation / window / knowledge 版本恢复."""

    def _phase_a(self, steps=(1, 2, 3, 4), checkpoint=4):
        responses = [
            switch("gigpo", {"gamma": 0.9}),
            knowledge("gigpo", "decision-0001-g2", configuration={"gamma": 0.9}),
            switch("rewardflow", {"gamma": 0.9, "alpha_act": 1.0}),
            knowledge("rewardflow", "decision-0002-g4", configuration={"gamma": 0.9, "alpha_act": 1.0}),
        ]
        loop = self._loop(responses)
        profiles = {1: "low", 2: "low", 3: "mid", 4: "mid", 5: "down", 6: "down"}
        for step in steps:
            self._step(
                loop,
                step,
                profiles[step],
                is_last=(step == 6),
                checkpoint=(step == checkpoint and step in steps),
            )
        return loop

    def test_resume_restores_pending_decision_window_and_knowledge(self):
        phase_a = self._phase_a()
        self.assertEqual(phase_a.knowledge.version, 1)
        self.assertEqual(str(phase_a.runtime.current().method), "rewardflow")
        loop_b = self._loop([knowledge("rewardflow", "decision-0002-g4", configuration={"gamma": 0.9, "alpha_act": 1.0})], global_steps=4)
        self.assertEqual(loop_b.resume_report["resumed_from_step"], 4)
        self.assertEqual(loop_b.resume_report["state_source"], "snapshots/step_000004.json")
        self.assertTrue(loop_b.resume_report["exact"])
        self.assertTrue(loop_b.resume_report["knowledge_version_restored"])
        self.assertEqual(loop_b.active.method, "rewardflow")
        self.assertEqual(loop_b.knowledge.version, 1)
        self.assertEqual(loop_b.last_intervention_step, 4)
        self.assertEqual(loop_b.window_start_step, 5)
        self.assertIsNone(loop_b.carryover_window)
        self.assertEqual(loop_b.pending["decision"]["decision_id"], "decision-0002-g4")
        self.assertEqual(loop_b.pending["baseline_window"]["step_range"], [3, 4])
        self._step(loop_b, 5, "down")
        self.assertEqual(loop_b.collector.steps[-1]["skill"], "rewardflow")
        self._step(loop_b, 6, "down", is_last=True)
        loop_b.finalize()
        records = self._history(loop_b, "evaluation")
        self.assertEqual(len(records), 1)
        evaluation = records[0]
        self.assertEqual(evaluation["decision_id"], "decision-0002-g4")
        self.assertEqual(evaluation["baseline_window"]["step_range"], [3, 4])
        self.assertEqual(evaluation["observed_window"]["step_range"], [5, 6])
        self.assertEqual(evaluation["outcome"], "negative")
        self.assertNotIn("resume", json.dumps(evaluation["evidence_notes"]))
        self.assertEqual(self.script.count("knowledge"), 1)
        self.assertEqual(self.script.count("controller"), 0)
        self.assertEqual(loop_b.knowledge.version, 2)

    def test_resume_does_not_leak_future_knowledge(self):
        phase_a = self._phase_a(steps=(1, 2, 3, 4, 5, 6), checkpoint=4)
        self.assertEqual(phase_a.knowledge.version, 2)
        self.assertIn("exp-0001", phase_a.knowledge.read("rewardflow"))
        loop_b = self._loop([LLMError("unused")], global_steps=4)
        self.assertEqual(loop_b.knowledge.version, 1)
        self.assertEqual(loop_b.resume_report["knowledge_version_restored"], True)
        self.assertIn("exp-0001", loop_b.knowledge.read("gigpo"))
        self.assertNotIn("exp-0001", loop_b.knowledge.read("rewardflow"))
        self.assertEqual(loop_b.runtime.current().method, "rewardflow")
        prompt = loop_b.knowledge.prompt_block()
        self.assertNotIn("decision-0002-g4", prompt)

    def test_mid_window_resume_flags_the_split_window(self):
        self._phase_a(steps=(1, 2, 3, 4, 5), checkpoint=5)
        loop_b = self._loop([knowledge("rewardflow", "decision-0002-g4")], global_steps=5)
        self.assertEqual(loop_b.window_start_step, 6)
        self.assertEqual(loop_b.carryover_window["num_steps"], 1)
        self.assertEqual(loop_b.last_intervention_step, 4)
        self._step(loop_b, 6, "down", is_last=True)
        loop_b.finalize()
        records = self._history(loop_b, "evaluation")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["outcome"], "insufficient_evidence")
        self.assertFalse(records[0]["eligible_for_knowledge_update"])
        notes = json.dumps(records[0]["evidence_notes"])
        self.assertIn("split by a resume", notes)
        self.assertIn("min_window_steps", notes)
        self.assertEqual(self.script.count("knowledge"), 0)
        self.assertEqual(loop_b.knowledge.version, 1)

    def test_strict_resume_refuses_a_missing_snapshot(self):
        self._phase_a()
        os.remove(os.path.join(self.si_dir, "snapshots", "step_000004.json"))
        with self.assertRaises(SelfImprovingStateError) as ctx:
            self._loop([switch("gigpo", {"gamma": 0.9})], global_steps=4)
        self.assertIn("strict_resume=false", str(ctx.exception))

    def test_non_strict_resume_falls_back_and_flags_the_gap(self):
        self._phase_a()
        os.rename(
            os.path.join(self.si_dir, "snapshots", "step_000004.json"),
            os.path.join(self.si_dir, "snapshots", "step_000003.json"),
        )
        loop = self._loop(
            [switch("grpo", {"norm_adv_by_std_in_grpo": True})],
            global_steps=4,
            si_over={"persistence": {"enabled": True, "strict_resume": False}},
        )
        self.assertFalse(loop.resume_report["exact"])
        self.assertEqual(loop.resume_report["state_source"], "snapshots/step_000003.json")
        self.assertEqual(loop.resume_report["restored_step"], 3)
        self.assertIn("cannot be restored exactly", loop.resume_report["reason"])
        self._step(loop, 5, "down")
        self._step(loop, 6, "down", is_last=True)
        loop.finalize()
        evaluation = self._history(loop, "evaluation")[-1]
        self.assertIn("not exactly aligned", json.dumps(evaluation["evidence_notes"]))
        self.assertEqual(evaluation["outcome"], "insufficient_evidence")
        self.assertEqual(self.script.count("knowledge"), 0)
        self.assertEqual(self.script.count("controller"), 0)

    def test_resume_from_a_run_without_self_improving_state(self):
        loop = self._loop([switch("gigpo", {"gamma": 0.9}), {"update_required": False}], global_steps=4)
        self.assertIsNone(loop.resume_report["state_source"])
        self.assertFalse(loop.resume_report["exact"])
        self.assertIn("no self_improving state", loop.resume_report["reason"])
        self.assertEqual(loop.last_intervention_step, 4)
        self.assertEqual(loop.active.method, "grpo")
        self.assertEqual(loop.window_start_step, 5)
        self.assertIsNone(loop.carryover_window)
        self._step(loop, 5, "mid")
        self.assertEqual(self.script.count("controller"), 0)
        self.assertEqual(loop.executed.method, "grpo")
        self._step(loop, 6, "mid")
        self.assertEqual(self.script.count("controller"), 1)
        self.assertEqual(self.script.count("knowledge"), 0)
        self.assertEqual(str(self.trainer.config.algorithm.credit_assignment.method), "gigpo")
        self._step(loop, 7, "mid")
        self.assertEqual(loop.executed.method, "gigpo")
        self._step(loop, 8, "mid", is_last=True)
        loop.finalize()
        records = self._history(loop, "evaluation")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["decision_id"], "decision-0001-g6")
        self.assertEqual(records[0]["previous_skill"], "grpo")
        self.assertEqual(records[0]["applied_skill"], "gigpo")
        self.assertEqual(records[0]["observed_window"]["step_range"], [7, 8])
        self.assertEqual(self.script.count("knowledge"), 1)
        self.assertEqual(loop.knowledge.version, 0)

    def test_resume_ignores_history_written_after_the_restored_step(self):
        """§M-5: 同目录里上一 session 写到更晚 step 的记录不能当成 controller 的当前证据."""
        phase_a = self._phase_a(steps=(1, 2, 3, 4, 5, 6), checkpoint=4)
        future = [r for r in phase_a.store.read_history() if int(r.get("step") or 0) > 4]
        self.assertTrue(future)  # phase A 确实在 checkpoint 之后又写了记录
        loop_b = self._loop([{"update_required": False}], global_steps=4)
        self.assertEqual(loop_b.history_max_step, 4)
        # 预取的 recent history 只取 step <= ckpt 的记录 (step=0 的 initialized 不携 step, 仍可保留).
        capped = loop_b.store.read_history(max_step=loop_b.history_max_step)
        self.assertTrue(capped)
        self.assertTrue(
            all(int(r.get("step") or 0) <= 4 for r in capped),
            [(r.get("record_type"), r.get("step")) for r in capped],
        )
        self.assertGreater(len(loop_b.store.read_history()), len(capped))
        # 本 session 新写的记录不受限制.
        self._step(loop_b, 5, "down")
        self._step(loop_b, 6, "down", is_last=True)
        loop_b.finalize()
        self.assertIsNone(loop_b.history_max_step)
        own = self._history(loop_b)
        self.assertTrue(any(int(r.get("step") or 0) > 4 for r in own))

    def test_strict_resume_refuses_research_config_drift(self):
        """§M-6: candidate_skills 等研究语义设置不一致 -> 显式失败."""
        self._phase_a()
        with self.assertRaises(SelfImprovingStateError) as ctx:
            self._loop(
                [switch("grpo", {"norm_adv_by_std_in_grpo": True})],
                global_steps=4,
                si_over={"candidate_skills": ["grpo", "gigpo"]},
            )
        self.assertIn("candidate_skills", str(ctx.exception))

    def test_resume_allows_operational_changes_and_records_them(self):
        """§M-6: endpoint / timeout 等运营设置可变, 但必须写进历史."""
        self._phase_a()
        loop_b = self._loop(
            [switch("grpo", {"norm_adv_by_std_in_grpo": True})],
            global_steps=4,
            si_over={"llm": {"client": "mock", "timeout_seconds": 3.0}},
        )
        self.assertIn("llm.timeout_seconds", loop_b.resume_report.get("operational_config_changes") or [])
        self.assertTrue(loop_b.resume_report["exact"])
        self._step(loop_b, 5, "down")
        changes = self._history(loop_b, "config_change")
        self.assertTrue(changes)

    def test_resume_then_new_knowledge_update_does_not_overwrite_old_versions(self):
        """§M-5 / §M-6: 恢复后开新记录, 旧 knowledge 版本保持不动, 新旧 session 可区分."""
        phase_a = self._phase_a(steps=(1, 2, 3, 4, 5, 6), checkpoint=4)
        baseline_versions = [v["version"] for v in phase_a.knowledge.load_index()["versions"]]
        loop_b = self._loop([knowledge("gigpo", "decision-0001-g2", configuration={"gamma": 0.9})], global_steps=4)
        self.assertNotEqual(loop_b.run_id, phase_a.run_id)
        # lineage 延续 (同一个实验链), session (run_id) 是新的: 新记录能按 run_id 分离.
        self.assertEqual(loop_b.store.load_state()["lineage"], phase_a.store.load_state()["lineage"])
        self.assertEqual([v["version"] for v in loop_b.knowledge.load_index()["versions"]], baseline_versions)
        self._step(loop_b, 5, "down")
        self._step(loop_b, 6, "down", is_last=True)
        loop_b.finalize()
        # 恢复后写的记录属于新的 run_id, phase A 自己的记录仍带旧 run_id.
        phase_a_runs = {r.get("run_id") for r in phase_a.store.read_history(only_run_id=phase_a.run_id)}
        self.assertEqual(phase_a_runs, {phase_a.run_id})
        # 旧版本目录内容不被覆写: phase A 自己的 knowledge 文本保持原样.
        self.assertIn("exp-0001", phase_a.knowledge.read("gigpo"))
        self.assertNotIn("exp-0001", phase_a.knowledge.read("rewardflow"))


class DisabledModeTest(LoopTestCase):
    def test_initialize_refuses_and_leaves_no_trace(self):
        script = ScriptedLLM([])
        trainer = FakeTrainer(self.local_dir, si_node(self.local_dir, enabled=False))
        before = OmegaConf.to_container(trainer.config, resolve=True)
        with self.assertRaises(SelfImprovingStateError):
            SelfImprovingLoop(llm=script).initialize(trainer)
        self.assertEqual(OmegaConf.to_container(trainer.config, resolve=True), before)
        self.assertEqual(script.calls, [])
        self.assertFalse(os.path.exists(self.local_dir))

    def test_disabled_config_does_not_touch_the_algorithm_node(self):
        cfg = OmegaConf.create(trainer_config(self.local_dir, {"enabled": False}))
        snapshot = copy.deepcopy(OmegaConf.to_container(cfg.algorithm, resolve=True))
        with self.assertRaises(SelfImprovingStateError):
            SelfImprovingLoop(llm=ScriptedLLM([])).initialize(FakeTrainer(self.local_dir, cfg.self_improving))
        self.assertEqual(OmegaConf.to_container(cfg.algorithm, resolve=True), snapshot)

    def _fit_nodes(self):
        with open(RAY_TRAINER, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        fit = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fit")
        parents = {}
        for node in ast.walk(fit):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        def guard(node):
            cursor = parents.get(node)
            while cursor is not None:
                if isinstance(cursor, ast.If):
                    test = ast.dump(cursor.test)
                    if "_si" in test or "self_improving" in test:
                        return cursor
                cursor = parents.get(cursor)
            return None

        si_calls = []
        for node in ast.walk(fit):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                base = node.func.value
                if isinstance(base, ast.Attribute) and base.attr == "_si":
                    si_calls.append(node)
        constructions = [
            node
            for node in ast.walk(fit)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "SelfImprovingLoop"
        ]
        return source, tree, fit, si_calls, constructions, guard

    def test_trainer_hooks_are_all_gated_on_the_enabled_flag(self):
        _, tree, fit, si_calls, constructions, guard = self._fit_nodes()
        self.assertGreaterEqual(len(si_calls), 5)
        self.assertEqual(len(constructions), 1)
        for node in si_calls + constructions:
            self.assertIsNotNone(guard(node), ast.dump(node)[:200])
        gate = guard(constructions[0])
        test = ast.dump(gate.test)
        self.assertIn("self_improving", test)
        self.assertIn("enabled", test)
        # 没有任何 else 分支在关闭时改动 config / 启动 loop.
        self.assertIsNone(gate.orelse or None)
        assignments = [
            stmt
            for stmt in ast.walk(fit)
            if isinstance(stmt, ast.Assign)
            and any(getattr(t, "attr", "") == "_si" for t in stmt.targets)
            and not isinstance(stmt.value, ast.Constant)
        ]
        for stmt in assignments:
            self.assertIsNotNone(guard(stmt))

    def test_module_level_imports_stay_clean(self):
        source, tree, fit, _, constructions, _ = self._fit_nodes()
        offenders = []
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom) and "self_improving" in str(stmt.module or ""):
                offenders.append(stmt.lineno)
            if isinstance(stmt, ast.Import):
                offenders.extend(a.name for a in stmt.names if "self_improving" in a.name)
        self.assertEqual(offenders, [])
        imports = [n for n in ast.walk(fit) if isinstance(n, ast.ImportFrom) and "self_improving" in str(n.module or "")]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "verl.trainer.credit_assignment.self_improving.slow_loop")
        self.assertFalse([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and "SelfImproving" in n.name])
        self.assertFalse([n for n in ast.walk(tree) if isinstance(n, (ast.Assign,)) and any(getattr(t, "id", "").startswith("_SI") for t in n.targets)])

    def test_no_llm_call_and_no_metric_when_disabled(self):
        trainer = FakeTrainer(self.local_dir, si_node(self.local_dir, enabled=False))
        metrics = {"actor/pg_loss": 0.1, "critic/score/mean": 0.5}
        self.assertIsNone(trainer._si)
        if trainer._si is not None:  # pragma: no cover - thin hook 的语义
            trainer._si.on_step_end(batch=None, metrics=metrics, global_step=1)
        self.assertEqual([k for k in metrics if k.startswith("self_improving/")], [])
        self.assertEqual(os.path.isdir(self.local_dir), False)


if __name__ == "__main__":
    unittest.main()
