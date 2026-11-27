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

"""Training Context 采集 (CPU).

覆盖 명령서 §21.5: 合成 DataProto / raw 与 shaped reward 区分 / masked advantage 统计 /
代表性 trajectory 抽取 / window 聚合 / 缺 metadata / NaN-Inf 序列化 / context 预算 /
不持有 GPU tensor.
"""

import json
import math
import unittest

import numpy as np
import torch

from verl.trainer.credit_assignment.self_improving.config import ContextLimits
from verl.trainer.credit_assignment.self_improving.context import ContextCollector, jsonable


def obj_array(rows):
    arr = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        arr[i] = row
    return arr


class FakeBatch:
    """只用到 DataProto 的 .batch / .non_tensor_batch 两个视图."""

    def __init__(self, tensors=None, non_tensor=None):
        self.batch = dict(tensors or {})
        self.non_tensor_batch = dict(non_tensor or {})


def make_batch(bs=4, length=12, with_meta=True, scores=None, shaped=None, advantages=None):
    spans = [[[0, 4], [4, 8], [8, length]] for _ in range(bs)]

    def tile(values, size):
        return (values * ((size + len(values) - 1) // len(values)))[:size]

    scores = scores if scores is not None else tile([1.0, 0.0, 0.5, -0.2], bs)
    shaped = shaped if shaped is not None else tile([0.9, -0.3, 0.4, -0.1], bs)
    rewards = torch.zeros(bs, length)
    rewards[:, -1] = torch.tensor(shaped)
    tensors = {
        "response_mask": torch.ones(bs, length),
        "responses": torch.arange(bs * length, dtype=torch.long).reshape(bs, length),
        "prompts": torch.zeros(bs, 4, dtype=torch.long),
        "token_level_scores": torch.zeros(bs, length),
        "token_level_rewards": rewards,
    }
    for i, value in enumerate(scores):
        tensors["token_level_scores"][i, -1] = value
    if advantages is not None:
        tensors["advantages"] = advantages
    non_tensor = {
        "uid": obj_array([f"g{i // 2}" for i in range(bs)]),
        "acc": obj_array(tile([1.0, 0.0, 1.0, 0.0], bs)),
        "raw_prompt": obj_array([[{"role": "user", "content": f"task {i}"}] for i in range(bs)]),
    }
    if with_meta:
        non_tensor.update(
            {
                "ca_step_spans": obj_array(spans),
                "ca_step_observations": obj_array([["obs a", "obs b", "obs c"] for _ in range(bs)]),
                "ca_step_transitions": obj_array([[{"pre": "sha256:a", "post": "sha256:b"}] * 3 for _ in range(bs)]),
                "ca_terminal_reason": obj_array(tile(["final_answer", "max_turns"], bs)),
                "ca_terminal_trusted": obj_array([True, True, False, True][:bs] if bs <= 4 else tile([True, True, False, True], bs)),
                "ca_final_state": obj_array(["sha256:b"] * bs),
                "ca_state_stats": obj_array([{"parse_ms": 12.5, "unknown": 0} for _ in range(bs)]),
                "eval_workbook_fingerprint": obj_array(["sha256:eval"] * bs),
            }
        )
    return FakeBatch(tensors, non_tensor), spans


class ContextCollectorTest(unittest.TestCase):
    def setUp(self):
        self.collector = ContextCollector(ContextLimits(), tokenizer=None)
        self.collector.reset(1)

    def _observe(self, step=1, **kw):
        batch, spans = make_batch(**kw)
        metrics = {
            "actor/pg_loss": 0.25,
            "critic/score/mean": 0.3,
            "response_length/mean": 9.5,
            "credit_assignment/gigpo/anchor_count": 3.0,
            "credit_assignment/gigpo/step_adv_nonzero_frac": 0.5,
            "val-core/spreadsheet/acc/mean@4": 0.42,
            "self_improving/executed_skill_id": 1.0,
        }
        return self.collector.observe(batch, metrics, step, skill="gigpo", params={"gamma": 0.95}), spans

    def test_synthetic_batch_produces_step_summary(self):
        summary, spans = self._observe()
        self.assertEqual(summary["step"], 1)
        self.assertEqual(summary["skill"], "gigpo")
        self.assertEqual(summary["params"], {"gamma": 0.95})
        self.assertEqual(summary["num_trajectories"], 4)
        self.assertEqual(summary["num_groups"], 2.0)
        self.assertEqual(len(self.collector.steps), 1)
        self.assertEqual(self.collector.observed_steps(), 1)
        self.assertEqual(self.collector.window_skill_ids(), ["gigpo"])
        self.assertEqual(summary["steps_per_trajectory"]["mean"], 3.0)
        self.assertEqual(summary["terminal_reasons"], {"final_answer": 2, "max_turns": 2})
        self.assertEqual(summary["trusted_terminal_ratio"], 0.75)

    def test_raw_and_shaped_rewards_stay_distinct(self):
        summary, _ = self._observe(scores=[1.0, 0.0, 0.5, -0.2], shaped=[0.9, -0.3, 0.4, -0.1])
        self.assertAlmostEqual(summary["raw_reward"]["mean"], float(np.mean([1.0, 0.0, 0.5, -0.2])), places=6)
        self.assertAlmostEqual(summary["train_reward"]["mean"], float(np.mean([0.9, -0.3, 0.4, -0.1])), places=6)
        self.assertNotEqual(summary["raw_reward"]["mean"], summary["train_reward"]["mean"])
        self.assertAlmostEqual(summary["raw_reward"]["min"], -0.2, places=5)
        self.assertIn("negative", summary["raw_reward_hist"])
        self.assertIn("(0.75,1)", summary["raw_reward_hist"])

    def test_advantage_statistics_respect_the_mask(self):
        length = 12
        adv = torch.full((4, length), 99.0)
        adv[:, :10] = torch.tensor([1.0, -1.0, 2.0, 0.0]).unsqueeze(-1)
        batch, _ = make_batch(advantages=adv)
        mask = torch.zeros(4, length)
        mask[:, :10] = 1
        batch.batch["response_mask"] = mask
        summary = self.collector.observe(batch, {}, 1, skill="grpo", params={})
        self.assertAlmostEqual(summary["advantage"]["mean"], 0.5, places=6)
        self.assertAlmostEqual(summary["advantage"]["abs_mean"], 1.0, places=6)
        self.assertAlmostEqual(summary["advantage"]["zero_ratio"], 0.25, places=6)
        self.assertEqual(summary["trajectory_length"]["mean"], 10.0)

    def test_representative_trajectories_are_deterministic(self):
        first, _ = self._observe()
        self.collector.reset(1)
        second, _ = self._observe()
        rows_a = [(s["kind"], s["row"]) for s in self.collector.samples]
        kinds = {k for k, _ in rows_a}
        self.assertTrue({"success", "failed", "truncated"} <= kinds, rows_a)
        self.assertEqual(rows_a, [(s["kind"], s["row"]) for s in self.collector.samples])
        for sample in self.collector.samples:
            self.assertEqual(sample["step"], 1)
            self.assertIn("task_instruction", sample)
            self.assertTrue(sample["has_eval_fingerprint"])
            self.assertEqual(len(sample["steps"]), 3)
            self.assertEqual(sample["steps"][0]["observation"], "obs a")

    def test_sample_budget_is_a_window_budget(self):
        collector = ContextCollector(ContextLimits(max_trajectory_samples=3, max_steps_per_trajectory=2), tokenizer=None)
        collector.reset(1)
        for step in (1, 2, 3):
            collector.observe(make_batch()[0], {}, step, skill="grpo", params={})
        self.assertLessEqual(len(collector.samples), 3)
        for sample in collector.samples:
            self.assertLessEqual(sum(1 for s in sample["steps"] if "index" in s), 2)
            self.assertTrue(any("steps_omitted" in s for s in sample["steps"]))

    def test_window_aggregation(self):
        self._observe(step=1, scores=[1.0, 1.0, 1.0, 1.0], shaped=[1.0, 1.0, 1.0, 1.0])
        self._observe(step=2, scores=[0.0, 0.0, 0.0, 0.0], shaped=[0.5, 0.5, 0.5, 0.5])
        self._observe(step=3, scores=[0.5, 0.5, 0.5, 0.5], shaped=[0.25, 0.25, 0.25, 0.25])
        agg = self.collector.aggregate()
        self.assertEqual(agg["num_steps"], 3)
        self.assertEqual(agg["step_range"], [1, 3])
        self.assertEqual(agg["skills"], ["gigpo"])
        self.assertAlmostEqual(agg["raw_reward"]["mean"], 0.5, places=6)
        self.assertAlmostEqual(agg["train_reward"]["mean"], float(np.mean([1.0, 0.5, 0.25])), places=6)
        self.assertEqual(agg["terminal_reasons"]["max_turns"], 6)
        self.assertAlmostEqual(agg["success_rate"], float(np.mean([0.5, 0.5, 0.5])), places=6)
        self.assertAlmostEqual(agg["credit_diagnostics"]["credit_assignment/gigpo/anchor_count"], 3.0, places=6)
        self.assertAlmostEqual(agg["training_metrics"]["actor/pg_loss"], 0.25, places=6)
        self.assertNotIn("self_improving/executed_skill_id", agg["training_metrics"])
        self.assertEqual(self.collector.window_start, 1)

    def test_empty_window_aggregate(self):
        self.assertEqual(self.collector.aggregate(), {"num_steps": 0})

    def test_missing_metadata_is_recorded_as_missing_not_zero(self):
        summary, _ = self._observe(with_meta=False)
        health = summary["metadata_health"]
        self.assertIsNone(health["step_transitions"])
        self.assertIsNone(health["eval_workbook_fingerprint"])
        self.assertFalse(health["structured_state_requested"])
        self.assertIsNone(summary["steps_per_trajectory"].get("mean"))
        self.assertIsNone(summary["trusted_terminal_ratio"])
        agg = self.collector.aggregate()
        self.assertFalse(agg["metadata_health"]["eval_fingerprint_any"])
        self.assertFalse(agg["metadata_health"]["structured_state_any"])
        self.assertIsNone(agg["steps_per_trajectory"].get("mean"))

    def test_nan_and_inf_never_reach_the_prompt(self):
        collector = ContextCollector(ContextLimits(), tokenizer=None)
        collector.reset(1)
        batch, _ = make_batch(scores=[float("nan"), math.inf, 0.5, 1.0], shaped=[float("nan"), 1.0, 2.0, 3.0])
        adv = torch.ones(4, 12)
        adv[0, 0] = float("nan")
        adv[1, 5] = math.inf
        batch.batch["advantages"] = adv
        collector.observe(batch, {"actor/pg_loss": float("nan"), "credit_assignment/x": float("inf")}, 1, skill="grpo", params={})
        step = collector.steps[-1]
        # 非有限 reward 被排除而不是当成 0: mean([0.5, 1.0]) = 0.75
        self.assertAlmostEqual(step["raw_reward"]["mean"], 0.75, places=6)
        self.assertIsNone(step["advantage"]["mean"])
        self.assertNotIn("actor/pg_loss", step["training_metrics"])
        self.assertNotIn("credit_assignment/x", step["credit_diagnostics"])
        payload = collector.build_context(global_step=1, skill="grpo", params={}).to_dict()
        text = json.dumps(payload, allow_nan=False)
        self.assertIn("null", text)

    def test_context_shape_and_budget(self):
        for step in range(1, 4):
            self._observe(step=step)
        context = self.collector.build_context(
            global_step=3,
            skill="gigpo",
            params={"gamma": 0.9},
            latest_validation={"step": 3, "metrics": {"val-core/x/acc/mean@4": 0.4}},
            previous_evaluation={"outcome": "positive"},
            extra_notes={"interval_steps": 20},
        )
        payload = context.to_dict()
        for key in (
            "global_step",
            "window_start_step",
            "window_end_step",
            "active_skill",
            "task_summary",
            "trajectory_statistics",
            "trajectory_samples",
            "training_metrics",
            "credit_diagnostics",
            "metadata_health",
            "window_skills",
            "latest_validation",
            "previous_evaluation",
            "notes",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["window_end_step"], 3)
        self.assertEqual(payload["active_skill"], {"method": "gigpo", "params": {"gamma": 0.9}})
        self.assertEqual(payload["notes"], {"interval_steps": 20})
        self.assertEqual(payload["window_skills"], ["gigpo"])
        json.dumps(payload, allow_nan=False)

    def test_context_respects_char_budget(self):
        limits = ContextLimits(max_context_chars=4000, max_trajectory_samples=8, max_text_chars_per_field=20000)
        collector = ContextCollector(limits, tokenizer=None)
        collector.reset(1)
        for step in range(1, 6):
            batch, _ = make_batch(bs=8)
            batch.non_tensor_batch["raw_prompt"] = obj_array(
                [[{"role": "user", "content": ("长任务描述 " * 400)}] for _ in range(8)]
            )
            batch.non_tensor_batch["ca_step_observations"] = obj_array(
                [["x" * 4000, "y" * 4000, "z" * 4000] for _ in range(8)]
            )
            collector.observe(batch, {"credit_assignment/m%d" % step: float(step)}, step, skill="gigpo", params={})
        payload = collector.build_context(global_step=5, skill="gigpo", params={}).to_dict()
        self.assertLessEqual(len(json.dumps(payload, ensure_ascii=False)), 4000 + 200)
        self.assertTrue(json.dumps(payload, allow_nan=False))

    def test_no_gpu_tensor_is_retained(self):
        self._observe()
        self._observe(step=2)
        payload = self.collector.build_context(global_step=2, skill="gigpo", params={"gamma": 0.9}).to_dict()

        def tensors_inside(node, path="root"):
            found = []
            if isinstance(node, torch.Tensor):
                found.append(path)
            elif isinstance(node, np.ndarray):
                found.append(path)
            elif isinstance(node, dict):
                for k, v in node.items():
                    found += tensors_inside(v, f"{path}.{k}")
            elif isinstance(node, (list, tuple)):
                for i, v in enumerate(node):
                    found += tensors_inside(v, f"{path}[{i}]")
            return found

        self.assertEqual(tensors_inside(self.collector.steps), [])
        self.assertEqual(tensors_inside(self.collector.samples), [])
        self.assertEqual(tensors_inside(payload), [])
        self.assertTrue(all(not hasattr(v, "detach") for v in self.collector.steps[-1].values()))

    def test_reset_starts_a_clean_window(self):
        self._observe(step=1)
        self._observe(step=2)
        self.assertEqual(self.collector.observed_steps(), 2)
        self.collector.reset(3)
        self.assertEqual(self.collector.observed_steps(), 0)
        self.assertEqual(self.collector.samples, [])
        self.assertEqual(self.collector.window_start, 3)
        self._observe(step=3)
        self.assertEqual(self.collector.aggregate()["step_range"], [3, 3])

    def test_jsonable_converts_numpy_and_containers(self):
        self.assertEqual(jsonable(np.float32(1.5)), 1.5)
        self.assertIsNone(jsonable(float("nan")))
        self.assertIsNone(jsonable(np.inf))
        self.assertEqual(jsonable(np.arange(3)), [0, 1, 2])
        self.assertEqual(jsonable({"a": np.int64(2), "b": (1, 2)}), {"a": 2, "b": [1, 2]})
        self.assertEqual(jsonable(torch.zeros(2)), [0.0, 0.0])
        self.assertEqual(jsonable({"x"}), ["x"])
        self.assertEqual(jsonable(None), None)


if __name__ == "__main__":
    unittest.main()
