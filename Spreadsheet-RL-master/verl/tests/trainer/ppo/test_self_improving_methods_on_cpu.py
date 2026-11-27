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

"""GRPO as a credit assignment method + RewardFlow regression (CPU).

覆盖 명령서 §21.2 (GRPO 与 vanilla 数值等价 / mask 与 dtype / singleton / 参数校验 /
注册) 与 §21.3 (RewardFlow 既有行为不变: graph credit, 评估 fingerprint 不符,
不可信 terminal, 缺评估 metadata, 非法 transition 结构, 现有诊断量).
"""

import hashlib
import unittest

import numpy as np
import torch

from verl.trainer.credit_assignment import available_methods, compute, run_on_inputs
from verl.trainer.credit_assignment.interface import CreditAssignmentInput
from verl.trainer.credit_assignment.methods.grpo.config import GRPOConfig
from verl.trainer.credit_assignment.registry import get_credit_assignment_class


def sid(name: str) -> str:
    """rollout 侧 state 身份是 sha256: 前缀指纹; 非该格式一律视为 unknown."""
    return "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()


def mask_from_spans(spans, length, dtype=torch.float32):
    mask = torch.zeros(len(spans), length, dtype=dtype)
    for i, row in enumerate(spans):
        for start, end in row:
            mask[i, min(start, length) : min(end, length)] = 1
    return mask


def scatter_last(rewards, mask):
    """vanilla GRPO 的输入形态: trajectory 分数放在最后一个有效 token."""
    out = torch.zeros(mask.shape, dtype=torch.float64)
    rewards_t = torch.as_tensor(np.asarray(rewards), dtype=torch.float64)
    lengths = (mask != 0).sum(dim=-1)
    pos = torch.where(lengths > 0, lengths - 1, torch.zeros_like(lengths)).to(torch.long)
    out.scatter_(1, pos.unsqueeze(-1), rewards_t.unsqueeze(-1))
    return out


class GrpoEquivalenceTest(unittest.TestCase):
    def setUp(self):
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        self.vanilla = compute_grpo_outcome_advantage
        self.spans = [[[0, 4], [4, 10]], [[0, 6]], [[0, 3], [3, 9], [9, 14]], [[0, 5], [5, 8]]]
        self.length = 16
        self.uids = np.array(["t0", "t0", "t1", "t1"], dtype=object)

    def _run(self, rewards, mask, norm=True, epsilon=1e-6):
        inputs = CreditAssignmentInput(
            episode_group_ids=self.uids,
            trajectory_rewards=np.asarray(rewards),
            assistant_mask=mask,
            step_spans=self.spans,
            step_observations=[["o"] * len(r) for r in self.spans],
            extra={},
        )
        out = run_on_inputs(
            "grpo", inputs, {"norm_adv_by_std_in_grpo": norm, "epsilon": epsilon}
        )
        return out, inputs

    def _vanilla(self, rewards, mask, norm=True, epsilon=1e-6):
        return self.vanilla(
            token_level_rewards=scatter_last(rewards, mask.float()),
            response_mask=mask,
            index=self.uids,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm,
        )

    def test_exact_equal_for_float_rewards(self):
        rewards = [1.0, 0.0, 0.5, 0.0]
        mask = mask_from_spans(self.spans, self.length)
        (out, _), (adv, ret) = self._run(rewards, mask), self._vanilla(rewards, mask)
        self.assertTrue(torch.equal(out.advantages, adv))
        self.assertTrue(torch.equal(out.returns, ret))

    def test_exact_equal_for_both_norm_modes_and_mask_dtypes(self):
        rewards = [2.0, -1.0, 0.25, 0.0]
        for norm in (True, False):
            for dtype in (torch.float32, torch.bool, torch.int64, torch.long):
                mask = mask_from_spans(self.spans, self.length, dtype=dtype)
                (out, _), (adv, _) = self._run(rewards, mask, norm=norm), self._vanilla(rewards, mask, norm=norm)
                self.assertTrue(
                    torch.equal(out.advantages.float(), adv.float()),
                    f"mismatch for dtype={dtype} norm={norm}",
                )

    def test_integer_rewards_are_not_truncated(self):
        # bool mask + int rewards: 若 dtype 取自 zeros_like(mask) 会被截成 0/1.
        rewards = np.array([3, 7, 1, 2], dtype=np.int64)
        mask = mask_from_spans(self.spans, self.length, dtype=torch.bool)
        (out, _), (adv, _) = self._run(rewards, mask), self._vanilla(rewards, mask)
        self.assertTrue(torch.equal(out.advantages.float(), adv.float()))
        # 3 / 7 被截成 1 / 1 时组内无差异 -> advantage 应为 0; 这里仍是关于 0 对称的 +/-1/sqrt(2)
        self.assertAlmostEqual(float(out.advantages[1, 0:6].mean()), 2**-0.5, places=5)
        self.assertAlmostEqual(float(out.advantages[0, 0:10].mean()), -(2**-0.5), places=5)

    def test_all_zero_rewards_and_singleton_groups(self):
        uids = np.array(["a", "b", "c", "d"], dtype=object)
        inputs = CreditAssignmentInput(
            episode_group_ids=uids,
            trajectory_rewards=np.zeros(4),
            assistant_mask=mask_from_spans(self.spans, self.length),
            step_spans=self.spans,
            step_observations=[["o"] * len(r) for r in self.spans],
        )
        out = run_on_inputs("grpo", inputs, {})
        self.assertTrue(torch.equal(out.advantages, torch.zeros_like(out.advantages)))
        self.assertEqual(out.metrics["singleton_group_ratio"], 1.0)
        self.assertEqual(out.metrics["num_groups"], 4.0)
        self.assertEqual(out.metrics["zero_adv_ratio"], 1.0)

    def test_fully_masked_row_keeps_reward_sum_and_zero_advantage(self):
        spans = [[[0, 0]], [[0, 4]], [[0, 4]], [[0, 4]]]
        mask = torch.zeros(4, self.length)
        mask[1:, :4] = 1
        inputs = CreditAssignmentInput(
            episode_group_ids=np.array(["t0", "t0", "t1", "t1"], dtype=object),
            trajectory_rewards=np.array([1.0, 1.0, 0.0, 0.0]),
            assistant_mask=mask,
            step_spans=spans,
            step_observations=[[], ["o"], ["o"], ["o"]],
        )
        out = run_on_inputs("grpo", inputs, {})
        # 空行的 advantage 必须严格为 0 (mask 全 0), 但 reward 不应丢失 -> 与 vanilla 一致
        vanilla = self.vanilla(
            token_level_rewards=scatter_last([1.0, 1.0, 0.0, 0.0], mask),
            response_mask=mask,
            index=np.array(["t0", "t0", "t1", "t1"], dtype=object),
        )
        self.assertTrue(torch.equal(out.advantages, vanilla[0]))
        self.assertTrue(torch.equal(out.advantages[0], torch.zeros(self.length)))

    def test_group_centering_is_within_group(self):
        rewards = [1.0, 1.0, 0.0, 0.0]
        mask = mask_from_spans(self.spans, self.length)
        (out, _), (adv, _) = self._run(rewards, mask), self._vanilla(rewards, mask)
        self.assertTrue(torch.equal(out.advantages, adv))
        per_row = out.advantages.detach().float().sum(dim=-1)
        self.assertAlmostEqual(float(per_row[0]), float(per_row[1]), places=5)
        self.assertAlmostEqual(float(per_row[0]), -float(per_row[2]), places=5)

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            GRPOConfig.from_params({"unknown": 1})
        with self.assertRaises(ValueError):
            GRPOConfig.from_params({"norm_adv_by_std_in_grpo": "yes"})
        with self.assertRaises(ValueError):
            GRPOConfig.from_params({"epsilon": 0.5})
        with self.assertRaises(ValueError):
            GRPOConfig.from_params({"epsilon": 0.0})
        cfg = GRPOConfig.from_params({"norm_adv_by_std_in_grpo": False, "epsilon": 1e-4})
        self.assertFalse(cfg.norm_adv_by_std_in_grpo)
        self.assertEqual(cfg.epsilon, 1e-4)

    def test_registry_registration_and_no_metadata_requirement(self):
        self.assertIn("grpo", available_methods())
        cls = get_credit_assignment_class("grpo")
        self.assertEqual(cls.required_metadata({}), frozenset())
        self.assertEqual(cls.requires, frozenset())

    def test_diagnostics_and_estimator_wiring(self):
        rewards = [1.0, 0.0, 1.0, 0.0]
        mask = mask_from_spans(self.spans, self.length)
        out = compute(
            method="grpo",
            token_level_rewards=mask * torch.tensor(rewards).unsqueeze(-1),
            response_mask=mask,
            index=self.uids,
            non_tensor_batch={},
            params={},
        )
        self.assertTrue(all(k in out.metrics for k in ("adv_mean", "adv_std", "zero_adv_ratio", "num_groups")))
        for value in out.metrics.values():
            self.assertTrue(np.isfinite(value))

        from verl.trainer.config.algorithm import AlgoConfig
        from verl.trainer.ppo.core_algos import compute_credit_assignment_advantage

        metrics_out: dict = {}
        config = AlgoConfig(adv_estimator="credit_assignment", credit_assignment={"method": "grpo", "params": {}})
        adv, ret = compute_credit_assignment_advantage(
            token_level_rewards=mask * torch.tensor(rewards).unsqueeze(-1),
            response_mask=mask,
            index=self.uids,
            config=config,
            non_tensor_batch={},
            batch={"token_level_rewards": mask * torch.tensor(rewards).unsqueeze(-1)},
            metrics_out=metrics_out,
        )
        self.assertTrue(torch.equal(adv, ret))
        self.assertTrue(metrics_out)
        self.assertTrue(all(k.startswith("credit_assignment/grpo/") for k in metrics_out))


class RewardFlowRegressionTest(unittest.TestCase):
    """RewardFlow 实现未被改动: 用真实 metadata 走 adapter, 校验锚点条件与诊断量."""

    def setUp(self):
        self.spans = [[[0, 4], [4, 8]], [[0, 4], [4, 8]]]
        self.length = 8
        self.mask = mask_from_spans(self.spans, self.length)
        self.s = [sid("S0"), sid("S1"), sid("S2")]

    def _trans(self, pre, post):
        return {"pre": pre, "post": post, "executed": True, "status": "executed", "graph_valid": True}

    def _non_tensor(self, finals, eval_fps, trusted, transitions=None):
        def obj(rows):
            arr = np.empty(len(rows), dtype=object)
            for i, r in enumerate(rows):
                arr[i] = r
            return arr

        node = {
            "ca_step_spans": obj(self.spans),
            "ca_step_observations": obj([["o", "o"], ["o", "o"]]),
            "ca_final_state": obj(finals),
            "ca_terminal_trusted": obj(trusted),
            "eval_workbook_fingerprint": obj(eval_fps),
            "ca_terminal_reason": obj(["final_answer", "final_answer"]),
        }
        if transitions is not None:
            node["ca_step_transitions"] = obj(transitions)
        return node

    def _run(self, raw, shaped, node, params=None):
        rewards = torch.zeros(2, self.length)
        rewards[:, -1] = torch.tensor(shaped)
        scores = torch.zeros(2, self.length)
        scores[:, -1] = torch.tensor(raw)
        return compute(
            method="rewardflow",
            token_level_rewards=rewards,
            response_mask=self.mask,
            index=np.array(["t0", "t0"], dtype=object),
            non_tensor_batch=node,
            params=params or {},
            raw_token_level_scores=scores,
        )

    def _happy_transitions(self):
        return [
            [self._trans(self.s[0], self.s[1]), self._trans(self.s[1], self.s[2])],
            [self._trans(self.s[0], self.s[1]), self._trans(self.s[1], sid("SX"))],
        ]

    def test_basic_graph_credit(self):
        node = self._non_tensor([self.s[2], sid("SX")], [self.s[2], sid("SX")], [True, True], self._happy_transitions())
        out = self._run([1.0, 0.0], [1.0, 0.0], node)
        self.assertEqual(out.metrics["anchor_count"], 1.0)
        self.assertEqual(out.metrics["edge_count"], 3.0)
        self.assertEqual(out.metrics["nonzero_action_adv_frac"], 0.5)
        # 同一 trajectory 内两个 step 的 credit 不同 (potential 差), 不是纯 trajectory 广播
        self.assertFalse(torch.equal(out.advantages[0, 0:4], out.advantages[0, 4:8]))
        self.assertAllClose(out.advantages[0, 0:4], torch.full((4,), 0.5), atol=1e-6)
        self.assertAllClose(out.advantages[0, 4:8], torch.full((4,), 1.0), atol=1e-6)
        # 失败 trajectory 靠近 terminal 的 step 得到更负的 credit
        self.assertAllClose(out.advantages[1, 0:4], torch.full((4,), -0.5), atol=1e-6)
        self.assertAllClose(out.advantages[1, 4:8], torch.full((4,), -1.0), atol=1e-6)

    def assertAllClose(self, actual, expected, atol=1e-6):
        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=atol), f"{actual} vs {expected}")

    def test_eval_fingerprint_mismatch_excludes_anchor(self):
        transitions = self._happy_transitions()
        node = self._non_tensor([self.s[2], sid("SX")], [sid("OTHER"), sid("SX")], [True, True], transitions)
        out = self._run([1.0, 0.0], [1.0, 0.0], node)
        self.assertEqual(out.metrics["anchor_count"], 0.0)
        self.assertEqual(out.metrics["anchors_eval_mismatch"], 1.0)
        # 无锚点 -> 无 potential -> action 分量全 0, 只剩 episode 分量 (每行常数)
        self.assertAllClose(out.advantages[0], torch.full((self.length,), 0.5))
        self.assertTrue(torch.equal(out.advantages[0, 0:4], out.advantages[0, 4:8]))
        self.assertEqual(out.metrics["nonzero_action_adv_frac"], 0.0)

    def test_untrusted_terminal_excludes_anchor(self):
        node = self._non_tensor([self.s[2], sid("SX")], [self.s[2], sid("SX")], [False, True], self._happy_transitions())
        out = self._run([1.0, 0.0], [1.0, 0.0], node)
        self.assertEqual(out.metrics["anchor_count"], 0.0)
        self.assertEqual(out.metrics["anchors_untrusted"], 1.0)

    def test_missing_evaluation_metadata_row_is_fail_closed_only(self):
        node = self._non_tensor([self.s[2], sid("SX")], [None, sid("SX")], [True, True], self._happy_transitions())
        out = self._run([1.0, 0.0], [1.0, 0.0], node)
        self.assertEqual(out.metrics["anchor_count"], 0.0)
        self.assertEqual(out.metrics["anchors_eval_missing"], 1.0)
        self.assertEqual(out.metrics["anchors_eval_mismatch"], 0.0)

    def test_unknown_final_state_excludes_anchor(self):
        node = self._non_tensor(["not-a-fingerprint", sid("SX")], ["not-a-fingerprint", sid("SX")], [True, True], self._happy_transitions())
        out = self._run([1.0, 0.0], [1.0, 0.0], node)
        self.assertEqual(out.metrics["anchor_count"], 0.0)
        self.assertEqual(out.metrics["anchors_missing"], 1.0)

    def test_missing_transition_row_raises(self):
        node = self._non_tensor([self.s[2], self.s[2]], [self.s[2], self.s[2]], [True, True])
        with self.assertRaises(ValueError):
            self._run([1.0, 1.0], [1.0, 1.0], node)

    def test_misaligned_transition_row_raises(self):
        bad = [[self._trans(self.s[0], self.s[1])], [self._trans(self.s[0], self.s[1]), self._trans(self.s[1], self.s[2])]]
        node = self._non_tensor([self.s[2], self.s[2]], [self.s[2], self.s[2]], [True, True], bad)
        with self.assertRaises(ValueError):
            self._run([1.0, 1.0], [1.0, 1.0], node)

    def test_missing_raw_outcome_raises_instead_of_using_shaped_reward(self):
        node = self._non_tensor([self.s[2], sid("SX")], [self.s[2], sid("SX")], [True, True], self._happy_transitions())
        rewards = torch.zeros(2, self.length)
        rewards[:, -1] = torch.tensor([1.0, 0.0])
        with self.assertRaises(ValueError):
            compute(
                method="rewardflow",
                token_level_rewards=rewards,
                response_mask=self.mask,
                index=np.array(["t0", "t0"], dtype=object),
                non_tensor_batch=node,
                params={},
            )

    def test_existing_diagnostics_are_present_and_finite(self):
        node = self._non_tensor([self.s[2], sid("SX")], [self.s[2], sid("SX")], [True, True], self._happy_transitions())
        out = self._run([1.0, 0.0], [1.0, 0.0], node)
        for key in ("graph_count", "node_count", "edge_count", "self_loop_count", "anchor_count", "zero_action_step_frac"):
            self.assertIn(key, out.metrics)
            self.assertTrue(np.isfinite(out.metrics[key]), key)
        self.assertEqual(out.advantages.shape, (2, self.length))
        self.assertTrue(torch.equal(out.advantages, out.returns))

    def test_required_metadata_declares_soft_eval_fingerprint(self):
        cls = get_credit_assignment_class("rewardflow")
        req = cls.required_metadata({})
        self.assertIn("step_transitions", req)
        self.assertIn("terminal_trusted", req)
        self.assertIn("eval_fingerprint", req)


if __name__ == "__main__":
    unittest.main()
