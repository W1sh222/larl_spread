# Copyright 2025 Spreadsheet-RL contributors
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

"""GiGPO -> 现有 vanilla token-level PPO 的连接回归 (CPU only).

GiGPO 官方实现 (d8a9b70 / f974dc5 / 20bd331) 不使用 step-level policy loss:
step advantage 广播到该 step 的有效 assistant token 后, 优化仍由 vanilla token-level
PPO 完成. 本文件把这条连接固化成可执行断言, 覆盖 8 项验证:

  D1 : 同一 trajectory 内不同 step 拿到不同 advantage
  D2 : 同一 step 的所有有效 assistant token 拿到相同 advantage
  D3 : tool observation / padding token 不进入 policy loss (advantage=0 且梯度=0)
  D4 : advantage tensor 的 shape / dtype / device 与现有 PPO 接口兼容
  D5 : GiGPO 与 GRPO 调用同一个 policy loss 函数
  D6 : actor / optimizer 侧不存在按 credit assignment 算法分支的代码
  D7 : anchor 无效时 episode advantage 仍然保留
  D8 : 关闭 step 项 (step_advantage_w=0) 时 GiGPO 的 advantage 与 GRPO 完全相同,
       即 GiGPO 只在 advantage tensor 上做增量, 不改 loss/optimizer

expected 数值全部按 GiGPO Eq.3/5/6/7/8 手算, 不引用被测实现的中间结果.
"""

import os
import unittest

import numpy as np
import torch

from verl.trainer.config.algorithm import AlgoConfig
from verl.trainer.ppo.core_algos import (
    POLICY_LOSS_REGISTRY,
    compute_credit_assignment_advantage,
    compute_grpo_outcome_advantage,
    compute_policy_loss_vanilla,
    get_policy_loss_fn,
)
from verl.workers.config import ActorConfig
from verl.workers.config.actor import PolicyLossConfig

# ---------------------------------------------------------------------------
# synthetic batch: 4 trajectory / 2 task, 每个 assistant decision = 1 step.
# 非 span 区间 (4-6, 10-12, 15-16 ...) 代表 tool observation 与 padding token.
# ---------------------------------------------------------------------------

BS, LEN = 4, 16
UIDS = np.array(["t0", "t0", "t1", "t1"], dtype=object)
TRAJ_REWARDS = [1.0, 0.0, 1.0, 0.0]
SPANS = [
    [[0, 4], [6, 10], [12, 15]],
    [[0, 4], [6, 10], [12, 15]],
    [[0, 5], [7, 11]],
    [[0, 5], [7, 11]],
]
# pre-action workbook fingerprint of each step (None = 状态未知 -> INVALID anchor).
ANCHORS = [
    ["S0", "SA", "SB"],
    ["S0", "SA2", "SC"],
    [None, "SD"],
    ["SE", "SD"],
]
GAMMA = 1.0
STEP_W = 1.0

# 手算 (gamma=1 且无中间 reward -> 每个 step 的 discounted return = trajectory reward):
#   Eq.3 episode 部分 (mode=mean_norm -> 组内减均值): A_E = [+0.5, -0.5, +0.5, -0.5]
#   Eq.6/7 step 分组 key = (uid, fingerprint):
#     t0::S0 = {1.0, 0.0} -> +/-0.5 ; t0::SA/SA2/SB/SC 均为 singleton -> 0
#     t1::SD = {1.0, 0.0} -> +/-0.5 ; t1::SE singleton -> 0 ; INVALID 不参与分组 -> 0
#   Eq.8 A_final(i,k) = A_E(i) + w * A_S(i,k)
EXPECTED_STEP_ADV = [
    [1.0, 0.5, 0.5],
    [-1.0, -0.5, -0.5],
    [0.5, 1.0],
    [-0.5, -1.0],
]


def _mask_from_spans():
    mask = torch.zeros(BS, LEN)
    for i, row in enumerate(SPANS):
        for start, end in row:
            mask[i, start:end] = 1.0
    return mask


def _object_array(rows):
    arr = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        arr[i] = row
    return arr


def _non_tensor_batch():
    transitions = [
        [{"pre": a, "post": f"P{k}", "status": "executed", "executed": True, "graph_valid": True}
         for k, a in enumerate(row)]
        for row in ANCHORS
    ]
    return {
        "ca_step_spans": _object_array(SPANS),
        "ca_step_observations": _object_array([["O%d" % k for k in range(len(r))] for r in SPANS]),
        "ca_step_transitions": _object_array(transitions),
        "ca_terminal_reason": _object_array(["natural"] * BS),
    }


def _token_level_rewards():
    # outcome reward 只落在该 trajectory 最后一个有效 token 上 (与 verl reward manager 一致).
    rewards = torch.zeros(BS, LEN)
    for i, row in enumerate(SPANS):
        rewards[i, row[-1][1] - 1] = TRAJ_REWARDS[i]
    return rewards


def _ca_config(step_w=STEP_W, anchor="state"):
    return AlgoConfig(
        adv_estimator="credit_assignment",
        credit_assignment={
            "method": "gigpo",
            "params": {
                "anchor": anchor,
                "gamma": GAMMA,
                "step_advantage_w": step_w,
                "mode": "mean_norm",
                "grouping": "exact",
            },
        },
    )


def _advantages(step_w=STEP_W, metrics_out=None):
    return compute_credit_assignment_advantage(
        token_level_rewards=_token_level_rewards(),
        response_mask=_mask_from_spans(),
        index=UIDS,
        config=_ca_config(step_w=step_w),
        non_tensor_batch=_non_tensor_batch(),
        batch={"response_mask": _mask_from_spans()},
        metrics_out=metrics_out if metrics_out is not None else {},
    )


def _actor_config():
    return ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        use_dynamic_bsz=True,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
        policy_loss=PolicyLossConfig(),
    )


class StepToTokenAdvantageTest(unittest.TestCase):
    """D1 / D2 / D3 / D7: step advantage 广播进 token-level tensor 的语义."""

    @classmethod
    def setUpClass(cls):
        cls.adv, cls.ret = _advantages()
        cls.mask = _mask_from_spans()

    def test_steps_within_trajectory_differ(self):
        """D1: 同一 trajectory 内不同 step 的 advantage 不相同."""
        for i, row in enumerate(SPANS):
            got = [float(self.adv[i, s:e][self.mask[i, s:e] > 0].max()) for s, e in row]
            self.assertEqual(got, EXPECTED_STEP_ADV[i], f"row {i} step-level advantage mismatch")
            self.assertGreater(len(set(got)), 1, f"row {i} 各 step advantage 未区分")

    def test_constant_within_step(self):
        """D2: 同一 step 的所有有效 assistant token 取同一 advantage."""
        for i, row in enumerate(SPANS):
            for k, (s, e) in enumerate(row):
                values = self.adv[i, s:e][self.mask[i, s:e] > 0]
                self.assertEqual(float(values.min()), float(values.max()))
                self.assertAlmostEqual(float(values[0]), EXPECTED_STEP_ADV[i][k], places=6)

    def test_non_assistant_tokens_excluded(self):
        """D3: tool observation 区间与 padding 位置的 advantage 恒为 0."""
        assistant = self.mask > 0
        self.assertTrue(torch.equal(self.adv[~assistant], torch.zeros_like(self.adv[~assistant])))
        # 逐行取 span 补集 = tool 响应区 + 尾部 padding (row0: 4-5/10-11/15; row2: 5-6/11-15)
        for i, row in enumerate(SPANS):
            gaps = [t for t in range(LEN) if not any(s <= t < e for s, e in row)]
            self.assertTrue(gaps, f"row {i} 没有非 assistant token, fixture 无效")
            for t in gaps:
                self.assertEqual(float(self.adv[i, t]), 0.0, f"row {i} token {t} 应被排除")

    def test_returns_equal_advantages(self):
        self.assertTrue(torch.equal(self.adv, self.ret))

    def test_invalid_anchor_keeps_episode_advantage(self):
        """D7: row2 step0 anchor=INVALID -> A_S=0, 但 A_E(+0.5) 仍在."""
        self.assertAlmostEqual(float(self.adv[2, 0]), 0.5, places=6)
        metrics = {}
        _advantages(metrics_out=metrics)
        self.assertAlmostEqual(metrics["credit_assignment/gigpo/anchor_invalid_frac"], 1.0 / 10.0, places=6)

    def test_tensor_interface_compatible_with_grpo(self):
        """D4: shape / dtype / device 与 GRPO 产出的 advantage tensor 一致."""
        grpo_adv, _ = compute_grpo_outcome_advantage(
            token_level_rewards=_token_level_rewards(),
            response_mask=self.mask,
            index=UIDS,
            norm_adv_by_std_in_grpo=False,
        )
        self.assertEqual(self.adv.shape, grpo_adv.shape)
        self.assertEqual(self.adv.dtype, grpo_adv.dtype)
        self.assertEqual(self.adv.device, grpo_adv.device)
        self.assertEqual(tuple(self.adv.shape), (BS, LEN))
        self.assertTrue(self.adv.is_floating_point())
        self.assertTrue(torch.isfinite(self.adv).all())


class SharedVanillaPpoPathTest(unittest.TestCase):
    """D5 / D6 / D8: GiGPO 与 GRPO 共用现有 vanilla PPO, 无算法分支."""

    def test_same_policy_loss_function(self):
        cfg = _actor_config()
        loss_mode = cfg.policy_loss.get("loss_mode", "vanilla")
        self.assertEqual(loss_mode, "vanilla")
        fn = get_policy_loss_fn(loss_mode)
        self.assertIs(fn, compute_policy_loss_vanilla)
        # registry 中不存在 GiGPO 专用 step objective
        for name in ("gigpo_joint", "gigpo_geomean"):
            self.assertNotIn(name, POLICY_LOSS_REGISTRY)

    def test_step_advantage_off_equals_grpo(self):
        """D8: step_advantage_w=0 时 GiGPO 的 advantage == GRPO(无 std 归一化)."""
        gigpo_adv, _ = _advantages(step_w=0.0)
        grpo_adv, _ = compute_grpo_outcome_advantage(
            token_level_rewards=_token_level_rewards(),
            response_mask=_mask_from_spans(),
            index=UIDS,
            norm_adv_by_std_in_grpo=False,
        )
        self.assertTrue(torch.allclose(gigpo_adv, grpo_adv, atol=1e-6))

    def test_identical_loss_for_identical_advantages(self):
        """同一 advantage tensor -> 同一 loss 数值, 与它由哪个 estimator 产生无关."""
        cfg = _actor_config()
        torch.manual_seed(0)
        old_log_prob = torch.randn(BS, LEN)
        log_prob_a = torch.randn(BS, LEN, requires_grad=True)

        gigpo_adv, _ = _advantages()
        loss_g, metrics_g = compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob_a,
            advantages=gigpo_adv,
            response_mask=_mask_from_spans().bool(),
            loss_agg_mode=cfg.loss_agg_mode,
            config=cfg,
        )
        # 把 GiGPO advantage 换成手工写入的同一 tensor, 结果必须逐位相同
        loss_r, metrics_r = compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob_a,
            advantages=gigpo_adv.clone(),
            response_mask=_mask_from_spans().bool(),
            loss_agg_mode=cfg.loss_agg_mode,
            config=cfg,
        )
        self.assertEqual(float(loss_g), float(loss_r))
        self.assertEqual(set(metrics_g), set(metrics_r))
        self.assertTrue(torch.isfinite(loss_g))

    def test_gradient_only_on_assistant_tokens(self):
        """D3(loss 侧): 被 mask 掉的 token 不产生 log-prob 梯度."""
        cfg = _actor_config()
        torch.manual_seed(1)
        gigpo_adv, _ = _advantages()
        log_prob = torch.zeros(BS, LEN, requires_grad=True)
        loss, _ = compute_policy_loss_vanilla(
            old_log_prob=torch.zeros(BS, LEN),
            log_prob=log_prob,
            advantages=gigpo_adv,
            response_mask=_mask_from_spans().bool(),
            loss_agg_mode=cfg.loss_agg_mode,
            config=cfg,
        )
        loss.backward()
        grad = log_prob.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertEqual(float(grad[~(_mask_from_spans() > 0)].abs().sum()), 0.0)
        self.assertGreater(float(grad[_mask_from_spans() > 0].abs().sum()), 0.0)
        # step 之间梯度符号由各自 advantage 决定 -> row0 step0 与 step2 的梯度不应相同
        self.assertNotAlmostEqual(float(grad[0, 1]), float(grad[0, 13]), places=6)

    def test_no_algorithm_branch_on_worker_side(self):
        """D6: workers / actor / loss 源码中不存在 gigpo / credit_assignment 分支."""
        import verl.workers.config as workers_config_pkg
        import verl.workers.utils.losses as losses_mod

        roots = [
            os.path.dirname(os.path.dirname(os.path.abspath(workers_config_pkg.__file__))),  # verl/workers
            os.path.dirname(os.path.abspath(losses_mod.__file__)),
        ]
        # 只区分“算法分支”: worker 侧不应知道 credit assignment 方法名。
        # (不限定 adv_estimator 等 verl 固有词, 避免跟 upstream 演化冲突)
        needle = ("gigpo", "credit_assignment", "step_advantage")
        checked = 0
        for root in roots:
            for dirpath, _dirnames, filenames in os.walk(root):
                if "__pycache__" in dirpath:
                    continue
                for name in filenames:
                    if not name.endswith(".py"):
                        continue
                    path = os.path.join(dirpath, name)
                    if os.path.abspath(root) == os.path.abspath(roots[1]) and os.path.basename(path) != "losses.py":
                        continue
                    with open(path) as fh:
                        text = fh.read()
                    checked += 1
                    for token in needle:
                        self.assertNotIn(token, text, f"{os.path.relpath(path, root)} 仍含 '{token}' 分支")
        # verl/workers 下共 69 个 .py, 确认扫描确实执行过而不是空转
        self.assertGreaterEqual(checked, 60)


class AdvantageDispatchTest(unittest.TestCase):
    """estimator 注册表: GiGPO 走 adv_estimator 入口, 不需要 policy loss 侧配合."""

    def test_estimator_entries_present(self):
        # 注意: 不要在这里读 ADV_ESTIMATOR_REGISTRY -- upstream 的
        # test_core_algos_on_cpu.TestRegisterAdvEst.setUp 会 clear/rebind 该模块属性且
        # tearDown 只 clear 不恢复, 目录级执行时它是脏的 (upstream 自身的用例隔离问题).
        # 因此改用装饰器源文本 + 可调用对象来固化注册关系.
        import verl.trainer.ppo.core_algos as core_algos
        from verl.trainer.ppo.core_algos import AdvantageEstimator

        with open(core_algos.__file__) as fh:
            src = fh.read()
        self.assertEqual(AdvantageEstimator.GRPO.value, "grpo")
        self.assertIn("@register_adv_est(AdvantageEstimator.GRPO)", src)
        self.assertIn('@register_adv_est("credit_assignment")', src)
        self.assertTrue(callable(core_algos.compute_grpo_outcome_advantage))
        self.assertTrue(callable(core_algos.compute_credit_assignment_advantage))
        # 被移除的 step objective 不得仍可被解析 (dangling config 会在启动时报 unknown loss mode)
        for name in ("gigpo_joint", "gigpo_geomean"):
            self.assertNotIn(name, POLICY_LOSS_REGISTRY)
            self.assertNotIn(f'@register_policy_loss("{name}")', src)
            with self.assertRaises(Exception):
                get_policy_loss_fn(name)

    def test_returns_are_token_level(self):
        adv, ret = _advantages()
        self.assertEqual(adv.dim(), 2)
        self.assertEqual(ret.dim(), 2)


if __name__ == "__main__":
    unittest.main()
