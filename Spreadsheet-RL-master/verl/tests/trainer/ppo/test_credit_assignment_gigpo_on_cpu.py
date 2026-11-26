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

"""GiGPO credit assignment (Eq.3/5/6/7/8) - CPU math tests.

覆盖确认事项 / 修正事项:
  4  : state anchor 默认 = pre-action full workbook fingerprint
  5  : anchor 种类只影响 Eq.7 分组, 与 policy objective (现有 token-level vanilla PPO) 无关
  B  : 截断 step 是否进入 Eq.7 baseline / 其他 step 的 A_S 如何变化 / A_E 处理 / 收尾区分
  C  : INVALID state != invalid action (A_S=0 时 episode 部分仍然保留)
  D  : GiGPOConfig <-> YAML parser schema 一致
  E  : reward-equivalent 状态不等于同一环境状态
  F3/F4/F5 : 上述对应的测试补充

golden 数值 (observation anchor) 来自修改前的基线抓取, 用于证明既有 Credit
Assignment 计算与 GRPO/GDPO/legacy_token 路径未被静默改动.
step-loss 侧 (gigpo_joint / gigpo_geomean) 已移除, 只保留 ConfigSchemaTest 中的
dangling guard.
"""

import hashlib
import os
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.credit_assignment import available_methods, compute, run_on_inputs
from verl.trainer.credit_assignment.interface import INITIAL_STATE_MARKER, CreditAssignmentInput
from verl.trainer.credit_assignment.methods.gigpo.config import GiGPOConfig
from verl.trainer.credit_assignment.anchor_fingerprint import INVALID_FINGERPRINT
from verl.trainer.credit_assignment.state_meta import (
    TRUNC_PARTIAL_ACTION,
    TRUNC_UNEXECUTED,
    build_step_meta,
    classify_truncation,
)

# ---------------------------------------------------------------------------
# golden fixture: 修改前基线 (observation anchor), 4 条 trajectory / 2 个 task
# ---------------------------------------------------------------------------

GOLDEN_B, GOLDEN_L = 4, 24
GOLDEN_SPANS = [
    [[0, 4], [6, 12], [14, 20]],
    [[0, 6], [8, 16]],
    [[0, 3], [5, 11], [13, 17], [19, 23]],
    [[0, 5], [7, 15]],
]
GOLDEN_OBS = [
    [INITIAL_STATE_MARKER, "OBS_A", "OBS_B"],
    [INITIAL_STATE_MARKER, "OBS_A"],
    [INITIAL_STATE_MARKER, "OBS_C", "OBS_B", ""],
    [INITIAL_STATE_MARKER, "OBS_C"],
]
GOLDEN_REWARDS = [1.0, 0.0, 1.0, 0.0]
GOLDEN_UIDS = np.array(["t0", "t0", "t1", "t1"], dtype=object)

GOLDENS = {
    "mean_norm": {
        "sum": -0.6123752594,
        "abs_sum": 50.9095039368,
        "nonzero": 60,
        "sha": "0a90c838394e5cd8",
        "per_row": [12.65500069, -13.50750160, 12.49356270, -12.25343800],
    },
    "mean_std_norm": {
        "sum": -1.4142115116,
        "abs_sum": 74.9532165527,
        "nonzero": 60,
        "sha": "a6001305fed3b0e6",
        "per_row": [18.38474846, -19.79895973, 18.38474846, -18.38474846],
    },
}


def _mask_from_spans(spans, bs, length):
    mask = torch.zeros(bs, length)
    for i, row in enumerate(spans):
        for start, end in row:
            mask[i, min(start, length) : min(end, length)] = 1.0
    return mask


def _golden_inputs():
    return CreditAssignmentInput(
        episode_group_ids=GOLDEN_UIDS,
        trajectory_rewards=np.array(GOLDEN_REWARDS),
        assistant_mask=_mask_from_spans(GOLDEN_SPANS, GOLDEN_B, GOLDEN_L),
        step_spans=GOLDEN_SPANS,
        step_observations=GOLDEN_OBS,
        step_rewards=None,
        extra={},
    )


def _inputs(spans, observations, rewards, uids, transitions=None, initial_states=None, length=None, bs=None):
    bs = bs or len(spans)
    length = length or max(max(end for _, end in row) for row in spans)
    mask = _mask_from_spans(spans, bs, length)
    extra = {}
    if transitions is not None:
        extra["step_transitions"] = transitions
    if initial_states is not None:
        extra["initial_states"] = initial_states
    return CreditAssignmentInput(
        episode_group_ids=np.array(uids, dtype=object),
        trajectory_rewards=np.array(rewards, dtype=np.float64),
        assistant_mask=mask,
        step_spans=spans,
        step_observations=observations,
        step_rewards=None,
        extra=extra,
    )


def trans(pre, post="S_NEXT", status="executed", executed=True, graph_valid=True, reason=None):
    row = {"pre": pre, "post": post, "status": status, "executed": executed, "graph_valid": graph_valid}
    if reason is not None:
        row["reason"] = reason
    return row


def span_values(out, spans, length):
    """按 span 取回每个 step 的 token advantage (span 内应为常数)."""
    adv = out.advantages
    values = []
    for i, row in enumerate(spans):
        values.append([float(adv[i, min(s, length) : min(e, length)].max()) for s, e in row])
    return values


class BackwardCompatibilityTest(unittest.TestCase):
    """确认事项 3: 既有 observation-anchor 计算数值不变."""

    def test_registered_methods_present(self):
        self.assertIn("gigpo", available_methods())

    def test_golden_advantages_observation_anchor(self):
        for mode, golden in GOLDENS.items():
            with self.subTest(mode=mode):
                out = run_on_inputs(
                    "gigpo", _golden_inputs(), {"mode": mode, "gamma": 0.95, "step_advantage_w": 1.0,
                                                 "anchor": "observation"}
                )
                adv = out.advantages
                sha = hashlib.sha256(adv.detach().cpu().numpy().tobytes()).hexdigest()[:16]
                self.assertEqual(sha, golden["sha"], f"{mode}: advantage tensor changed")
                self.assertAlmostEqual(float(adv.sum()), golden["sum"], places=6)
                self.assertAlmostEqual(float(adv.abs().sum()), golden["abs_sum"], places=6)
                self.assertEqual(int((adv != 0).sum()), golden["nonzero"])
                per_row = [round(float(adv[i].sum()), 8) for i in range(GOLDEN_B)]
                for got, want in zip(per_row, golden["per_row"], strict=True):
                    self.assertAlmostEqual(got, want, places=5)

    def test_state_anchor_is_the_new_default_and_requires_transitions(self):
        """确认事项 4: 默认 anchor=state; 缺 transitions 时明确报错而非静默退化."""
        with self.assertRaisesRegex(ValueError, "ca_step_transitions"):
            run_on_inputs("gigpo", _golden_inputs(), {"mode": "mean_norm", "gamma": 0.95})
        # 显式 observation 才回到旧行为
        out = run_on_inputs("gigpo", _golden_inputs(), {"anchor": "observation"})
        self.assertAlmostEqual(float(out.advantages.sum()), GOLDENS["mean_norm"]["sum"], places=6)


class AnchorSemanticsTest(unittest.TestCase):
    """确认事项 4/5 + 修正事项 E: state anchor 的分组语义."""

    def setUp(self):
        # 两行同 uid; row0/row1 的 step0 具有相同 pre-action state, 但观测不同
        self.spans = [[[0, 4], [4, 8]], [[0, 4], [4, 8]]]
        self.obs = [[INITIAL_STATE_MARKER, "OBS_X"], ["OBS_Y", "OBS_Z"]]
        self.rewards = [1.0, 0.0]
        self.uids = ["t0", "t0"]
        self.trans = [
            [trans("S_A"), trans("S_B")],
            [trans("S_A"), trans("S_B")],
        ]

    def _run(self, anchor):
        inputs = _inputs(self.spans, self.obs, self.rewards, self.uids, transitions=self.trans, length=8)
        return run_on_inputs("gigpo", inputs, {"anchor": anchor, "gamma": 1.0, "step_advantage_w": 1.0})

    def test_same_state_different_observation_grouped_by_state_anchor(self):
        state_out = self._run("state")
        obs_out = self._run("observation")
        # mean_norm: A_E = reward - group_mean = +/-0.5; A_S = G - group_mean = +/-0.5
        s_vals = span_values(state_out, self.spans, 8)
        self.assertAlmostEqual(s_vals[0][0], 0.5 + 0.5, places=6)
        self.assertAlmostEqual(s_vals[1][0], -0.5 + -0.5, places=6)
        # observation anchor: 两处观测不同 -> singleton -> A_S=0 -> 只剩 episode adv
        o_vals = span_values(obs_out, self.spans, 8)
        self.assertAlmostEqual(o_vals[0][0], 0.5, places=6)
        self.assertAlmostEqual(o_vals[1][0], -0.5, places=6)
        self.assertEqual(state_out.metrics["anchor_state_frac"], 1.0)
        self.assertEqual(obs_out.metrics["anchor_observation_frac"], 1.0)

    def test_reward_equivalent_states_are_not_the_same_state(self):
        """修正事项 E: 观测 (常含 reward 信息) 相同不代表环境状态相同."""
        self.trans = [[trans("S_A"), trans("S_B")], [trans("S_C"), trans("S_B")]]
        self.obs = [["OBS_SAME", "OBS_Z"], ["OBS_SAME", "OBS_Z"]]
        state_vals = span_values(self._run("state"), self.spans, 8)
        obs_vals = span_values(self._run("observation"), self.spans, 8)
        # state anchor: S_A != S_C -> 两个 singleton -> A_S=0
        self.assertAlmostEqual(state_vals[0][0], 0.5, places=6)
        self.assertAlmostEqual(state_vals[1][0], -0.5, places=6)
        # observation anchor: 观测相同 -> 合组 -> A_S = +/-0.5 (与 state 臂形成对比)
        self.assertAlmostEqual(obs_vals[0][0], 0.5 + 0.5, places=6)
        self.assertAlmostEqual(obs_vals[1][0], -0.5 + -0.5, places=6)

    def test_invalid_state_anchor_zeroes_step_advantage_only(self):
        """修正事项 C + F4: pre 状态未知 -> A_S=0, 但 episode advantage 仍保留."""
        self.trans = [[trans(None, executed=False, graph_valid=False, reason="state_unknown"), trans("S_B")],
                      [trans("S_A"), trans("S_B")]]
        out = self._run("state")
        vals = span_values(out, self.spans, 8)
        # row0 step0 anchor INVALID -> A_S=0, 但 episode adv (+0.5) 仍在
        self.assertAlmostEqual(vals[0][0], 0.5, places=6)
        self.assertNotEqual(vals[0][0], 0.0)
        self.assertAlmostEqual(out.metrics["anchor_invalid_frac"], 0.25, places=6)
        self.assertAlmostEqual(out.metrics["anchor_state_frac"], 0.75, places=6)
        # 判定分离: 即使 anchor 为 INVALID, 该 step 仍是"完整的 decision"
        # (span 有效且未被截断 -> 不会被排除在 advantage 分配之外).
        metas = build_step_meta(
            spans=self.spans[0], response_length=8, anchor_mode="state", transitions_row=self.trans[0]
        )
        self.assertEqual(metas[0].anchor_fp, INVALID_FINGERPRINT)
        self.assertFalse(metas[0].truncated)
        self.assertGreater(metas[0].num_tokens, 0)

    def test_initial_state_used_for_step_zero_pre_only_when_recorded(self):
        inputs = _inputs(
            [[[0, 4]]], [["OBS_X"]], [1.0], ["t0"],
            transitions=[[trans(None)]], initial_states=["S_INIT"], length=4,
        )
        out = run_on_inputs("gigpo", inputs, {"anchor": "state"})
        metas = build_step_meta(spans=[[0, 4]], response_length=4, anchor_mode="state",
                                transitions_row=[trans(None)], initial_state="S_INIT")
        self.assertEqual(metas[0].anchor_fp, "S_INIT")
        self.assertEqual(out.metrics["anchor_state_frac"], 1.0)


class TruncationPolicyTest(unittest.TestCase):
    """修正事项 B + F3: 截断与 baseline / loss 的两个独立开关."""

    def setUp(self):
        # row0 的 step1 span 越过 response 上限 (截断), row1 正常; 两者 anchor 相同
        self.spans = [[[0, 4], [4, 12]], [[0, 4], [4, 8]]]
        self.length = 8
        self.obs = [["O1", "O2"], ["O1", "O2"]]
        self.trans = [[trans("S_A"), trans("S_B")], [trans("S_A"), trans("S_B")]]
        self.rewards = [1.0, 0.0]
        self.uids = ["t0", "t0"]

    def _run(self, include_in_baseline):
        inputs = _inputs(self.spans, self.obs, self.rewards, self.uids, transitions=self.trans, length=self.length)
        return run_on_inputs(
            "gigpo",
            inputs,
            {"anchor": "state", "gamma": 1.0, "step_advantage_w": 1.0,
             "include_truncated_in_baseline": include_in_baseline},
        )

    def test_truncated_steps_stay_in_baseline_by_default(self):
        out = self._run(True)
        vals = span_values(out, self.spans, self.length)
        # G(step0)=traj_reward (gamma=1), 组 {1,0} -> A_S = +/-0.5; A_E 同为 +/-0.5
        self.assertAlmostEqual(vals[0][0], 1.0, places=6)
        self.assertAlmostEqual(vals[1][0], -1.0, places=6)
        # step1 组 {1 (truncated), 0} -> 同样 +/-0.5: 截断 step 默认仍进入 baseline
        self.assertAlmostEqual(vals[0][1], 1.0, places=6)
        self.assertAlmostEqual(vals[1][1], -1.0, places=6)
        self.assertEqual(out.metrics["baseline_excluded_truncated"], 0.0)

    def test_excluding_truncated_from_baseline_changes_other_steps_advantage(self):
        included = span_values(self._run(True), self.spans, self.length)
        excluded = span_values(self._run(False), self.spans, self.length)
        # (ii) 排除后 row1 step1 变 singleton -> A_S=0 (advantage 从 -1.0 回到 -0.5)
        self.assertAlmostEqual(included[1][1], -1.0, places=6)
        self.assertAlmostEqual(excluded[1][1], -0.5, places=6)
        # 截断 step 自身也失去组内 baseline
        self.assertAlmostEqual(excluded[0][1], 0.5, places=6)
        # (iii) episode 部分完全不受影响: step0 未截断 -> 数值不变
        self.assertAlmostEqual(excluded[0][0], included[0][0], places=6)
        self.assertAlmostEqual(excluded[1][0], included[1][0], places=6)
        self.assertEqual(self._run(False).metrics["baseline_excluded_truncated"], 1.0)

    def test_fully_cut_step_gets_no_token_advantage(self):
        inputs = _inputs(
            [[[0, 4], [4, 12]]], [["O1", "O2"]], [1.0], ["t0"],
            transitions=[[trans("S_A"), trans("S_B")]], length=8,
        )
        out = run_on_inputs("gigpo", inputs, {"anchor": "state"})
        self.assertGreater(float(out.advantages[0, 4:8].max()), 0.0)
        self.assertAlmostEqual(out.metrics["truncated_step_frac"], 0.5, places=6)
        # 完全被切掉的 step (起点已在上限之外) -> 无 token 可写
        metas = build_step_meta(spans=[[4, 8], [8, 14]], response_length=8, anchor_mode="state",
                                transitions_row=[trans("S_A"), trans("S_B")])
        self.assertEqual(metas[1].num_tokens, 0)
        self.assertEqual(metas[1].span_visible, (8, 8))
        self.assertTrue(metas[1].truncated)

    def test_terminal_situations_are_distinguishable(self):
        """(iv) 正常结束 / response 截断 / tool 未执行 三态可区分."""
        self.assertEqual(classify_truncation((4, 12), "executed", 8), TRUNC_PARTIAL_ACTION)
        self.assertEqual(classify_truncation((4, 8), "terminated_before_tools", 8), TRUNC_UNEXECUTED)
        self.assertEqual(classify_truncation((4, 8), "executed", 8), "none")
        self.assertEqual(classify_truncation((4, 8), "final_answer", 8), "none")

        metas = build_step_meta(
            spans=[[0, 4], [4, 8], [8, 12]],
            response_length=12,
            anchor_mode="state",
            transitions_row=[
                trans("S_A"),
                trans("S_B", status="terminated_before_tools", executed=False, graph_valid=False,
                    reason="max_assistant_turns"),
                trans("S_C", status="final_answer", executed=False, graph_valid=False),
            ],
            terminal_reason="response_length",
        )
        self.assertEqual([m.truncated for m in metas], [False, False, False])
        self.assertEqual([m.status for m in metas][1:], ["terminated_before_tools", "final_answer"])
        # 最后一个 span 越界才算截断
        metas2 = build_step_meta(
            spans=[[0, 8], [8, 20]], response_length=12, anchor_mode="state",
            transitions_row=[trans("S_A"), trans("S_B")],
        )
        self.assertEqual([m.truncated for m in metas2], [False, True])
        self.assertEqual(metas2[1].span_visible, (8, 12))
        self.assertTrue(metas2[1].span_clipped)

    def test_baseline_flag_does_not_change_step_metadata(self):
        """B/C: include_truncated_in_baseline 只调节 Eq.7 分组, step 分类本身不变."""
        metas = build_step_meta(
            spans=[[0, 4], [4, 12]], response_length=8, anchor_mode="state",
            transitions_row=[trans("S_A"), trans("S_B")],
        )
        self.assertEqual([m.truncated for m in metas], [False, True])
        self.assertEqual([m.num_tokens for m in metas], [4, 4])
        # 同一份 metadata 必须同时能表达 baseline 计入/不计入两种配置 (开关与 metadata 无关)
        for flag in (True, False):
            out = run_on_inputs(
                "gigpo",
                _inputs([[[0, 4], [4, 12]]], [["O1", "O2"]], [1.0], ["t0"],
                        transitions=[[trans("S_A"), trans("S_B")]], length=8),
                {"anchor": "state", "include_truncated_in_baseline": flag},
            )
            self.assertEqual(out.metrics["truncated_step_frac"], 0.5)
            self.assertEqual(out.metrics["baseline_excluded_truncated"], 0.0 if flag else 1.0)


class ConfigSchemaTest(unittest.TestCase):
    """修正事项 D + F5: GiGPOConfig <-> YAML 结构统一, 以及旧配置向后兼容."""

    YAML_PATH = "verl/trainer/config/ppo_trainer.yaml"
    ACTOR_YAML_PATH = "verl/trainer/config/actor/actor.yaml"

    def test_params_round_trip_through_parser(self):
        raw = OmegaConf.create(
            {
                "algorithm": {
                    "credit_assignment": {
                        "method": "gigpo",
                        "params": {
                            "gamma": 0.9,
                            "step_advantage_w": 0.5,
                            "mode": "mean_std_norm",
                            "grouping": "exact",
                            "epsilon": 1e-5,
                            "anchor": "state",
                            "anchor_scope": "full",
                            "include_truncated_in_baseline": False,
                        },
                    }
                }
            }
        )
        params = OmegaConf.to_container(raw.algorithm.credit_assignment.params, resolve=True)
        cfg = GiGPOConfig.from_params(params)
        self.assertEqual(cfg, GiGPOConfig(**params))
        # 再喂回 parser 得到同一配置 (round-trip)
        again = GiGPOConfig.from_params(OmegaConf.to_container(OmegaConf.create(dict(params)), resolve=True))
        self.assertEqual(again, cfg)
        self.assertTrue(again.remove_std is False)

    def test_all_dataclass_fields_are_documented_in_ppo_trainer_yaml(self):
        path = os.path.join(os.path.dirname(__file__), "..", "..", "..", self.YAML_PATH)
        with open(os.path.abspath(path)) as fh:
            text = fh.read()
        block = text.split("credit_assignment:", 1)[1].split("# config for the trainer", 1)[0]
        comments = "\n".join(line for line in block.splitlines() if line.strip().startswith("#"))
        for field in GiGPOConfig.__dataclass_fields__:
            self.assertIn(field, comments, f"field '{field}' undocumented in ppo_trainer.yaml")

    def test_legacy_params_still_accepted(self):
        """F5 backward compat: 修改前的参数集合不含 anchor 也能构造, 且取新默认."""
        cfg = GiGPOConfig.from_params({"gamma": 0.95, "step_advantage_w": 1.0, "mode": "mean_norm",
                                       "grouping": "exact", "epsilon": 1e-6})
        self.assertEqual(cfg.anchor, "state")
        self.assertEqual(cfg.anchor_scope, "full")
        self.assertTrue(cfg.include_truncated_in_baseline)

    def test_unknown_or_unsupported_params_raise(self):
        with self.assertRaisesRegex(ValueError, "unknown params"):
            GiGPOConfig.from_params({"anchors": "state"})
        with self.assertRaisesRegex(ValueError, "anchor"):
            GiGPOConfig.from_params({"anchor": "answer_range"})
        with self.assertRaisesRegex(ValueError, "anchor_scope"):
            GiGPOConfig.from_params({"anchor_scope": "answer_range"})
        with self.assertRaisesRegex(ValueError, "anchor_scope"):
            GiGPOConfig.from_params({"anchor_scope": "observation_projection"})
        with self.assertRaisesRegex(ValueError, "grouping"):
            GiGPOConfig.from_params({"grouping": "similarity"})
        with self.assertRaisesRegex(ValueError, "mode"):
            GiGPOConfig.from_params({"mode": "median_norm"})
        with self.assertRaisesRegex(ValueError, "gamma"):
            GiGPOConfig.from_params({"gamma": 0.0})

    def test_removed_step_loss_config_keys_are_absent(self):
        """被移除的 step-loss 配置的 dangling guard: 若在 actor.yaml 或 dataclass 残留则失败."""
        from verl.workers.config.actor import PolicyLossConfig

        removed = (
            "step_gradient",
            "step_reduction",
            "include_truncated_steps",
            "exclude_unexecuted_steps",
            "exclude_invalid_graph_steps",
            "clip_ratio_step",
        )
        path = os.path.join(os.path.dirname(__file__), "..", "..", "..", self.ACTOR_YAML_PATH)
        with open(os.path.abspath(path)) as fh:
            yaml_cfg = OmegaConf.load(fh)
        block = yaml_cfg.policy_loss
        defaults = PolicyLossConfig()
        for field in removed:
            self.assertNotIn(field, set(block.keys()), f"{field} 已被移除, 不应残留在 actor.yaml")
            self.assertFalse(hasattr(defaults, field), f"{field} 已被移除, 不应残留在 PolicyLossConfig")
        # 必须保留的键 (vanilla PPO 原有 policy_loss 设置) 也一并确认
        for field in ("loss_mode", "clip_cov_ratio", "clip_cov_lb", "clip_cov_ub", "kl_cov_ratio", "ppo_kl_coef"):
            self.assertIn(field, set(block.keys()))
            self.assertEqual(OmegaConf.to_container(block, resolve=True)[field], getattr(defaults, field))
        # loss_mode 默认值是 vanilla, 且 registry 中不存在 gigpo_* step objective
        self.assertEqual(defaults.loss_mode, "vanilla")
        from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY

        for name in ("gigpo_joint", "gigpo_geomean"):
            self.assertNotIn(name, POLICY_LOSS_REGISTRY, f"{name} step loss 仍然被注册")


class DiagnosticsAndWiringTest(unittest.TestCase):
    """指标旁路: CreditAssignmentOutput.metrics -> estimator metrics_out."""

    def test_metrics_bypass_through_estimator(self):
        from verl.trainer.config.algorithm import AlgoConfig
        from verl.trainer.ppo.core_algos import compute_credit_assignment_advantage

        inputs = _inputs(
            [[[0, 4], [4, 12]], [[0, 4], [4, 8]]],
            [["O1", "O2"], ["O1", "O2"]],
            [1.0, 0.0],
            ["t0", "t0"],
            transitions=[[trans("S_A"), trans("S_B")], [trans("S_A"), trans("S_B")]],
            length=8,
        )
        spans = inputs.step_spans
        obs = inputs.step_observations
        trans_rows = inputs.extra["step_transitions"]
        bs, length = 2, 8

        def obj_array(rows):
            arr = np.empty(bs, dtype=object)
            for i, r in enumerate(rows):
                arr[i] = r
            return arr

        non_tensor_batch = {
            "ca_step_spans": obj_array(spans),
            "ca_step_observations": obj_array(obs),
            "ca_step_transitions": obj_array(trans_rows),
            "ca_initial_state": np.array([None, None], dtype=object),
            "ca_terminal_reason": obj_array(["response_length", "natural"]),
        }
        token_rewards = torch.zeros(bs, length)
        token_rewards[0, 7] = 1.0
        metrics_out: dict = {}
        config = AlgoConfig(
            adv_estimator="credit_assignment",
            credit_assignment={"method": "gigpo", "params": {"anchor": "state", "gamma": 1.0}},
        )
        adv, ret = compute_credit_assignment_advantage(
            token_level_rewards=token_rewards,
            response_mask=inputs.assistant_mask,
            index=np.array(["t0", "t0"], dtype=object),
            config=config,
            non_tensor_batch=non_tensor_batch,
            batch={"token_level_rewards": token_rewards},
            metrics_out=metrics_out,
        )
        self.assertTrue(torch.equal(adv, ret))
        self.assertTrue(metrics_out, "metrics_out should be filled")
        self.assertTrue(all(k.startswith("credit_assignment/gigpo/") for k in metrics_out))
        self.assertIn("credit_assignment/gigpo/truncated_step_frac", metrics_out)
        self.assertAlmostEqual(metrics_out["credit_assignment/gigpo/truncated_step_frac"], 0.25, places=6)
        # 不传 metrics_out 时行为不变 (返回值仍是 2-tuple)
        adv2, _ = compute_credit_assignment_advantage(
            token_level_rewards=token_rewards,
            response_mask=inputs.assistant_mask,
            index=np.array(["t0", "t0"], dtype=object),
            config=config,
            non_tensor_batch=non_tensor_batch,
        )
        self.assertTrue(torch.equal(adv, adv2))

    def test_compute_entry_point_uses_non_tensor_batch(self):
        spans = [[[0, 4], [4, 8]], [[0, 4], [4, 8]]]
        obs = [["O1", "O2"], ["O1", "O2"]]
        rows = [[trans("S_A"), trans("S_B")], [trans("S_A"), trans("S_B")]]
        non_tensor_batch = {
            "ca_step_spans": _object_array(spans),
            "ca_step_observations": _object_array(obs),
            "ca_step_transitions": _object_array(rows),
        }
        mask = _mask_from_spans(spans, 2, 8)
        rewards = torch.zeros(2, 8)
        rewards[0, 7] = 1.0
        out = compute(
            method="gigpo",
            token_level_rewards=rewards,
            response_mask=mask,
            index=np.array(["t0", "t0"], dtype=object),
            non_tensor_batch=non_tensor_batch,
            params={"anchor": "state", "gamma": 1.0},
        )
        self.assertEqual(out.advantages.shape, (2, 8))
        self.assertAlmostEqual(float(out.advantages[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(out.advantages[1, 0]), -1.0, places=6)


def _object_array(rows):
    arr = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        arr[i] = r
    return arr


if __name__ == "__main__":
    unittest.main()
