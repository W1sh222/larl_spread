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

"""Runtime Credit Skill 切换门禁 (CPU).

覆盖 명령서 §21.4: 三种 Skill 之间的来回切换 / 同 Skill 校准 / 参数整体替换 /
未知 Skill / 未知参数 / 非法类型 / 越界 / metadata 需求冲突 / 失败时不留半改状态.
"""

import unittest
import warnings

from omegaconf import OmegaConf

from verl.trainer.credit_assignment.registry import get_credit_assignment_class
from verl.trainer.credit_assignment.self_improving.config import load_self_improving_config, validate_self_improving_config
from verl.trainer.credit_assignment.self_improving.runtime import (
    SkillConfig,
    SkillRuntime,
    SkillValidationError,
    bootstrap_metadata_signature,
    canonical_params,
    collected_metadata,
    maximal_required_metadata,
)


def algo_config(method="rewardflow", params=None, adv_estimator="credit_assignment", **extra):
    ca = {"method": method, "params": dict(params or {})}
    ca.update(extra)
    return OmegaConf.create({"adv_estimator": adv_estimator, "credit_assignment": ca})


def si_config(initial="grpo", candidates=("grpo", "gigpo", "rewardflow"), initial_params=None, **node):
    raw = {
        "enabled": False,
        "initial_skill": {"method": initial, "params": dict(initial_params or {})},
        "candidate_skills": list(candidates),
        "llm": {"client": "mock"},
    }
    raw.update(node)
    return load_self_improving_config(raw)


class RuntimeSwitchingTest(unittest.TestCase):
    def setUp(self):
        # bootstrap method 用 rewardflow: 它需要全部 metadata, 三种 Skill 都可达.
        self.algorithm_cfg = algo_config("rewardflow")
        self.runtime = SkillRuntime(si_config(), self.algorithm_cfg)
        self.runtime.apply_initial()

    def _active(self):
        ca = self.algorithm_cfg.credit_assignment
        return str(ca.method), OmegaConf.to_container(ca.params, resolve=True) or {}

    def test_grpo_to_gigpo_to_rewardflow_to_grpo(self):
        seen = []
        for method in ("gigpo", "rewardflow", "grpo"):
            applied = self.runtime.apply(SkillConfig(method=method, params={}), from_llm=True)
            name, params = self._active()
            seen.append(name)
            self.assertEqual(name, method)
            self.assertEqual(params, canonical_params(method, {}))
            self.assertEqual(applied.params, params)
            # 切换后的 config 必须能被 registry 真正执行 (metadata 门禁与 method 一致).
            cls = get_credit_assignment_class(name)
            self.assertTrue(set(cls.required_metadata(params)) <= set(self.runtime.collected))
        self.assertEqual(seen, ["gigpo", "rewardflow", "grpo"])
        self.assertEqual(self.runtime.current().method, "grpo")

    def test_same_skill_parameter_calibration(self):
        self.runtime.apply(SkillConfig("gigpo", {"gamma": 0.9, "mode": "mean_std_norm"}), from_llm=True)
        before = self.runtime.current().params
        self.assertEqual(before["gamma"], 0.9)
        self.assertEqual(before["mode"], "mean_std_norm")
        self.runtime.apply(SkillConfig("gigpo", {"gamma": 0.7}), from_llm=True)
        _, params = self._active()
        self.assertEqual(params["gamma"], 0.7)
        self.assertEqual(params["mode"], "mean_norm")  # 未提交的参数回到 dataclass 默认值

    def test_parameters_are_replaced_not_merged(self):
        self.runtime.apply(SkillConfig("grpo", {"norm_adv_by_std_in_grpo": False}), from_llm=False)
        self.assertIn("norm_adv_by_std_in_grpo", self._active()[1])
        self.runtime.apply(SkillConfig("gigpo", {"gamma": 0.8}), from_llm=True)
        params = self._active()[1]
        self.assertNotIn("norm_adv_by_std_in_grpo", params)
        self.assertIn("step_advantage_w", params)
        self.runtime.apply(SkillConfig("rewardflow", {"alpha_act": 0.5}), from_llm=True)
        params = self._active()[1]
        self.assertNotIn("step_advantage_w", params)
        self.assertNotIn("anchor", params)
        self.assertEqual(params["alpha_act"], 0.5)

    def test_unknown_skill_rejected(self):
        for method in ("gae", "ppo", "", None):
            with self.assertRaises(SkillValidationError):
                self.runtime.apply(SkillConfig(method, {}), from_llm=True)
        with self.assertRaises(SkillValidationError):
            self.runtime.apply("grpo", from_llm=True)  # type: ignore[arg-type]

    def test_skill_outside_candidate_list_rejected(self):
        runtime = SkillRuntime(si_config(candidates=("grpo",)), algo_config("grpo"))
        runtime.apply_initial()
        with self.assertRaises(SkillValidationError) as ctx:
            runtime.apply(SkillConfig("gigpo", {}), from_llm=True)
        self.assertIn("candidate_skills", str(ctx.exception))

    def test_unregistered_method_raises_value_error(self):
        with self.assertRaises(ValueError):
            canonical_params("no_such_method", {})

    def test_registered_but_unmapped_method_rejected(self):
        # 已注册但没有 calibration 映射的 method -> 明确报错, 不静默放行.
        from verl.trainer.credit_assignment import registry
        from verl.trainer.credit_assignment.interface import CreditAssignmentMethod

        class _Unmapped(CreditAssignmentMethod):
            requires = frozenset()

            def compute(self, inputs, params):  # pragma: no cover - 永不调用
                raise NotImplementedError

        registry._CREDIT_ASSIGNMENT_REGISTRY["unmapped"] = _Unmapped
        try:
            with self.assertRaises(SkillValidationError) as ctx:
                canonical_params("unmapped", {})
            self.assertIn("no calibration config class mapping", str(ctx.exception))
            with self.assertRaises(SkillValidationError) as ctx:
                runtime = SkillRuntime(si_config(candidates=("grpo", "unmapped")), algo_config("grpo"))
                runtime.apply_initial()
                runtime.apply(SkillConfig("unmapped", {}), from_llm=True)
            self.assertIn("no calibration config class mapping", str(ctx.exception))
        finally:
            registry._CREDIT_ASSIGNMENT_REGISTRY.pop("unmapped", None)

    def test_unknown_parameter_rejected(self):
        with self.assertRaises(SkillValidationError):
            self.runtime.apply(SkillConfig("grpo", {"epsilon": 1e-5}), from_llm=True)  # 未暴露给 LLM
        with self.assertRaises(SkillValidationError):
            self.runtime.apply(SkillConfig("grpo", {"learning_rate": 1e-6}), from_llm=True)
        with self.assertRaises(SkillValidationError) as ctx:
            self.runtime.apply(SkillConfig("grpo", {"learning_rate": 1e-6}), from_llm=False)
        self.assertIn("unsupported params", str(ctx.exception))
        # 非暴露参数仍可由 initial_skill 显式设置
        applied = self.runtime.apply(SkillConfig("gigpo", {"gamma": 0.9, "epsilon": 1e-5}), from_llm=False)
        self.assertEqual(applied.params["epsilon"], 1e-5)

    def test_invalid_parameter_type_rejected(self):
        cases = [
            SkillConfig("gigpo", {"gamma": "0.9"}),
            SkillConfig("gigpo", {"gamma": True}),
            SkillConfig("gigpo", {"mode": "median"}),
            SkillConfig("gigpo", {"grouping": "fuzzy"}),
            SkillConfig("gigpo", {"anchor": "hidden_state"}),
            SkillConfig("grpo", {"norm_adv_by_std_in_grpo": 1}),
            SkillConfig("rewardflow", {"normalization": 3}),
        ]
        for skill in cases:
            with self.assertRaises(SkillValidationError, msg=f"{skill.method} {skill.params}"):
                self.runtime.apply(skill, from_llm=True)

    def test_out_of_range_parameter_rejected(self):
        for params in ({"gamma": 1.5}, {"gamma": 0.1}, {"step_advantage_w": 3.0}):
            with self.assertRaises(SkillValidationError):
                self.runtime.apply(SkillConfig("gigpo", params), from_llm=True)
        for params in ({"alpha_act": -0.5}, {"alpha_traj": 2.5}, {"gamma": 0.0}):
            with self.assertRaises(SkillValidationError):
                self.runtime.apply(SkillConfig("rewardflow", params), from_llm=True)
        # 边界值本身合法
        self.runtime.apply(SkillConfig("gigpo", {"gamma": 1.0, "step_advantage_w": 0.0}), from_llm=True)

    def test_unexposed_parameters_cannot_be_reset_by_llm(self):
        # initial_skill 可以设定未暴露参数; LLM 只能原样回显, 不能改.
        runtime = SkillRuntime(si_config(initial="gigpo"), algo_config("rewardflow"))
        runtime.apply_initial(restore=SkillConfig("gigpo", {"include_truncated_in_baseline": False, "epsilon": 1e-5}))
        self.assertEqual(runtime.current().params["include_truncated_in_baseline"], False)
        runtime.apply(SkillConfig("gigpo", {"gamma": 0.8}), from_llm=True)
        self.assertEqual(runtime.current().params["include_truncated_in_baseline"], False)
        self.assertEqual(runtime.current().params["epsilon"], 1e-5)
        with self.assertRaises(SkillValidationError):
            runtime.apply(SkillConfig("gigpo", {"include_truncated_in_baseline": True}), from_llm=True)
        # 已归一化的 canonical 决策重新过门禁必须是幂等的 (loop 会再校验一次).
        runtime.apply(runtime.current(), from_llm=True)
        # 切到其他 method 时回到该 method 自己的默认值
        runtime.apply(SkillConfig("rewardflow", {"gamma": 0.9}), from_llm=True)
        self.assertEqual(runtime.current().params["success_threshold"], 1.0)

    def test_failed_apply_leaves_no_partial_mutation(self):
        self.runtime.apply(SkillConfig("gigpo", {"gamma": 0.9}), from_llm=True)
        snapshot = OmegaConf.to_container(self.algorithm_cfg, resolve=True)
        for skill in (
            SkillConfig("nope", {}),
            SkillConfig("rewardflow", {"not_a_param": 1}),
            SkillConfig("rewardflow", {"gamma": 42}),
            SkillConfig("grpo", {"epsilon": 0.5}),
        ):
            with self.assertRaises(SkillValidationError):
                self.runtime.apply(skill, from_llm=True)
            self.assertEqual(OmegaConf.to_container(self.algorithm_cfg, resolve=True), snapshot)
            self.assertEqual(self.runtime.current().method, "gigpo")

    def test_metadata_requirement_conflict_at_preflight(self):
        # bootstrap 用 gigpo anchor=observation -> 不采集 structured state -> anchor=state 不可达.
        cfg = algo_config("gigpo", {"anchor": "observation"})
        with self.assertRaises(SkillValidationError) as ctx:
            SkillRuntime(si_config(candidates=("grpo", "gigpo")), cfg)
        self.assertIn("step_transitions", str(ctx.exception))

    def test_metadata_requirement_conflict_per_parameter_variant(self):
        runtime = SkillRuntime(si_config(candidates=("grpo", "gigpo")), algo_config("rewardflow"))
        runtime.apply_initial()
        runtime.collected = frozenset({"step_spans", "step_observations"})
        with self.assertRaises(SkillValidationError) as ctx:
            runtime.apply(SkillConfig("gigpo", {"anchor": "state"}), from_llm=True)
        self.assertIn("only collects", str(ctx.exception))
        self.assertEqual(runtime.current().method, "grpo")

    def test_calibration_spec_only_lists_exposed_parameters(self):
        spec = self.runtime.calibration_spec()
        self.assertEqual(sorted(spec), ["gigpo", "grpo", "rewardflow"])
        for method, entry in spec.items():
            self.assertEqual(sorted(entry["defaults"]), sorted(canonical_params(method, {})))
            for name in entry["calibratable"]:
                self.assertNotIn(name, ("epsilon", "success_threshold", "success_tolerance", "include_truncated_in_baseline"))
        self.assertIn("anchor", spec["gigpo"]["calibratable"])
        self.assertNotIn("anchor_scope", spec["grpo"]["calibratable"])
        self.assertNotIn("norm_adv_by_std_in_grpo", spec["rewardflow"]["calibratable"])

    def test_initial_skill_is_applied_on_a_clean_run(self):
        cfg = algo_config("rewardflow")
        runtime = SkillRuntime(si_config(initial="gigpo"), cfg)
        applied = runtime.apply_initial()
        self.assertEqual(applied.method, "gigpo")
        self.assertEqual(str(cfg.credit_assignment.method), "gigpo")
        restored = runtime.apply_initial(restore=SkillConfig("rewardflow", {"gamma": 0.6}))
        self.assertEqual(restored.params["gamma"], 0.6)
        self.assertEqual(str(cfg.credit_assignment.method), "rewardflow")


class MetadataBootstrapTest(unittest.TestCase):
    def test_collected_metadata_mirrors_rollout_bootstrap(self):
        self.assertEqual(
            collected_metadata(algo_config("grpo")),
            frozenset({"step_spans", "step_observations"}),
        )
        self.assertEqual(
            collected_metadata(algo_config("gigpo", {"anchor": "observation"})),
            frozenset({"step_spans", "step_observations"}),
        )
        self.assertEqual(
            collected_metadata(algo_config("gigpo", {"anchor": "observation"}, collect_step_states=True)),
            frozenset({"step_spans", "step_observations", "step_transitions", "terminal_trusted"}),
        )
        full = collected_metadata(algo_config("rewardflow"))
        self.assertIn("eval_fingerprint", full)
        self.assertIn("step_transitions", full)
        self.assertEqual(collected_metadata(algo_config(adv_estimator="gae", method="rewardflow")), frozenset())
        self.assertEqual(collected_metadata(OmegaConf.create({"adv_estimator": "gae"})), frozenset())

    def test_maximal_required_metadata_bounds_calibration_space(self):
        self.assertEqual(maximal_required_metadata("grpo"), frozenset())
        self.assertIn("step_transitions", maximal_required_metadata("gigpo"))
        self.assertIn("eval_fingerprint", maximal_required_metadata("rewardflow"))

    def test_bootstrap_signature_records_launch_config(self):
        sig = bootstrap_metadata_signature(algo_config("rewardflow"))
        self.assertEqual(sig["bootstrap_method"], "rewardflow")
        self.assertIsNone(sig["collect_step_states"])
        self.assertIn("eval_fingerprint", sig["collected"])

    def test_non_credit_assignment_estimator_is_refused(self):
        cfg = algo_config("rewardflow", adv_estimator="gigpo")
        with self.assertRaises(SkillValidationError) as ctx:
            SkillRuntime(si_config(), cfg)
        self.assertIn("adv_estimator", str(ctx.exception))

    def test_unregistered_candidate_is_refused(self):
        with self.assertRaises(SkillValidationError) as ctx:
            SkillRuntime(si_config(candidates=("grpo", "imaginary")), algo_config("rewardflow"))
        self.assertIn("not registered", str(ctx.exception))

    def test_rewardflow_bootstrap_also_covers_gigpo_and_grpo(self):
        runtime = SkillRuntime(si_config(), algo_config("rewardflow"))
        self.assertEqual(runtime.metadata_gaps(), {})
        self.assertEqual(
            runtime.candidate_metadata(),
            {
                "grpo": [],
                "gigpo": sorted(maximal_required_metadata("gigpo")),
                "rewardflow": sorted(maximal_required_metadata("rewardflow")),
            },
        )

    def test_default_bootstrap_configuration_is_allowed(self):
        # §C-1: `params: {}` 是默认状态, 不能被拒.
        cfg = algo_config("rewardflow", params={})
        SkillRuntime(si_config(), cfg).apply_initial()
        # 未赋值 params 仍走默认.
        self.assertEqual(OmegaConf.to_container(cfg.credit_assignment.params, resolve=True) or {}, canonical_params("grpo", {}))

    def test_non_default_bootstrap_algorithm_params_are_refused(self):
        # §C-1: launch 节点的 algorithm params 不会静默合并到 initial_skill -> _preflight 报错.
        cfg = algo_config("rewardflow", params={"gamma": 0.42})
        with self.assertRaises(SkillValidationError) as ctx:
            SkillRuntime(si_config(), cfg).apply_initial()
        self.assertIn("algorithm.credit_assignment.params", str(ctx.exception))
        self.assertIn("self_improving.initial_skill.params", str(ctx.exception))

    def test_metadata_relevant_bootstrap_params_are_allowed(self):
        # anchor 会改变 required_metadata -> 属于 bootstrap 语义, 允许出现在 launch 节点.
        # 候选限定为 grpo/gigpo, 让 rewardflow (需 eval_fingerprint) 不干扰本用例的 C-1 意图.
        cfg = algo_config("gigpo", params={"anchor": "observation"}, collect_step_states=True)
        SkillRuntime(si_config(candidates=("grpo", "gigpo")), cfg).apply_initial()
        # anchor 只决定采集范围, 不会静默填进 initial_skill.
        self.assertNotIn("anchor", self._active(cfg)[1])

    def test_explicit_initial_skill_parameters_are_preserved(self):
        # §C-1: 训练参数只从 self_improving.initial_skill.params 走, apply_initial() 不越界.
        cfg = algo_config("rewardflow")
        runtime = SkillRuntime(si_config(initial="gigpo", initial_params={"gamma": 0.8, "mode": "mean_std_norm"}), cfg)
        applied = runtime.apply_initial()
        self.assertEqual(applied.method, "gigpo")
        self.assertEqual(applied.params["gamma"], 0.8)
        self.assertEqual(applied.params["mode"], "mean_std_norm")

    def test_bootstrap_params_are_not_inherited_by_initial_skill(self):
        # §C-1: 不允许 launch 节点被静默 merge 进 initial_skill.
        cfg = algo_config("rewardflow", params={}, collect_step_states=True)
        runtime = SkillRuntime(si_config(initial="grpo"), cfg)
        applied = runtime.apply_initial()
        # grpo 不应继承 rewardflow 的 gamma / alpha_act 等 launch 语义参数.
        self.assertEqual(applied.method, "grpo")
        self.assertNotIn("gamma", applied.params)
        self.assertNotIn("alpha_act", applied.params)

    def test_initial_grpo_rewardflow_gigpo_all_run_through_runtime(self):
        # §C-1: 三种 initial Skill 都要能走 apply_initial, 不被 _preflight 拒.
        for method in ("grpo", "gigpo", "rewardflow"):
            cfg = algo_config("rewardflow", collect_step_states=True)
            runtime = SkillRuntime(si_config(initial=method), cfg)
            applied = runtime.apply_initial()
            self.assertEqual(applied.method, method)

    def test_interval_smaller_than_min_window_warns_but_does_not_raise(self):
        # §10 minor: 该组合会让每个窗口都判 insufficient_evidence, 只警告不阻断 (CPU 快速实验合法).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_self_improving_config(si_config(enabled=True, interval_steps=1, evaluation={"min_window_steps": 5}))
        self.assertTrue(
            any("min_window_steps" in str(x.message) for x in caught),
            [str(x.message) for x in caught],
        )
        # 反向 (interval >= min_window) 不产生该警告.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_self_improving_config(si_config(enabled=True, interval_steps=5, evaluation={"min_window_steps": 5}))
        self.assertFalse([x for x in caught if "min_window_steps" in str(x.message)])
        # enabled=false 时不警告 (不跑 SI 就没有评估窗口可言).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_self_improving_config(si_config(enabled=False, interval_steps=1, evaluation={"min_window_steps": 5}))
        self.assertFalse([x for x in caught if "min_window_steps" in str(x.message)])

    def _active(self, cfg):
        return str(cfg.credit_assignment.method), OmegaConf.to_container(cfg.credit_assignment.params, resolve=True) or {}


def _with_initial_params(**over):
    node = {"enabled": False, "initial_skill": {"method": "grpo", "params": {}}, "candidate_skills": ["grpo", "gigpo", "rewardflow"], "llm": {"client": "mock"}}
    node.update(over)
    return node


if __name__ == "__main__":
    unittest.main()
