# GRPO (Group Relative Policy Optimization)

## Mechanism

GRPO is a trajectory-level (outcome) credit assignment method. A single scalar reward is
attached per rollout; that scalar is turned into an advantage by comparing rollouts that
share the same episode group (same prompt / same task instance), then the resulting scalar
is broadcast to the response tokens.

Reference implementation: `verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage`,
called unchanged by `methods/grpo/algorithm.py`.

Computation, per group of rollouts indexed by `episode_group_ids`:

1. `scores[i] = token_level_rewards[i].sum()` — the trajectory return.
2. Group mean `mu_g` and group std `sigma_g` over the trajectories in the group.
3. `advantage[i] = scores[i] - mu_g`, optionally divided by `sigma_g`
   (`norm_adv_by_std_in_grpo=True`).
4. A singleton group contributes `mean = 0.0`, `std = 1.0`, so its advantage is `0`
   after centering — a lone rollout neither helps nor hurts.
5. The scalar advantage is multiplied by the response mask, so only generated response
   tokens carry credit and padding carries exactly zero.

Because centering happens inside the group, GRPO needs no learned value function and no
environment feedback beyond the terminal outcome.

## Required Inputs

| Input | Source | Notes |
| --- | --- | --- |
| `token_level_rewards` | reward path | summed per row to obtain the trajectory return |
| `response_mask` | rollout | defines which tokens receive the broadcast advantage, and its row length |
| `episode_group_ids` | agent loop | the grouping key; trajectories with the same id are compared with each other |

## Required Metadata

None. GRPO declares an empty `requires` set, so it runs on the plain outcome reward and is
always available regardless of the launch-time metadata bootstrap. It does not read step
spans, step observations, structured state transitions, trusted-terminal flags or
evaluation fingerprints.

## Configurable Parameters

| Parameter | Type / domain | Default | Meaning |
| --- | --- | --- | --- |
| `norm_adv_by_std_in_grpo` | bool | `true` | divide the group-centered score by the group std |
| `epsilon` | number, numeric-stability only | `1e-6` | denominator guard; not a performance knob and not exposed for calibration |

Effect of `norm_adv_by_std_in_grpo`:

- `true` — advantages are scale-normalized inside each group. Groups whose outcomes are
  nearly identical (small std) get their tiny score differences amplified, which keeps the
  gradient magnitude uniform across tasks but can turn noise into a large update.
- `false` — advantages stay proportional to the raw score gap. High-variance tasks dominate
  the mean update magnitude; low-variance tasks contribute almost nothing.

## Expected Strengths

- Works with the least available signal: only a terminal outcome per trajectory.
- Group-relative centering removes the need for a critic and for cross-task reward scaling.
- Expected to be the most stable choice when step-level metadata is missing, corrupted or
  inconsistent, because no step signal is consumed at all.
- Expected to behave well when the outcome is sparse but reliably measured.

## When to Use

- The credit problem is essentially "which whole trajectory was better", with no trustworthy
  per-step signal.
- Step-level metadata coverage is low, so a step-aware skill would be operating on gaps.
- Groups contain enough rollouts per task instance for the group statistics to be meaningful;
  with many singleton groups the method returns mostly zero advantages.
- As a conservative fallback when a step-aware skill has produced unstable or contradictory
  evidence.

## Limitations

- All tokens inside a trajectory receive the same credit, so intermediate actions that were
  wrong in an ultimately successful trajectory are reinforced (the long-horizon credit
  assignment problem GRPO does not address).
- Group-relative baselines give no signal for tasks whose whole group has the same outcome —
  such groups produce zero advantage and contribute nothing.
- No use of environment state means the method cannot distinguish "recovered from a bad
  action" from "ruined an otherwise good trajectory".
- With `norm_adv_by_std_in_grpo=true`, groups with a near-zero std can be over-amplified.

## Calibration Guidelines

- Only `norm_adv_by_std_in_grpo` is calibratable. Consider toggling it when the distribution
  of group std across task instances is very wide (e.g. some tasks have binary success and
  others have graded partial credit).
- Evidence to inspect before toggling: the dispersion of `raw_reward` inside groups, and
  whether the advantage magnitude differs by orders of magnitude across batches.
- `epsilon` must not be treated as a tuning knob; changing it alters only numerical guards.
- Any expectation about the effect of this parameter on success rate is an untested
  hypothesis for the current run; it must be confirmed by a window comparison before being
  recorded as experience.

## Empirical Experience

_No verified experience recorded yet._
