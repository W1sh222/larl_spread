# RewardFlow (Potential-based Step Credit)

## Mechanism

RewardFlow assigns step-level credit from the *potential difference* of workbook states
inside a per-task state graph, rather than from anchor-grouped returns.

1. **Episode term**: the training trajectory reward (possibly KL-shaped) is normalized
   within the episode group (same uid), giving `A_traj`.
2. **State graph**: for each uid (disambiguated by initial state), every step transition
   `s_t -> s_{t+1}` becomes an edge. Self-loops (`s_t == s_{t+1}`) are not inserted as
   edges, but the corresponding `r_t = 0` sample still participates in the normalization
   group — "no state change" is treated as a valid comparison signal.
3. **Success anchors (Eq.1)**: a terminal state becomes a `Phi = 1` anchor only when the
   *raw environment outcome* (not the KL-shaped reward) reaches `success_threshold` within
   `success_tolerance`, the terminal transition is trusted, and — when an evaluation
   fingerprint exists — the submitted bytes' canonical fingerprint equals the terminal
   state fingerprint. Anchors are never invented from missing data.
4. **Potentials**: `Phi(s) = gamma^d(s, success)` over graph distance to a success anchor.
   States with no path to any anchor have no potential and are simply excluded from
   propagation (recorded as a connectivity diagnostic, not filled in).
5. **Action reward (Eq.3)**: `r_t = Phi(s_{t+1}) - Phi(s_t)`; normalized within the
   `(uid, pre-state)` group.
6. **Combination (Eq.6)**: `A = alpha_traj * A_traj + alpha_act * A_act`, broadcast onto the
   assistant span of each step. Steps with no usable transition (final answer, truncated,
   not-executed) get `A_act = 0` and keep only the episode term.

Reference implementation: `methods/rewardflow/algorithm.py`, `graph.py`, `state.py`.

## Required Inputs

| Input | Source | Notes |
| --- | --- | --- |
| `step_spans` | agent loop | assistant token ranges for broadcasting |
| `step_transitions` | structured rollout | pre/post state fingerprints per step |
| `trajectory_rewards` | reward path | training reward, drives the episode term |
| `raw_trajectory_rewards` | `token_level_scores` | raw environment outcome, drives success anchoring |
| `episode_group_ids` | agent loop | uid, the graph and normalization key |
| `terminal_trusted` | rollout | whether the terminal transition may be trusted |
| `eval_fingerprint` | reward side | optional per-row submitted-bytes fingerprint |
| `assistant_mask` | rollout | restricts credit to assistant-generated tokens |

## Required Metadata

`step_spans`, `step_transitions`, `terminal_trusted`, `eval_fingerprint`.
`eval_fingerprint` is a *soft* requirement: when it is absent for a row, only that row's
success anchor is excluded fail-closed; the rest of the batch still computes. It is never
substituted by the shaped reward.

RewardFlow refuses to run if a transition metadata row is `None` or its length disagrees
with the number of step spans — silent per-row degradation to episode-only credit is
deliberately not allowed. It does not consume `step_observations`.

## Configurable Parameters

| Parameter | Type / domain | Default | Meaning |
| --- | --- | --- | --- |
| `gamma` | number in [0.5, 1.0] | `0.95` | potential discount, `Phi = gamma^d` |
| `alpha_act` | number in [0.0, 2.0] | `1.0` | weight of the action/potential term (Eq.6) |
| `alpha_traj` | number in [0.0, 2.0] | `1.0` | weight of the episode term (Eq.6) |
| `normalization` | `mean_norm` / `mean_std_norm` | `mean_norm` | whether normalization divides by group std |
| `epsilon` | numeric stability | `1e-6` | not exposed for calibration |
| `success_threshold` | number | `1.0` | definition of success — a measurement criterion, not a tuning knob; not exposed |
| `success_tolerance` | number | `1e-6` | tolerance around the threshold; not exposed |

Expected effects:

- `gamma` → how strongly distance-to-success decays. Smaller `gamma` concentrates positive
  credit on the steps immediately preceding success; `gamma` near 1 spreads it far back,
  which weakens discrimination but increases coverage.
- `alpha_act = 0` degenerates to episode-only credit; `alpha_traj = 0` removes the trajectory
  term and relies entirely on potential differences (fragile when the graph is sparse).
- `normalization='mean_std_norm'` equalizes scale across pre-state groups, which amplifies
  groups whose potentials are almost identical.

## Expected Strengths

These are hypotheses implied by the mechanism, not measured results:

- Credit is dense along the path that actually leads to a successful terminal state, so steps
  that made measurable progress are separated from steps that merely preceded success.
- Potential shaping is state-based rather than visit-count-based, so repeated or detour
  rollouts toward the same state are not rewarded twice for the same progress.
- The episode term keeps the trustworthy terminal outcome as the anchor of the potential, so
  the shaping cannot invent progress that no successful rollout demonstrates.
- Steps with no path to any success anchor receive no action credit, which should keep
  exploration outside the reachable graph from being reinforced.

## When to Use

- Actions change observable workbook state and successful trajectories converge toward a
  recognizable terminal state, so potentials can propagate.
- The terminal outcome is trustworthy (`terminal_trusted`) and evaluation fingerprints are
  emitted, so success anchors are actually created.
- Episode-level credit alone gives too many zero advantages, and a state-mediated signal is
  available to break the tie.
- Task distributions where "progress toward the goal" is meaningful, e.g. multi-step file
  edits whose intermediate states are comparable across rollouts.

## Limitations

- Depends on the richest metadata of the three skills. Low structured-transition coverage or
  missing evaluation fingerprints means few anchors, and without anchors there is no
  potential at all — the action term contributes essentially nothing.
- Sparse or disconnected graphs: states with no path to a success anchor receive no credit,
  so long exploratory trajectories may be credited only at episode level.
- The success anchor is defined on the raw outcome, so a reward pipeline that does not keep
  the raw environment outcome distinct from the shaped reward invalidates the anchors.
- Binary-outcome assumption: non-binary raw rewards still work through the threshold, but the
  semantic of "success" becomes approximate and is warned about.
- Duplicate/repeated states yield zero potential differences, which can dominate the
  normalization group and shrink the effective step signal.

## Calibration Guidelines

- Before changing `alpha_act`, look at the fraction of steps that actually received a
  non-zero action reward and at transition coverage. If most steps fall back to the episode
  term, raising `alpha_act` only amplifies noise.
- Adjust `gamma` when the credit appears too concentrated (only the last steps differentiate)
  or too diffuse (nearly all steps share similar potential).
- Keep `alpha_traj` around `1.0` unless there is evidence the episode term is dominating;
  dropping it entirely is a high-variance move given the graph's sparsity.
- Never tune `success_threshold` / `success_tolerance` through the controller: they define
  what counts as success and must stay aligned with the evaluation pipeline.
- Any predicted effect on success rate is a hypothesis until a window comparison confirms it.

## Empirical Experience

_No verified experience recorded yet._
