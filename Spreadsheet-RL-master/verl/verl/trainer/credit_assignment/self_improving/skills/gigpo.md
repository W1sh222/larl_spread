# GiGPO (Group-in-Group Policy Optimization)

## Mechanism

GiGPO combines a trajectory-level (episode) advantage with a step-level advantage computed
by grouping *steps* — not trajectories — across rollouts. Two nested groupings:

- **Episode level (Eq.3)**: the terminal trajectory return is normalized within the episode
  group (same task instance). Normalization is per group, so a trajectory with many steps
  does not receive more weight than a short one.
- **Step level (Eq.5-7)**: each step gets a discounted return
  `G[k] = r[k] + gamma * G[k+1]`, where the terminal outcome reward is placed only on the
  last step and intermediate steps default to zero. Steps are then grouped by an *anchor*
  fingerprint (an equivalence class of the environment state at the step) and the returns
  are normalized inside each `(uid, anchor)` group.
- **Combination (Eq.8)**: `A = A_episode + step_advantage_w * A_step`, broadcast onto the
  assistant token span of each step. Tool-observation tokens and padding are masked to zero.

Anchor states are classified by shared code (`state_meta.build_step_meta`) into
`INITIAL / OBSERVED / INVALID`. An `INVALID` anchor is never grouped: its step advantage is
zero and only the episode term survives, so "missing observation" is not silently treated as
a comparable state. Singleton groups and zero-variance groups yield step advantage `0`.

Reference implementation: `methods/gigpo/algorithm.py` (equations follow arXiv:2505.10978).

## Required Inputs

| Input | Source | Notes |
| --- | --- | --- |
| `step_spans` | agent loop | token range of each step, needed to broadcast step credit |
| `step_observations` | agent loop | text used as anchor when `anchor='observation'` |
| `trajectory_rewards` | reward path | terminal outcome per trajectory, placed on the last step |
| `episode_group_ids` | agent loop | uid for the episode grouping and part of the step group key |
| `assistant_mask` | rollout | restricts credit to assistant-generated tokens |
| `step_transitions` | structured rollout | only when `anchor='state'` |

## Required Metadata

Always: `step_spans`, `step_observations`.
Additionally with the default `anchor='state'`: `step_transitions` (structured pre/post-state
workbook fingerprints). GiGPO does **not** declare `terminal_trusted` or `eval_fingerprint`
as required, and does not consume them.

If `anchor='state'` but structured transitions were not collected, the method raises rather
than degrading silently to the observation anchor — the launch configuration must provide
the metadata (the self-improving metadata bootstrap makes this condition impossible to hit).

## Configurable Parameters

| Parameter | Type / domain | Default | Meaning |
| --- | --- | --- | --- |
| `gamma` | number in [0.5, 1.0] | `0.95` | discount of the step return (Eq.5) |
| `step_advantage_w` | number in [0.0, 2.0] | `1.0` | weight of the step term in Eq.8 |
| `mode` | `mean_norm` / `mean_std_norm` | `mean_norm` | whether group normalization removes the std |
| `grouping` | `exact` | `exact` | anchor grouping implementation (only exact is implemented) |
| `anchor` | `state` / `observation` | `state` | what defines step equivalence |
| `anchor_scope` | `full` | `full` | state projection of the anchor (only `full` implemented) |
| `epsilon` | numeric stability | `1e-6` | not exposed for calibration |
| `include_truncated_in_baseline` | bool | `true` | whether truncated steps join the Eq.7 baseline group; changes grouping semantics, so it is not exposed for calibration |

Expected effects:

- `gamma` → how far ahead a step's credit reaches. Lower `gamma` confines credit to the
  immediate step; `gamma=1.0` makes the terminal outcome dominate every earlier step, which
  approaches trajectory-only behaviour.
- `step_advantage_w = 0` degenerates GiGPO into episode-only credit (closer to GRPO without
  group-std normalization); larger values let the anchor-grouped step signal dominate.
- `anchor='state'` groups steps by workbook content, which should produce larger, more
  meaningful step groups in spreadsheet tasks; `anchor='observation'` groups by tool output
  text and is the cheaper A/B baseline. Which one is better for a given task distribution is
  a hypothesis, not an established result.
- `mode='mean_std_norm'` scales each anchor group to unit variance, which amplifies groups
  where all returns are nearly equal.

## Expected Strengths

These are hypotheses implied by the mechanism, not measured results:

- Discriminates *which* tool call within a trajectory mattered, so credit is not spread
  uniformly over every assistant token of the episode.
- Steps that reach a commonly visited anchor get a baseline from other rollouts, which should
  reduce the variance caused by task difficulty alone.
- The episode term keeps the terminal outcome dominant, so a step signal cannot on its own
  reward a trajectory that never finishes the task.
- Anchor grouping is batch-local and adds no model or environment cost beyond the fingerprint
  computation already performed during rollout.

## When to Use

- The action space produces observable intermediate state changes, so a step-level signal
  plausibly exists.
- Rollouts share task instances and revisit similar states, so anchor groups contain more
  than one member (otherwise step advantages collapse to zero).
- Structured state metadata is available and has good coverage.
- Trajectory-level credit appears too coarse: e.g. successful and failed trajectories become
  indistinguishable early on, or many zero-advantage groups are observed.

## Limitations

- The whole step term vanishes when anchors are unique: with singleton groups, GiGPO
  behaves like episode-only credit regardless of `step_advantage_w`.
- Anchor quality depends entirely on the fingerprint; states that differ only in irrelevant
  content fall into different groups, while semantically different states that hash alike
  are grouped together.
- Credit is still assigned by grouping within the current batch, so the signal depends on
  batch composition and on how many rollouts per task instance are sampled.
- No use of trusted-terminal or evaluation-fingerprint metadata means an unreliable terminal
  outcome propagates directly into both the episode and step terms.

## Calibration Guidelines

- Check `anchor_group_count` and `step_adv_nonzero_frac` before touching `step_advantage_w`:
  if the non-zero fraction is near zero, increasing the weight amplifies a signal that barely
  exists.
- Prefer one change at a time (`gamma`, `step_advantage_w`, or `mode`) so a later window
  comparison can attribute the observed delta.
- Treat `anchor` switching as a semantic change of the grouping key, not a small tuning step:
  it should be recorded as a distinct configuration in history.
- Keep `grouping` and `anchor_scope` at `exact` / `full`; other values are not implemented and
  are rejected.
- Expectations about how these changes affect success rate remain hypotheses until a verified
  window comparison exists.

## Empirical Experience

_No verified experience recorded yet._
