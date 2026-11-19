# RewardFlow Credit Assignment Method

RewardFlow (论文 / paper Eq.1-6) 作为 `credit_assignment` adv_estimator 的一个 registered method。
Step-level 信号来自 **uid 组内 workbook state graph 的 potential difference**,不是 observation 文本分组。

```
A(row, t) = alpha_traj * A_episode(row) + alpha_act * A_action(row, t)
A_action  = normalize_{(uid, pre-state)} [ Phi(s_{t+1}) - Phi(s_t) ]
Phi(s)    = gamma^{d_G(s, S_success)}      # reverse BFS, 与官方 max-plus iteration 在 binary seed 下等价 (App. E.7)
```

## File layout

| File | Role |
|---|---|
| `config.py` | `RewardFlowConfig`;未知 param 直接报错 |
| `graph.py` | `StateGraph`:per-(uid, initial-state) 有向图 + reverse-BFS potentials |
| `state.py` | transition metadata 规范化;unknown state 严格区分、永不合并 |
| `algorithm.py` | `@register_credit_assignment("rewardflow")`;纯计算模块 |

Rollout 侧采集在 `verl/experimental/agent_loop/tool_agent_loop.py`
(`collect_step_states` 门控);fingerprint 工具在 `verl/utils/workbook_fingerprint.py`
(parse → canonical repr → sha256;原始 xlsx 字节因 zip metadata 非确定,禁止直接 hash)。
Adapter (`verl/trainer/credit_assignment/adapter.py`) 将 `ca_step_*` 字段透传进 `inputs.extra`。

## Metadata schema (extra_fields → non_tensor_batch)

每个 rollout trajectory 输出(仅 `collect_step_states` 启用时):

```jsonc
// list 长度必须 == len(ca_step_spans), 每 assistant decision step 一条
"ca_step_transitions": [
  {
    "pre":  "sha256:<hex>" | null,   // tool batch 执行前的 workbook fingerprint
    "post": "sha256:<hex>" | null,   // tool batch 完成后立即采集 (含 truncation 路径)
    "executed": true,                // 该 step 是否真的有 tool 调用被执行
    "status": "success",             // tool batch 聚合 status
    "graph_valid": true,             // 是否可作为 graph edge (由 rollout 侧 policy 决定)
    "reason": null,                  // 不可入图原因: state_unknown / external_writeback_unverified / ...
    "tool_status": [{"tool": "write_range", "status": "success"}]
  }, ...
],
"ca_initial_state":  "sha256:<hex>" | null,   // workspace 准备后的初始 fingerprint
"ca_final_state":    "sha256:<hex>" | null,   // rollout 结束时的 fingerprint (success anchor 用)
"ca_terminal_trusted": true | false,          // fingerprint 是否可视为 reward 评估对象 (见下节; 与 graph_valid 分离)
"ca_terminal_reason": "final_answer" | "response_length" | "max_assistant_turns" | ...,
"ca_state_stats": {"compute": 7, "cache_hit": 2, "unknown": 0, "parse_ms": 123.4}
```

Row 为 `None`(采集缺失)与 `[]`(无 step)在 adapter 中严格区分。RewardFlow 对 metadata
缺失**拒绝 silent episode-only degradation**:key 缺失、行级 `None`、数量不一致均为
结构性错误直接 raise(可能原因:mixed agent-loop batch 或采集开关部分生效)。
仅当单个 transition 的 fingerprint 为 unknown(行内字段级)时,才排除该 transition、
其余正常计算。

## Rollout-side policies

- **Fingerprint failure / 超限 (file >100MB 或 >400k cells) → `unknown`**,不做 observation fallback;
  unknown 端点的 transition 不入图,不参与归一化(action advantage 0),不污染同组统计。
- **Self-loop** (pre == post):edge 不入图(对齐官方 `clean_trajectory` discard),
  但 r̃=0 作为样本参与 (uid, pre-state) normalization 分组 — "无状态变化"是有效比较信号。此为有意策略。
- **Empty-cell formatting 计入 fingerprint**:值 `None` 但 `_style` 已分配 (fill / border /
  alignment / number_format 等) 的 cell 以 `e:` 标记参入 canonical representation
  (yellow-fill 空格 ≠ 无格式空格);从未分配 style 的 cell 跳过,避免碎片化。
  Merged range 非 anchor 格的显式格式同样计入。
- **Row/column dimensions 计入 fingerprint**:`format_range` 可直接写 `row_dimensions[].height` /
  `column_dimensions[].width` 与 hidden,这些是可成为任务目标的 workbook state,故纳入稀疏 dim
  (hidden / height / width / customHeight / bestFit + sheet_format 默认行高列宽)。仅采集已存在的
  dim entry (`.items()` 不创建)。**未纳入**:dimension-level *style index* (整行/列套用的 numFmt/字体)
  与 outline level — 理由:openpyxl 的 style 为跨文件不稳定的 index,直接 hash 会破坏决定论;而
  format_range 的格式化是逐 cell 写入(已被 cell-level style 覆盖)。风险:仅改整列 numFmt 而未改任何
  cell 显式格式的极端场景可能不被区分 (见 Deviations)。
- **`code_interpreter` 等外部 sandbox 写回时机未验证** → transition `graph_valid=false`,
  `reason="external_writeback_unverified"`(conservative)。recalc 的本地原子写回已验证,可入图。
- **Terminal state trust (`ca_terminal_trusted`)**:known (final fingerprint 采集成功) 且最后一次执行
  tool 的 step 未触达 `_UNVERIFIED_WRITEBACK_TOOLS` (=code_interpreter) 时为 true。in-process / recalc
  写回同步且本地重读即评估对象,故可信;不假定 fingerprint 存在即代表远程 writeback 已完成。此判定
  与单条 transition 的 graph_valid **语义分离**(parse 失败等中间 excluded step 不影响 terminal 可信)。
- **Truncation 路径**同样在 tool batch 后采集 post state (`response_length_after_tools`)。
- Workspace 在 reward 打分后即被删除,因此 fingerprint 必须 rollout 侧完成 — 已如此实现。

## Success anchoring 与 graph connectivity

- Spreadsheet-RL 终局 reward 严格 binary {0.0, 1.0} → anchor 条件 `reward >= success_threshold(1.0) - tolerance`。
- **Anchor 的三个必要且充分条件**:(1) 成功 reward;(2) `ca_final_state` known(非 None);
  (3) `ca_terminal_trusted` 为 true。三者同时满足才以 terminal workbook state 为 Φ=1 锚点。
- **graph-valid edge 是否存在 NOT 是锚定前置条件**:terminal 无 incoming eligible edge 的孤立
  anchor 仍是合法 anchor(Φ=1),只是不向其他 state 传播 — 这属于 connectivity 诊断,不是错误。
- **禁止的回退**:`ca_final_state` 为 unknown 或 untrusted 时,**不回退锚到 last eligible post 或
  任何更早的中间 state**(那可能不是成功 state),也不把 excluded transition 恢复为通向 success 的
  edge。unknown → 计 `anchors_missing`;known 但 writeback 完成时序不可证 → 计 `anchors_untrusted`;
  两者均退化为 episode-only(action advantage 0),不凭空赋予 potential。
- **Connectivity 只诊断不补馈**:propagation 行为不变,仅记录 `anchors_without_valid_incoming_edge` /
  `unreachable_valid_states` 统计;不给不可达 state 凭空赋 potential。
- reward 0.0 **不作为 negative anchor**(可能是评估基础设施失败,语义不确定)。

## Deviations from 论文 / 官方 reference implementation

以下均为有意偏离,**不应被描述为与官方实现完全相同**:

1. **State identity = workbook content fingerprint** (官方: observation 文本 hash)。理由:本环境
   observation 是 tool 摘要文本,相同文本可对应不同 workbook state,会污染 graph 拓扑。
2. **Potential = reverse-BFS Φ(s)=γ^{d(s,S_succ)}** (官方: max-plus value iteration,
   `V[s]=max(V[s], γ·max_succ V[s])`,max_iter 上限)。**等价条件**:binary seed(anchors=1,
   其余无种子)且 max-aggregation — 此时不动点唯一且等于 γ^{最短跳数}(论文 App. E.7)。本实现
   兼得了精确收敛(O(V+E),无迭代上限);非 binary / 非 max 聚合场景官方行为与 BFS 不同。
3. **Singleton episode group 保留 raw reward**(官方 `episode_norm_reward` 同义:baseline=0 → adv=raw)。此为对齐项,列出以防误读为 GRPO-style /std。
4. **Final-answer step 的 action advantage = 0**(官方将 last step-row 的 step_reward 设为原始
   terminal reward 作为 fallback,`core_rewardflow.py` L408-409)。理由:本实现中 final answer 不
   构成 state transition,强行赋 raw reward 会把 episode 信号重复计入 action 通道。代价:官方
   设计中 last action 获得的最大信号在本实现中完整由 episode 通道承担。
5. **Code-interpreter transitions 保守排除 + terminal trust 显式化**:`ca_terminal_trusted` 与 transition
   `graph_valid` 语义分离—仅“最后一次执行 step 触达 unverified-writeback tool”时降级 terminal 为
   untrusted(不因存在任意 excluded transition 而自动不信任)。代价:经由外部 sandbox 完成最后修改的
   成功 trajectory 不锚定,其所在 (uid, init) 组 action 信号可能受限或归零(诊断字段可观测) —
   精度/保守权衡,不是实现疏漏,也不声称已保证外部 writeback 完成。
6. **Empty-cell formatting 参与 state identity**(官方 observation-hash 无此概念)。提升拓扑保真;
   代价是纯格式冗余(如全网格刷格式)会计入 cell cap。
7. **数值/格式 identity 的已知偏差**:openpyxl 将数值统一读为 float(实测 int/1.0 同指纹);若不同
   writer 将日期/布尔存为不同类型,逻辑等价 workbook 可能不同指纹(方向安全:分裂而非误合并)。
   number_format 中 `None`/"General" 归一为同一默认,但 `@` (Text) 与 "General" **语义不同,已区分为不同
   fingerprint**(不再同价)。自定义 format code 字符串差异保留。dimension-level style index 未纳入(见
   Rollout-side policies),因此“仅改整行/列套用的 numFmt 而不动任何 cell 显式格式”极端场景可能不被区分。

## Configuration

```yaml
algorithm:
  adv_estimator: credit_assignment
  credit_assignment:
    method: rewardflow          # 自动启用 collect_step_states
    collect_step_states: null   # null=auto;显式 true/false 覆盖
    params:
      gamma: 0.95               # potential decay
      alpha_act: 1.0            # Eq.6 action 权重
      alpha_traj: 1.0           # Eq.6 episode 权重
      normalization: mean_norm  # 官方默认;mean_std_norm 可选
      success_threshold: 1.0    # binary reward 下的成功判定
```

Rollback to GiGPO:`method: gigpo` + 删除/置空新增 params 即可;`collect_step_states`
对 GiGPO 无影响(默认 auto-off,不产生任何新 metadata)。GiGPO 代码路径零改动。

## Diagnostics

`VERL_LOGGING_LEVEL=INFO` 时每次 compute 输出一行:groups / nodes / edges / self_loops /
shared_in_nodes / anchors(_missing, _untrusted, _no_incoming) / unreachable_states / eligible /
unknown_state / invalid / zero_action_steps / adv mean·std。其中 terminal fingerprint 相关分两类:
`anchors_missing` = terminal fingerprint **本身缺失** (采集失败/None);`anchors_untrusted` =
fingerprint 存在但 **writeback 完成时序不可证** (最后修改来自 code_interpreter)。两者都不锚定、
不回退。`anchors_no_incoming` (= `anchors_without_valid_incoming_edge`) 为孤立 anchor 数(合法但不
传播),`unreachable_states` 为 Φ≤0 的 valid state 数。Rollout 侧每行带 `ca_state_stats`。
_untrusted · _no_incoming · unreachable 持续偏高说明 code_interpreter 依赖重或 unknown 比例高
(action 信号受限)。

## Validation status

静态检查 (py_compile / import wiring / openpyxl 指纹 CPU 决定论脚本) 通过;**GPU rollout +
训练端到端尚未在本环境运行**(无 GPU)。上机验证 checklist:
1. 小 batch rollout 确认 `ca_step_transitions` 数量 == `ca_step_spans`、unknown 比例可接受;
2. INFO 日志确认 anchors>0、edges>0,并观察 anchors_untrusted / unreachable_states;
3. `method=gigpo` 对照运行确认输出与本改动前一致。
