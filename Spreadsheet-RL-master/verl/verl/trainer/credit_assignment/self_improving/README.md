# Self-Improving Credit Assignment (MVP)

本目录实现的是一个 **慢循环 (Slow Loop)**：在不改动 RL 快循环的前提下，周期性观察训练窗口，
评估上一次 Credit Skill 变更的效果，把经验写入版本化的 Markdown 知识文件，再让外部 LLM
在候选 Skill 之间做一次 *Selection & Calibration* 决策，并由 Driver 在下一个 training step
生效。

整个 cycle：**Observe → Evaluate → Learn from Experience → Select & Calibrate → Apply → Observe Again**

---

## 1. Overview

### 1.1 Research Objective

Spreadsheet-RL 的 advantage 由 Credit Assignment 模块（GRPO / GiGPO / RewardFlow）决定，
但哪一个更适合当前训练阶段，事先并不知道。本项目做的是：在一次训练 run 内部，让
Credit Skill 及其参数成为 **可被 LLM 周期性调整的对象**，同时保证

- 训练数值路径仍然完全走现有 `credit_assignment` 库（不新写公式、不做算法组合）；
- 决策、证据、知识版本全部可追溯、可复现（含 checkpoint 对齐的 resume）；
- 任何 Slow Loop / LLM 故障都不会打断 RL 训练，也 **永不自动回滚**。

### 1.2 Fast Loop（未改动）

```
rollout (tool_agent, 产出 ca_step_* metadata)
  → reward (Windows Excel 重算)
  → advantage = credit_assignment.compute(method, ...)
  → 现有 vanilla token-level PPO loss + optimizer
```

快循环里唯一与本项目相关的对象是 `algorithm.credit_assignment` 配置节点：
Slow Loop 只重写该节点的 `method` / `params`，因此 Skill 切换是通过现有 estimator 生效的。

### 1.3 Slow Loop

每个 training step 结束后 `on_step_end()` 采集一次 step summary；距上次介入累计
`interval_steps`（默认 20）个 step 后，按以下顺序执行一次介入：

1. 聚合当前窗口 → `TrainingContext`
2. 评估上一次 decision（`DecisionEvaluator`）
3. 知识更新（Knowledge LLM 提议 + Python 门禁校验）
4. 新 decision（Controller LLM + 运行时三重门禁）
5. 追加 history / contexts、写 `state.json`（必要时写 checkpoint sidecar snapshot）
6. 重置窗口，从 **下一个** step 使用新 Skill

### 1.4 Skill Knowledge

每个候选 Skill 对应一份 Markdown（`skills/<method>.md` → run-local 副本）。前 8 个章节是人写的
静态说明，`## Empirical Experience` 与 `## Calibration Guidelines` 由本项目追加。
LLM 只能提议"解释/经验文本"，**所有数值一律取 Python 侧的 evaluation 记录**。

### 1.5 Overall Workflow

```
launch (method=rewardflow, collect_step_states=True)   ← 只为 worker 冻结 metadata 采集范围
  → initialize: SkillRuntime 校验 bootstrap + 应用 initial_skill / 恢复 snapshot
  → step 1..N: fast loop + ContextCollector.observe
  → 每 interval_steps: evaluate → knowledge → select → apply
  → finalize: 落盘 state + finalized 记录
```

---

## 2. Architecture

### 2.1 Directory Structure

```
self_improving/
  config.py           # self_improving.* schema、未知 key 报错、跨字段校验
  runtime.py          # SkillConfig / SkillRuntime / CALIBRATION_SPACE / metadata bootstrap 校验
  context.py          # ContextCollector：step summary、窗口聚合、轨迹采样、metadata health
  llm_client.py       # OpenAI 兼容客户端 + Mock 客户端（仅标准库）
  controller.py       # CreditSkillController：单次 LLM 调用完成 selection + calibration
  evaluator.py        # DecisionEvaluator：窗口对比 + evidence quality + outcome
  knowledge.py        # KnowledgeStore：版本化 Markdown 知识 + 更新门禁
  persistence.py      # SelfImprovingStore：state / snapshot / history / contexts / run 隔离
  slow_loop.py        # 编排层，trainer 只接触 4 个入口
  prompts/            # controller.md, knowledge_update.md
  skills/             # grpo.md, gigpo.md, rewardflow.md（模板）
  examples/           # run_self_improving.sh, run_smoke_mock.sh, mock_responses.jsonl
  README.md           # 本文件
  IMPLEMENTATION.md   # 实际变更范围 / Integration Point / 测试结果 / 产物样例
```

### 2.2 Module Responsibilities

| 模块 | 责任 | 关键约束 |
|---|---|---|
| `config.py` | 解析 + 校验 `self_improving` 节点 | 未知 key 直接 `SelfImprovingConfigError`；`client=api` 时缺 env 变量在开训前报错 |
| `runtime.py` | 候选集合 / 参数域 / metadata 覆盖三重门禁下切换 Skill；写回 `algorithm.credit_assignment` | `adv_estimator != credit_assignment` 时启动即报错 |
| `context.py` | 有界 Training Context（窗口指标、诊断、轨迹样本、metadata health） | 全部字段截断 + JSON 可序列化；不复制整批数据 |
| `llm_client.py` | 一次 HTTP chat completion；解析任意 JSON 围栏 | 不重试 4xx；key 只进 header；无新依赖 |
| `controller.py` | 渲染 prompt → 一次调用 → 结构化 decision | 任何失败都返回 `status=fallback` 的 *keep* 决策，不改 Skill |
| `evaluator.py` | baseline window vs observed window，训练指标 + 复用已有 validation | 只产出 observational 结论，措辞受 `summary` 模板限制 |
| `knowledge.py` | run-local 知识文件、版本快照、index、prompt 预算 | 数值取 evaluation；target 必须等于本窗口实际执行的 skill；按 decision_id 去重 |
| `persistence.py` | `state.json` / `snapshots/step_*.json` / `history.jsonl` / `contexts.jsonl` / `config.json` | 原子写；fresh run 遇到旧状态整目录归档到 `previous/`，不覆盖 |
| `slow_loop.py` | 调度与编排；错误隔离 | 可恢复的介入失败只回退为 keep、当前 Skill 不变，并在**下一个预定周期重试**（不再有"连续 3 次失败即停用"）；`SelfImprovingStateError` / `SelfImprovingConfigError` / `MemoryError` 等致命错误照常向上抛出，不被吞掉 |

### 2.3 Trainer Integration Points

`verl/trainer/ppo/ray_trainer.py`（`RayPPOTrainer.fit()`）中只有 6 处 thin hook（4 个入口方法，
`finalize()` 在两个出口各调一次），全部在 `self_improving.enabled` 判断之内（初始化点额外加了
`trainer.val_only` 守卫）：

| 位置 | 调用 | 作用 |
|---|---|---|
| `_load_checkpoint()` 之后 | `SelfImprovingLoop()` + `initialize(self)` | bootstrap 校验、应用 initial/restored Skill |
| 初始 validation 之后 | `observe_validation(val_metrics, global_steps)` | 把 val#0 记为比较基线 |
| `_save_checkpoint()` 之后 | `on_checkpoint_saved(self.global_steps)` | 标记本 step 需要 sidecar snapshot |
| step 结束、`logger.log` 之前 | `on_step_end(batch, metrics, timing_raw, global_step, is_last_step)` | 采集 + 按需介入 |
| `is_last_step` 分支 / epoch 循环退出 | `finalize()`（共 2 处） | 落盘最终状态与 `finalized` 记录 |

`verl/trainer/config/ppo_trainer.yaml` 只新增了 `self_improving:` 配置节点（默认 `enabled: false`）。
除此之外未改动任何现有文件。

`trainer.val_only=true` 时 Slow Loop 不会初始化（也不会 `finalize()`）：这种跑法没有训练窗口，
也就没有介入语义；若照样启动会把上一 run 的 `outputs/self_improving/` 归档并写下无意义的状态。

### 2.4 Runtime Skill Switching

`SkillRuntime.apply()` 在写回配置前依次校验：method 已注册 → 在 `candidate_skills` 内 →
参数可规范化（`canonical_params`，未知/越界参数报错，未提供项取默认值）→ 该方法所需的
metadata 都在 launch 时冻结的采集范围内。四项全过才写
`algorithm.credit_assignment.{method,params}`，因此下一个 step 的 advantage 计算立即使用新 Skill。
Decision 在 step *末尾* 生效，实际首个使用它的 step 是 `global_step + 1`。

### 2.5 Metadata Bootstrap Strategy

worker 在初始化时按当时的 `algorithm.credit_assignment` 冻结采集范围，Driver 之后换 Skill 不会
重新采集。因此启动示例统一用 **rewardflow（要求最多 metadata 的候选）** 作为 bootstrap 配置：

```
algorithm.adv_estimator=credit_assignment
algorithm.credit_assignment.method=rewardflow      # 仅用于采集范围
algorithm.credit_assignment.collect_step_states=True
self_improving.initial_skill.method=grpo           # 真正的第一个 step 的 Skill
```

两个配置含义不同，必须区分。`algorithm.credit_assignment.method` / `collect_step_states` 只决定 worker 的
**metadata 采集范围**（bootstrap）；真正用于训练第一个 step 的 Skill 与其参数来自
`self_improving.initial_skill.{method,params}`，active Skill 之后由验证过的 Controller 决策决定，
resume 时优先采用对应 snapshot 的 active Skill。

SI 启用时，若在 `algorithm.credit_assignment.params` 里写了 **非默认的算法参数**，会被
`_preflight()` 判为配置错误并拒绝开训（这些参数会被静默忽略、既不进 bootstrap 采集也不会传给初始
Skill）：训练用的参数请显式写在 `self_improving.initial_skill.params`。Hydra schema 自带的默认值不会
被误判，`params: {}` 等默认状态照常放行；bootstrap 参数**不会**自动 merge/继承到初始 Skill。

`bootstrap_metadata_signature()` 在 `initialize()` 中、应用初始
Skill **之前** 冻结一次，之后所有 context / state / config 记录都复用这份签名（不会随 active
Skill 漂移）。候选 Skill 若需要超出采集范围的 metadata，会在开训前 `SkillValidationError`。

---

## 3. Configuration

`self_improving.*`（Hydra override，全部为新增 key，无需 `+`）：

| Key | Default | 说明 |
|---|---|---|
| `enabled` | `false` | 总开关；false 时 trainer 行为与之前完全一致 |
| `interval_steps` | `20` | 两次介入之间完成的 training step 数 |
| `initial_skill.method` | `grpo` | 必须属于 `candidate_skills`；resume 时被恢复的 Skill 覆盖 |
| `initial_skill.params` | `{}` | 可只给部分参数，Python 补全为 canonical 值 |
| `candidate_skills` | `[grpo, gigpo, rewardflow]` | LLM 可选择的范围 |
| `context.max_trajectory_samples` | `4` | 每窗口保留的代表性轨迹数 |
| `context.max_steps_per_trajectory` | `20` | 每条轨迹渲染的 step 数 |
| `context.max_text_chars_per_field` | `1000` | 单个自由文本字段截断长度 |
| `context.max_context_chars` | `40000` | prompt 中 context 段的硬上限 |
| `llm.endpoint_env` | `SPREADSHEET_RL_SI_LLM_URL` | 环境变量 **名**，不是值 |
| `llm.model_env` | `SPREADSHEET_RL_SI_LLM_MODEL` | 同上 |
| `llm.api_key_env` | `SPREADSHEET_RL_SI_LLM_API_KEY` | 同上；缺失时按无鉴权请求 |
| `llm.timeout_seconds` | `60` | 单请求超时 |
| `llm.max_retries` | `1` | 仅网络/超时/解析失败重试；4xx 不重试 |
| `llm.temperature` | `0.0` | 取值范围 `[0, 2]` |
| `llm.client` | `api` | `api` \| `mock` |
| `llm.mock_script` | `null` | `client=mock` 时按行读取的响应脚本（JSONL） |
| `evaluation.use_training_metrics` | `true` | 窗口训练指标对比 |
| `evaluation.use_existing_validation` | `true` | 复用 trainer 已有 validation，不额外 `_validate()` |
| `evaluation.validation_metric_key` | `null` | 显式指定 val 指标；null 时自动取唯一的 `val-core/.../mean@N` |
| `evaluation.min_window_steps` | `5` | 短于此的窗口判为 `insufficient_evidence` |
| `evaluation.improvement_threshold` | `0.01` | 正/负判定的相对阈值 |
| `knowledge.enabled` | `true` | 关闭后知识文件在整个 run 内冻结 |
| `knowledge.update_interval` | `1` | 每 N 次介入做一次知识更新 |
| `knowledge.max_prompt_chars` | `20000` | 嵌入 prompt 的知识预算 |
| `persistence.enabled` | `true` | 关闭后不写任何状态 |
| `persistence.strict_resume` | `true` | 找不到与 checkpoint 对齐的 snapshot 时拒绝恢复 |

### 3.1 LLM 环境变量

```bash
export SPREADSHEET_RL_SI_LLM_URL="https://<host>/v1/chat/completions"   # 只填到 /v1 会自动补全
export SPREADSHEET_RL_SI_LLM_MODEL="your-model"
export SPREADSHEET_RL_SI_LLM_API_KEY="your-key"      # 只进请求头，不写日志/prompt/state
```

### 3.2 可校准参数（CALIBRATION_SPACE）

`exposed=false` 的参数不进入 prompt，LLM 无法改动（保持 bootstrap/驱动值）；越界或未注册参数
在 runtime 门禁处被拒绝并回退为 keep。

| Skill | LLM 可校准 | 固定（不暴露） | 所需 metadata |
|---|---|---|---|
| `grpo` | `norm_adv_by_std_in_grpo` | `epsilon` | 无 |
| `gigpo` | `gamma`, `step_advantage_w`, `mode`, `grouping`, `anchor`, `anchor_scope` | `epsilon`, `include_truncated_in_baseline` | `step_spans`, `step_observations`, `step_transitions` |
| `rewardflow` | `gamma`, `alpha_act`, `alpha_traj`, `normalization` | `epsilon`, `success_threshold`, `success_tolerance` | `step_spans`, `step_transitions`, `terminal_trusted`, `eval_fingerprint` |

---

## 4. Usage

以下命令都在 `Spreadsheet-RL-master/` 下执行；脚本只追加 override，复用 `deploy/run_grpo.sh`
→ `configs/qwen3-4b.sh` → `verl.trainer.main_ppo` 这条已验证路径（现有 launch script 未修改）。

```bash
SI=Spreadsheet-RL-master/verl/verl/trainer/credit_assignment/self_improving/examples

# 1) 标准 Self-Improving 训练
export SPREADSHEET_RL_SI_LLM_URL=... SPREADSHEET_RL_SI_LLM_MODEL=... SPREADSHEET_RL_SI_LLM_API_KEY=...
bash "$SI/run_self_improving.sh"

# 2) 初始 Skill 改为 GiGPO（可加部分参数）
INITIAL_SKILL=gigpo INITIAL_PARAMS='{"gamma":0.9}' bash "$SI/run_self_improving.sh"

# 3) 改 Slow Loop 周期
SI_INTERVAL=5 bash "$SI/run_self_improving.sh"

# 4) 限制候选 Skill
CANDIDATES="[grpo,gigpo]" bash "$SI/run_self_improving.sh"

# 5) 关闭 Knowledge Evolution（仍会切换 Skill，只是不更新知识）
KNOWLEDGE=false bash "$SI/run_self_improving.sh"

# 6) Mock LLM smoke（不发起任何外部请求；DRY=1 只打印命令）
LLM_CLIENT=mock bash "$SI/run_self_improving.sh"
DRY=1 bash "$SI/run_smoke_mock.sh"        # GPU smoke：6 step / interval 2 / mock reward

# 7) Resume：从 policy checkpoint 恢复（Slow Loop 状态按同 step 的 snapshot 复原）
RESUME_PATH="$CKPTS_DIR/global_step_20" bash "$SI/run_self_improving.sh"
#    GPU smoke 的第二段：bash "$SI/run_smoke_mock.sh" --resume

# 关闭 Self-Improving：见下方 §4.1（走 vanilla GRPO，不要引用 SI 内部的 GRPO wrapper）
```

### 4.1 关闭 Self-Improving（vanilla GRPO）

SI 关闭时 trainer 行为与接入前完全一致。要跑 **vanilla GRPO** 请使用现有内置估计器：

```bash
algorithm.adv_estimator=grpo \
self_improving.enabled=false
```

注意区分两个不同的 "GRPO"：

- **vanilla GRPO**：`algorithm.adv_estimator=grpo`，走 `core_algos.compute_grpo_outcome_advantage`，
  是仓库既有的、SI 未启用时也存在的路径。
- **SI 内部 GRPO wrapper**：`credit_assignment/methods/grpo`，只在 `self_improving.enabled=true` 时由
  `runtime.py` 通过 `importlib` 显式注册；SI 关闭时它**未注册**，因此
  `algorithm.adv_estimator=credit_assignment algorithm.credit_assignment.method=grpo`
  在关闭状态下会因 "not registered" 失败，不是有效的 vanilla 运行方式。

`run_smoke_mock.sh` 会复用 `gigpo_smoke/` 的环境（venv、9B 模型、8 任务子集、mock reward）；
本机若无这些环境，脚本以 exit 3 报 `BLOCKED` 并给出缺失项 —— 此时它属于"未执行"，不代表验证通过。

CPU 测试（无需 GPU）：

```bash
cd Spreadsheet-RL-master/verl
PYTHONPATH=. python -m pytest tests/trainer/ppo/test_self_improving_*.py -q   # 151 passed
PYTHONPATH=. python -m pytest tests/trainer/ppo -q                            # 269 passed（含既有回归）
```

---

## 5. Output

所有产物在 `trainer.default_local_dir/self_improving/` 下，**run-local**，不跨 run 共享知识。

```
config.json          # 冻结的 SI 配置 + experiment identity + metadata bootstrap 签名
state.json           # 最新一步的完整 Slow Loop 状态
history.jsonl        # append-only 事件流（含 run_id / ts）
contexts.jsonl       # 每次介入用到的完整 Training Context
snapshots/step_XXXXXX.json   # 与 policy checkpoint 同 step 对齐的状态副本（resume 用）
knowledge/current/<skill>.md # 运行时可写的知识副本
knowledge/versions/vNNNN/    # 每个版本的完整快照（全部候选 skill）
knowledge/index.json # 版本索引：version / step / decision_id / evaluation_id / lineage
previous/<run_id>/   # 在同目录起 fresh run 时，上一 run 的整目录归档
```

`state.json` 关键字段：`completed_global_step`, `active_skill`/`active_parameters`（下一步将用）,
`executed_skill`/`executed_parameters`（本 step 实际用）, `last_intervention_step`,
`intervention_count`, `window_start_step`, `current_window_summary`, `pending_decision`,
`latest_validation`, `validation_baseline`, `knowledge_version`,
`windows_since_knowledge_update`, `metadata_collection_signature`, `skill_id_codes`,
`run_id`, `lineage`, `config_version`, `config_summary`, `consecutive_intervention_failures`
（仅供诊断，不再是停用条件）, `persistence_failures`, `state_misaligned`,
`last_persistence_error`。历史 state.json 里若仍带 `slow_loop_disabled`，`initialize()` 只打印一条
兼容提示、不据此停用（旧状态不再表示停用意图）。

`history.jsonl` 的 `record_type`：`initialized`, `config_change`, `decision`, `evaluation`,
`knowledge_attempt`, `knowledge_update`, `selection_skipped`, `decision_error`, `window_summary`,
`checkpoint_snapshot`, `error`, `persistence_error`, `finalized`（`contexts.jsonl` 里是 `context`
记录）。已不再有 `disabled` 记录类型。

### 5.1 Metrics（前缀 `self_improving/`）

| Metric | 含义 |
|---|---|
| `executed_skill_id` / `next_skill_id` | 本 step 实际执行的 Skill 序号 / 下一步将使用的 Skill 序号（`candidate_skills` 下标） |
| `intervention_triggered` | 本 step 末尾是否发生介入（0/1） |
| `intervention_count` | 本 run 累计介入次数 |
| `knowledge_version` / `knowledge_updated` | 介入结束后的知识版本 / 本次是否写入经验 |
| `eval_delta_reward` / `eval_delta_success_rate` | 上一次 decision 的窗口差值 |
| `decision_valid` | Controller 决策是否通过全部门禁（0 = keep/fallback/rejected） |
| `metadata_available_ratio` | 当前 Skill 所需 metadata 在该 batch 的平均覆盖率 |
| `metadata_missing_count` / `metadata_partial_count` | 必需 metadata 中"完全未采集" / "采集了但覆盖不全" 的项数（§M-3 诊断） |
| `llm_call_seconds` / `slow_loop_seconds` | Controller 调用耗时 / 本 step 慢循环总耗时 |
| `consecutive_failures` | 连续可恢复介入失败次数（诊断用；失败后仍在下一周期重试，不停用） |
| `persistence_failures` | 累计持久化失败次数（诊断用） |

---

## 6. Known Limitations

- **评估是观察性对比**：policy 一直在变、task/trajectory 分布也在变，窗口差值不能证明 Credit Skill
  的因果效果；措辞限定为 *observed improvement / degradation under the current context*。
- **RewardFlow bootstrap 提高 metadata 采集成本**：即使初始 Skill 是 GRPO，worker 仍按 rewardflow
  的范围采集 `ca_step_*` 与 evaluation fingerprint（额外序列化与内存开销）。
- **未修复现有 row-0 reward metadata aggregation 问题**：某些失败/timeout 轨迹会导致
  `eval_fingerprint` 缺失；本项目只在 `metadata_health` 里观测并降级证据质量，不改原代码。
- **外部 LLM 调用会阻塞 Driver**：介入发生在 step 末尾，超时上限 `llm.timeout_seconds`。
- **知识演化是经验文本级 Markdown 追加**：无 RAG、无向量库、无跨 run 共享，也没有统计置信检验。
- **候选 Skill 限于已注册算法**：GRPO / GiGPO / RewardFlow；不做算法组合或代码生成。
- **Resume 精度受限**：只有 policy checkpoint 保存成功的 step 才有 sidecar snapshot；
  非严格恢复会回退到最近 snapshot，并把受影响的 evaluation 标为 `insufficient_evidence`。
- **崩溃时刻仍可能留下"有 policy ckpt、无同 step SI 状态"的中间态**：`_save_checkpoint()` 与随后的
  step-end 处理不是原子事务。本轮只保证正常 step-end 后的对齐，以及缺 snapshot 时显式失败
  （不静默采用更新的状态），未改造现有 checkpoint 写入顺序。
- **GPU / 外部 API 未在本轮验证**：CPU 集成测试与 mock 路径已验证；真实 GPU smoke 与外部
  LLM 端点的执行结果需按实际运行记录（见 `IMPLEMENTATION.md` §测试结果的 Not Run / Blocked 项）。
