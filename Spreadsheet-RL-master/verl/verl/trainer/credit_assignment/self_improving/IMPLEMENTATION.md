# Self-Improving Credit Assignment — 实现记录

本文件记录本次实现（명령서.md 的 MVP）在仓库中的**实际**落地情况：代码结构、Integration Point、
每个阶段的完成状态、真实测试命令与结果、以及从测试中导出的产物样例。
面向使用者的说明文档见 [README.md](./README.md)；本文件只写"实际做了什么、验证到什么程度"。

生成时间基准：全部测试在本机（macOS，无 GPU）以 CPU 方式执行。

---

## 1. 变更范围

### 1.1 新增文件

| 路径 | 行数 | 责任 |
|---|---|---|
| `self_improving/__init__.py` | 16 | 只导出 `SelfImprovingConfig` / `SelfImprovingLoop` / `SkillConfig` / `SkillRuntime` / `load_self_improving_config` |
| `self_improving/config.py` | 293 | `self_improving.*` schema、未知 key 报错、跨字段校验、env 变量存在性预检 |
| `self_improving/runtime.py` | 465 | `SkillConfig` / `SkillRuntime` / `CALIBRATION_SPACE` / `bootstrap_metadata_signature()`；写回 `algorithm.credit_assignment` |
| `self_improving/context.py` | 673 | `ContextCollector`：step summary、窗口聚合、代表性轨迹、metadata health、字符预算 |
| `self_improving/llm_client.py` | 241 | OpenAI 兼容客户端（仅标准库）+ `MockLLMClient` + 错误分类 + JSON 提取 |
| `self_improving/controller.py` | 307 | `CreditSkillController`：单次 LLM 调用产生 selection + calibration；schema/语义门禁；失败回退 keep |
| `self_improving/evaluator.py` | 355 | `DecisionEvaluator`：baseline/observed 窗口对比、validation 复用、outcome + evidence quality、必需 metadata 的 missing / partial 分类 |
| `self_improving/knowledge.py` | 452 | `KnowledgeStore`：run-local Markdown、版本快照、index、更新门禁、prompt 预算 |
| `self_improving/persistence.py` | 365 | `SelfImprovingStore`：state / snapshot / history / contexts / config、原子写、run 归档、`config_diff` / `experiment_identity` |
| `self_improving/slow_loop.py` | 973 | 编排层：调度、错误隔离与重试、指标、resume 校验、finalize |
| `self_improving/prompts/controller.md` | 93 | Controller prompt 模板（占位符由 Python 填充） |
| `self_improving/prompts/knowledge_update.md` | 68 | Knowledge 更新 prompt 模板 |
| `self_improving/skills/grpo.md` | 100 | GRPO 静态知识模板（8 个必需章节） |
| `self_improving/skills/gigpo.md` | 126 | GiGPO 静态知识模板 |
| `self_improving/skills/rewardflow.md` | 129 | RewardFlow 静态知识模板 |
| `self_improving/examples/run_self_improving.sh` | 83 | launch 示例（只追加 Hydra override） |
| `self_improving/examples/run_smoke_mock.sh` | 121 | GPU smoke（mock reward + mock controller，复用 `gigpo_smoke/` 环境） |
| `self_improving/examples/mock_responses.jsonl` | 7 | smoke 用的脚本化 LLM 响应 |
| `self_improving/README.md` / `IMPLEMENTATION.md` | 345 / 本文件 | 文档 |
| `credit_assignment/methods/grpo/__init__.py` | 5 | 触发 `@register_credit_assignment` 注册（命令书 §8.2 指定路径） |
| `credit_assignment/methods/grpo/algorithm.py` | 84 | GRPO wrapper：直接调用 `core_algos.compute_grpo_outcome_advantage` |
| `credit_assignment/methods/grpo/config.py` | 29 | `GRPOConfig`（`norm_adv_by_std_in_grpo`, `epsilon`） |
| `tests/trainer/ppo/test_self_improving_methods_on_cpu.py` | 357 | §21.2 GRPO 等价性 + §21.3 RewardFlow regression |
| `tests/trainer/ppo/test_self_improving_runtime_on_cpu.py` | 386 | §21.4 runtime switching + metadata bootstrap + §C-1 / §10 |
| `tests/trainer/ppo/test_self_improving_context_on_cpu.py` | 321 | §21.5 context collection |
| `tests/trainer/ppo/test_self_improving_controller_on_cpu.py` | 566 | §21.6 controller + §21.7 evaluation + §M-3 + §10 |
| `tests/trainer/ppo/test_self_improving_knowledge_on_cpu.py` | 507 | §21.8 knowledge + §21.9 persistence |
| `tests/trainer/ppo/test_self_improving_loop_on_cpu.py` | 925 | §21.10 full CPU integration + §21.11 disabled-mode + §M-1/§M-5/§M-6/§M-7 + §10 |

### 1.2 修改的既有文件（全部改动范围）

| File | Modified Function | Change | Reason |
|---|---|---|---|
| `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer.fit()` | +31 / -0 行，6 个 thin hook block | §18 要求的 Slow Loop 接入点；全部在 `enabled`（且非 `val_only`）判断之内 |
| `verl/trainer/config/ppo_trainer.yaml` | — | +105 / -0 行，新增 `self_improving:` 节点（`enabled: false`） | §7 配置 schema |

`git diff --numstat` 实测只有这两个既有文件被改动：

```
105     0       verl/trainer/config/ppo_trainer.yaml
31      0       verl/trainer/ppo/ray_trainer.py
```

未被改动（`git status` 无记录）的关键文件：`core_algos.py`、`credit_assignment/adapter.py`、
`credit_assignment/registry.py`、`credit_assignment/methods/__init__.py`、`methods/gigpo/*`、
`methods/rewardflow/*`、`ppo/dp_actor.py`、rollout / tool / reward 相关模块、`deploy/*`、`configs/*`。
GRPO wrapper 通过 `self_improving/runtime.py:108` 的 `importlib.import_module(...)` 显式注册，
因此不需要修改 `methods/__init__.py`。

---

## 2. Integration Point（实际行号以当前工作树为准）

`ray_trainer.py` 中 6 处 hook，全部只在 `self.config.self_improving.enabled` 为真时构造/调用：

| 行 | 位置 | 代码 |
|---|---|---|
| 1322–1332 | `_load_checkpoint()` / `checkpoint_manager.update_weights()` 之后、初始 validation 之前 | `self._si = None`；enabled 且非 `val_only` 时 `SelfImprovingLoop()` + `initialize(self)` |
| 1360–1361 | 初始 `_validate()` 的 `logger.log()` 之后 | `self._si.observe_validation(val_metrics, self.global_steps)` |
| 1638–1640 | `with marked_timer("save_checkpoint")` 内 `_save_checkpoint()` 之后 | `self._si.on_checkpoint_saved(self.global_steps)` |
| 1714–1721 | step 末尾、`logger.log(data=metrics, ...)` **之前** | `self._si.on_step_end(batch, metrics, timing_raw, global_step, is_last_step)` |
| 1738–1739 | `is_last_step` 分支 `return` 之前 | `self._si.finalize()` |
| 1750–1751 | `epoch` 循环正常退出后（无 `is_last_step` 提前返回时） | `self._si.finalize()` |

设计要点：

- `initialize()` 内部**永不**抛出到 trainer：配置非法 → 打印 + 抛 `SelfImprovingConfigError`（开训前失败，
  这是有意为之）；其余运行期异常在 `on_step_end()` / `finalize()` 内部被捕获并转成 `error` 记录。
- `on_step_end()` 在 `logger.log` 之前调用，因此 Slow Loop 写入的 `self_improving/*` 指标与训练指标同批上报。
- Skill 生效时点：decision 在 step *末尾* 写回配置 → 首个使用新 Skill 的 step 是 `global_step + 1`；
  `state.json` 因此同时记录 `executed_skill`（本 step 实际用的）与 `active_skill`（下一步将用的）。
- `finalize()` 幂等：两个出口都调用，第二次直接返回。

Disabled-mode 兼容性由 `DisabledModeTest`（5 个测试）用 AST 静态解析 `ray_trainer.py` 验证：所有 `self._si`
引用都在 `if self._si is not None` 之内，初始化点在 `enabled` / `val_only` 判断之内，且模块级 import 不引入新依赖。
（本测试环境无 `torchdata`，`import verl.trainer.ppo.ray_trainer` 本身会失败，因此用 AST 而不是运行时导入。）

`trainer.val_only=true` 时不再构造 Slow Loop：这种跑法没有训练窗口 / 也没有介入槽位，若照样 `initialize()`
会把上一 run 的 `outputs/self_improving/` 归档到 `previous/` 并写下无意义的 `state.json`（本轮唯一一次
改动核心文件，+5 行守卫）。

---

## 3. 运行时数据流（实际调用链）

```text
worker 初始化: algorithm.credit_assignment.method=rewardflow + collect_step_states=True
    → rollout 产出 ca_step_* metadata（采集范围在此冻结）
driver:
  fit() → SelfImprovingLoop.initialize()
      → load_self_improving_config()            # config.py: 校验 + env 预检
      → bootstrap_metadata_signature()          # runtime.py: 冻结采集签名（在应用初始 Skill 之前）
      → SkillRuntime.apply(initial/restored)    # runtime.py: 四道门禁 → 写回 algorithm 节点
      → SelfImprovingStore.prepare/write_config # persistence.py
      → KnowledgeStore.bootstrap                # knowledge.py: skills/*.md → knowledge/current/
  每个 step 末尾: on_step_end → _step()
      → ContextCollector.observe(batch, metrics, step)      # 累积 step summary
      → _intervention_due()?  否 → 只写 metrics + _persist()
      → 是 → _intervene():
            aggregate_window() → TrainingContext
            DecisionEvaluator.evaluate(pending_decision, baseline, observed)   # evaluator.py
            _maybe_update_knowledge(): KnowledgeLLM → KnowledgeStore.apply()   # knowledge.py 门禁
            _select(): CreditSkillController.decide() → SkillRuntime.apply()   # controller.py + runtime.py
            _record_intervention() → history.jsonl / contexts.jsonl
            _start_next_window()
            _persist() → state.json（+ checkpoint step 的 sidecar snapshot）
  下一个 step: compute_advantage → credit_assignment.compute(method=新 Skill)
```

LLM 调用点只有两处：`controller.decide()`（selection + calibration 一次调用）与
`_maybe_update_knowledge()`（知识文本提议）。其余数值全部由 Python 计算并覆盖 LLM 输出。

---

## 4. 阶段完成状态

| Phase | 完成条件（命令书 §25） | 状态 | 证据 |
|---|---|---|---|
| 1 Runtime Foundation | 三个 Credit Skill 在统一接口下运行与切换 | 完成 | `test_self_improving_methods_on_cpu.py` + `test_self_improving_runtime_on_cpu.py`（48 passed） |
| 2 Training Context | 生成与真实 Trainer 数据兼容的 bounded context | 完成 | `test_self_improving_context_on_cpu.py`（14 passed）；轨迹样本取自 `non_tensor_batch` 的 `ca_step_*` / `tool_rewards` / `raw_reward` |
| 3 Controller | Mock LLM 下产出并应用有效 decision | 完成 | `test_self_improving_controller_on_cpu.py`（38 passed） |
| 4 Slow Loop | 单个 run 内按周期切换 Skill | 完成 | `test_self_improving_loop_on_cpu.py::FullWorkflowTest`（10 passed） |
| 5 Evaluation & Knowledge | 上一 decision 被评估，有效经验进入下一 decision | 完成 | 同上 + `test_self_improving_knowledge_on_cpu.py`（35 passed） |
| 6 Persistence & Verification | 整个 cycle 可执行、可追踪、可复原 | CPU 部分完成；GPU 未验证 | resume 10 passed；smoke 脚本已提供但**未执行**（无 GPU 环境） |

---

## 5. 测试结果

### 5.1 实际执行

```bash
cd Spreadsheet-RL-master/verl
PYTHONPATH=. python -m pytest tests/trainer/ppo -q
# 285 passed, 6 warnings, 2 subtests passed               （本机：Python 3.10 / torch 2.2.2 / CPU）
```

按文件（同环境）：

| Test | Status | Command | Result |
|---|---|---|---|
| GRPO 数值等价 / mask / dtype / singleton / 参数校验 / 注册（§21.2） | Passed | `pytest tests/trainer/ppo/test_self_improving_methods_on_cpu.py::GrpoEquivalenceTest -q` | 9 passed |
| RewardFlow regression（§21.3） | Passed | `... test_self_improving_methods_on_cpu.py::RewardFlowRegressionTest -q` | 10 passed |
| Runtime switching + metadata bootstrap（§21.4 + §C-1 + §10） | Passed | `pytest tests/trainer/ppo/test_self_improving_runtime_on_cpu.py -q` | 29 passed |
| Training context collection（§21.5） | Passed | `pytest tests/trainer/ppo/test_self_improving_context_on_cpu.py -q` | 14 passed |
| Controller / decision validation（§21.6） | Passed | `pytest tests/trainer/ppo/test_self_improving_controller_on_cpu.py::ControllerDecisionTest -q` | 16 passed |
| Evaluation（§21.7 + §M-3 + §10） | Passed | `pytest tests/trainer/ppo/test_self_improving_controller_on_cpu.py::EvaluatorTest -q` | 17 passed |
| LLM client 失败分层（§21.6 前置） | Passed | `pytest tests/trainer/ppo/test_self_improving_controller_on_cpu.py::LLMResponseLayerTest -q` | 5 passed |
| Knowledge evolution（§21.8） | Passed | `pytest tests/trainer/ppo/test_self_improving_knowledge_on_cpu.py::KnowledgeStoreTest -q` | 20 passed |
| Persistence & resume 原语（§21.9） | Passed | `pytest tests/trainer/ppo/test_self_improving_knowledge_on_cpu.py::PersistenceStoreTest -q` | 15 passed |
| Full CPU integration（§21.10） | Passed | `pytest tests/trainer/ppo/test_self_improving_loop_on_cpu.py -q` | 32 passed（FullWorkflow 10 / LoopSafety 7 / Resume 10 / Disabled 5） |
| Disabled-mode 不变性（§21.11） | Passed | 同上 `DisabledModeTest` | 5 passed |
| 既有 GiGPO 回归（§21.1） | Passed | `pytest tests/trainer/ppo/test_credit_assignment_gigpo_on_cpu.py -q` | 19 passed（+2 subtests） |
| 既有 GiGPO↔vanilla 链接（§21.1） | Passed | `pytest tests/trainer/ppo/test_gigpo_vanilla_link_on_cpu.py -q` | 13 passed |
| 既有 `core_algos`（GRPO 原函数，未被修改） | Passed | `pytest tests/trainer/ppo/test_core_algos_on_cpu.py -q` | 22 passed |
| 新增 SI 测试合计 | Passed | `pytest tests/trainer/ppo/test_self_improving_*.py -q` | **167 passed** |

SI 之外的既有测试**未被修改**以适配新增实现（`git status` 显示 `tests/trainer/ppo/` 下只有 6 个新增
`test_self_improving_*` 文件）。Review 后的 bug fix 只在这 6 个 SI 测试文件内部增删用例；
其中 3 个用例的断言按修正后的语义改写（失败重试不再产生停用标记、落盘失败升级路径、
bootstrap 参数校验的候选集），**没有为迁就实现而放宽断言**。

### 5.2 未执行 / 受阻

| Test | Status | 原因 |
|---|---|---|
| GPU Smoke Test（§21.12，`examples/run_smoke_mock.sh`） | **Not Run** | 本机为 macOS，无 NVIDIA GPU / venv / Qwen3.5-9B 权重 / 子集数据。脚本会在缺项时以 exit 3 报 `BLOCKED`；`DRY=1` 只打印命令。已在本机验证 `bash -n`、`DRY=1` 输出与 BLOCKED 退出码，**不代表训练被验证** |
| Resume 第二段（`run_smoke_mock.sh --resume`） | **Blocked** | 依赖上一项产生的真实 checkpoint |
| 外部 LLM API（`llm.client=api`）真实调用 | **Not Run** | 无端点。该路径只有单元测试（错误分类、超时、4xx 不重试、JSON 围栏解析）与 mock 客户端被验证 |
| 长时训练 / 真实 reward（Windows Excel 重算）下的 Slow Loop | **Not Run** | 无实验环境；`checkpoint 对齐的 sidecar snapshot` 只在 CPU 伪 checkpoint 下验证 |
| Ray / 多 GPU worker 上的 metadata 采集开销实测 | **Not Run** | 需要真实集群；因此 README 中"bootstrap 增加采集成本"仍是设计推断 |

---

## 6. 关键设计决策（命令书未完全指定、按现有 convention 补齐的部分）

1. **bootstrap 签名只在 `initialize()` 里冻结一次**，且在 `SkillRuntime.apply_initial()` 之前取值。
   worker 的采集范围在初始化时确定，Driver 之后换 Skill 不会重新采集；若在应用初始 Skill 之后惰性重算，
   签名会缩水成当前 Skill 的需求集（例如切到 grpo 后只剩 2 项），使 resume 校验与 context 记录失真。
2. **失败的介入同样占用 interval 槽位**（`_attempt_intervention()` 在异常时仍推进
   `last_intervention_step`），但**不会停用 Slow Loop**：可恢复的介入失败只消耗本次槽位，到下一个
   预定周期重试；连续失败计数 `_failure_streak` 仅用于诊断（`self_improving/consecutive_failures`），
   在一次干净介入后归零。旧 schema 里可能残留的 `slow_loop_disabled` 标记只打印一条兼容提示并忽略。
   只有 `FATAL_ERRORS`（`SelfImprovingStateError` / `SelfImprovingConfigError` / `MemoryError`）会向上抛，
   以免把状态不一致、既有算法错误藏进大范 exception handler。
3. **非严格恢复的窗口不作为知识依据**：当 `state.json` 来自与 policy checkpoint 不同 step 的 snapshot
   （`state_source` 存在且 `exact=False`）时，baseline 与 policy 不同源 → evaluation 降级为
   `insufficient_evidence` 且 `eligible_for_knowledge_update=false`。
4. **每个 knowledge 版本目录都是全部候选 Skill 的完整快照**，`index.json` 里的 `skills` 只列本次实际变更项。
   否则 `adopt(v)` 会让未被改动的 Skill 继续读到未来内容（§19.5）。
5. **TrainingContext 的 `notes` 字段承载调度信息**（interval、窗口内已观测 step 数、skill 编号表、
   bootstrap 签名），避免为此再扩一个顶层 schema 字段。
6. **不新增依赖**：LLM 客户端只用 `urllib`；`numpy` 走已有的 jsonable 转换；测试用标准库 `unittest` + 仓库既有 pytest convention。
7. **prompt 与知识文件分离**：`prompts/*.md` 是模板，`skills/*.md` 是知识；Tool Specification 不被复制进知识文件
   （§12.3），两者内容不同源，避免与 `configs/tool/*.yaml` 漂移。
8. **launch 脚本不改既有 `deploy/run_grpo.sh`**：`run_self_improving.sh` 只组装 Hydra override 数组并透传 `"$@"`，
   最终 `bash "${REPO}/deploy/run_grpo.sh" "${OVERRIDES[@]}" "$@"`（§22）。
9. **§C-1：bootstrap 节点与 initial_skill 节点职责分离**。`algorithm.credit_assignment.method` 只决定 worker
   启动时采集哪些 metadata；训练参数只从 `self_improving.initial_skill.params` 走。`_preflight()` 把 launch
   节点里"非默认的 algorithm params"（与该方法默认值不同、且不属于 metadata 语义的项）视为配置错误直接 raise，
   而不是静默丢弃；Hydra schema 自动填入的默认值仍放行，因此默认配置不会被拒。
10. **§M-5：恢复只往回看**。`resume_state()` 只取 `step <= ckpt_step` 的状态；knowledge 按 snapshot 指定的
    version `adopt()`（写入 step 晚于 ckpt 的版本直接报错），不会自动采纳 index 里的最新版；history 预取时传
    `max_step=ckpt_step`；一次完整介入落盘后这些上限才清空。
11. **§M-6：resume 兼容性分三层**。`RESEARCH_CRITICAL`（adv_estimator / bootstrap_method / collect_step_states /
    candidate_skills）与 metadata 采集签名不一致 -> 严格模式下 raise；`llm.*` / `persistence.*` 前缀归为运营设置，
    允许改变但写一条 `config_change` 记录；其余 `research` 类变更只在日志和 `resume_report` 里体现。
12. **§M-7：`runtime.apply()` 之后 runtime 是权威状态**。active / `next_skill_id` 都从 runtime 回读，记录写入
    失败只能新增诊断（`record_error` / `persistence_failures`），不能把一个已生效的 decision 改写成失败；
    反复落盘失败累计到 `PERSISTENCE_FAILURE_LIMIT` 才升级为 `SelfImprovingStateError`。
13. **§10：`interval_steps < min_window_steps` 只警告不报错**，因为 CPU 快速实验 / 故障重试确实需要短 interval，
    代价是每个窗口都会被评估为 `insufficient_evidence`。

---

## 7. 产物样例

> 以下 JSON 全部由 `tests/trainer/ppo/test_self_improving_loop_on_cpu.py` 的
> **合成 batch + Mock LLM** 导出（6 个 step、`interval_steps=2`、`checkpoints=(4,)`）。
> 它们用于说明 schema 与字段含义，**不是真实训练结果**。

### 7.1 Console

```text
[self_improving] enabled: run=20261127-024857-72455-d4155ded initial/active skill=grpo
  params={'norm_adv_by_std_in_grpo': True, 'epsilon': 1e-06} interval=2
  collected_metadata=['eval_fingerprint','step_observations','step_spans','step_transitions','terminal_trusted']
  candidates=['grpo','gigpo','rewardflow'] knowledge_version=0
[self_improving] finalized: 3 intervention(s), knowledge version 2, active skill rewardflow
```

### 7.2 Training Context（`contexts.jsonl` 第 2 条，截断）

```json
{
  "global_step": 4,
  "window_start_step": 3,
  "window_end_step": 4,
  "active_skill": {"method": "gigpo", "params": {"gamma": 0.9, "step_advantage_w": 1.0, "mode": "mean_norm", "grouping": "exact", "epsilon": 1e-06, "anchor": "state", "anchor_scope": "full", "include_truncated_in_baseline": true}},
  "window_skills": ["gigpo"],
  "task_summary": {"prompt_example": "fix workbook 0", "batch_size_per_step": 4.0},
  "training_metrics": {"actor/pg_loss": 0.25, "critic/score/mean": 0.5, "response_length/mean": 12.0},
  "trajectory_statistics": {
    "num_steps": 2, "step_range": [3, 4], "num_trajectories": 8.0,
    "raw_reward": {"mean": 0.5, "std": 0.5, "min": 0.0, "max": 1.0, "n_steps": 2},
    "train_reward": {"mean": 0.45, "std": 0.45, "min": 0.0, "max": 0.9, "n_steps": 2},
    "raw_reward_hist": {"zero": 0.5, "(0.75,1)": 0.5},
    "success_rate": 0.5, "within_group_reward_std": 0.0,
    "trajectory_length": {"mean": 12.0, "p50": 12.0, "p90": 12.0, "n_steps": 2}
  },
  "credit_diagnostics": {"window": {"credit_assignment/gigpo/anchor_count": 3.0}, "latest_step": {"credit_assignment/gigpo/anchor_count": 3.0}},
  "metadata_health": {"step_spans": 1.0, "step_observations": 1.0, "step_transitions": 1.0, "terminal_trusted": 1.0, "terminal_reason": 1.0, "eval_workbook_fingerprint": 1.0, "unknown_state_ratio": 0.0, "state_parse_ms_mean": 10.0, "eval_fingerprint_any": true, "structured_state_any": true},
  "trajectory_samples": [
    {"kind": "success", "step": 3, "row": 0, "group_id": "g0", "raw_reward": 1.0, "train_reward": 0.9,
     "terminal_reason": "final_answer", "terminal_trusted": true, "step_count": 3, "trajectory_length": 11,
     "has_eval_fingerprint": true, "task_instruction": "fix workbook 0",
     "steps": [{"index": 0, "observation": "cell A1=1"}, {"index": 1, "observation": "sheet2 opened"}, {"index": 2, "observation": "formula written"}]}
  ],
  "notes": {"interval_steps": 2, "steps_since_last_intervention": 2, "observed_steps_in_window": 2,
            "is_last_step": false, "skill_codes": {"grpo": 0.0, "gigpo": 1.0, "rewardflow": 2.0},
            "metadata_bootstrap": {"bootstrap_method": "rewardflow", "collected": ["eval_fingerprint", "step_observations", "step_spans", "step_transitions", "terminal_trusted"]}}
}
```

### 7.3 Controller Decision（`history.jsonl` 中的 `decision` 记录）

```json
{
  "decision_id": "decision-0002-g4",
  "step": 4,
  "action": "switch",
  "status": "applied",
  "error": null,
  "previous_skill": {"method": "gigpo", "params": {"gamma": 0.9, "step_advantage_w": 1.0, "mode": "mean_norm", "grouping": "exact", "epsilon": 1e-06, "anchor": "state", "anchor_scope": "full", "include_truncated_in_baseline": true}},
  "applied_skill": "rewardflow",
  "applied_parameters": {"gamma": 0.9, "alpha_act": 1.0, "alpha_traj": 1.0, "normalization": "mean_norm", "epsilon": 1e-06, "success_threshold": 1.0, "success_tolerance": 1e-06},
  "changed": true,
  "reasoning": "step level credit should separate individual tool calls",
  "evidence": ["raw reward mean 0.25", "many zero-advantage assistant tokens"],
  "expected_improvement": "better per-step credit assignment",
  "knowledge_version": 1,
  "llm_latency_seconds": 0.01,
  "knowledge_update": {"status": "applied", "entry_id": "exp-0001", "knowledge_version": 1, "latency_seconds": 0.01},
  "run_id": "20261127-024857-72455-d4155ded",
  "ts": "2026-11-27T02:48:57Z"
}
```

`status` 取值：`applied` / `keep` / `fallback`（LLM 或解析失败）/ `rejected`（schema 或门禁失败）。
四种都不会改变当前 Skill 之外的任何训练状态，也**不存在**回滚语义。

### 7.4 Evaluation Result（`record_type=evaluation`）

```json
{
  "evaluation_id": "eval-0003-g6",
  "decision_id": "decision-0002-g4",
  "step": 6,
  "previous_skill": "gigpo",
  "applied_skill": "rewardflow",
  "action": "switch",
  "baseline_window": {"step_range": [3, 4], "num_steps": 2, "skills": ["gigpo"], "success_rate": 0.5},
  "observed_window": {"step_range": [5, 6], "num_steps": 2, "skills": ["rewardflow"], "success_rate": 0.125},
  "metric_deltas": {"success_rate": -0.375, "raw_reward_mean": -0.375, "train_reward_mean": -0.35,
                    "trajectory_length_mean": 0.0, "steps_per_trajectory_mean": 0.0,
                    "zero_advantage_ratio": null, "advantage_abs_mean": null, "trusted_terminal_ratio": 0.0},
  "validation_comparison": {"status": "unavailable", "reason": "no validation run was recorded in this window"},
  "evidence_quality": "medium",
  "evidence_notes": [],
  "outcome": "negative",
  "primary_metric": "success_rate",
  "eligible_for_knowledge_update": true,
  "summary": "skill 'rewardflow' observed over steps 5-6; outcome=negative; primary=success_rate delta=-0.375; evidence=medium; observational only (not a causal effect estimate)",
  "run_id": "20261127-024857-72455-d4155ded"
}
```

`outcome=negative` 不触发任何动作：下一次介入照常进行，Skill 保持 `rewardflow`（见 §5.1 的
`test_negative_evaluation_does_not_roll_back`）。

### 7.5 Knowledge Update

```json
{"record_type": "knowledge_update", "step": 6, "target_skill": "rewardflow", "entry_id": "exp-0001",
 "knowledge_version": 2, "decision_id": "decision-0002-g4", "evaluation_id": "eval-0003-g6"}
```

追加进 `knowledge/current/gigpo.md` 的 `## Empirical Experience`（数值一律来自 Python 的 evaluation 记录）：

```markdown
### exp-0001 · step 4 · decision decision-0001-g2

- outcome: positive
- configuration: `{"anchor": "state", "anchor_scope": "full", "epsilon": 1e-06, "gamma": 0.9, "grouping": "exact", "include_truncated_in_baseline": true, "mode": "mean_norm", "step_advantage_w": 1.0}`
- metric deltas: `{"advantage_abs_mean": null, "raw_reward_mean": 0.25, "steps_per_trajectory_mean": 0.0, "success_rate": 0.25, "train_reward_mean": 0.25, "trajectory_length_mean": 0.0, "trusted_terminal_ratio": 0.0, "zero_advantage_ratio": null}`
- evidence quality: medium
- observation: the metric moved together with the switch
- interpretation: consistent with the documented mechanism
- limitations: single window, observational only
```

版本索引 `knowledge/index.json`（每个版本目录含**全部**候选 Skill 的快照）：

```json
{"lineage": "20261127-024857-72455-d4155ded", "versions": [
  {"version": 0, "step": 0, "target_skill": null, "previous_version": null, "skills": ["gigpo", "grpo", "rewardflow"]},
  {"version": 1, "step": 4, "decision_id": "decision-0001-g2", "evaluation_id": "eval-0002-g4", "target_skill": "gigpo", "previous_version": 0, "skills": ["gigpo"]},
  {"version": 2, "step": 6, "decision_id": "decision-0002-g4", "evaluation_id": "eval-0003-g6", "target_skill": "rewardflow", "previous_version": 1, "skills": ["rewardflow"]}]}
```

### 7.6 state.json（run 结束）

```json
{
  "completed_global_step": 6,
  "active_skill": "rewardflow",
  "active_parameters": {"gamma": 0.9, "alpha_act": 1.0, "alpha_traj": 1.0, "normalization": "mean_norm", "epsilon": 1e-06, "success_threshold": 1.0, "success_tolerance": 1e-06},
  "executed_skill": "rewardflow",
  "executed_parameters": {"gamma": 0.9, "alpha_act": 1.0, "alpha_traj": 1.0, "normalization": "mean_norm", "epsilon": 1e-06, "success_threshold": 1.0, "success_tolerance": 1e-06},
  "last_intervention_step": 6,
  "intervention_count": 3,
  "window_start_step": 7,
  "current_window_summary": {"num_steps": 0},
  "pending_decision": null,
  "latest_validation": null,
  "validation_baseline": null,
  "knowledge_version": 2,
  "windows_since_knowledge_update": 0,
  "metadata_collection_signature": {"bootstrap_method": "rewardflow", "collect_step_states": null, "bootstrap_params": {}, "collected": ["eval_fingerprint", "step_observations", "step_spans", "step_transitions", "terminal_trusted"]},
  "experiment_identity": {"adv_estimator": "credit_assignment", "bootstrap_method": "rewardflow", "collect_step_states": null, "project": "larl", "experiment": "si-integration", "interval_steps": 2, "candidate_skills": ["grpo", "gigpo", "rewardflow"], "total_training_steps": 6, "digest": "5a3444eddce11113"},
  "skill_id_codes": {"grpo": 0.0, "gigpo": 1.0, "rewardflow": 2.0},
  "config_summary": {"enabled": true, "interval_steps": 2, "...": "本 session 的 self_improving 配置，resume 时用 config_diff 分类"},
  "consecutive_intervention_failures": 0,
  "persistence_failures": 0,
  "state_misaligned": false,
  "last_persistence_error": null,
  "run_id": "20261127-024857-72455-d4155ded",
  "lineage": "20261127-024857-72455-d4155ded",
  "config_version": 1
}
```

已移除旧版的 `slow_loop_disabled` 字段（不再有永久停用语义）；历史 state 里仍然存在时，`initialize()`
只打印一条兼容提示并忽略。`consecutive_intervention_failures` / `persistence_failures` / `state_misaligned`
仅供诊断与恢复提示，不作为停用条件。

`snapshots/step_000004.json` 与上同构，额外带 `snapshot_step: 4` 和 `pending_decision`
（其中 `executed_skill="gigpo"` / `active_skill="rewardflow"`，正好展示 step 末尾决策、下一 step 生效）；
resume 时按 policy checkpoint 的 step 精确取用该文件。

### 7.7 Experiment History 记录类型序列

```text
initialized, window_summary, decision, knowledge_update, window_summary, evaluation,
knowledge_attempt, decision, checkpoint_snapshot, knowledge_update, selection_skipped,
window_summary, evaluation, knowledge_attempt, finalized
```

`window_summary` 无论有无 decision 都会写（§10：没有 decision 时不伪造 evaluation，但窗口摘要与诊断仍保留）；
`checkpoint_snapshot` 只在政策 checkpoint 对齐的那一步出现；`config_change` 只在该 session 的
self_improving 配置与被恢复的状态不同时出现（§M-6，本例没有）。

`selection_skipped` 只在最后一个 step 出现：`{"reason": "no further training step exists; running evaluation and persistence only"}`
—— 末步不再向 LLM 征求 decision（§17.3）。

### 7.8 指标（介入 step 的 `self_improving/*`）

```json
{
  "self_improving/consecutive_failures": 0.0,
  "self_improving/decision_valid": 1.0,
  "self_improving/eval_delta_reward": 0.25,
  "self_improving/eval_delta_success_rate": 0.25,
  "self_improving/executed_skill_id": 1.0,
  "self_improving/intervention_count": 2.0,
  "self_improving/intervention_triggered": 1.0,
  "self_improving/knowledge_updated": 1.0,
  "self_improving/knowledge_version": 1.0,
  "self_improving/llm_call_seconds": 0.00275,
  "self_improving/metadata_available_ratio": 1.0,
  "self_improving/metadata_missing_count": 0.0,
  "self_improving/metadata_partial_count": 0.0,
  "self_improving/next_skill_id": 2.0,
  "self_improving/persistence_failures": 0.0,
  "self_improving/slow_loop_seconds": 0.02023
}
```

`metadata_missing_count` / `metadata_partial_count` 是 §M-3 诊断：前者是本窗口"完全未采集"的必需 metadata
项数，后者是"采集了但覆盖 < 1.0"的项数；两者一起参与 Evidence Quality 降级。

---

## 8. Review 后的 bug fix

> 本节只记录本轮修改的状态与边界，不新增研究功能。

| Issue | 状态 | 实际修改 | Regression Test |
|---|---|---|---|
| C-1 bootstrap params | Fixed | `runtime.bootstrap_param_overrides()` + `_preflight()` 拒绝 launch 节点里的非默认 algorithm params；训练参数只走 `self_improving.initial_skill.params` | `test_self_improving_runtime_on_cpu.py::MetadataBootstrapTest`（新增 6 个） |
| M-1 / M-2 永久停用 | Fixed | 删除连续 3 次失败停用；`FATAL_ERRORS` 向上抛，其余可恢复失败下一周期重试；旧 `slow_loop_disabled` 只提示并忽略 | `LoopSafetyTest::test_repeated_recoverable_intervention_failures_retry_at_the_next_interval` / `test_state_inconsistency_is_not_swallowed_by_the_generic_handler` / `test_llm_failure_falls_back_without_touching_the_skill` |
| M-3 metadata 健度 / Evidence Quality | Fixed（硬校验语义不变） | `HEALTH_KEY_OF` + `_metadata_health_signals()` 区分 hard-missing / partial；`evaluate(observed_requirements=…)`；无 observed_requirements 时沿用旧启发式 | `EvaluatorTest::test_hard_required_metadata_missing_vs_partial_new_path` + 既有 contaminated-window 用例 |
| M-4 checkpoint 对齐 | Fixed | 每个 step 末尾写 state；`on_checkpoint_saved()` 置位后同 step 写 sidecar snapshot；缺 snapshot 时严格 resume 明确失败 | `ResumeTest::test_strict_resume_refuses_a_missing_snapshot` / `test_non_strict_resume_falls_back_and_flags_the_gap` |
| M-5 未来信息泄漏 | Fixed | `resume_state()` 只取 `step <= ckpt`；`adopt()` 拒绝未来版本；history 预取传 `max_step`；`start_new_lineage()` 隔离旧知识 | `ResumeTest::test_resume_does_not_leak_future_knowledge` / `test_resume_ignores_history_written_after_the_restored_step` |
| M-6 实验身份 / resume 兼容 | Fixed | `experiment_identity()` + `config_diff()`（operational / research 分类）+ `RESEARCH_CRITICAL`；`config_change` 入库 | `ResumeTest::test_strict_resume_refuses_research_config_drift` / `test_resume_allows_operational_changes_and_records_them` |
| M-7 apply 后状态一致 | Fixed | runtime 为权威：active / `next_skill_id` 回读；写入失败走 `_safe_history` / `_note_persistence_error`，触顶才 `SelfImprovingStateError` | `LoopSafetyTest::test_persistence_failures_eventually_escalate_to_state_error` + FullWorkflow 的 executed/next 归属断言 |
| M-8 disabled-mode 文档 | Fixed | README 明确 vanilla GRPO 用 `algorithm.adv_estimator=grpo`（SI 的 GRPO wrapper 只在 `enabled=true` 时注册） | `DisabledModeTest`（5 passed，未改 registry） |
| §10 minor | Fixed / Mitigated | interval &lt; min_window 改为警告；`missing_ratio` → `metadata_missing_count` / `metadata_partial_count`；修正不存在的 diagnostic key 引用；清理 dead code（`context.py` 读取从未产生的 `anchor_missing_frac`）；`val_only` 下不再 initialize Slow Loop | `test_interval_smaller_than_min_window_warns_but_does_not_raise` |
| §10 keep / switch / calibrate 的"是否真的改了配置" | Fixed | `ControllerDecision.changed` 由 method + params 实际比对得出并入库；`_evaluate()` 在 `changed=false` 时把窗口差异记为 policy drift，不作为 Skill 变更效果，也不进知识 | `LoopSafetyTest::test_keep_decision_is_not_attributed_as_a_skill_change` + `EvaluatorTest::test_no_actual_configuration_change_is_not_attributed` |
| row-0 reward metadata aggregation | Accepted Limitation（按任务约束未修） | 仅观测与降级 | 既有 metadata 相关用例 |
| Crash Recovery 中间态（state 已写 / policy ckpt 未写） | Accepted Limitation | 未改现有 checkpoint 结构 | — |
| GPU / 外部 API / fsync / 轨迹解码性能 | Deferred | 无设备环境，待 smoke + profiling | — |

## 9. 未实现项（命令书 §24 明确排除，此处仅作核对）

算法组合、LLM 生成算法代码、动态改算法结构、新 advantage 公式、per-trajectory / per-task 选择、
policy checkpoint 分支、反事实训练、任何形式的自动回滚、meta-learning、Controller RL、
RAG / 向量库 / 跨 run 知识共享、自动超参搜索框架、新 reward shaping、context management 改动、
rollout 架构重构 —— 均未实现，也不在本 MVP 范围内。

## 10. 剩余问题

- 真实训练环境下的 Slow Loop 端到端行为（GPU smoke、外部 LLM 端点、resume 第二段）**未验证**；
  验证前不应声称"集成训练已可用"。
- `deploy/run_grpo.sh` 是否把未知 Hydra override 原样透传，只在 `DRY=1` 下检查了拼出的命令行，
  未在实际 `verl.trainer.main_ppo` 启动下确认。
- rewardflow bootstrap 的额外序列化 / 内存开销未实测（缺集群环境）。
- 既有 row-0 reward metadata aggregation 缺陷（失败/timeout 轨迹缺 `eval_fingerprint`）未修复，
  只做观测与证据降级。
- **Policy checkpoint 与 sidecar snapshot 的写入时点仍有窗口**：`_save_checkpoint()` 与随后的
  `on_step_end()` 之间崩溃，会得到"有 policy ckpt、无同 step SI 状态"的中间态。本轮只保证
  正常 step-end 后的对齐与失败时的显式报错，未改造现有 checkpoint 架构（需先改 ckpt 写入顺序）。
- 非严格 resume 从 initial template 开新 lineage 时，旧 knowledge / history 整目录归档到 `previous/`，
  不会被覆写，但也不会自动合回：需要人工确认归档目录才能接回旧经验。
- 性能相关项（每 step GPU→CPU 同步、轨迹解码、每步 fsync、knowledge 版本 retention）本轮**未优化**，
  等 GPU smoke / profiling 实测后再判。
