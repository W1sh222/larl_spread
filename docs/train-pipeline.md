# 训练链路：Spreadsheet-RL GRPO

链路：verl GRPO 训练器 → vLLM 多轮 rollout（工具调用）→ SandboxFusion 执行代码 → Windows Excel 重算 + 比对得到 reward → 策略更新。

## 0. 前置

| 服务 | 地址 | 启动方式 |
| --- | --- | --- |
| SandboxFusion 代码执行 | `http://127.0.0.1:8080/run_code` | `deploy/serve_sandbox.sh` |
| Windows 奖励 / 重算服务 | `http://172.16.6.155:5000` | Windows 交互式计划任务（见 [windows.md](windows.md)） |
| 数据集 parquet | `data/train_hermes.parquet` | 见 [environment.md](environment.md) |

## 1. 起服务

```bash
# SandboxFusion（CPU，无容器隔离的 hpc 配置）
PORT=8080 bash Spreadsheet-RL-master/deploy/serve_sandbox.sh

# 验证
curl -s -X POST http://127.0.0.1:8080/run_code \
     -H 'Content-Type: application/json' \
     -d '{"code": "print(1+1)", "language": "python"}' | head -c 200

# 验证 Windows 端（必须在跑）
curl -s http://172.16.6.155:5000/health
```

## 2. 启动训练

```bash
NGPUS=4 GEN_TP=1 GMU=0.85 bash Spreadsheet-RL-master/deploy/run_grpo.sh
```

脚本做的事：

1. `source deploy/env.sh` 注入 `SPREADSHEET_RL_DATA_ROOT`、`SANDBOX_FUSION_URL`、`SPREADSHEET_RL_REWARD_URL`、`SPREADSHEET_RL_RECALC_URL`、`PYTHONPATH`
2. 切到 `spread` 环境
3. 调 `configs/qwen3-4b.sh`，并**强制覆盖** A100 不适用的项目：
   - `attn_implementation=sdpa`（A100 无 FlashAttention-3）
   - `use_remove_padding=False`
   - `use_fused_kernels=False`
   - `n_gpus_per_node` / `fsdp_size` / `tensor_model_parallel_size` / `gpu_memory_utilization`
   - `trainer.logger=["console"]`（不用 wandb 时）

常用覆盖示例：

```bash
# 2 卡 + 小 batch + 12 小时档位
NGPUS=2 GEN_TP=1 GMU=0.6 bash Spreadsheet-RL-master/deploy/run_grpo.sh \
    data.train_files=$PWD/data/train_12h_768.parquet \
    data.train_batch_size=4 \
    actor_rollout_ref.rollout.n=4 \
    trainer.save_freq=-1

# 4 卡正式训练
NGPUS=4 GMU=0.85 bash Spreadsheet-RL-master/deploy/run_grpo.sh \
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n=8
```

日志：`Spreadsheet-RL-master/logs/qwen3-4b-<日期>_<时间>.log`

终止：

```bash
pkill -f "verl.trainer.main_pp[o]"    # 会连带结束 raylet 与 vLLM worker
```

## 3. 关键参数约束

| 参数 | 约束 | 原因 |
| --- | --- | --- |
| `data.max_prompt_length` | 建议 ≥ 8192 | 默认 4096 会把 60%+ 样本过滤掉（本数据集 prompt 中位数约 5.3k token） |
| `data.max_response_length` | 建议 24576 | 8192 时实测 62% 响应被截断 |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | > 12k | 多轮对话最长长度，太小触发断言失败 |
| `rollout.log_prob_max_token_len_per_gpu` / `ref.log_prob_max_token_len_per_gpu` | 24576~65536 | 同上 |
| `train_batch_size × rollout.n` | 必须是 agent_loop worker 数整数倍 | 否则 rollout 分片不均匀报错 |
| `rollout.gpu_memory_utilization` | 2 卡时 ≤ 0.6 | 默认 0.85 在 compute_log_prob 阶段 OOM |
| `attn_implementation` | `sdpa` | A100 不支持 `flash_attention_3` |
| `trainer.save_freq` | 空间紧张时设 `-1` | 每 5 步存 checkpoint 占用大量磁盘 |

## 4. 单步耗时参考（2×A100，train_batch_size=4，rollout.n=4）

| 阶段 | 耗时 |
| --- | --- |
| gen（多轮 rollout + 工具 + 沙箱） | 83~122 s |
| compute old_log_prob | 17~21 s |
| compute ref | ~13.6 s |
| update_actor | 61~74 s |
| update_weights | ~6 s |
| **合计** | **~190 s/step** |

实测 74 步、平均得分 0.196，其中 65% 的 step 至少有一条 rollout 拿到满分 1.0；Windows 端 257 次重算 + 635 次评分调用全部成功。

## 5. 训练数据准备

`train_hermes.parquet` 由上游数据集直接提供。若需自建子集，用 `spread` 环境读全量 parquet 后按行采样导出，例如：

```python
import pandas as pd
df = pd.read_parquet('data/train_hermes.parquet')
df.head(768).to_parquet('data/train_12h_768.parquet', index=False)
```

注意：`filter_overlong_prompts=True` 会再过滤一轮，**最终有效样本数约为导出量的 40%**（`max_prompt_length=4096` 时）。要跑满预定步数，请把 `max_prompt_length` 调到 8192 或按比例多导样本。

## 6. reward 是怎么算的

每条 rollout 结束后，verl 把工作簿提交到 Windows 的 `/reward/submit`，拿 `job_id` 轮询 `/reward/result/{job_id}`：

1. 服务用 Excel COM 打开模型产出的 xlsx
2. 强制重算全部公式（`Application.CalculateFullRebuild`）
3. 与 `target.xlsx` 按 `answer_position` 逐单元格比对
4. 返回分数

因此**模型产出的 xlsx 必须能被 Excel 打开**；打不开的文件直接判 0（这是预期行为，不是环境故障）。

## 7. 常见失败模式

| 现象 | 原因 / 处理 |
| --- | --- |
| 启动即报 `flash_attn` 导入失败 | `use_remove_padding=False` 未生效，或 verl 回退补丁丢失 |
| `FusedMoE` ImportError | vLLM 0.32 移除该类，需要 `verl/utils/vllm/vllm_fp8_utils.py` 里的占位类 |
| tied embedding `ValueError` | 需要把分桶同步的硬报错降级为 warning（见 troubleshooting） |
| reward 全部超时 | Windows 主机注销/关机，计划任务没有交互式会话 |
| `assert` 失败于 token 长度 | `*_max_token_len_per_gpu` 小于实际多轮长度 |
