# larl_spread

本仓库是**在单机 Linux（8×A100）+ 一台 Windows 主机（Microsoft Excel）**上部署并跑通的两套代码的完整落地版本：

| 目录 | 用途 | 上游 |
| --- | --- | --- |
| [`SpreadsheetBench/`](SpreadsheetBench/README.md) | 基准评测：LLM 推理 → 代码执行 → 结果 xlsx → OJ 式逐用例比对 | [RUCKBReasoning/SpreadsheetBench](https://github.com/RUCKBReasoning/SpreadsheetBench) |
| [`Spreadsheet-RL-master/`](Spreadsheet-RL-master/README.md) | GRPO 训练：多轮 rollout → 沙箱代码执行 → Windows Excel 重算打分 → 策略更新 | [Spreadsheet-RL/Spreadsheet-RL](https://github.com/Spreadsheet-RL/Spreadsheet-RL) |

两套代码共用同一条"Windows Excel 重算"能力，但是**彼此独立的两条链路**，可以单独部署、单独运行。

```
              ┌─────────────────── Linux 单机 ───────────────────┐        ┌─── Windows ───┐
              │                                                  │        │               │
评测链路       │  vLLM(Qwen3.5-9B) ──► inference_single.py         │        │               │
              │        ▲                    │                    │        │               │
              │        └──── /execute ── code_exec_local.py(8081)│        │               │
              │                             │ 生成 outputs/*.xlsx │        │               │
              │                     evaluation.py（纯本地比对）  │        │               │
              │                                                  │        │               │
              ├──────────────────────────────────────────────────┤        │               │
              │                                                  │        │               │
训练链路       │  verl GRPO ──► vLLM 多轮 rollout                 │  HTTP  │  async_reward │
              │       │              │                           │ ─────► │  _api (5000)  │
              │       │              └─► SandboxFusion (8080)     │        │  Excel COM    │
              │       └── reward ◄───────────────────────────────┼────────┤  重算 + 比对  │
              └──────────────────────────────────────────────────┘        └───────────────┘
```

## 环境一览（实测版本）

| 组件 | 位置 | 版本 / 说明 |
| --- | --- | --- |
| conda 环境 `spread` | Linux | Python 3.11.18，verl 0.8.0.dev + torch 2.13.0+cu129 + vllm 0.32.0 |
| conda 环境 `sandbox` | Linux | Python 3.11.18，SandboxFusion 代码执行服务 |
| conda 环境 `vllm` | Linux | Python 3.12.16，vllm 0.32.0（仅评测链路用） |
| 模型 | Linux | `/remote-home/yichengfu/models/Qwen3-4B-Thinking-2507`、`Qwen3.5-9B` |
| 重算/奖励服务 | Windows | Python 3.11.9 venv，Excel COM，端口 5000 |

> ⚠️ 数据集（约 6.6GB）与模型权重**不在本仓库内**，`.gitignore` 已排除。获取方式见 [docs/environment.md](docs/environment.md)。

## 快速开始

三条链路分别对应三个入口脚本，**每一步都要先确认上游服务已经在跑**：

| 目标 | 命令 | 前置服务 |
| --- | --- | --- |
| 跑 SpreadsheetBench 评测 | `cd SpreadsheetBench && python inference/inference_single.py ...` 再 `python evaluation/evaluation.py ...` | vLLM(8000) + code_exec(8081) |
| 跑 GRPO 训练 | `bash Spreadsheet-RL-master/deploy/run_grpo.sh` | SandboxFusion(8080) + Windows reward(5000) |
| 起 Windows 奖励服务 | 交互式计划任务 `SRL-RewardAPI` | Windows 已登录桌面 |

详细步骤：

- [docs/environment.md](docs/environment.md)：环境搭建、依赖版本、数据集与模型获取
- [docs/eval-pipeline.md](docs/eval-pipeline.md)：SpreadsheetBench 推理 + 评测链路
- [docs/train-pipeline.md](docs/train-pipeline.md)：Spreadsheet-RL GRPO 训练链路
- [docs/windows.md](docs/windows.md)：Windows 侧服务部署、SSH 密钥、计划任务
- [docs/troubleshooting.md](docs/troubleshooting.md)：**踩坑记录**（部署时踩过的坑与修法）

## 部署脚本

所有本地新增/改写的启动脚本集中在 `deploy/` 目录，原始上游脚本未做删除：

```
Spreadsheet-RL-master/deploy/
├── env.sh                   # 环境变量（服务地址、PYTHONPATH、flashinfer 开关）
├── serve_sandbox.sh         # 起 SandboxFusion 代码执行服务 :8080
├── run_grpo.sh              # 起 GRPO 训练（自动套 A100 适配参数）
├── win_install_reward_task.ps1  # 注册 Windows 交互式计划任务
└── win_start_reward_api.ps1     # Windows 奖励服务实际启动入口

SpreadsheetBench/deploy/
├── serve_vllm.sh            # 起 vLLM OpenAI 兼容服务 :8000
└── code_exec_local.py       # 无 Docker 的本地代码执行服务 :8081
```

## 关键前提

1. **Windows 主机必须保持登录状态**（锁屏可以，注销/关机会让 Excel COM 失效）。这是整个训练链路的硬约束。
2. Linux 与 Windows 需在同一内网互通（本例 `172.16.6.155:5000`）。
3. A100 不支持 FlashAttention-3，训练必须用 `sdpa`，脚本已默认处理。
4. 训练数据需先在本地生成 parquet（依赖 Windows 重算服务），见 [docs/train-pipeline.md](docs/train-pipeline.md)。
