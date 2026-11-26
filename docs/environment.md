# 环境搭建

本文件描述从零复现本部署所需的全部环境。实测机器：Linux 单机 8×A100 80GB + Windows 10。

## 1. Linux conda 环境

机器上共存 3 个环境，**职责不同，不要合并**：

| 环境 | Python | 用途 |
| --- | --- | --- |
| `spread` | 3.11.18 | GRPO 训练（verl）+ 评测链路推理脚本 |
| `sandbox` | 3.11.18 | SandboxFusion 代码执行服务 :8080 |
| `vllm` | 3.12.16 | 评测链路专用 vLLM 推理服务 :8000 |

### 1.1 `spread`（训练 + 评测脚本）

```bash
conda create -n spread python=3.11 -y
conda activate spread
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
pip install vllm==0.32.0 openai==2.54.0 pyarrow pandas openpyxl tqdm
# verl 用仓库内置源码，不要 pip install verl
pip install -e Spreadsheet-RL-master/verl   # 视需要，常规做法是只加 PYTHONPATH
```

实测关键版本（务必对齐，否则会踩 [troubleshooting](troubleshooting.md) 里的坑）：

```
torch     2.13.0+cu129
vllm      0.32.0
numpy     1.26.4
pandas    3.0.6
openai    2.54.0
pyarrow   21.0.0
verl      0.8.0.dev（仓库内源码）
```

注意：

- **不要装 `torchcodec` 0.17**，它与当前 torch 组合不兼容，会直接让 vLLM 启动失败。若已装：`pip uninstall -y torchcodec`。
- `numpy` 固定在 1.26.x，上游 `SpreadsheetBench/requirements.txt` 锁的 1.22.2 与 torch 2.13 冲突，以 1.26.4 为准。
- 上游 `SpreadsheetBench/requirements.txt` 里的 `transformers==4.38.0`、`openai==1.30.4`、`docker==7.1.0` 在本部署中**不是必需的**：评测链路走的是本地 `code_exec_local.py`，不需要 docker SDK；`openai` 用 2.54.0。

### 1.2 `sandbox`（SandboxFusion）

```bash
conda create -n sandbox python=3.11 -y
conda activate sandbox
pip install fastapi "uvicorn==0.25.0" pydantic structlog psutil aiofiles aiohttp tenacity transformers
```

`Spreadsheet-RL-master/sandbox/` 是 vendored 源码，通过 `PYTHONPATH` 加载，不需要 poetry install。

### 1.3 `vllm`（评测用推理服务）

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm==0.32.0 openai
```

## 2. 模型权重

放在 `/remote-home/yichengfu/models/` 下（脚本默认路径，可用 `MODEL_PATH` 覆盖）：

| 模型 | 用途 |
| --- | --- |
| `Qwen3-4B-Thinking-2507` | GRPO 训练的 actor（`Qwen/Qwen3-4B-Thinking-2507`） |
| `Qwen3.5-9B` | SpreadsheetBench 评测的被测模型 |

```bash
# 示例：从 HuggingFace 拉取
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-4B-Thinking-2507 --local-dir /remote-home/yichengfu/models/Qwen3-4B-Thinking-2507
```

## 3. 数据集

**数据集不随仓库分发**（约 6.6GB，且含真实业务表格，已加入 `.gitignore`）。

### 3.1 SpreadsheetBench（评测用）

```bash
cd SpreadsheetBench/data
wget https://github.com/RUCKBReasoning/SpreadsheetBench/raw/main/data/all_data_912.tar.gz
tar -xzf all_data_912.tar.gz -C all_data_912_v0.1   # 解出 dataset.json + spreadsheet/ + outputs/
```

评测脚本要求代码里能访问 `/mnt/data/`（模型生成的代码硬编码了这个路径），建软链接：

```bash
sudo ln -sfn /remote-home/yichengfu/larl/SpreadsheetBench/data/all_data_912_v0.1 /mnt/data
```

### 3.2 Spreadsheet-RL（训练用）

从 HuggingFace 拉：[Spreadsheet-RL/Spreadsheet-RL](https://huggingface.co/datasets/Spreadsheet-RL/Spreadsheet-RL)，解压到 `Spreadsheet-RL-master/data/`，目录结构需为：

```
Spreadsheet-RL-master/data/
├── train_hermes.parquet        # 训练集（configs/qwen3-4b.sh 默认读取）
├── test_hermes.parquet         # 验证集
├── excelforum/                 # 工作簿
├── domain/
├── spreadsheetbench/
├── spreadsheetbench_2/
├── spreadsheetbench_verified/
└── _workspaces/                # 运行时按 rollout 生成，可不存在
```

Windows 侧需要**同一份数据**（奖励服务要读 `target.xlsx` 比对），部署在 `E:\SpreadsheetRL\data`。

## 4. Windows 侧环境

见 [windows.md](windows.md)。

## 5. 环境自检

```bash
# Linux
/remote-home/yichengfu/anaconda3/envs/spread/bin/python -c "import torch, vllm, verl; print(torch.__version__, vllm.__version__)"
curl -s http://127.0.0.1:8080/openapi.json | head -c 100          # SandboxFusion
curl -s http://127.0.0.1:8081/execute -d '{"convid":"t","code":"print(1)"}' -H 'Content-Type: application/json'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# Windows（从 Linux）
ssh fyclab@172.16.6.155 'curl.exe -s http://127.0.0.1:5000/health'
```
