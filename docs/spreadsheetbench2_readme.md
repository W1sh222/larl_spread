# SpreadsheetBench 2 本地部署指南

本文档介绍如何在本地环境（无 Docker）运行 SpreadsheetBench 2。

## 目录结构

```
SpreadsheetBench-2/
├── SWE-agent/
│   ├── scripts/
│   │   ├── run_inference_local.sh   # 本地推理脚本
│   │   └── run_evaluation_local.sh  # 本地评估脚本
│   ├── config/
│   │   ├── spreadsheet.yaml         # Docker 配置（原版）
│   │   └── spreadsheet_local.yaml   # 本地配置（新增）
│   └── results/                     # 运行结果
├── evaluation/
│   ├── evaluation.py                # 评估逻辑
│   └── open_spreadsheet.py          # LibreOffice 重算
└── data/
    └── spreadsheetbench-v2/         # 数据集
```

---

## 环境要求

参考 SpreadsheetBench 2 官方仓库：https://github.com/RUCKBReasoning/SpreadsheetBench-2

核心依赖：
- Python 3.11 + Conda
- LibreOffice 7.5+
- SWE-agent

---

## 快速开始

### 1. 环境设置

```bash
conda create -n ssb-v2 python=3.11 -y
conda activate ssb-v2

cd SpreadsheetBench-2/SWE-agent
pip install --upgrade pip
pip install --editable .
pip install openpyxl numpy pandas matplotlib xlsxwriter
```

### 2. 下载数据

```bash
pip install -U "huggingface_hub[cli]"

cd SpreadsheetBench-2
huggingface-cli download KAKA22/SpreadsheetBench-v2 \
    --repo-type dataset \
    --local-dir data/spreadsheetbench-v2
```

数据类别：
- Debugging (100 题)
- Financial_Model (100 题)
- Template (97 题)
- Visualization (需特殊评估，当前计划排除)

### 3. 运行推理

```bash
cd SpreadsheetBench-2/SWE-agent/scripts

bash run_inference_local.sh \
    --api-key YOUR_API_KEY \
    --category Debugging \
    --model openrouter/z-ai/glm-5 \
    --num-workers 4
```

参数说明：
- `--api-key`: LLM API 密钥（必需）
- `--category`: 数据类别（默认 Debugging）
- `--model`: 模型名称（默认 openrouter/z-ai/glm-5）
- `--num-workers`: 并行工作数（默认 1）

### 4. 评估结果

```bash
bash run_evaluation_local.sh \
    --category Debugging \
    --model glm-5 \
    --outputs-dir /path/to/results
```

评估流程：
1. LibreOffice 重新计算公式
2. openpyxl 读取并比较结果

---

## 重要参考事项

### Visualization 评估除外

当前计划中 Visualization 类别将从评估中排除。

原因：
- Visualization 评估需要 Windows 环境的 Excel/WPS COM 接口
- 必须导出图表图像并进行 VLM（视觉语言模型）评估
- 本地部署方案难以完全自动化支持此流程

应对方案：
- 仅使用本地部署评估 Debugging、Financial_Model、Template 3个类别
- Visualization 类别将在后续使用官方 Docker 环境单独评估

如需完整评估（将来参考）：
```bash
# 在 Windows 环境下运行
cd SpreadsheetBench-2

python evaluation/run_visual_vlm_checklist_eval.py \
    --tasks-json data/spreadsheetbench-v2/Visualization/dataset.json \
    --output-dir /path/to/results \
    --api-key YOUR_BIGMODEL_API_KEY \
    --model glm-4.6v
```

---

## 配置文件说明

本地部署配置文件：`SpreadsheetBench-2/SWE-agent/config/spreadsheet_local.yaml`

关键配置项：
```yaml
agent:
  type: default
  templates:
    system_template: "..."
    instance_template: "..."
  
  tools:
    enable_bash_tool: true
    execution_timeout: 60
    bundles:
      - path: tools/submit
      - path: tools/view_xlsx

env:
  deployment:
    type: local  # 关键：使用本地而非 Docker
```

与 Docker 配置对比：
| 配置项 | Docker 版本 | 本地版本 |
|--------|------------|--------|
| `deployment.type` | `docker` | `local` |
| `deployment.image` | `spreadsheetbench-v2` | N/A |
| 执行环境 | 容器隔离 | 本地系统 |

---

## 日志查看

推理日志位置：
```
SpreadsheetBench-2/SWE-agent/results/
└── <timestamp>/
    ├── traj/
    │   └── <task_id>.traj  # 详细轨迹
    └── logs/
        └── <task_id>.log   # 执行日志
```

查看实时日志：
```bash
tail -f results/<timestamp>/logs/<task_id>.log
```

---

## 参考资源

- 原项目 README: `SpreadsheetBench-2/README.md`
- SWE-agent 文档: `SpreadsheetBench-2/SWE-agent/README.md`
- 执行脚本: `SpreadsheetBench-2/SWE-agent/scripts/`
- 评估代码: `SpreadsheetBench-2/evaluation/`
