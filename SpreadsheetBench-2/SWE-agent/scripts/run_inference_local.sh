#!/bin/bash
# SpreadsheetBench 2 本地推理执行脚本（无 Docker）

set -e

echo "=========================================="
echo "SpreadsheetBench 2 本地推理"
echo "=========================================="

# 默认值
MODEL_NAME="openrouter/z-ai/glm-5"
API_KEY=""
CATEGORY="Debugging"
NUM_WORKERS=1
CONFIG_FILE="config/spreadsheet_local.yaml"

# 参数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model MODEL         模型名称 (默认: $MODEL_NAME)"
            echo "  --api-key KEY         API 密钥 (必需)"
            echo "  --category CATEGORY   数据类别 (默认: $CATEGORY)"
            echo "                        可选: Debugging, Financial_Model, Template, Visualization"
            echo "  --num-workers N       并行工作数 (默认: $NUM_WORKERS)"
            echo "  --config FILE         配置文件 (默认: $CONFIG_FILE)"
            echo "  --help                显示此帮助"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 检查 API 密钥
if [ -z "$API_KEY" ]; then
    echo "错误: 需要 --api-key 选项"
    echo "用法: $0 --help"
    exit 1
fi

# 激活 Conda 环境
CONDA_ENV="ssb-v2"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# 脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SWE_AGENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SWE_AGENT_DIR")"

cd "$SWE_AGENT_DIR"

# 数据路径
DATA_PATH="$PROJECT_ROOT/data/spreadsheetbench-v2/$CATEGORY"

if [ ! -d "$DATA_PATH" ]; then
    echo "错误: 数据目录不存在: $DATA_PATH"
    echo "请先下载数据（见 README）"
    exit 1
fi

echo ""
echo "配置:"
echo "  模型: $MODEL_NAME"
echo "  类别: $CATEGORY"
echo "  工作数: $NUM_WORKERS"
echo "  数据路径: $DATA_PATH"
echo ""

# Visualization 任务使用不同配置
if [ "$CATEGORY" = "Visualization" ]; then
    CONFIG_FILE="config/visualisation.yaml"
    echo "使用 Visualization 配置文件: $CONFIG_FILE"
fi

# 运行 SWE-agent
echo "开始推理..."
echo ""

sweagent run \
    --config "$CONFIG_FILE" \
    --env.deployment.type local \
    --agent.model.name="$MODEL_NAME" \
    --agent.model.api_key="$API_KEY" \
    --agent.model.completion_kwargs='{"extra_body": {"reasoning": {"enabled": true}}}' \
    --num_workers "$NUM_WORKERS" \
    --dataset_path "$DATA_PATH"

echo ""
echo "=========================================="
echo "推理完成！"
echo "=========================================="
echo ""
echo "结果位置: $SWE_AGENT_DIR/results/"
echo ""
echo "下一步:"
echo "  1. 使用 LibreOffice 重新计算结果: python evaluation/open_spreadsheet.py --dir_path <结果路径>"
echo "  2. 运行评估: python evaluation/evaluation.py --model $MODEL_NAME --dataset $CATEGORY --outputs-dir <结果路径>"
echo ""
