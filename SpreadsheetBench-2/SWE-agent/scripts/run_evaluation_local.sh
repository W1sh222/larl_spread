#!/bin/bash
# SpreadsheetBench 2 本地评估执行脚本

set -e

echo "=========================================="
echo "SpreadsheetBench 2 本地评估"
echo "=========================================="

# 默认值
MODEL_NAME="glm-5"
CATEGORY="Debugging"
OUTPUTS_DIR=""

# 参数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --outputs-dir)
            OUTPUTS_DIR="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model MODEL         模型名称 (默认: $MODEL_NAME)"
            echo "  --category CATEGORY   数据类别 (默认: $CATEGORY)"
            echo "                        可选: Debugging, Financial_Model, Template, Visualization"
            echo "  --outputs-dir DIR     输出目录路径 (必需)"
            echo "  --help                显示此帮助"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 检查输出目录
if [ -z "$OUTPUTS_DIR" ]; then
    echo "错误: 需要 --outputs-dir 选项"
    echo "用法: $0 --help"
    exit 1
fi

if [ ! -d "$OUTPUTS_DIR" ]; then
    echo "错误: 输出目录不存在: $OUTPUTS_DIR"
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
EVAL_DIR="$PROJECT_ROOT/evaluation"

cd "$PROJECT_ROOT"

echo ""
echo "配置:"
echo "  模型: $MODEL_NAME"
echo "  类别: $CATEGORY"
echo "  输出目录: $OUTPUTS_DIR"
echo ""

# 步骤 1: 使用 LibreOffice 重新计算表格
echo "[1/2] 正在使用 LibreOffice 重新计算表格..."
python "$EVAL_DIR/open_spreadsheet.py" --dir_path "$OUTPUTS_DIR"

echo ""

# 步骤 2: 运行评估
if [ "$CATEGORY" = "Visualization" ]; then
    echo "[2/2] Visualization 评估..."
    echo ""
    echo "⚠️ 重要：当前计划中 Visualization 类别暂不评估"
    echo ""
    echo "原因："
    echo "- Visualization 任务需要视觉语言模型 (VLM) 评估"
    echo "- 需要 Windows 环境下的 Excel/WPS COM 接口导出图表"
    echo "- 本地部署方案暂不完全自动化支持此流程"
    echo ""
    echo "如需完整评估，请在 Windows 环境下运行："
    echo ""
    echo "python evaluation/run_visual_vlm_checklist_eval.py \\"
    echo "  --tasks-json data/spreadsheetbench-v2/Visualization/dataset.json \\"
    echo "  --output-dir $OUTPUTS_DIR \\"
    echo "  --api-key <your_bigmodel_api_key> \\"
    echo "  --model glm-4.6v"
    echo ""
else
    echo "[2/2] 运行 $CATEGORY 评估..."
    python "$EVAL_DIR/evaluation.py" \
        --model "$MODEL_NAME" \
        --dataset "$CATEGORY" \
        --outputs-dir "$OUTPUTS_DIR" \
        --workers 4
    
    echo ""
    echo "=========================================="
    echo "评估完成！"
    echo "=========================================="
    echo ""
    echo "结果位置: $PROJECT_ROOT/results/$CATEGORY/"
fi

echo ""
