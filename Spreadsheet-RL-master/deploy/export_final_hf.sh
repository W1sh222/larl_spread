#!/bin/bash
# 把最终 FSDP checkpoint 合并成 HF BF16 safetensors（纯 CPU，不需要 GPU）。
# 只读 actor ckpt 里的 model_world_size_*_rank_*.pt 与 huggingface/ 配置/tokenizer，
# 不读 optimizer 状态，也不修改训练用的 full checkpoint。
#
# 用法: bash deploy/export_final_hf.sh [ckpt_root] [target]
#   ckpt_root: 实验目录（自动取最新 global_step_*）或某个 global_step_*/actor 目录
#   target   : 输出目录，产物落在 <target>/
#   DRY=1    : 只打印将要执行的命令
#
# 例:
#   bash deploy/export_final_hf.sh
#   bash deploy/export_final_hf.sh outputs/checkpoints/spreadsheet-rl/exp/global_step_50/actor exports/step50
#   DRY=1 bash deploy/export_final_hf.sh
set -e
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$REPO/deploy/env.sh"

CKPT_ROOT=${1:-${CKPT_ROOT:-$REPO/outputs/checkpoints}}
TARGET=${2:-${TARGET:-$REPO/outputs/hf_export}}

# 1. 定位含 fsdp_config.json 的 actor 目录（不跨目录猜测，找不到就报错）
if [ -f "$CKPT_ROOT/fsdp_config.json" ]; then
    ACTOR_DIR="$CKPT_ROOT"
elif [ -f "$CKPT_ROOT/actor/fsdp_config.json" ]; then
    ACTOR_DIR="$CKPT_ROOT/actor"
else
    LATEST_STEP=$(ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null | sed 's#.*global_step_##' | sort -n | tail -1)
    if [ -z "$LATEST_STEP" ]; then
        echo "[export] 错误: $CKPT_ROOT 下既不是 FSDP checkpoint，也没有 global_step_* 子目录" >&2
        exit 1
    fi
    ACTOR_DIR="$CKPT_ROOT/global_step_$LATEST_STEP/actor"
fi

# 2. 校验输入（merger 需要 config/tokenizer + 每个 rank 的 model shard）
if [ ! -f "$ACTOR_DIR/fsdp_config.json" ]; then
    echo "[export] 错误: $ACTOR_DIR 缺 fsdp_config.json，不是 FSDP checkpoint" >&2
    exit 1
fi
if [ ! -d "$ACTOR_DIR/huggingface" ]; then
    echo "[export] 错误: $ACTOR_DIR 缺 huggingface/（config + tokenizer），无法写出可用的 HF 目录" >&2
    exit 1
fi
SHARD_COUNT=$(ls "$ACTOR_DIR"/model_world_size_*_rank_*.pt 2>/dev/null | wc -l | tr -d ' ')
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "[export] 错误: $ACTOR_DIR 下没有 model_world_size_*_rank_*.pt" >&2
    exit 1
fi

# 3. CPU merge：world_size/DTensor placement 由 ckpt 自身决定，无需知道训练时的 GPU 数
CMD=("${PYTHON:-python3}" -m verl.model_merger merge --backend fsdp \
    --local_dir "$ACTOR_DIR" --target_dir "$TARGET")

echo "[export] actor ckpt: $ACTOR_DIR ($SHARD_COUNT shards)"
echo "[export] target    : $TARGET"
echo "[export] command   : ${CMD[*]}"

if [ "${DRY:-0}" = "1" ]; then
    echo "[export] DRY=1: 不执行"
    exit 0
fi

mkdir -p "$(dirname "$TARGET")"
"${CMD[@]}"
echo "[export] 完成: $TARGET"