#!/bin/bash
# 用指定权重在指定验证集上做 val-only 评估（不训练、不存档）
# 用法: bash deploy/eval_ckpt.sh <base|ckpt_dir> <val_parquet> <exp_name> [val_batch]
set -e
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL_PATH="${MODEL_PATH:-$REPO/models/Qwen3-4B-Thinking-2507}"
CKPT=${1:?需要 base 或 checkpoint 目录}
VALF=${2:?需要验证集 parquet}
EXP=${3:?需要实验名}
VAL_BATCH=${4:-100}

if [ "$CKPT" = "base" ]; then
  RESUME_ARGS="trainer.resume_mode=disable"
else
  [ -f "$CKPT/data.pt" ] || { echo "错误：$CKPT 不是完整 checkpoint"; exit 1; }
  RESUME_ARGS="trainer.resume_mode=resume_path trainer.resume_from_path=$CKPT"
fi

cd "$REPO"
env CUDA_VISIBLE_DEVICES=4,5,6,7 \
  MODEL_PATH="$MODEL_PATH" \
  TRAIN_BATCH_SIZE=32 PPO_MINI_BATCH=16 N_SAMPLE=8 \
  TOTAL_STEPS=1 SAVE_FREQ=10000 TEST_FREQ=10000 VAL_BEFORE_TRAIN=True VAL_ONLY=True \
  VAL_FILES="$VALF" VAL_BATCH_SIZE="$VAL_BATCH" \
  LR=2e-6 KL_COEF=0.001 TEMP=0.6 \
  MAX_RESP_LEN=24576 MAX_MODEL_LEN=33792 ACTOR_MAX_TOKEN_LEN=33792 \
  GMU=0.6 USE_FUSED_KERNELS=True FUSED_BACKEND=torch \
  SPREADSHEET_RL_REWARD_MAX_CONCURRENT=8 \
  EXP_NAME="$EXP" \
  EXTRA_ARGS="$RESUME_ARGS" \
  bash deploy/run_debug.sh
