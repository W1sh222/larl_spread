#!/bin/bash
# 单机 GRPO 训练入口（A100 默认用 sdpa 注意力，config 里默认的 flash_attention_3 只适用于 Hopper）
# 用法: NGPUS=4 GEN_TP=1 GMU=0.85 bash run_grpo.sh [额外 hydra 覆盖参数]
set -e
REPO=${REPO:-/remote-home/yichengfu/larl/Spreadsheet-RL-master}
CONDA_ROOT=${CONDA_ROOT:-/remote-home/yichengfu/anaconda3}

source "$REPO/deploy/env.sh"
export PATH="$CONDA_ROOT/envs/spread/bin:$PATH"
export MODEL_PATH=${MODEL_PATH:-/remote-home/yichengfu/models/Qwen3.5-9B}
export WANDB_MODE=${WANDB_MODE:-offline}

NGPUS=${NGPUS:-4}
GEN_TP=${GEN_TP:-1}
GMU=${GMU:-0.85}
ATTN_IMPL=${ATTN_IMPL:-sdpa}

cd "$REPO"
bash configs/qwen3-4b.sh "$@" \
    actor_rollout_ref.model.override_config.attn_implementation="$ATTN_IMPL" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    trainer.n_gpus_per_node="$NGPUS" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="$NGPUS" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$GEN_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GMU" \
    trainer.logger='["console"]'
