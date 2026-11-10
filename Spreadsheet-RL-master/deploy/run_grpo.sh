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
# 长度预算必须满足 max_prompt_length + max_response_length <= max_model_len，
# 否则 rollout 会被 max_model_len 截断。9B 显存紧张，默认按 8k+16k=24k 配置。
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-8192}
MAX_RESP_LEN=${MAX_RESP_LEN:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-24576}
# Qwen3.5 默认会输出思考过程，会吃掉大段 response 预算，默认关闭（与已验证的 9B 配置一致）
ENABLE_THINKING=${ENABLE_THINKING:-False}

cd "$REPO"
bash configs/qwen3-4b.sh "$@" \
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESP_LEN" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LEN" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_MODEL_LEN" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_MODEL_LEN" \
    actor_rollout_ref.model.override_config.attn_implementation="$ATTN_IMPL" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    trainer.n_gpus_per_node="$NGPUS" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="$NGPUS" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$GEN_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GMU" \
    trainer.logger='["console"]'
