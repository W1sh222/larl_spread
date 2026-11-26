#!/bin/bash
# 单机 GRPO 训练入口（A100 默认用 sdpa 注意力，config 里默认的 flash_attention_3 只适用于 Hopper）
# 用法: NGPUS=4 GEN_TP=1 GMU=0.85 bash run_grpo.sh [额外 hydra 覆盖参数]
set -e
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONDA_ROOT=${CONDA_ROOT:-$HOME/anaconda3}

source "$REPO/deploy/env.sh"
export PATH="$CONDA_ROOT/envs/spread/bin:$PATH"
export MODEL_PATH=${MODEL_PATH:-$REPO/models/Qwen3.5-9B}
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
# 训练侧（actor 反传 / ref 与 rollout 的 log_prob 重算）每个 micro-batch 的 token 上限。
# lm_head logits = seq_len × vocab(248320) × 2B，长序列下单次分配可达 12+ GiB，
# 是 update_actor 末期 OOM 的主因。把训练 token 预算降到 MAX_MODEL_LEN 以下，
# 让反传按更小的 token 块切分（数学等价，仅多几个 micro-batch），峰值显存随 token 数线性下降。
# rollout 生成长度仍由 MAX_MODEL_LEN 决定，不受影响。默认对齐 MAX_MODEL_LEN，可被环境变量覆盖。
ACTOR_MAX_TOKEN_LEN=${ACTOR_MAX_TOKEN_LEN:-$MAX_MODEL_LEN}
LOGPROB_MAX_TOKEN_LEN=${LOGPROB_MAX_TOKEN_LEN:-$MAX_MODEL_LEN}
# Qwen3.5 默认会输出思考过程，会吃掉大段 response 预算，默认关闭（与已验证的 9B 配置一致）
ENABLE_THINKING=${ENABLE_THINKING:-False}
# 9B 的 embed_tokens 张量约 2.03GB（248320×4096 bf16），超过默认 1024MB 传输桶会导致
# update_weights 断言失败 "Weight ... too large to fit in the bucket"
BUCKET_MB=${BUCKET_MB:-4096}
# A100 上 Qwen3.5 的 CUDA graph（FULL_AND_PIECEWISE）会偶发卡死：TP worker 在
# sample_tokens 阶段无响应，EngineCore 报 RPC 超时后引擎死亡，整卡进入错误态。
# 默认用 eager 模式（关闭 cuda graph），吞吐略降但稳定。
ENFORCE_EAGER=${ENFORCE_EAGER:-True}
# 是否去除 padding 走变长注意力（省激活显存）。默认 False（与已验证的 A100 sdpa 路径一致）；
# 设 REMOVE_PADDING=True 开启，需验证与 sdpa/无 flash_attn 的兼容性。
REMOVE_PADDING=${REMOVE_PADDING:-False}
# 是否用 fused linear-cross-entropy（不物化全量 lm_head logits，峰值省 10+ GiB）。
# 上推上下文（max_resp≥20k）后单 micro-batch logits 可达 13-16GiB，是 update_actor OOM 主因，
# 开启 fused 后 logits 按 512-token 分块计算，峰值降到 ~0.5GiB。默认 False（保守）；
# 设 USE_FUSED_KERNELS=True 开启。注意：triton backend 在 remove_padding 路径下返回
# 非连续 log_probs，反传会触发 is_contiguous 断言 → 必须配 FUSED_BACKEND=torch。
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-False}
FUSED_BACKEND=${FUSED_BACKEND:-triton}

cd "$REPO"
bash configs/qwen3-4b.sh "$@" \
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESP_LEN" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LEN" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_MAX_TOKEN_LEN" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKEN_LEN" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKEN_LEN" \
    actor_rollout_ref.model.override_config.attn_implementation="$ATTN_IMPL" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="$BUCKET_MB" \
    actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER" \
    actor_rollout_ref.model.use_remove_padding="${REMOVE_PADDING:-False}" \
    actor_rollout_ref.model.use_fused_kernels="$USE_FUSED_KERNELS" \
    actor_rollout_ref.model.fused_kernel_options.impl_backend="$FUSED_BACKEND" \
    trainer.n_gpus_per_node="$NGPUS" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="$NGPUS" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$GEN_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GMU" \
    trainer.logger='["console"]'
