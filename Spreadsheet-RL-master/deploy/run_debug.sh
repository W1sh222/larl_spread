#!/bin/bash
# 调试训练（LoRA 版）：验证「修复坏文件 + LoRA 省显存」后训练是否正向/不退化
# 用法: bash deploy/run_debug.sh   [可选 LORA_RANK=64 TOTAL_STEPS=6 ...]
set -e
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export LOG_DIR=${LOG_DIR:-$REPO/logs/debug}
mkdir -p "$LOG_DIR"

# PYTORCH_CUDA_ALLOC_CONF / TRITON_CACHE_DIR / HF 离线等统一在 deploy/env.sh 设置
# （run_grpo.sh 会 source env.sh），避免这里覆盖掉更完整的默认值。

# ---- 可调超参 ----
# gate 跑默认全参（LORA_RANK=0）—— 已验证能跑到 step8 不 OOM 的 golden 路径。
# 确认正向/非退化后，再设 LORA_RANK>0 开启 LoRA 以省下梯度显存、上推上下文。
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-32}        # alpha == rank → 缩放因子 1，LoRA 常见稳健设置
# 对齐 golden（存活 8 步 0 OOM）：gmu=0.50 / micro_batch=2 / val_before_train=True。
# 此前覆盖成 gmu=0.40 + micro_batch=1 + val_before_train=False 会让 update_actor
# 在 PyTorch 进程内累积 ~66GB 后再分配 13GiB logits 触发 OOM（vLLM cache 已正确释放，
# 656MiB；66GB 全在训练进程内）。
GMU=${GMU:-0.50}
MICRO_BATCH=${MICRO_BATCH:-2}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}   # debug 默认 64；正式跑设 128
TOTAL_STEPS=${TOTAL_STEPS:-6}
LR=${LR:-5e-7}                      # 保守起步，确认不退化后再考虑上调
KL_COEF=${KL_COEF:-0.01}
TEMP=${TEMP:-1.0}                   # 与 old_log_prob 重算温度一致，避免 importance-ratio 偏置
N_SAMPLE=${N_SAMPLE:-4}             # debug 用 n=4 省时间；正式跑 n=8
# val 集 / mini_batch 可调：gate 求快时用小 val 子集 + 小 batch 加速，
# update_actor 的 OOM 代表性由 ppo_max_token_len_per_gpu(dynamic_bsz 单 micro-batch token 上限)
# 决定，与 train_batch_size 无关 → 小 batch 仍能验证 OOM。
VAL_FILES=${VAL_FILES:-$REPO/data/debug_val_300.parquet}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-300}
PPO_MINI_BATCH=${PPO_MINI_BATCH:-64}
# 省显存：去 padding（变长注意力，最大杠杆）。
# balance_batch 默认 False：实测 balance_batch=True 会把长序列聚集到同一 rank，
# 叠加 dynamic_bsz 的 workload 均衡打包，单 micro-batch 可达 29749 tokens
# （超过 ppo_max_token_len_per_gpu=24576），lm_head logits 分配 13.76GiB 触发 OOM。
# 对齐已验证能跑到 step8 的 golden 配置（balance_batch=False）。
# 可通过环境变量开启以快速对比：BALANCE_BATCH=True
REMOVE_PADDING=${REMOVE_PADDING:-True}
BALANCE_BATCH=${BALANCE_BATCH:-False}
# fused linear-cross-entropy：不物化全量 logits，是上推上下文后规避 update_actor OOM 的关键。
# 默认 False（gate 全参路径已验证不 OOM）；上推上下文时设 USE_FUSED_KERNELS=True。
# triton backend 在 remove_padding 下反传会 is_contiguous 断言失败 → 必须 FUSED_BACKEND=torch。
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-False}
FUSED_BACKEND=${FUSED_BACKEND:-torch}
# True = 只跑 val_before_train 就退出，不进入训练循环。用于判别 val-only 实验
# （避免 total_training_steps=0 仍可能尝试一步训练 → compute_log_prob/update_actor）。
VAL_ONLY=${VAL_ONLY:-False}
# 存档/验证频率（冒烟 gate 用 SAVE_FREQ=1 测保存路径、TEST_FREQ 调大跳过 val 节省时间）
SAVE_FREQ=${SAVE_FREQ:-2}
TEST_FREQ=${TEST_FREQ:-1}
# 额外 hydra 覆盖（空格分隔），便于不改脚本直接迭代。例：EXTRA_ARGS="actor.fsdp_config.x=y"
EXTRA_ARGS=${EXTRA_ARGS:-}
# 终止约束：单条轨迹最多允许的 assistant 轮数（每轮≈1 次工具调用）。
# 已验证病态：预算变大后模型做完不停、继续"验证式改表"，把正确状态覆盖掉（详见日志分析）。
# 用小值可做 A/B，验证截断尾部是否提升 val。
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-15}
# 实验名（决定 validation/rollout/ckpt 落盘目录），不同 arm 用不同名字避免互相覆盖。
EXP_NAME=${EXP_NAME:-debug-qwen35-9b}

# LoRA 仅在 LORA_RANK>0 时启用；LORA_RANK=0 走全参（对齐已验证不 OOM 的 golden 配置）。
# 实测 LoRA(冻结 base) + activation checkpointing 会让 update_actor 反传时显存逐 micro-batch
# 泄漏（30GB→78GB）触发 OOM，而全参 balance_batch=False 可稳定跑到 step8。
LORA_ARGS=()
if [ "${LORA_RANK:-0}" -gt 0 ]; then
  LORA_ARGS+=(actor_rollout_ref.model.lora_rank="$LORA_RANK")
  LORA_ARGS+=(actor_rollout_ref.model.lora_alpha="$LORA_ALPHA")
  LORA_ARGS+=(actor_rollout_ref.model.target_modules=all-linear)
  LORA_ARGS+=(actor_rollout_ref.model.lora.merge=True)
fi

# 上下文预算：默认 response 16384 + model_len 24576（prompt 8192 + resp 16384 不截断），
# 对齐已验证能跑到 step8 的 golden 配置；上探 baseline（max_resp=24576）时用：
#   MAX_RESP_LEN=24576 MAX_MODEL_LEN=33792 \
#   ACTOR_MAX_TOKEN_LEN=33792 LOGPROB_MAX_TOKEN_LEN=65536 USE_FUSED_KERNELS=True
# 关键约束：多轮 agent loop 里 tool 响应 token 也计入 response_mask（loss 掩码为 0），
# 因此单样本总长上界 = max_prompt(8192) + max_response(24576) = 32768，再叠加 chat-template
# 每轮 wrapper 开销（≈几十 token）。MAX_MODEL_LEN 必须 ≥ 此上界，否则 vLLM 在 generate 时
# 抛 "Prompt length exceeds max context"（实测 28834 > 28672 即因此崩溃）。
# 故 MAX_MODEL_LEN 取 32768+1024=33792；ACTOR_MAX_TOKEN_LEN 必须 ≥ 最长单样本（33792），
# 否则 rearrange_micro_batches 的 max_token_len>=max_seq_len 断言失败。
# 上推后单 micro-batch 全量 logits 可达 16GiB，必须开 USE_FUSED_KERNELS=True（torch backend）
# 才不 OOM。
# 训练侧 token 上限 ACTOR_MAX_TOKEN_LEN/LOGPROB_MAX_TOKEN_LEN 默认跟随 MAX_MODEL_LEN，
# 由 run_grpo.sh 传给 ppo/log_prob_max_token_len_per_gpu（dynamic_bsz 按此切 micro-batch）。
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-8192} \
MAX_RESP_LEN=${MAX_RESP_LEN:-16384} \
MAX_MODEL_LEN=${MAX_MODEL_LEN:-24576} \
ACTOR_MAX_TOKEN_LEN=${ACTOR_MAX_TOKEN_LEN:-$MAX_MODEL_LEN} \
LOGPROB_MAX_TOKEN_LEN=${LOGPROB_MAX_TOKEN_LEN:-$MAX_MODEL_LEN} \
GMU="$GMU" NGPUS=4 GEN_TP=1 REMOVE_PADDING="$REMOVE_PADDING" \
USE_FUSED_KERNELS="$USE_FUSED_KERNELS" FUSED_BACKEND="$FUSED_BACKEND" bash "$REPO/deploy/run_grpo.sh" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.rollout.n="$N_SAMPLE" \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.rollout.temperature="$TEMP" \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.val_only="$VAL_ONLY" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_ASSISTANT_TURNS" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="${CKPT_ROOT:-$REPO/outputs/checkpoints}/spreadsheet-rl/$EXP_NAME" \
    trainer.rollout_data_dir="$REPO/rollout_data/spreadsheet-rl/$EXP_NAME" \
    trainer.validation_data_dir="$REPO/validation_data/spreadsheet-rl/$EXP_NAME" \
    "${LORA_ARGS[@]}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH" \
    trainer.balance_batch="$BALANCE_BATCH" \
    $EXTRA_ARGS
