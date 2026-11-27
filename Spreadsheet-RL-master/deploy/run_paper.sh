#!/bin/bash
# 按论文 A.14（arXiv 2605.22642v1, Table 3）的超参训练 Qwen3-4B-Thinking-2507。
# 与仓库 configs/qwen3-4b.sh 的差异（仓库值 → 论文值）：
#   lr 2e-6 → 1e-6；prompts/step 128 → 32；N 8 → 16；mini-batch 64 → 32；
#   每步 2 次优化更新（ppo_epochs=2）；prompt/response 8192/24576 → 4096/27648；
#   max turns 15 → 20；训练 60 步（论文）。
# 为控制耗时（A100 单步 4.5h → 60 步约 11 天），在两处偏离论文：
#   ppo_epochs 2 → 1、max_response 27648 → 24576。
# 用户额外要求：每 2 步存 ckpt + 每 4 步在 300 条验证集上评估
# （验证集按 4096 prompt 长度过滤后实际参与评估 153 条）。
set -e
REPO=${REPO:-/remote-home/yichengfu/larl/Spreadsheet-RL-master}
EXP=${EXP_NAME:-fullparam-4b-32x16-paper}
CKPT_ROOT=${CKPT_ROOT:-/remote-home/share/yichengfu/checkpoints}

cd "$REPO"
env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7} \
  MODEL_PATH=/remote-home/yichengfu/models/Qwen3-4B-Thinking-2507 \
  NGPUS=4 GEN_TP=1 GMU=${GMU:-0.6} \
  ATTN_IMPL=sdpa \
  MAX_PROMPT_LEN=4096 MAX_RESP_LEN=24576 MAX_MODEL_LEN=28672 \
  ACTOR_MAX_TOKEN_LEN=29696 LOGPROB_MAX_TOKEN_LEN=65536 \
  ENABLE_THINKING=False BUCKET_MB=4096 ENFORCE_EAGER=True \
  REMOVE_PADDING=True USE_FUSED_KERNELS=True FUSED_BACKEND=torch \
  bash deploy/run_grpo.sh \
    data.train_files="$REPO/data/train_hermes.parquet" \
    data.val_files="$REPO/data/val_300_seed42.parquet" \
    data.train_batch_size=32 \
    data.val_batch_size=64 \
    data.seed=42 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
    trainer.total_training_steps=60 \
    trainer.save_freq=2 \
    trainer.test_freq=4 \
    trainer.val_before_train=True \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$CKPT_ROOT/spreadsheet-rl/$EXP" \
    trainer.rollout_data_dir="$REPO/rollout_data/spreadsheet-rl/$EXP" \
    trainer.validation_data_dir="$REPO/validation_data/spreadsheet-rl/$EXP"
