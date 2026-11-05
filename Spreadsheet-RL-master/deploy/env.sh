#!/bin/bash
# Spreadsheet-RL 运行环境变量（source 后使用）
# 数据集: /remote-home/yichengfu/larl/Spreadsheet-RL-master/data
# 奖励/重算服务: Windows 172.16.6.155:5000（Excel COM，需保持登录状态）

export REPO=${REPO:-/remote-home/yichengfu/larl/Spreadsheet-RL-master}
export SPREADSHEET_RL_DATA_ROOT=${SPREADSHEET_RL_DATA_ROOT:-$REPO/data}
export SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL:-http://127.0.0.1:8080/run_code}
export SPREADSHEET_RL_REWARD_URL=${SPREADSHEET_RL_REWARD_URL:-http://172.16.6.155:5000/reward/submit}
export SPREADSHEET_RL_RECALC_URL=${SPREADSHEET_RL_RECALC_URL:-http://172.16.6.155:5000/recalculate}
export SPREADSHEET_RL_REWARD_TIMEOUT_S=${SPREADSHEET_RL_REWARD_TIMEOUT_S:-600}
export SPREADSHEET_RL_REWARD_MAX_CONCURRENT=${SPREADSHEET_RL_REWARD_MAX_CONCURRENT:-4}
export PYTHONPATH="$REPO/verl${PYTHONPATH:+:$PYTHONPATH}"

# CUDA 12 的 nvcc 不支持 flashinfer 采样 kernel 的 --compress-mode=size，
# 关闭 flashinfer 采样，回退到 vLLM 原生采样实现
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
