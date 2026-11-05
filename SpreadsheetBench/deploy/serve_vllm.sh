#!/bin/bash
# 启动 Qwen3.5-9B 的 vLLM OpenAI 兼容服务
# 用法: PORT=8000 GPU=4 bash serve_vllm.sh
set -e

MODEL=${MODEL:-/remote-home/yichengfu/models/Qwen3.5-9B}
PORT=${PORT:-8000}
GPU=${GPU:-4}
TP=${TP:-1}
SERVED_NAME=${SERVED_NAME:-Qwen3.5-9B}
MAX_LEN=${MAX_LEN:-32768}
GMU=${GMU:-0.90}
CONDA_ROOT=${CONDA_ROOT:-/remote-home/yichengfu/anaconda3}

export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
# flashinfer 采样 kernel 需要 JIT 编译，这里直接用原生采样避免编译依赖
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$CONDA_ROOT/envs/vllm/bin:$PATH"

exec "$CONDA_ROOT/envs/vllm/bin/vllm" serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GMU"
