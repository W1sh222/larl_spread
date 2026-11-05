#!/bin/bash
# 启动 SandboxFusion 代码执行服务（hpc 配置 = 无容器隔离，直接子进程执行）
# 用法: PORT=8080 bash serve_sandbox.sh
set -e
PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}
CONDA_ROOT=${CONDA_ROOT:-/remote-home/yichengfu/anaconda3}
REPO=${REPO:-/remote-home/yichengfu/larl/Spreadsheet-RL-master}

export SANDBOX_CONFIG=${SANDBOX_CONFIG:-hpc}
export PYTHONPATH="$REPO/sandbox${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO/sandbox"
exec "$CONDA_ROOT/envs/sandbox/bin/uvicorn" sandbox.server.server:app --host "$HOST" --port "$PORT"
