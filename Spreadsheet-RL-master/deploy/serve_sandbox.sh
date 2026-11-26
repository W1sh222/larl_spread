#!/bin/bash
# 启动 SandboxFusion 代码执行服务（hpc 配置 = 无容器隔离，直接子进程执行）
# 用法: PORT=8080 bash serve_sandbox.sh
set -e
PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}
CONDA_ROOT=${CONDA_ROOT:-$HOME/anaconda3}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export SANDBOX_CONFIG=${SANDBOX_CONFIG:-hpc}
# workspace 请求依赖该数据根；不设则回落到不存在的 /opt/sandboxfusion/data，
# 导致所有带 workspace_id 的调用报 "workspace requested but could not be prepared"。
export SANDBOX_DATA_ROOT=${SANDBOX_DATA_ROOT:-$REPO/data}
export PYTHONPATH="$REPO/sandbox${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO/sandbox"
exec "$CONDA_ROOT/envs/sandbox/bin/uvicorn" sandbox.server.server:app --host "$HOST" --port "$PORT"
