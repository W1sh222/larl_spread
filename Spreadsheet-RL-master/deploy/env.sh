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
# 打分并发必须 <= Windows 侧 Excel 实例数：超出会让单实例收到并发 COM 调用，
# Excel（STA）直接抛 RPC_E_CALL_REJECTED，服务端无重试 → 该轨迹判 0 分。
# 当前 Windows 侧实际起了 2 个实例（health: configured_instances=2），这里就设 2。
export SPREADSHEET_RL_REWARD_MAX_CONCURRENT=${SPREADSHEET_RL_REWARD_MAX_CONCURRENT:-2}
# 工具侧同时上传的重算请求上限：与服务端 Excel 实例数匹配。
# 不设限时，每个 AgentLoopWorker 会把上百个工作簿同时推给 Windows，形成上传风暴。
export SHEET_ARENA_RECALC_MAX_CONCURRENT=${SHEET_ARENA_RECALC_MAX_CONCURRENT:-2}
export PYTHONPATH="$REPO/verl${PYTHONPATH:+:$PYTHONPATH}"

# 临时目录统一放共享盘：根分区 7T 已用 97%，raylet 在 /tmp 超 95% 时会持续告警并拒绝 object spill
# 注意路径必须短：ray dashboard 的 AF_UNIX socket 路径上限 107 字节，超长会导致 MetricsHead 启动失败
export SPREADSHEET_RL_TMP_ROOT=${SPREADSHEET_RL_TMP_ROOT:-/remote-home/share/yichengfu/rt}
export RAY_TMPDIR=${RAY_TMPDIR:-$SPREADSHEET_RL_TMP_ROOT/ray}
export TMPDIR=${TMPDIR:-$SPREADSHEET_RL_TMP_ROOT/py}
mkdir -p "$RAY_TMPDIR" "$TMPDIR" 2>/dev/null || true

# CUDA 12 的 nvcc 不支持 flashinfer 采样 kernel 的 --compress-mode=size，
# 关闭 flashinfer 采样，回退到 vLLM 原生采样实现
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

# 减少显存碎片：update_actor 末期长序列反传需要大块连续显存，
# expandable_segments 让分配器复用/扩展已有段而非索要新连续块，避免碎片型 OOM。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}

# fla / 线性注意力 Triton 内核的 JIT 编译缓存持久化（避开 /tmp，重启不丢），
# 省去每次启动 ~50s 的重编译；所有 ray worker 共享同一份缓存。
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/remote-home/yichengfu/.triton_cache}
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true

# 本环境 HuggingFace Hub 不可达（DNS 被拦），强制离线避免 transformers 拉取 hub kernel
# 时网络超时/挂起；优化内核走本地已装的 fla / causal_conv1d。
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
