#!/bin/bash
# GiGPO 초소형 스모크 공통 환경 (실측값 기준).
# 모든 0x_/run_/tmux_ 스크립트는 맨 먼저 이 파일을 source 한다.
#
# 이 머신의 실측 상태 (2026-11-22 계측):
#   - GPU 0-3 : 다른 작업 점유(약 49GB씩) -> 사용 금지.  GPU 4-7 : A100 80GB, 18MiB(유휴)
#   - 학습 venv : 기존 Spreadsheet-RL-master/.venv 는 torch 2.6.0+cu124 / vllm 0.8.5
#                 -> vLLM 0.8.5 는 Qwen3_5 arch 를 registry 에 가지고 있지 않아 9B 로드 불가.
#                 docs/environment.md + PyPI 실측 조합(torch 2.13.0+cu129 / vllm 0.32.0,
#                 vllm 0.32.0 이 transformers<5.20 요구, Qwen3_5ForConditionalGeneration 지원)을
#                 아래 VENV_DIR 에 새로 만든다.
#   - Qwen3.5-9B : arch=Qwen3_5ForConditionalGeneration (vision 포함 멀티모달), 4 shards.
#                 deploy/run_grpo.sh 가 이미 9B 전용 옵션(sdpa/eager/bucket4096/thinking off)을 갖고
#                 있어 실행은 그 스크립위를 얹는다.
#   - 보상 서비스 : Windows 172.16.6.155:5000 (healthy, Excel 4) - Linux에선 구동 불가
#   - 네트워크 : PyPI / download.pytorch.org / hf-mirror / modelscope / raw.githubusercontent 도달,
#                huggingface.co 차단
#   - HF_ENDPOINT : 이 컨테이너 값이 "[https://hf-mirror.com](https://hf-mirror.com)" 라는
#                   마크다운 문자열 -> 그대로 쓰면 전부 실패. 여기서 강제 정상화.

SMOKE_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SMOKE_HOME

REPO=${REPO:-"/root/work/larl_spread/Spreadsheet-RL-master"}
export REPO
export VERL_SRC="${REPO}/verl"

# ---------------------------------------------------------------- GPU (4,5,6,7 만)
export GPUS=${GPUS:-"4,5,6,7"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"${GPUS}"}
export NGPUS=${NGPUS:-$(awk -F',' '{print NF}' <<< "${GPUS}")}
export GEN_TP=${GEN_TP:-1}
export GMU=${GMU:-0.5}           # rollout gpu_memory_utilization (스모크는 KV 적게)
export ATTN_IMPL=${ATTN_IMPL:-"sdpa"}
export ENFORCE_EAGER=${ENFORCE_EAGER:-"True"}
export BUCKET_MB=${BUCKET_MB:-4096}   # 9B embed_tokens ~2.03GB > 기본 1024MB 버킷

# ---------------------------------------------------------------- 경로
export VENV_DIR=${VENV_DIR:-"${SMOKE_HOME}/.venv-spread"}
export PY=${PY:-"${VENV_DIR}/bin/python"}
export UV=${UV:-"$(command -v uv || echo /root/.local/bin/uv)"}
export CONDA_SHIM=${CONDA_SHIM:-"${SMOKE_HOME}/.conda_shim"}   # deploy/run_grpo.sh 의 PATH 주입용

# 설치할 고정 조합 (docs/environment.md + PyPI 실측)
export PY_VERSION=${PY_VERSION:-"3.12"}
export TORCH_VERSION=${TORCH_VERSION:-"2.13.0"}
export TORCH_INDEX=${TORCH_INDEX:-"https://download.pytorch.org/whl/cu129"}
export VLLM_VERSION=${VLLM_VERSION:-"0.32.0"}

# ---------------------------------------------------------------- 모델 (Qwen3.5-9B)
export SMOKE_MODEL=${SMOKE_MODEL:-"Qwen/Qwen3.5-9B"}
export MODEL_PATH=${MODEL_PATH:-"${SMOKE_HOME}/models/Qwen3.5-9B"}
export MODEL_SOURCE=${MODEL_SOURCE:-"auto"}     # auto | hf(=hf-mirror) | modelscope

# ---------------------------------------------------------------- 데이터
export SPREADSHEET_RL_DATA_ROOT=${SPREADSHEET_RL_DATA_ROOT:-"${SMOKE_HOME}/data"}
export HF_PARQUET_TRAIN="${SPREADSHEET_RL_DATA_ROOT}/train_hermes.parquet"
export DATA_ZIP="${SPREADSHEET_RL_DATA_ROOT}/spreadsheets.zip"
export TRAIN_FILE=${TRAIN_FILE:-"${SPREADSHEET_RL_DATA_ROOT}/train_smoke.parquet"}
export TEST_FILE=${TEST_FILE:-"${SPREADSHEET_RL_DATA_ROOT}/test_smoke.parquet"}
export SUBSET_TASKS=${SUBSET_TASKS:-"${OUT_DIR:-${SMOKE_HOME}/outputs}/subset_tasks.txt"}

# ---------------------------------------------------------------- 실행 아티팩트
export LOG_DIR=${LOG_DIR:-"${SMOKE_HOME}/logs"}
export OUT_DIR=${OUT_DIR:-"${SMOKE_HOME}/outputs"}
export CKPTS_DIR=${CKPTS_DIR:-"${SMOKE_HOME}/checkpoints"}
export TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-"${OUT_DIR}/tools_no_sandbox.yaml"}
export CONFIG_PATH=${CONFIG_PATH:-"${REPO}/configs"}

# ---------------------------------------------------------------- 보상 경로
# REWARD=windows : 실제 Excel 재계산 보상(Windows 172.16.6.155, 태스크가 Windows 쪽에도 있어야 >0)
# REWARD=mock    : 로컬 워크북 diff 기반 의사보상. Windows 없이 GiGPO 배선만 검증.
export REWARD=${REWARD:-"mock"}
export SPREADSHEET_RL_REWARD_URL=${SPREADSHEET_RL_REWARD_URL:-"http://172.16.6.155:5000/reward/submit"}
export SPREADSHEET_RL_RECALC_URL=${SPREADSHEET_RL_RECALC_URL:-"http://172.16.6.155:5000/recalculate"}
export SPREADSHEET_RL_REWARD_TIMEOUT_S=${SPREADSHEET_RL_REWARD_TIMEOUT_S:-1200}
export SPREADSHEET_RL_REWARD_MAX_CONCURRENT=${SPREADSHEET_RL_REWARD_MAX_CONCURRENT:-4}
export SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL:-"http://127.0.0.1:8080/run_code"}
export MOCK_REWARD_PATH=${MOCK_REWARD_PATH:-"${SMOKE_HOME}/scripts/mock_reward.py"}

# ---------------------------------------------------------------- 다운로드 캐시
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_HOME=${HF_HOME:-"${OUT_DIR}/hf_cache"}
export TRANSFORMERS_CACHE="${HF_HOME}"
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-"${OUT_DIR}/modelscope_cache"}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"${OUT_DIR}/triton_cache"}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"${OUT_DIR}/torch_extensions"}
export RAY_TMPDIR=${RAY_TMPDIR:-"${OUT_DIR}/ray"}
export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_DIR=${WANDB_DIR:-"${OUT_DIR}/wandb"}

# ---------------------------------------------------------------- verl 런타임
export PYTHONPATH="${VERL_SRC}:${PYTHONPATH:-}"
export SPREADSHEET_RL_VERL_CONFIG_DIR=${SPREADSHEET_RL_VERL_CONFIG_DIR:-"${VERL_SRC}/verl/trainer/config"}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export ENABLE_THINKING=${ENABLE_THINKING:-"False"}

# ---------------------------------------------------------------- 스모크 크기 (초소형)
export MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-8192}
export MAX_RESP_LEN=${MAX_RESP_LEN:-8192}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}      # >= prompt+response
export MAX_STEPS=${MAX_STEPS:-4}                  # multi_turn assistant/user turn 상한 (스모크용)
export SUBSET_SIZE=${SUBSET_SIZE:-8}              # 태스크 수
export TRAIN_BS=${TRAIN_BS:-8}                    # prompt 수 (=SUBSET_SIZE 로 두면 1 step/epoch)
export ROLLOUT_N=${ROLLOUT_N:-4}                  # trajectory per prompt -> GiGPO 그룹 크기
export MINI_BS=${MINI_BS:-8}
export TOTAL_STEPS=${TOTAL_STEPS:-2}
export LR=${LR:-1e-6}

# ---------------------------------------------------------------- GiGPO 모듈 설정
# adv_estimator 은 항상 credit_assignment (모듈 진입점), method 는 gigpo.
export CA_METHOD=${CA_METHOD:-"gigpo"}
export CA_ANCHOR=${CA_ANCHOR:-"state"}            # state | observation
export CA_MODE=${CA_MODE:-"mean_norm"}            # mean_norm | mean_std_norm
export CA_GAMMA=${CA_GAMMA:-0.95}
export CA_STEP_W=${CA_STEP_W:-1.0}
export CA_GROUPING=${CA_GROUPING:-"exact"}
export CA_COLLECT_STATES=${CA_COLLECT_STATES:-"True"}   # anchor=state 이면 transitions 수집 필수
export CA_INCLUDE_TRUNCATED_BASELINE=${CA_INCLUDE_TRUNCATED_BASELINE:-"True"}
# policy loss 는 건드리지 않는다. 공식 GiGPO(d8a9b70 / f974dc5 / 20bd331) 는 step-level
# objective 를 쓰지 않고, step advantage 를 유효 assistant token 에 broadcast 한 뒤
# 기존 vanilla token-level PPO 를 그대로 호출한다. 고로 여기엔 LOSS_MODE / STEP_* 계열
# override 가 존재하지 않으며, actor.policy_loss.loss_mode 는 config 기본값 "vanilla".
# (구현 근거: verl/tests/trainer/ppo/test_gigpo_vanilla_link_on_cpu.py D1-D8 단언)

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${CKPTS_DIR}" "${SPREADSHEET_RL_DATA_ROOT}" \
         "${MODEL_PATH%/*}" "${HF_HOME}" "${RAY_TMPDIR}" "${TRITON_CACHE_DIR}"

log(){ echo "[gigpo_smoke] $*"; }
die(){ echo "[gigpo_smoke][ERROR] $*" >&2; exit 1; }

# GiGPO 모듈 관련 hydra override 묶음 (run_gigpo_smoke.sh / check 시 reused).
# actor/policy_loss 쪽 override 는 의도적으로 없음: Credit Assignment 모듈은 advantage 텐서만
# 교체하고 손실·optimizer 는 기존 vanilla 경로를 쓴다.
gigpo_overrides(){
    cat <<EOF
algorithm.adv_estimator=credit_assignment
algorithm.credit_assignment.method=${CA_METHOD}
algorithm.credit_assignment.collect_step_states=${CA_COLLECT_STATES}
++algorithm.credit_assignment.params.anchor=${CA_ANCHOR}
++algorithm.credit_assignment.params.anchor_scope=full
++algorithm.credit_assignment.params.grouping=${CA_GROUPING}
++algorithm.credit_assignment.params.mode=${CA_MODE}
++algorithm.credit_assignment.params.gamma=${CA_GAMMA}
++algorithm.credit_assignment.params.step_advantage_w=${CA_STEP_W}
++algorithm.credit_assignment.params.include_truncated_in_baseline=${CA_INCLUDE_TRUNCATED_BASELINE}
EOF
}
export -f gigpo_overrides 2>/dev/null || true

# 스모크 전체 hydra override (크기 + 모듈 + 보상 + 산출물). run_gigpo_smoke.sh 와 00_preflight.sh,
# scripts/check_module.py 가 같은 목록을 공유한다 (설정 drifting 방지).
smoke_args(){
    local rundir=${RUN_DIR:-${OUT_DIR}/runs/check}
    cat <<EOF
data.train_files=${TRAIN_FILE}
data.val_files=${TEST_FILE}
data.train_batch_size=${TRAIN_BS}
data.val_batch_size=${TRAIN_BS}
actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BS}
actor_rollout_ref.rollout.n=${ROLLOUT_N}
actor_rollout_ref.actor.optim.lr=${LR}
actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_STEPS}
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_STEPS}
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
actor_rollout_ref.rollout.multi_turn.enable=True
trainer.rollout_data_dir=${rundir}/rollout_data
trainer.validation_data_dir=${rundir}/validation_data
trainer.default_local_dir=${rundir}/checkpoints
trainer.val_before_train=False
trainer.test_freq=-1
trainer.save_freq=-1
trainer.total_training_steps=${TOTAL_STEPS}
trainer.total_epochs=99
EOF
    [ -f "${TOOL_CONFIG_PATH}" ] && echo "actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH}"
    if [ "${REWARD}" = "mock" ]; then
        echo "reward.custom_reward_function.path=${MOCK_REWARD_PATH}"
        echo "reward.custom_reward_function.name=compute_score"
    fi
    gigpo_overrides
}
export -f smoke_args 2>/dev/null || true

# 시각별 런 디렉토리: 로그/롤아웃/체크포인트를 한 곳에 모은다.
new_run_dir(){
    local tag=${1:-run}
    local ts; ts=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="${OUT_DIR}/runs/${tag}_${ts}"
    mkdir -p "${RUN_DIR}/rollout_data" "${RUN_DIR}/validation_data" "${RUN_DIR}/checkpoints"
    export RUN_DIR
    ln -sfn "${RUN_DIR}" "${OUT_DIR}/runs/last"
    echo "${RUN_DIR}"
}
export -f new_run_dir 2>/dev/null || true
