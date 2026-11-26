#!/bin/bash
# 06(실행). GiGPO 모듈 초소형 스모크 본편.
#   repo 의 검증된 9B 실행 경로(deploy/run_grpo.sh -> configs/qwen3-4b.sh -> verl.trainer.main_ppo)를
#   그대로 쓰고, 마지막에 GiGPO 모듈 override 를 얹는다:
#     algorithm.adv_estimator=credit_assignment + algorithm.credit_assignment.{method,params}
#  policy loss 는 건드리지 않는다. 공식 GiGPO 처럼 step advantage 를 유효 assistant token 에
#   broadcast 한 뒤 기존 vanilla token-level PPO + 기존 optimizer 가 그대로 학습한다.
#   → rollout(tool_agent) 이 ca_step_* 메타데이터를 내고, driver 가 credit_assignment adv 로
#     교체하고, actor/step_* 없이 vanilla 지표가 나온다. 모든 산출물은 RUN_DIR 에 모인다.
#
# 사용:
#   bash run_gigpo_smoke.sh                        (REWARD=mock, 8태스크 x n4 x 2step)
#   CA_ANCHOR=observation bash run_gigpo_smoke.sh  (앵커 A/B)
#   CA_STEP_W=0 bash run_gigpo_smoke.sh            (GRPO 와 동일한 advantage 대조군)
#   REWARD=windows  bash run_gigpo_smoke.sh        (실제 Excel 보상; 05 에서 1.0 확인된 뒤)
#   DRY=1 bash run_gigpo_smoke.sh                  (조립된 명령만 출력, 실행 안 함)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

export PATH="${VENV_DIR}/bin:${PATH}"      # ray / python / hydra CLI

new_run_dir smoke >/dev/null               # 하위 shell 이 아니라 현재 shell 에 RUN_DIR 을 넣는다
LOG="${RUN_DIR}/train.log"
say(){ echo "[run $(date +%H:%M:%S)] $*"; }
die(){ echo "[run][ERROR] $*" >&2; echo "로그: ${LOG}" >&2; exit 1; }

# ---------------------------------------------------------------- 사전조건
[ -x "${PY}" ]            || die "venv 없음 -> bash 01_setup_env.sh"
[ -f "${MODEL_PATH}/config.json" ] || die "모델 없음 -> bash 02_get_model.sh"
[ -f "${TRAIN_FILE}" ]    || die "서브셋 없음 -> bash 04_pick_subset.sh"
[ -f "${TOOL_CONFIG_PATH}" ] || say "[WARN] ${TOOL_CONFIG_PATH} 없음 -> repo 기본 tools(sandbox 호출 가능)"
NFREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v g="${GPUS}" 'BEGIN{n=split(g,a,",");for(i=1;i<=n;i++)want[a[i]]=1} want[$1]&&$2<2000' | wc -l)
[ "${NFREE}" -ge "${NGPUS}" ] || die "GPU 4-7 중 유휴 ${NFREE}장 < NGPUS=${NGPUS} (GPUS=... 로 축소 가능)"
[ -f "${MOCK_REWARD_PATH}" ] || die "mock reward 스크립트 없음: ${MOCK_REWARD_PATH}"

say "RUN_DIR=${RUN_DIR}"
say "actor=${SMOKE_MODEL}  GPUs=${GPUS}(유휴 ${NFREE})  NGPUS=${NGPUS} GEN_TP=${GEN_TP} GMU=${GMU}"
say "크기: tasks=${SUBSET_SIZE} train_bs=${TRAIN_BS} rollout_n=${ROLLOUT_N} steps=${TOTAL_STEPS} "
say "      prompt=${MAX_PROMPT_LEN} resp=${MAX_RESP_LEN} model_len=${MAX_MODEL_LEN} max_turns=${MAX_STEPS}"
say "모듈: adv=credit_assignment method=${CA_METHOD} anchor=${CA_ANCHOR} step_w=${CA_STEP_W} gamma=${CA_GAMMA}"
say "      loss=vanilla(고정, override 없음) reward=${REWARD}"

# ---------------------------------------------------------------- 이전 잔여물 정리
ray stop --head=false >/dev/null 2>&1 || true
pkill -f "verl.trainer.main_ppo" 2>/dev/null || true
rm -rf "${SPREADSHEET_RL_DATA_ROOT}/_workspaces" 2>/dev/null || true
mkdir -p "${SPREADSHEET_RL_DATA_ROOT}/_workspaces"

# ---------------------------------------------------------------- override 조립
# 싯수/모듈/보상/산출물 경로는 env.sh:smoke_args 가 단 하나의 소스 (preflight·check 와 동일 목록).
mapfile -t ARGS < <(smoke_args)
ARGS+=(trainer.experiment_name="gigpo_smoke_${CA_METHOD}_${CA_ANCHOR}_$(date +%H%M%S)"
       trainer.logger='["console"]')

# 실행 환경 (deploy/run_grpo.sh 의 env contract)
export REPO CONDA_ROOT="${CONDA_SHIM}" MODEL_PATH
export SPREADSHEET_RL_DATA_ROOT SPREADSHEET_RL_VERL_CONFIG_DIR PYTHONPATH
export NGPUS GEN_TP GMU ATTN_IMPL ENFORCE_EAGER BUCKET_MB ENABLE_THINKING
export MAX_PROMPT_LEN MAX_RESP_LEN MAX_MODEL_LEN
export SPREADSHEET_RL_REWARD_URL SPREADSHEET_RL_RECALC_URL SANDBOX_FUSION_URL
export CUDA_VISIBLE_DEVICES WANDB_MODE

CMD=(bash "${REPO}/deploy/run_grpo.sh" "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"})
say "명령:"
printf '    %q\n' "${CMD[@]}" | sed -n '1,200p'
if [ "${DRY:-0}" = "1" ]; then
    say "DRY=1 -> 실행하지 않았습니다. 위 명령을 그대로 복사해 쓸 수 있습니다."
    exit 0
fi

# ---------------------------------------------------------------- 실행
say "실행 시작 (로그: ${LOG})"
"${CMD[@]}" 2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}
say "학습 종료 rc=${rc}"

# ---------------------------------------------------------------- 검증
say "결과 검증: scripts/verify_run.py"
"${PY}" "${SMOKE_HOME}/scripts/verify_run.py" --log "${LOG}" --run-dir "${RUN_DIR}" \
        --method "${CA_METHOD}" --anchor "${CA_ANCHOR}" --reward "${REWARD}" \
        --expect-steps "${TOTAL_STEPS}" \
        2>&1 | tee "${RUN_DIR}/verdict.txt"
vrc=${PIPESTATUS[0]}
say "RUN_DIR=${RUN_DIR}"
say "verdict=${RUN_DIR}/verdict.txt (rc=${vrc}), train.log rc=${rc}"
[ ${rc} -eq 0 ] && [ ${vrc} -eq 0 ] || { say "실패. 로그/verdict 확인 후 README 의 '흔한 실패' 항목 참조."; exit 1; }
say "[성공] GiGPO 모듈 경로가 end-to-end 으로 동작했다."
