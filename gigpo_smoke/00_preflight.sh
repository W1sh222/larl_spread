#!/bin/bash
# 00. 사전 점검. 아무 것도 설치/다운로드하지 않고, 지금 스모크를 돌릴 수 있는지만 판정한다.
#   4단계에서 GPU 없이 scripts/check_module.py 를 돌려 "GiGPO 모듈 override 가 실제 hydra
#   구성에서 해석되고 step 텐서/registered loss 가 동작하는지" 까지 확인한다 (설정의 단일 소스: env.sh:smoke_args).
# 사용: bash 00_preflight.sh          (항목별 PASS/WARN/FAIL 출력)
#       STRICT=1 bash 00_preflight.sh (FAIL 하나라도 있으면 exit 1)
#       SKIP_MODULE_CHECK=1 bash 00_preflight.sh   (verl import 생략, 수십초 단축)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

STRICT=${STRICT:-0}
LOG="${LOG_DIR}/preflight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
N_PASS=0; N_WARN=0; N_FAIL=0

ok(){   printf "  [PASS] %s\n" "$*"; N_PASS=$((N_PASS+1)); }
warn(){ printf "  [WARN] %s\n" "$*"; N_WARN=$((N_WARN+1)); }
bad(){  printf "  [FAIL] %s\n" "$*"; N_FAIL=$((N_FAIL+1)); }
head_(){ printf "\n=== %s\n" "$*"; }

echo "GiGPO 스모크 사전점검  $(date -Is)"
echo "SMOKE_HOME=${SMOKE_HOME}"
echo "RUN artifacts -> ${OUT_DIR}/runs/last"

# ---------------------------------------------------------------- 1. GPU
head_ "GPU (요청: ${GPUS} / 허용: 4,5,6,7)"
BAD_GPU=$(awk -F',' -v allow="4,5,6,7" 'BEGIN{n=split(allow,a,",");for(i=1;i<=n;i++)ok[a[i]]=1}
            {for(i=1;i<=NF;i++) if(!ok[$i]) printf "%s%s",sep,$i,sep=","}' <<<"${GPUS}")
[ -n "${BAD_GPU}" ] && bad "GPUS=${GPUS} -> 허용 범위(4,5,6,7) 밖의 GPU: ${BAD_GPU}" \
                    || ok "GPUS=${GPUS} 는 허용 범위 안 (NGPUS=${NGPUS})"
[ "${CUDA_VISIBLE_DEVICES}" = "${GPUS}" ] \
    && ok "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    || bad "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} != GPUS=${GPUS} (다른 GPU 에 노출될 수 있음)"
if ! command -v nvidia-smi >/dev/null; then
    bad "nvidia-smi 없음"
else
    SMIP=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ -z "${SMIP}" ]; then
        bad "nvidia-smi 가 출력이 없음(드라이버/권한 확인)"
    else
        gpusel(){ awk -F', ' -v g="${GPUS}" -v mode="$1" '
            BEGIN{n=split(g,a,",");for(i=1;i<=n;i++)want[a[i]+0]=1}
            want[$1+0] && ((mode=="free" && $2<2000) || (mode=="busy" && $2>=2000)) {printf "%s%s",sep,$1; sep=","}
            END{print ""}' <<<"${SMIP}"; }
        FREE=$(gpusel free); BUSY=$(gpusel busy)
        NFREE=0; [ -n "${FREE}" ] && NFREE=$(awk -F',' '{print NF}' <<<"${FREE}")
        echo "       유휴(<2000MiB): ${FREE:-없음} | 점유: ${BUSY:-없음} | 필요 NGPUS=${NGPUS}"
        if [ "${NFREE}" -ge "${NGPUS}" ]; then ok "요청 GPU 중 유휴 ${NFREE}장 >= NGPUS=${NGPUS}"
        else bad "유휴 ${NFREE}장 < NGPUS=${NGPUS} -> 다른 작업 대기 후 재실행하거나 NGPUS/GEN_TP 조정"; fi
    fi
fi

# ---------------------------------------------------------------- 2. 디스크
head_ "디스크"
AVAIL_GB=$(df -BG --output=avail "${SMOKE_HOME}" | tail -1 | tr -dc '0-9')
NEED_GB=60   # 모델 19GB + zip 1.2GB + 해제분 + 로그/롤아웃
if [ "${AVAIL_GB}" -ge "${NEED_GB}" ]; then ok "여유 ${AVAIL_GB}GB (>= ${NEED_GB}GB)"; else bad "여유 ${AVAIL_GB}GB < ${NEED_GB}GB"; fi

# ---------------------------------------------------------------- 3. 외부 서비스
# REWARD=mock(기본) 일 때 Windows 는 선택사항. 단 재계산 도구의 관측값이 error 로 채워진다.
head_ "Windows 보상 / 재계산 서비스  (REWARD=${REWARD})"
BASE_URL="${SPREADSHEET_RL_REWARD_URL%/reward/submit}"
HEALTH=$(timeout 15 curl -s --max-time 12 "${BASE_URL}/health" 2>/dev/null)
if [ -z "${HEALTH}" ]; then
    if [ "${REWARD}" = "windows" ]; then
        bad "REWARD=windows 인데 응답 없음: ${BASE_URL}/health (Windows 서비스/로그인 확인, docs/windows.md)"
    else
        warn "reward API 응답 없음 -> REWARD=mock 으로 진행 가능 (GiGPO 배선 검증은 성립). "
        echo "       다만 fill_formula/clear_range/recalculate_and_read 관측이 error 가 되어 step 내용이 단조로워질 수 있음"
    fi
else
    echo "  ${HEALTH}" | head -c 300; echo
    if echo "${HEALTH}" | grep -q '"ready":true'; then
        ok "reward API ready=true (재계산 도구 + 실제 보상 사용 가능)"
    elif [ "${REWARD}" = "windows" ]; then
        bad "reward API ready=false -> Excel 세션 재시작(win_start_reward_api.ps1) 후 재실행"
    else
        warn "reward API 응답하나 ready=false -> REWARD=mock 으로 진행 가능"
    fi
fi
echo "       REWARD_URL = ${SPREADSHEET_RL_REWARD_URL}"
echo "       RECALC_URL = ${SPREADSHEET_RL_RECALC_URL}"
echo "       SANDBOX_FUSION_URL = ${SANDBOX_FUSION_URL} (tools_no_sandbox.yaml 면 미사용)"

# ---------------------------------------------------------------- 4. venv / 코드
head_ "학습 venv 및 GiGPO 모듈"
if [ ! -x "${PY}" ]; then
    bad "venv 없음: ${VENV_DIR} -> bash 01_setup_env.sh 실행"
else
    VER=$(${PY} - <<'PY' 2>/dev/null
import importlib.metadata as m
out = []
for p in ("torch", "vllm", "transformers", "ray"):
    try:
        out.append(f"{p}={m.version(p)}")
    except Exception as e:
        out.append(f"{p}=MISSING")
print(" ".join(out))
PY
)
    if [ -z "${VER}" ]; then bad "venv 는 있으나 torch/vllm import 실패 (${VENV_DIR}) -> bash 01_setup_env.sh"
    else
        ok "venv: ${VER}"
        echo "${VER}" | grep -q "vllm=${VLLM_VERSION}" \
            && ok "vllm ${VLLM_VERSION} (docs/environment.md 검증 조합)" \
            || warn "vllm 이 ${VLLM_VERSION} 아님 -> Qwen3_5 미지원 가능성. bash 01_setup_env.sh 로 재확인"
    fi
    Q35=$(${PY} - <<'PY' 2>/dev/null
from vllm.model_executor.models.registry import ModelRegistry
print(any('Qwen3_5' in a for a in ModelRegistry.get_supported_archs()))
PY
)
    case "${Q35}" in
        True)  ok "vLLM 이 Qwen3_5 arch 지원 (9B 구동 가능)" ;;
        False) bad "vLLM 이 Qwen3_5 arch 미지원 -> 9B rollout 실패. vllm>=0.32 필요 (bash 01_setup_env.sh)" ;;
        *)     warn "vLLM arch 지원 여부 확인 불가 (import 실패)" ;;
    esac
    G=$(${PY} - <<'PY' 2>/dev/null
from verl.trainer.ppo import core_algos as C
from verl.trainer.credit_assignment import available_methods
adv = "credit_assignment" in getattr(C, "ADV_ESTIMATOR_REGISTRY", {})
# step objective 는 기본 구현에 존재하면 안 된다 (공식 GiGPO 는 vanilla token-level PPO 를 쓴다).
dead = [n for n in ("gigpo_joint", "gigpo_geomean") if n in C.POLICY_LOSS_REGISTRY]
try:
    import verl.trainer.ppo.step_tensors  # noqa: F401
    dead.append("step_tensors")
except ImportError:
    pass
print(adv, "vanilla" in C.POLICY_LOSS_REGISTRY, len(dead), "gigpo" in available_methods())
PY
)
    if [ -z "${G}" ]; then
        bad "verl import 실패 (PYTHONPATH=${PYTHONPATH})"
    else
        read -r ADV VAN DEAD MTH <<< "${G}"
        [ "${ADV}" = "True" ]  && ok "adv_estimator=credit_assignment 등록됨"  || bad "credit_assignment estimator 없음"
        [ "${VAN}" = "True" ]  && ok "policy loss vanilla 등록됨 (GiGPO 가 실제로 쓰는 손실)" || bad "vanilla loss 없음 (core_algos 확인)"
        [ "${DEAD}" = "0" ]    && ok "제거 확인: gigpo_joint/gigpo_geomean/step_tensors 없음" || bad "제거된 step objective 잔존 ${DEAD} 개 (config/code dangling 확인)"
        [ "${MTH}" = "True" ]  && ok "credit_assignment method 'gigpo' 등록됨" || bad "gigpo method 없음"
    fi
fi

# ---------------------------------------------------------------- 4b. override -> 실제 config 해석 확인
head_ "GiGPO 모듈 배선 (scripts/check_module.py, GPU 불필요)"
CHK_LOG="${OUT_DIR}/preflight_check_module.txt"
if [ "${SKIP_MODULE_CHECK:-0}" = "1" ]; then
    warn "SKIP_MODULE_CHECK=1 -> 건너뜀"
elif [ ! -x "${PY}" ]; then
    warn "venv 없음 -> 건너뜀 (01_setup_env.sh 후 재실행 권장)"
else
    OVR=()
    while IFS= read -r _line; do [ -n "${_line}" ] && OVR+=(--override "${_line}"); done < <(smoke_args)
    echo "       smoke_args 키 $((${#OVR[@]} / 2)) 개  주입  (method=${CA_METHOD} anchor=${CA_ANCHOR} step_w=${CA_STEP_W} loss=vanilla reward=${REWARD})"
    if CHECK_NGPUS="${NGPUS}" CA_ANCHOR="${CA_ANCHOR}" timeout 900 "${PY}" "${SMOKE_HOME}/scripts/check_module.py" \
            "${OVR[@]}" >"${CHK_LOG}" 2>&1; then
        grep -E '^  \[(ok|WARN|FAIL)\]|^  \[PASS\]' "${CHK_LOG}" | sed 's/^/       /'
        ok "check_module.py 통과 (상세: ${CHK_LOG})"
    else
        tail -n 40 "${CHK_LOG}" | sed 's/^/       /'
        bad "check_module.py 실패 -> 위 출력 참조 (${CHK_LOG})"
    fi
fi

# ---------------------------------------------------------------- 5. 모델
head_ "모델 가중치 (Qwen3.5-9B)"
if [ -f "${MODEL_PATH}/config.json" ] && [ -f "${MODEL_PATH}/model.safetensors.index.json" ]; then
    NSHARD=$(ls "${MODEL_PATH}"/*.safetensors 2>/dev/null | wc -l)
    SIZE=$(du -sh "${MODEL_PATH}" 2>/dev/null | cut -f1)
    IDX=${MODEL_PATH}/model.safetensors.index.json
    NEED=$(${PY:-python3} - <<PY 2>/dev/null
import json
print(len(set(json.load(open("${IDX}"))["weight_map"].values())))
PY
)
    if [ -n "${NEED}" ] && [ "${NSHARD}" -ge "${NEED}" ]; then
        ok "${MODEL_PATH} (${SIZE}, shard ${NSHARD}/${NEED})"
    else
        bad "shard 부족: ${NSHARD}/${NEED:-?} -> bash 02_get_model.sh 재실행"
    fi
    [ -f "${MODEL_PATH}/tokenizer_config.json" ] && ok "tokenizer 존재" || bad "tokenizer_config.json 없음"
else
    bad "모델 없음: ${MODEL_PATH} -> bash 02_get_model.sh 실행"
fi

# ---------------------------------------------------------------- 6. 데이터
head_ "학습 데이터"
if [ -f "${SPREADSHEET_RL_DATA_ROOT}/train_hermes.parquet" ]; then
    ok "train_hermes.parquet ($(du -h "${SPREADSHEET_RL_DATA_ROOT}/train_hermes.parquet" | cut -f1))"
else
    bad "data/train_hermes.parquet 없음 -> bash 03_get_data.sh 실행"
fi
if [ -f "${DATA_ZIP}" ]; then ok "spreadsheets.zip 존재"; else warn "spreadsheets.zip 없음 -> bash 03_get_data.sh (1.22GB)"; fi
if [ -f "${TRAIN_FILE}" ]; then
    ok "스모크 서브셋: ${TRAIN_FILE}"
    if [ -f "${SUBSET_TASKS}" ]; then
        MISS=0; TOT=0
        while read -r td; do
            [ -z "${td}" ] && continue
            TOT=$((TOT+1))
            [ -f "${SPREADSHEET_RL_DATA_ROOT}/${td}/output.xlsx" ] || MISS=$((MISS+1))
        done < "${SUBSET_TASKS}"
        if [ "${MISS}" -eq 0 ]; then ok "서브셋 태스크 ${TOT}개 워크북 해제 완료 (workspace seed 용 output.xlsx 있음)"
        else bad "서브셋 워크북 ${MISS}/${TOT}개 누락 -> bash 04_pick_subset.sh 재실행"; fi
    else
        warn "${SUBSET_TASKS} 없음 -> bash 04_pick_subset.sh 실행"
    fi
else
    warn "스모크 서브셋 미생성 -> bash 04_pick_subset.sh 실행 (filter_overlong_prompts 대비 길이 검사 포함)"
fi

# ---------------------------------------------------------------- 6b. 보상 arm 준비
head_ "보상 경로 준비 (REWARD=${REWARD})"
if [ "${REWARD}" = "mock" ]; then
    if [ -f "${MOCK_REWARD_PATH}" ]; then
        ok "mock scorer: ${MOCK_REWARD_PATH}"
        PYP=${PY}
        [ -x "${PYP}" ] || PYP=$(command -v python3 || true)
        if [ -n "${PYP}" ] && "${PYP}" -c "import ast,sys;ast.parse(open('${MOCK_REWARD_PATH}').read())" 2>/dev/null; then
            ok "mock scorer 문법 OK (${PYP})"
        else
            bad "mock scorer 파싱 실패 (${PYP})"
        fi
    else
        bad "MOCK_REWARD_PATH 없음: ${MOCK_REWARD_PATH}"
    fi
    [ -d "${SPREADSHEET_RL_DATA_ROOT}/_workspaces" ] && ok "_workspaces 디렉토리 존재" \
        || warn "_workspaces 없음 (run_gigpo_smoke.sh 이 생성) - mock 보장은 롤아웃 워크스북을 여기서 찾는다"
elif [ "${REWARD}" = "windows" ]; then
    ok "REWARD=windows -> 05_probe_reward.sh 에서 정답 제출이 1.0 이어야 의미 있음"
else
    bad "REWARD 는 mock | windows 중 하나 (현재 ${REWARD})"
fi

# ---------------------------------------------------------------- 7. 도구 설정
head_ "도구 설정 (code_interpreter 제외)"
if [ -f "${TOOL_CONFIG_PATH}" ]; then
    grep -q "SandboxFusionTool" "${TOOL_CONFIG_PATH}" \
        && bad "${TOOL_CONFIG_PATH} 에 SandboxFusionTool 잔존 (sandbox 미기동 시 오류 남)" \
        || ok "${TOOL_CONFIG_PATH} : sandbox 없음, 도구 $(grep -c 'class_name:' "${TOOL_CONFIG_PATH}") 개"
    grep -q "RecalculateAndReadTool" "${TOOL_CONFIG_PATH}" \
        && ok "recalculate_and_read 포함 (Windows 재계산 경로)" || warn "recalculate_and_read 없음 -> step 전환이 약해질 수 있음"
else
    warn "도구 설정 없음 -> 04_pick_subset.sh 가 생성 (또는 TOOL_CONFIG_PATH 지정)"
fi

# ---------------------------------------------------------------- 8. tmux / 실행 수단
head_ "실행 수단"
command -v tmux >/dev/null && ok "tmux $(tmux -V | awk '{print $2}')" || bad "tmux 없음"
command -v uv >/dev/null 2>&1 && ok "uv 있음 ($(/root/.local/bin/uv --version 2>/dev/null || uv --version))" \
    || warn "uv 가 PATH 에 없음 -> 01_setup_env.sh 에서 uv 설치 필요"
command -v unzip >/dev/null && ok "unzip 있음" || bad "unzip 없음"

# ---------------------------------------------------------------- 결론
printf "\n=== 결론: PASS=%d WARN=%d FAIL=%d   (상세: %s)\n" "${N_PASS}" "${N_WARN}" "${N_FAIL}" "${LOG}"
echo "       구성 한곳: env.sh (GPUS=${GPUS} NGPUS=${NGPUS} SUBSET_SIZE=${SUBSET_SIZE} TRAIN_BS=${TRAIN_BS} "
echo "       ROLLOUT_N=${ROLLOUT_N} TOTAL_STEPS=${TOTAL_STEPS} MAX_STEPS=${MAX_STEPS} CA_STEP_W=${CA_STEP_W} loss=vanilla)"
if [ "${N_FAIL}" -gt 0 ]; then
    echo "FAIL 항목을 먼저 해결하세요. 순서: 01_setup_env.sh -> 02_get_model.sh -> 03_get_data.sh -> 04_pick_subset.sh -> 05_probe_reward.sh"
    echo "전부 통과하면: bash tmux_smoke.sh    (또는 단독으로 bash run_gigpo_smoke.sh)"
    [ "${STRICT}" = "1" ] && exit 1
else
    echo "FAIL 없음. 05_probe_reward.sh 로 Windows 쪽 태스크 데이터 유무까지 확인 후 실행 권장."
fi
exit 0
