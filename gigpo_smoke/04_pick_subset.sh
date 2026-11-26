#!/bin/bash
# 04. 초소형 스모크 서브셋 만들기 (parquet 8태스크 + 해당 워크북만 zip 해제 + sandbox 없는 tool config).
# 상세 로직은 scripts/make_subset.py. 반복 실행 안전(멱등).
# 사용: bash 04_pick_subset.sh
#       SUBSET_SIZE=4 MAX_WORKBOOK_MB=10 bash 04_pick_subset.sh    (더 가볍게)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/pick_subset_${TS}.log"
exec > >(tee -a "${LOG}") 2>&1
say(){ echo "[04 $(date +%H:%M:%S)] $*"; }

# pyarrow/pandas/yaml 이 필요하므로 01 에서 만든 venv 를 우선 쓰고, 없으면 기존 .venv 로 대체한다.
CAND="${PY} ${REPO}/.venv/bin/python /usr/bin/python3"
PYBIN=""
for c in ${CAND}; do
    [ -x "${c}" ] || continue
    if "${c}" -c "import pyarrow, pandas, yaml" 2>/dev/null; then PYBIN=${c}; break; fi
done
[ -n "${PYBIN}" ] || die "pyarrow/pandas/yaml 있는 python 없음 -> bash 01_setup_env.sh 실행"
say "python=${PYBIN}"

MODEL_PATH="${MODEL_PATH}" SPREADSHEET_RL_DATA_ROOT="${SPREADSHEET_RL_DATA_ROOT}" \
SUBSET_SIZE="${SUBSET_SIZE}" MAX_PROMPT_LEN="${MAX_PROMPT_LEN}" OUT_DIR="${OUT_DIR}" \
REPO="${REPO}" TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH}" \
"${PYBIN}" "${SMOKE_HOME}/scripts/make_subset.py" || die "서브셋 생성 실패. 로그: ${LOG}"

say "생성물:"
say "  ${TRAIN_FILE}"
say "  ${TEST_FILE}"
say "  ${SUBSET_TASKS}"
say "  ${TOOL_CONFIG_PATH}"
[ -f "${SUBSET_TASKS}" ] && say "태스크 $(wc -l < "${SUBSET_TASKS}")개:" && sed 's/^/      /' "${SUBSET_TASKS}"
say "완료 -> 다음: bash 05_probe_reward.sh (보상 경로 확인), 그 후 bash tmux_smoke.sh"
say "로그: ${LOG}"
