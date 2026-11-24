#!/bin/bash
# 03. 학습 데이터 준비. 방법 자체는 리포 문서 그대로다(docs/environment.md 3.2, 데이터셋 README):
#     Spreadsheet-RL/Spreadsheet-RL 리포에서 parquet + spreadsheets.zip 을 받아 data/ 에 둔다.
#   이 네트워크엔 huggingface.co 가 막혀 있고 hf CLI 가 없으므로 hf-mirror 의 resolve URL 을 쓴다.
#   zip(1.22GB, 11,006 태스크)은 통째로 해제하지 않는다 -> 04_pick_subset.sh 가 스모크 태스크만 해제.
# 사용: bash 03_get_data.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/get_data_${TS}.log"
exec > >(tee -a "${LOG}") 2>&1
say(){ echo "[03 $(date +%H:%M:%S)] $*"; }

BASE="https://hf-mirror.com/datasets/Spreadsheet-RL/Spreadsheet-RL/resolve/main"
cd "${SPREADSHEET_RL_DATA_ROOT}" || die "데이터 디렉토리 없음: ${SPREADSHEET_RL_DATA_ROOT}"

# 파일별 기대 크기(실측) -> 크기가 맞으면 건너뛴다.
fetch(){
    local name=$1 want=$2
    if [ -f "${name}" ] && [ "$(stat -c %s "${name}")" = "${want}" ]; then
        say "[skip] ${name} 이미 있음 (${want} B)"; return 0
    fi
    say "[get ] ${name} (want ${want} B)"
    for a in 1 2 3; do
        # -C - : 이어받기. 이미 완본이면 416 -> rc=22
        code=$(curl -sSL --max-time 3600 -C - -o "${name}.part" -w "%{http_code}" "${BASE}/${name}" || echo 000)
        have=$( [ -f "${name}.part" ] && stat -c %s "${name}.part" || echo 0 )
        if [ "${have}" = "${want}" ]; then
            mv -f "${name}.part" "${name}"; say "[ok  ] ${name} (${have} B, attempt ${a}, http=${code})"; return 0
        fi
        say "[retry ${a}] ${name}: http=${code} have=${have} want=${want}"
        sleep $((5*a))
    done
    die "${name} 다운로드 실패"
}

# 1) parquet (train 3,551,543 B / test 441,790 B, tree API 실측)
fetch train_hermes.parquet 3551543
fetch test_hermes.parquet  441790

# 2) 워크북 zip (1,221,102,713 B) - tree API 가 보고한 크기 우선 사용
ZIP_WANT=$(timeout 60 curl -s --max-time 50 \
    "https://hf-mirror.com/api/datasets/Spreadsheet-RL/Spreadsheet-RL/tree/main?recursive=false" \
    | /usr/bin/python3 -c "import json,sys;print(next(((x.get('lfs') or {}).get('size') or x.get('size') or 0) for x in json.load(sys.stdin) if x['path']=='spreadsheets.zip'))" 2>/dev/null)
ZIP_WANT=${ZIP_WANT:-0}
if [ "${ZIP_WANT}" -gt 0 ]; then
    fetch spreadsheets.zip "${ZIP_WANT}"
else
    if [ -f spreadsheets.zip ]; then
        say "[skip] spreadsheets.zip 존재 ($(du -h spreadsheets.zip | cut -f1)) - 크기 검증만 수행"
    else
        say "[get ] spreadsheets.zip (크기 미확인 -> 이어받기 후 unzip -t 로 검증)"
        curl -L --fail --max-time 7200 -C - -o spreadsheets.zip "${BASE}/spreadsheets.zip" || die "zip 실패"
    fi
fi

# 3) 검증: 중앙 디렉토리만 읽어 태스크 개수와 스모크에 필요할 경로 샘플을 확인한다(해제 없음).
say "zip 중앙 디렉토리 검사 중..."
unzip -Z1 spreadsheets.zip 2>/dev/null | head -3 > "${OUT_DIR}/zip_sample.txt" || die "unzip -Z1 실패 (zip 손상?)"
NENT=$(unzip -Z1 spreadsheets.zip 2>/dev/null | wc -l)
say "zip 엔트리 ${NENT}개 (11,006 태스크 x ~4 파일 + 디렉토리)"
[ "${NENT}" -gt 1000 ] || die "zip 엔터리가 너무 적습니다: ${NENT} (다시 받아야 함)"

say "parquet 스키마 확인"
PYBIN=${PY}; [ -x "${PYBIN}" ] || PYBIN=/usr/bin/python3
"${PYBIN}" - <<'PY' || die "parquet 검증 실패 (pyarrow 필요: 01_setup_env.sh 먼저)"
import pyarrow.parquet as pq
for name in ("train_hermes.parquet", "test_hermes.parquet"):
    f = pq.ParquetFile(name)
    print(f"  {name}: rows={f.metadata.num_rows} cols={f.schema_arrow.names}")
PY

say "완료 -> 다음: bash 04_pick_subset.sh"
say "로그: ${LOG}"
