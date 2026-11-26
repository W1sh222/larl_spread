#!/bin/bash
# 05. 보상 경로 사전 확인.
#   (a) Windows reward API health
#   (b) /recalculate 는 업로드한 워크북만으로 동작 -> 태스크가 Windows 에 없어도 도구는 동작해야 함
#   (c) /reward/submit 은 thread_dir 아래 instruction.json/target.xlsx 가 Windows 에 있어야 >0
#       -> 스모크 서브셋의 "정답 output.xlsx"를 그대로 제출해 1.0 이 나오는지 본다.
# 나오지 않으면 REWARD=mock 으로 실행하면 된다(보상만 가짜, GiGPO 배선은 그대로 실경로).
# 사용: bash 05_probe_reward.sh
#       PROBE_N=1 bash 05_probe_reward.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/probe_reward_${TS}.log"
exec > >(tee -a "${LOG}") 2>&1
say(){ echo "[05 $(date +%H:%M:%S)] $*"; }
PROBE_N=${PROBE_N:-2}
BASE_URL="${SPREADSHEET_RL_REWARD_URL%/reward/submit}"

# ---------------------------------------------------------------- (a) health
say "=== (a) health: ${BASE_URL}/health"
HEALTH=$(timeout 20 curl -s --max-time 15 "${BASE_URL}/health" || echo "")
if [ -z "${HEALTH}" ]; then
    say "[FAIL] 응답 없음. Windows 172.16.6.155 의 reward API/Excel 로그인 상태 확인 (docs/windows.md)"
else
    echo "  ${HEALTH}" | head -c 600; echo
    echo "${HEALTH}" | grep -q '"ready":true' \
        && say "[ok]   ready=true" \
        || say "[WARN] ready=false -> Excel 세션 재시작 필요 (win_start_reward_api.ps1)"
fi

PYBIN=${PY}
[ -x "${PYBIN}" ] || { say "[SKIP] venv python 없음 (${PYBIN}) -> 01_setup_env.sh 후 재실행"; exit 0; }
[ -f "${SUBSET_TASKS}" ] || { say "[SKIP] ${SUBSET_TASKS} 없음 -> 04_pick_subset.sh 먼저"; exit 0; }

# ---------------------------------------------------------------- (b) recalculate
say "=== (b) /recalculate (로컬 워크북 업로드, thread_dir 불필요)"
FIRST_TD=$(head -n1 "${SUBSET_TASKS}")
WB="${SPREADSHEET_RL_DATA_ROOT}/${FIRST_TD}/output.xlsx"
if [ -f "${WB}" ]; then
    RC=$(timeout 200 curl -s --max-time 190 -o "${OUT_DIR}/recalc_probe.json" -w "%{http_code}" \
            -F "file=@${WB}" "${BASE_URL}/recalculate" || echo ERR)
    say "  POST /recalculate http=${RC} ($(wc -c < "${OUT_DIR}/recalc_probe.json" 2>/dev/null || echo 0) B)"
    head -c 300 "${OUT_DIR}/recalc_probe.json" 2>/dev/null; echo
    [ "${RC}" = "200" ] && say "[ok]   재계산 경로 동작 (fill_formula/clear_range/recalculate_and_read 사용 가능)" \
                        || say "[WARN] 재계산 실패(http=${RC}) -> 도구 관측이 error 로 채워질 수 있음"
else
    say "[SKIP] 워크북 없음: ${WB}"
fi

# ---------------------------------------------------------------- (c) reward submit
say "=== (c) /reward/submit : 서브셋의 정답 output.xlsx 를 그대로 제출 (기대 reward=1.0)"
REWARD_PROBE_TASKS="${SUBSET_TASKS}" PROBE_N="${PROBE_N}" "${PYBIN}" - <<'PY' || say "[FAIL] 보상 프로비 실패"
import os, sys, time
from pathlib import Path

root = Path(os.environ["SPREADSHEET_RL_DATA_ROOT"])
tasks = [t.strip() for t in open(os.environ["REWARD_PROBE_TASKS"]) if t.strip()][: int(os.environ["PROBE_N"])]
sys.path.insert(0, os.environ["VERL_SRC"])
from verl.utils.reward_score import sheet_arena  # noqa: E402

print(f"제출 태스크 {len(tasks)}개 (정답본 제출 -> Windows 에 같은 태스크가 없으면 0.0)", flush=True)
hits = 0
for td in tasks:
    t0 = time.monotonic()
    try:
        r, msg = sheet_arena.compute_score(td, {"primary_ext": ".xlsx"}, return_msg=True)
    except Exception as e:
        r, msg = 0.0, f"EXC {type(e).__name__}: {e}"
    ok = float(r) >= 0.999
    hits += int(ok)
    print(f"  [{'ok' if ok else 'zero'}] reward={r:.3f} {time.monotonic()-t0:5.1f}s  {td}"
          f"  | {str(msg)[:150]}", flush=True)
print(f"\n정답-1.0 성공 {hits}/{len(tasks)}", flush=True)
if hits == 0:
    print("=> Windows 쪽에 이 태스크들이 없다(또는 preflight 실패). 스모크는 REWARD=mock 으로 돌리세요.", flush=True)
    sys.exit(2)
PY
rc=$?
echo
if [ ${rc} -eq 0 ]; then
    say "[결론] Windows 보상 사용 가능 -> REWARD=windows 로 실행 가능"
elif [ ${rc} -eq 2 ]; then
    say "[결론] Windows 보상은 0점 -> REWARD=mock(기본값) 으로 실행하면 GiGPO 배선 검증은 그대로 진행됨"
else
    say "[결론] 보상 프로비 실패. REWARD=mock 으로 진행하세요."
fi
say "REWARD=${REWARD} (현재 기본값)  |  로그: ${LOG}"
