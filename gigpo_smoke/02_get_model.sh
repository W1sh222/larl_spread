#!/bin/bash
# 02. Qwen3.5-9B 가중치 준비 ( 순재현 없이 파일별 이어받기 ).
# 소스: hf-mirror 기본, 실패/지연 시 modelscope. huggingface.co 는 이 네트워크에서 차단돼 있다.
# 저장 위치: ${MODEL_PATH} (= gigpo_smoke/models/Qwen3.5-9B)
# 사용: bash 02_get_model.sh
#       MODEL_SOURCE=modelscope bash 02_get_model.sh   (모델스코페 강제)
#       SMOKE_MODEL=Qwen/Qwen3-4B-Thinking-2507 bash 02_get_model.sh  (다른 actor 실험용)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/get_model_${TS}.log"
exec > >(tee -a "${LOG}") 2>&1
say(){ echo "[02 $(date +%H:%M:%S)] $*"; }

SYS_PY=${SYS_PY:-/usr/bin/python3}
[ -x "${PY}" ] && SYS_PY="${PY}"
say "python=${SYS_PY}  model=${SMOKE_MODEL}  ->  ${MODEL_PATH}  (source=${MODEL_SOURCE})"

"${SYS_PY}" - "${SMOKE_MODEL}" "${MODEL_PATH}" "${MODEL_SOURCE}" <<'PY'
import json, os, sys, time, urllib.request

model_id, dest, source = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dest, exist_ok=True)

HF = "https://hf-mirror.com"
MS = "https://www.modelscope.cn"


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "gigpo-smoke"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def list_hf():
    # hf-mirror tree API: recursive=true 가 평면 목록 + lfs size 를 준다.
    entries = get(f"{HF}/api/models/{model_id}/tree/main?recursive=true", timeout=90)
    files = [(e["path"], int((e.get("lfs") or {}).get("size") or e.get("size") or 0))
             for e in entries if e.get("type") == "file"]
    return files, "hf"


def list_ms():
    d = get(f"{MS}/api/v1/models/{model_id}/repo/files?Recursive=true", timeout=90)
    out = []
    for f in (d.get("Data") or {}).get("Files") or []:
        if f.get("type") == "tree":
            continue
        out.append((f["Path"], int(f.get("Size") or 0)))
    return out, "ms"


def fetch(source_kind, path, size):
    url = f"{HF}/{model_id}/resolve/main/{path}" if source_kind == "hf" \
        else f"{MS}/models/{model_id}/resolve/master/{path}"
    dst = os.path.join(dest, path)
    os.makedirs(os.path.dirname(dst) or dest, exist_ok=True)
    if size > 0 and os.path.isfile(dst) and os.path.getsize(dst) == size:
        print(f"  [skip] {path} 이미 완전 ({size/1e6:.1f} MB)", flush=True)
        return True
    tmp = dst + ".part"
    for attempt in range(1, 4):
        cmd = f'curl -sSL --fail --max-time 3600 -C - -o "{tmp}" "{url}"'
        rc = os.system(cmd)
        got = os.path.getsize(tmp) if os.path.exists(tmp) else -1
        if (rc == 0 and size <= 0) or (size > 0 and got == size):
            os.replace(tmp, dst)
            print(f"  [ok]   {path} {got/1e6:.1f} MB (attempt {attempt})", flush=True)
            return True
        # 이미 완본 파일에 -C - 를 걸면 416(rc=22) 이 난다 -> 크기가 맞으면 성공 취급
        print(f"  [retry {attempt}] {path}: rc={rc} got={got} want={size}", flush=True)
        time.sleep(3 * attempt)
    return False


print(f"=== 파일 목록 조회 ({source}) ===", flush=True)
try_list = list_hf if source in ("auto", "hf") else list_ms
try:
    files, kind = try_list()
except Exception as e:
    print(f"  목록 실패({source}): {e}", flush=True)
    files, kind = list_ms() if source == "auto" else ([], "?")
    if not files:
        sys.exit(1)
if not files:
    files, kind = list_ms()

total = sum(s for _, s in files)
print(f"{len(files)} files, {total/1e9:.2f} GB via {kind}", flush=True)
weights = [p for p, _ in files if p.endswith(".safetensors")]
print("weights:", weights, flush=True)

failed = []
for path, size in sorted(files, key=lambda x: -x[1]):     # 큰 파일 먼저
    if not fetch(kind, path, size):
        failed.append(path)

if failed:
    print(f"\n[FAIL] 실패 {len(failed)}건: {failed}", flush=True)
    print("다시 실행하면 이어받기된다. 혹은 MODEL_SOURCE=modelscope 로 재시도.", flush=True)
    sys.exit(1)

print("\n=== 무결성 검사 ===", flush=True)
need = ["config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"]
idx_path = os.path.join(dest, "model.safetensors.index.json")
missing = [n for n in need if not os.path.exists(os.path.join(dest, n))]
shards = sorted(f for f in os.listdir(dest) if f.endswith(".safetensors"))
if os.path.exists(idx_path):
    want = set(json.load(open(idx_path))["weight_map"].values())
    absent = sorted(want - set(shards))
    print(f"index 가 요구하는 shard {len(want)}개, 설치 {len(shards)}개, 누락 {absent or '없음'}", flush=True)
    missing += absent
else:
    print("[WARN] index.json 없음 (단일 shard 모델일 수 있음)", flush=True)
cfg = json.load(open(os.path.join(dest, "config.json")))
print("architectures:", cfg.get("architectures"), "model_type:", cfg.get("model_type"), flush=True)
if missing:
    print(f"[FAIL] 누락 파일: {missing}", flush=True)
    sys.exit(1)
print(f"[OK] {dest} 준비 완료 ({sum(os.path.getsize(os.path.join(dest,f)) for f in shards)/1e9:.2f} GB weights)", flush=True)
PY
rc=$?
[ ${rc} -eq 0 ] || die "모델 준비 실패 (rc=${rc}). 로그: ${LOG}"
say "로그: ${LOG}"
