#!/usr/bin/env python3
"""GiGPO 스모크 서브셋 생성 (04_pick_subset.sh 가 호출).

做的事:
  1. train_hermes.parquet 에서 ground_truth(=thread_dir) 가 zip 안에 실재하는 태스크만 고른다.
  2. 프롬프트가 짧은 순서대로 SUBSET_SIZE 개를 뽑는다 (rollout 시간/토큰 예산 확보).
  3. 토크나이저가 있으면 프롬프트 토큰수를 재고, tool schema 텍스트 비용을 더해
     MAX_PROMPT_LEN 에 들어올 수 있는지 판정한다 (verl filter_overlong_prompts 대비).
  4. 선택된 태스크만 zip 에서 해제한다 (11,006개 통째 해제 금지).
  5. train_smoke.parquet / test_smoke.parquet / subset_tasks.txt / subset_summary.json 을 쓴다.
  6. sandbox 없는 tool config 를 생성한다 (code_interpreter 호출 시 8080 이 없어 죽는다).

env: SPREADSHEET_RL_DATA_ROOT, SUBSET_SIZE, MAX_PROMPT_LEN, MODEL_PATH,
     REPO, TOOL_CONFIG_PATH, OUT_DIR
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path(os.environ["SPREADSHEET_RL_DATA_ROOT"]).resolve()
ZIP = DATA_ROOT / "spreadsheets.zip"
TRAIN = DATA_ROOT / "train_hermes.parquet"
OUT_DIR = Path(os.environ.get("OUT_DIR", DATA_ROOT.parent / "outputs")).resolve()
N_TASKS = int(os.environ.get("SUBSET_SIZE", "8"))
MAX_PROMPT_LEN = int(os.environ.get("MAX_PROMPT_LEN", "8192"))
MODEL_PATH = os.environ.get("MODEL_PATH", "")
REPO = Path(os.environ.get("REPO", "/root/work/larl_spread/Spreadsheet-RL-master"))
SRC_TOOL_YAML = Path(os.environ.get("SRC_TOOL_YAML", REPO / "configs/tool/spreadsheet_tools.yaml"))
DST_TOOL_YAML = Path(os.environ.get("TOOL_CONFIG_PATH", OUT_DIR / "tools_no_sandbox.yaml"))
PROMPT_MARGIN = int(os.environ.get("PROMPT_MARGIN", "1024"))

fails = []


def check(cond, msg):
    print(("  [ok]   " if cond else "  [FAIL] ") + msg, flush=True)
    if not cond:
        fails.append(msg)
    return cond


# ---------------------------------------------------------------- 0. 의존
try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover
    sys.exit(f"pyarrow/pandas 필요 ({e}). 01_setup_env.sh 를 먼저 실행하세요. / "
             f"혹은 PYTHONPATH 에 venv 를 지정하세요: {sys.executable}")

if not check(TRAIN.is_file(), f"{TRAIN} 존재"):
    sys.exit(1)
if not check(ZIP.is_file(), f"{ZIP} 존재"):
    sys.exit(1)

# ---------------------------------------------------------------- 1. zip 내용 인덱스
print(f"=== zip 중앙 디렉토리 읽기: {ZIP} ===", flush=True)
listing = subprocess.run(["unzip", "-v", str(ZIP)], capture_output=True, text=True).stdout
entries = {}       # task_dir -> {basename: uncompressed_bytes}
for line in listing.splitlines():
    parts = line.split()
    if len(parts) < 8 or not parts[-1].endswith(("/xlsx", ".json", ".csv", ".txt")):
        continue
    name = parts[-1]
    if name.endswith("/") or "/" not in name:
        continue
    size = parts[6]
    if not size.isdigit():       # 첫 행은 Length/method/... 헤더
        continue
    task_dir, base = name.rsplit("/", 1)
    entries.setdefault(task_dir, {})[base] = int(size)
print(f"zip 안의 태스크 디렉토리 {len(entries)}개", flush=True)
if not check(len(entries) > 100, "zip 에서 태스크 100개 이상 발견"):
    sys.exit(1)

REQUIRED = ("instruction.json", "input.xlsx", "output.xlsx", "target.xlsx")


def complete(task_dir):
    d = entries.get(task_dir) or {}
    return all(k in d for k in REQUIRED)


# ---------------------------------------------------------------- 2. 후보 선별
df = pd.read_parquet(TRAIN)
print(f"=== {TRAIN.name}: {len(df)} rows, cols={list(df.columns)} ===", flush=True)


def thread_dir(row):
    gt = (row.get("reward_model") or {}).get("ground_truth")
    if isinstance(gt, list) and len(gt) == 1:
        gt = gt[0]
    return gt if isinstance(gt, str) and gt else None


def prompt_chars(msgs):
    return sum(len(str(m.get("content", ""))) for m in msgs)


rows = []
for i, row in df.iterrows():
    td = thread_dir(row.to_dict())
    if not td or not complete(td):
        continue
    wb_mb = max(entries[td].get("input.xlsx", 0), entries[td].get("output.xlsx", 0)) / 1e6
    if wb_mb > float(os.environ.get("MAX_WORKBOOK_MB", "30")):     # 큰 워크북은 rollout 이 느리다
        continue
    rows.append((i, td, prompt_chars(row["prompt"]), wb_mb))
rows.sort(key=lambda r: r[2])
print(f"parquet + zip 교차 가능한 태스크: {len(rows)}개 (작은 프롬프트 순)", flush=True)
if not check(len(rows) >= N_TASKS, f"후보 >= {N_TASKS}개"):
    print("[FAIL] 후보가 부족하다. SUBSET_SIZE 를 줄이세요.", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------- 3. 토큰 길이 측정 (가능하면)
tokenizer = None
if MODEL_PATH and Path(MODEL_PATH, "tokenizer_config.json").is_file():
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"토크나이저 로드: {MODEL_PATH}", flush=True)
    except Exception as e:
        print(f"[WARN] 토크나이저 로드 실패({type(e).__name__}: {e}) -> 문자 추산 사용", flush=True)
else:
    print("[WARN] MODEL_PATH 토크나이저 없음 -> 문자 추산 사용 (02_get_model.sh 후 재실행 권장)", flush=True)


def tool_text():
    """tool schema 의 description/param 문자열이 rollout 프롬프트에 더해진다."""
    try:
        import yaml
        cfg = yaml.safe_load(SRC_TOOL_YAML.read_text())
        out = []
        for t in cfg.get("tools", []):
            fn = (t.get("tool_schema") or {}).get("function") or {}
            if "sandbox_fusion" in str(t.get("class_name", "")):
                continue                       # 스모크에선 code_interpreter 를 쓰지 않는다
            out.append(json.dumps(fn, ensure_ascii=False))
        return "\n".join(out)
    except Exception as e:
        print(f"[WARN] tool schema 파싱 실패: {e}", flush=True)
        return ""


tools_str = tool_text()


def token_len(msgs):
    if tokenizer is None:
        return int((prompt_chars(msgs) + len(tools_str)) / 3.5)     # 거친 추산
    try:
        base = len(tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True))
    except Exception:
        base = len(tokenizer("".join(str(m.get("content", "")) for m in msgs))["input_ids"])
    extra = len(tokenizer(tools_str)["input_ids"]) if tools_str else 0
    return int(base + extra)


budget = MAX_PROMPT_LEN - PROMPT_MARGIN
picked = []
for i, td, chars, mb in rows:
    tl = token_len(list(df.iloc[i]["prompt"]))
    picked.append((i, td, chars, mb, tl))
    if len(picked) >= N_TASKS:
        break
if not tokenizer:
    print("[WARN] 문자 추산 기반 -> 실제 토큰수와 ±20% 오차 가능", flush=True)
over = [p for p in picked if p[4] > budget]
for i, td, chars, mb, tl in picked:
    print(f"  {td}  chars={chars:6d}  wb={mb:5.1f}MB  prompt~={tl} tok"
          f"{'  [예산초과]' if tl > budget else ''}", flush=True)
check(not over, f"선별된 프롬프트 모두 ~= {budget} tok 이하 (tool schema 포함)")

# ---------------------------------------------------------------- 4. parquet 쓰기
sel = df.iloc[[p[0] for p in picked]].reset_index(drop=True)
sel.to_parquet(DATA_ROOT / "train_smoke.parquet", index=False)
# val 은 스모크에서 꺼두지만, val_before_train=True 로 돌릴 수 있게 같은 태스크 2개로 채운다.
sel.iloc[: min(2, len(sel))].to_parquet(DATA_ROOT / "test_smoke.parquet", index=False)
print(f"\ntrain_smoke.parquet {len(sel)} rows / test_smoke.parquet 작성", flush=True)

# ---------------------------------------------------------------- 5. 선택 태스크만 zip 에서 해제
tasks = [p[1] for p in picked]
(OUT_DIR / "subset_tasks.txt").write_text("\n".join(tasks) + "\n")
missing = [t for t in tasks
           if not (DATA_ROOT / t / "output.xlsx").is_file()
           or not (DATA_ROOT / t / "instruction.json").is_file()]
if missing:
    print(f"해제 필요 {len(missing)}개: {missing}", flush=True)
    args = [f"{t}/*" for t in missing]
    r = subprocess.run(["unzip", "-o", "-q", str(ZIP), *args, "-d", str(DATA_ROOT)])
    check(r.returncode in (0, 11), f"unzip rc={r.returncode}")
still = [t for t in tasks if not all((DATA_ROOT / t / f).is_file() for f in REQUIRED)]
check(not still, f"모든 선택 태스크에 {REQUIRED} 존재 (누락 {still or '없음'})")

# ---------------------------------------------------------------- 6. sandbox 없는 tool config
if SRC_TOOL_YAML.is_file():
    text = SRC_TOOL_YAML.read_text()
    blocks = re.split(r"(?m)^(?=  - class_name:)", text)
    keep, dropped = [], []
    for b in blocks:
        if not b.strip():
            continue
        if "sandbox_fusion_tools.SandboxFusionTool" in b or "code_interpreter" in b:
            dropped.append(re.search(r'class_name: "([^"]+)"', b).group(1) if "class_name" in b else "?")
        else:
            keep.append(b)
    header = keep[0].split("  - class_name:")[0] if keep[0].lstrip().startswith("tools:") else "tools:\n"
    DST_TOOL_YAML.parent.mkdir(parents=True, exist_ok=True)
    DST_TOOL_YAML.write_text(header + "".join(keep))
    n = sum(1 for b in keep if "class_name" in b)
    print(f"tool config: {DST_TOOL_YAML} ({n} 도구, 제외: {dropped})", flush=True)
    try:
        import yaml as _y
        _y.safe_load(DST_TOOL_YAML.read_text())
        check(True, "생성된 tool yaml parse 가능")
    except Exception as e:
        check(False, f"tool yaml 손상: {e}")
else:
    print(f"[WARN] {SRC_TOOL_YAML} 없음 -> repo 기본 tool config 사용", flush=True)

(OUT_DIR / "subset_summary.json").write_text(json.dumps({
    "tasks": tasks,
    "prompt_token_est": [p[4] for p in picked],
    "tokenizer": bool(tokenizer),
    "budget": budget,
    "max_workbook_mb": max(p[3] for p in picked),
}, indent=2))

print("\n=== 결론 ===", flush=True)
if fails:
    print(f"[FAIL] {len(fails)}건: {fails}", flush=True)
    sys.exit(1)
print(f"[OK] 스모크 서브셋 준비 완료: {len(tasks)}태스크 x rollout_n, "
      f"최대 워크북 {max(p[3] for p in picked):.1f}MB", flush=True)
