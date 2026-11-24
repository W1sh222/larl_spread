#!/bin/bash
# 01. 학습 venv 준비. 기존 .venv(vllm 0.8.5)는 Qwen3_5 arch가 없어 9B를 못 올린다 -> 새 venv.
# 조합: python3.12 + torch 2.13.0(cu129) + vllm 0.32.0 (PyPI 실측: vllm 0.32.0이 torch==2.13.0,
#       transformers<5.20 고정, Qwen3_5ForConditionalGeneration 지원) + repo requirements.
# 사용: bash 01_setup_env.sh            (이미 준비돼 있으면 검증만 하고 끝난다)
#       FORCE=1 bash 01_setup_env.sh    (venv 삭제 후 재생성)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/setup_env_${TS}.log"
exec > >(tee -a "${LOG}") 2>&1
log(){ echo "[01 $(date +%H:%M:%S)] $*"; }
err(){ echo "[01][ERROR] $*" >&2; }

log "SMOKE_HOME=${SMOKE_HOME}"
log "VENV_DIR=${VENV_DIR}  PY=${PY}"
log "torch=${TORCH_VERSION} index=${TORCH_INDEX}  vllm=${VLLM_VERSION}  python=${PY_VERSION}"

# ---------------------------------------------------------------- uv
if [ ! -x "${UV}" ]; then
    log "uv 없음 -> pip --user 로 설치 시도"
    /usr/bin/python3 -m pip install --user uv || die "uv 설치 실패. 수동 설치 후 재실행: curl -LsSf https://astral.sh/uv/install.sh | sh"
    UV="$(/usr/bin/python3 -m site --user-base)/bin/uv"; export UV
    [ -x "${UV}" ] || UV=$(command -v uv) || die "uv 경로 확인 실패"
fi
log "uv = $("${UV}" --version)"

# ---------------------------------------------------------------- venv 생성
if [ "${FORCE:-0}" = "1" ] && [ -d "${VENV_DIR}" ]; then
    log "FORCE=1 -> ${VENV_DIR} 삭제"
    rm -rf "${VENV_DIR}"
fi

if [ ! -x "${PY}" ]; then
    log "venv 생성: $("${UV}" venv "${VENV_DIR}" --python "${PY_VERSION}" 2>&1 | tail -1)"
else
    log "venv 이미 존재: ${VENV_DIR} ($("${PY}" -V 2>&1))"
fi
[ -x "${PY}" ] || die "venv python 없음: ${PY}"

install(){
    log "설치: $*"
    "${UV}" pip install --python "${PY}" "$@" || die "설치 실패: $*"
}

# ---------------------------------------------------------------- torch -> vllm -> verl deps
MARK="${VENV_DIR}/.gigpo_smoke_installed"
if [ -f "${MARK}" ]; then
    log "설치 마커 존재(${MARK}) -> 설치 건너뛰고 검증만 수행"
else
    # vllm 0.32.0 의 고정본(PyPI 실측): torch==2.13.0, torchaudio==2.11.0, torchvision==0.28.0
    install --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" "torchaudio==2.11.0" "torchvision==0.28.0"
    # vllm 은 이미 잡힌 torch 를 만족시키므로 재설치하지 않는다.
    install "vllm==${VLLM_VERSION}" "openai==2.54.0" tqdm pytest
    install -r "${VERL_SRC}/requirements.txt"
    # sheet_arena 보상/도구가 여는 xlsx + 스모크 스크립트 자체 의존
    install openpyxl pyarrow pandas
    touch "${MARK}"
fi

# ---------------------------------------------------------------- 검증
log "검증 시작"
"${PY}" - <<'PY'
import importlib.metadata as m
import sys
ok = True
def show(label, fn):
    global ok
    try:
        v = fn()
        print(f"  [ok]   {label}: {v}")
    except Exception as e:
        ok = False
        print(f"  [FAIL] {label}: {type(e).__name__}: {e}")

show("python", lambda: sys.version.split()[0])
show("torch", lambda: m.version("torch"))
show("vllm", lambda: m.version("vllm"))
show("transformers", lambda: m.version("transformers"))
show("ray", lambda: m.version("ray"))
show("numpy", lambda: m.version("numpy"))

def cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() == False")
    return f"{torch.__version__}, devices={torch.cuda.device_count()}, cc={torch.cuda.get_device_capability(0)}"
show("cuda", cuda)

def qwen35():
    from vllm.model_executor.models.registry import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    hits = [a for a in archs if a.startswith("Qwen3_5")]
    if not hits:
        raise AssertionError("이 vLLM 은 Qwen3_5 arch 미지원 -> VLLM_VERSION 상향 필요")
    return ",".join(hits)
show("vllm Qwen3_5 arch", qwen35)

def verl_import():
    import verl
    from verl.trainer.ppo import core_algos as C
    from verl.trainer.credit_assignment import available_methods
    assert "credit_assignment" in C.ADV_ESTIMATOR_REGISTRY, "adv estimator 미등록"
    assert "vanilla" in C.POLICY_LOSS_REGISTRY, "policy loss vanilla 미등록"
    # 공식 GiGPO 는 step-level objective 를 쓰지 않는다. 제거된 확장이 다시 생기면 여기서 잡는다.
    for n in ("gigpo_joint", "gigpo_geomean"):
        assert n not in C.POLICY_LOSS_REGISTRY, f"제거된 step loss {n} 가 다시 등록됨"
    try:
        import verl.trainer.ppo.step_tensors  # noqa: F401
        raise AssertionError("step_tensors 모듈이 아직 존재 -> 제거 대상 (policy loss 는 vanilla 하나)")
    except ImportError:
        pass
    assert "gigpo" in available_methods(), f"gigpo method 없음: {available_methods()}"
    return f"verl at {verl.__file__}"
show("GiGPO 모듈 등록", verl_import)
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || die "검증 실패. 로그: ${LOG}"

# ---------------------------------------------------------------- deploy/run_grpo.sh 용 PATH shim
# run_grpo.sh 는 "$CONDA_ROOT/envs/spread/bin" 을 PATH 앞쪽에 넣는다. conda 가 없는 이 머신에서
# 그 경로를 새 venv 의 bin 으로 symlink 해 두면 repo 스크립트를 수정하지 않고 같은 venv 를 쓴다.
mkdir -p "${CONDA_SHIM}/envs"
rm -f "${CONDA_SHIM}/envs/spread"
ln -s "${VENV_DIR}" "${CONDA_SHIM}/envs/spread"
log "CONDA_SHIM: ${CONDA_SHIM}/envs/spread -> ${VENV_DIR}  (spread/bin = venv/bin)"

log "완료. 다음: bash 02_get_model.sh && bash 03_get_data.sh && bash 04_pick_subset.sh"
log "로그: ${LOG}"
