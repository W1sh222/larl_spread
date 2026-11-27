#!/bin/bash
# Self-Improving GPU smoke test (명령서 §21.12).
#
# 复用 gigpo_smoke 的最小实验环境 (venv / Qwen3.5-9B / 8 任务子集 / mock reward),
# 只把 override 换成 Self-Improving: REWARD=mock, CONTROLLER=mock script,
# TOTAL_STEPS=6, SLOW_INTERVAL=2 -> 预期 3 次 intervention, 2 次 knowledge 调用,
# 至少一个 checkpoint + 对应的 self_improving snapshot.
#
# 用法:
#   DRY=1 bash run_smoke_mock.sh     # 只打印将要执行的命令 (无 GPU 环境时的默认行为)
#   bash run_smoke_mock.sh           # 真正执行 (需要 GPU / 模型 / 数据都已就绪)
#   bash run_smoke_mock.sh --resume  # 第二段: 从上一次最后的 checkpoint 恢复, 验证 Slow Loop 状态复原
#
# 环境前提与 gigpo_smoke 相同 (见 gigpo_smoke/README 与 00_preflight.sh):
#   bash 01_setup_env.sh; bash 02_get_model.sh; bash 03_get_data.sh; bash 04_pick_subset.sh
set -uo pipefail

SI_EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${REPO:-"$(cd "${SI_EXAMPLES_DIR}/../../../../../../" && pwd)"}   # Spreadsheet-RL-master/
SMOKE_HOME=${SMOKE_HOME:-"$(cd "${REPO}/.." && pwd)/gigpo_smoke"}

TOTAL_STEPS=${TOTAL_STEPS:-6}
SLOW_INTERVAL=${SLOW_INTERVAL:-2}
MIN_WINDOW_STEPS=${MIN_WINDOW_STEPS:-2}   # smoke 的窗口只有 2 step, 默认 5 会让 evaluation 全部 insufficient
SAVE_FREQ=${SAVE_FREQ:-2}
REWARD=${REWARD:-"mock"}
MOCK_SCRIPT=${MOCK_SCRIPT:-"${SI_EXAMPLES_DIR}/mock_responses.jsonl"}
RUN_DIR=${RUN_DIR:-"${SMOKE_HOME:-${REPO}/../gigpo_smoke}/outputs/runs/si_smoke_$(date +%Y%m%d_%H%M%S)"}

say() { echo "[si_smoke] $*"; }
die() { echo "[si_smoke][BLOCKED] $*" >&2; exit 3; }

[ -f "${SMOKE_HOME}/env.sh" ] || die "找不到 ${SMOKE_HOME}/env.sh (设 SMOKE_HOME 指向 gigpo_smoke 目录)"
# shellcheck disable=SC1091
source "${SMOKE_HOME}/env.sh"    # env.sh 也定义了 say/die/log, 因此下面重新定义为本脚本的版本

say() { echo "[si_smoke] $*"; }
die() { echo "[si_smoke][BLOCKED] $*" >&2; exit 3; }

PY=${PY:-"${VENV_DIR}/bin/python"}
[ -x "${PY}" ] || die "venv 不存在 -> bash ${SMOKE_HOME}/01_setup_env.sh"
[ -f "${MODEL_PATH}/config.json" ] || die "模型不存在 -> bash ${SMOKE_HOME}/02_get_model.sh"
[ -f "${TRAIN_FILE}" ] || die "子集数据不存在 -> bash ${SMOKE_HOME}/04_pick_subset.sh"
[ -f "${MOCK_REWARD_PATH}" ] || die "mock reward 脚本不存在: ${MOCK_REWARD_PATH}"
[ -f "${MOCK_SCRIPT}" ] || die "mock LLM 脚本不存在: ${MOCK_SCRIPT}"
command -v nvidia-smi >/dev/null 2>&1 || die "没有 nvidia-smi: 本机无法跑 GPU smoke, 用 DRY=1 取命令, 结果记为 Not Run"
NFREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v g="${GPUS}" 'BEGIN{n=split(g,a,",");for(i=1;i<=n;i++)want[a[i]]=1} want[$1]&&$2<2000' | wc -l)
[ "${NFREE}" -ge "${NGPUS}" ] || die "GPU ${GPUS} 中空闲 ${NFREE} 张 < NGPUS=${NGPUS} (可用 GPUS=... 缩小)"

mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/rollout_data" "${RUN_DIR}/validation_data"
SI_DIR="${RUN_DIR}/checkpoints/self_improving"
LAST_POINTER="${SMOKE_HOME}/outputs/runs/si_last_run.txt"

EXTRA_ARGS=(
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TEST_FILE}"
    "data.train_batch_size=${TRAIN_BS}"
    "data.val_batch_size=${TRAIN_BS}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BS}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_STEPS}"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_STEPS}"
    "actor_rollout_ref.rollout.multi_turn.enable=True"
    "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent"
    "trainer.project_name=si-smoke"
    "trainer.experiment_name=si_smoke_$(date +%H%M%S)"
    "trainer.logger=[console]"
    "trainer.rollout_data_dir=${RUN_DIR}/rollout_data"
    "trainer.validation_data_dir=${RUN_DIR}/validation_data"
    "trainer.default_local_dir=${RUN_DIR}/checkpoints"
    "trainer.val_before_train=False"
    "trainer.test_freq=-1"
    "trainer.total_epochs=99"
)
[ -f "${TOOL_CONFIG_PATH}" ] && EXTRA_ARGS+=("actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH}")
if [ "${REWARD}" = "mock" ]; then
    EXTRA_ARGS+=("reward.custom_reward_function.path=${MOCK_REWARD_PATH}" "reward.custom_reward_function.name=compute_score")
fi

if [ "${1:-}" = "--resume" ]; then
    # 第二局必须写回同一个 default_local_dir: Slow Loop 的 snapshot / knowledge 版本存在
    # <default_local_dir>/self_improving/ 下, 换新目录就变成另一段全新训练.
    [ -f "${LAST_POINTER}" ] || die "没有上一次 smoke 记录 (${LAST_POINTER}), 先不携 --resume 跑第一段"
    RUN_DIR="$(cat "${LAST_POINTER}")"
    SI_DIR="${RUN_DIR}/checkpoints/self_improving"
    LAST_CKPT=$(ls -1d "${RUN_DIR}/checkpoints/global_step_"* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    [ -n "${LAST_CKPT}" ] || die "${RUN_DIR}/checkpoints 下没有 checkpoint"
    RESUME_FROM="${RUN_DIR}/checkpoints/global_step_${LAST_CKPT}"
    TOTAL_STEPS=$((LAST_CKPT + SLOW_INTERVAL * 2))
    say "resume from ${RESUME_FROM} -> total_training_steps=${TOTAL_STEPS}"
fi
EXTRA_ARGS+=("trainer.total_training_steps=${TOTAL_STEPS}")
[ -n "${RESUME_FROM:-}" ] && EXTRA_ARGS+=("trainer.resume_mode=resume_path" "trainer.resume_from_path=${RESUME_FROM}")

export REPO MODEL_PATH SMOKE_HOME RUN_DIR
export SI_INTERVAL="${SLOW_INTERVAL}" MIN_WINDOW_STEPS KNOWLEDGE=true LLM_CLIENT=mock MOCK_SCRIPT SAVE_FREQ

say "RUN_DIR=${RUN_DIR}"
say "steps=${TOTAL_STEPS} interval=${SLOW_INTERVAL} min_window_steps=${MIN_WINDOW_STEPS} reward=${REWARD} controller=mock"
mkdir -p "$(dirname "${LAST_POINTER}")" && echo "${RUN_DIR}" > "${LAST_POINTER}"

if [ "${DRY:-0}" = "1" ]; then
    DRY=1 bash "${SI_EXAMPLES_DIR}/run_self_improving.sh" "${EXTRA_ARGS[@]}"
    say "DRY=1 -> GPU smoke 未执行, 报告里应记为 Not Run."
    exit 0
fi

LOG="${RUN_DIR}/train.log"
DRY=0 bash "${SI_EXAMPLES_DIR}/run_self_improving.sh" "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG}" || die "训练进程失败, 见 ${LOG}"

say "---------------- 产物检查"
for f in config.json state.json history.jsonl contexts.jsonl; do
    [ -f "${SI_DIR}/${f}" ] && say "ok  ${f}" || say "MISS ${f}"
done
[ -d "${SI_DIR}/snapshots" ] && say "ok  snapshots/: $(ls -1 "${SI_DIR}/snapshots" | tr '\n' ' ')"
[ -d "${SI_DIR}/knowledge/versions" ] && say "ok  knowledge versions: $(ls -1 "${SI_DIR}/knowledge/versions" | tr '\n' ' ')"
say "console: $(grep -c '\[self_improving\]' "${LOG}") 条 [self_improving] 日志, "
say "        interventions=$(grep -o 'finalized: [0-9]* intervention' "${LOG}" | tail -1)"
say "后续 A/B: 用 mock 响应换顺序或 INITIAL_SKILL=gigpo 再跑一次即可."
