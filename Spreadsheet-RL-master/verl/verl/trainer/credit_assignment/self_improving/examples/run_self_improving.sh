#!/bin/bash
# Self-Improving Credit Assignment 启动示例 (명령서 §22).
#
# 不修改任何现有 launch script: 复用已验证的 deploy/run_grpo.sh -> configs/qwen3-4b.sh
# -> verl.trainer.main_ppo 路径, 只在末尾追加 self_improving 的 Hydra override.
#
# 先设置外部 controller LLM (不要写进脚本):
#   export SPREADSHEET_RL_SI_LLM_URL="YOUR_CHAT_COMPLETIONS_ENDPOINT"
#   export SPREADSHEET_RL_SI_LLM_MODEL="YOUR_MODEL_NAME"
#   export SPREADSHEET_RL_SI_LLM_API_KEY="YOUR_API_KEY"
#
# 七种用法 (命令 §22):
#   bash run_self_improving.sh                          # 1. 默认 Self-Improving 训练
#   INITIAL_SKILL=gigpo bash run_self_improving.sh      # 2. 初始 Skill 改为 GiGPO
#   SI_INTERVAL=5 bash run_self_improving.sh            # 3. 改 Slow Loop 周期
#   CANDIDATES="[grpo,gigpo]" bash run_self_improving.sh # 4. 限制候选 Skill
#   KNOWLEDGE=false bash run_self_improving.sh          # 5. 关闭 Knowledge Evolution
#   LLM_CLIENT=mock bash run_self_improving.sh          # 6. Mock LLM smoke (无外部 API)
#   RESUME_PATH="$CKPTS_DIR/global_step_20" bash run_self_improving.sh   # 7. 从 policy checkpoint 恢复
#
# 其它变量: DRY=1 只打印命令; BOOTSTRAP_METHOD / SAVE_FREQ 见下方说明.
# 末尾可以直接追加任意 Hydra override (透传给 deploy/run_grpo.sh -> configs/qwen3-4b.sh).
# GPU / 模型 / 数据 / conda 环境等按 deploy/run_grpo.sh 的约定注入 (REPO, MODEL_PATH, NGPUS ...).
set -uo pipefail

SI_EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${REPO:-"$(cd "${SI_EXAMPLES_DIR}/../../../../../../" && pwd)"}   # Spreadsheet-RL-master/

LLM_CLIENT=${LLM_CLIENT:-"api"}          # api | mock
SI_INTERVAL=${SI_INTERVAL:-20}           # 每多少个 training step 触发一次 Slow Loop
INITIAL_SKILL=${INITIAL_SKILL:-"grpo"}   # 第一个 training step 真正使用的 Skill
INITIAL_PARAMS=${INITIAL_PARAMS:-""}     # 部分参数即可, Python 侧补全成 canonical 值
CANDIDATES=${CANDIDATES:-"[grpo,gigpo,rewardflow]"}
KNOWLEDGE=${KNOWLEDGE:-"true"}
MIN_WINDOW_STEPS=${MIN_WINDOW_STEPS:-5}
BOOTSTRAP_METHOD=${BOOTSTRAP_METHOD:-"rewardflow"}
SAVE_FREQ=${SAVE_FREQ:-5}
RESUME_PATH=${RESUME_PATH:-""}           # global_step_N 目录; 留空则续跑 default_local_dir 里最新的 checkpoint
MOCK_SCRIPT=${MOCK_SCRIPT:-"${SI_EXAMPLES_DIR}/mock_responses.jsonl"}

if [ "${LLM_CLIENT}" = "api" ]; then
    : "${SPREADSHEET_RL_SI_LLM_URL:?Set SPREADSHEET_RL_SI_LLM_URL to the chat completions endpoint.}"
    : "${SPREADSHEET_RL_SI_LLM_MODEL:?Set SPREADSHEET_RL_SI_LLM_MODEL.}"
    export SPREADSHEET_RL_SI_LLM_URL SPREADSHEET_RL_SI_LLM_MODEL
    [ -n "${SPREADSHEET_RL_SI_LLM_API_KEY:-}" ] && export SPREADSHEET_RL_SI_LLM_API_KEY
fi

OVERRIDES=(
    # 必须走 credit_assignment 估计器, 否则 Driver 换 method 不会改变 advantage.
    algorithm.adv_estimator=credit_assignment
    # launch 时的 rewardflow 只是 metadata bootstrap 配置: worker 在初始化时冻结采集范围,
    # 之后 Driver 换 Skill 也不会重新采集. 真正的 step method 由 self_improving.initial_skill
    # (或恢复的 snapshot) 决定 —— §9.2.
    "algorithm.credit_assignment.method=${BOOTSTRAP_METHOD}"
    algorithm.credit_assignment.collect_step_states=True
    "self_improving.enabled=true"
    "self_improving.interval_steps=${SI_INTERVAL}"
    "self_improving.initial_skill.method=${INITIAL_SKILL}"
    "self_improving.candidate_skills=${CANDIDATES}"
    "self_improving.evaluation.min_window_steps=${MIN_WINDOW_STEPS}"
    "self_improving.knowledge.enabled=${KNOWLEDGE}"
    "self_improving.llm.client=${LLM_CLIENT}"
    "self_improving.persistence.enabled=true"
    "self_improving.persistence.strict_resume=true"
    "trainer.save_freq=${SAVE_FREQ}"
)
[ -n "${INITIAL_PARAMS}" ] && OVERRIDES+=("++self_improving.initial_skill.params=${INITIAL_PARAMS}")
[ "${LLM_CLIENT}" = "mock" ] && OVERRIDES+=("self_improving.llm.mock_script=${MOCK_SCRIPT}")
if [ -n "${RESUME_PATH}" ]; then
    # Slow Loop 的 snapshot / knowledge 版本跟着 policy checkpoint 一起恢复 (§19.3);
    # verl 要求 resume_from_path 指向 global_step_N 目录.
    OVERRIDES+=("trainer.resume_mode=resume_path" "trainer.resume_from_path=${RESUME_PATH}")
fi

CMD=(bash "${REPO}/deploy/run_grpo.sh" "${OVERRIDES[@]}" "$@")
echo "[run_self_improving] REPO=${REPO}"
echo "[run_self_improving] 命令:"
printf '    %q\n' "${CMD[@]}"
if [ "${DRY:-0}" = "1" ]; then
    echo "[run_self_improving] DRY=1 -> 未执行."
    exit 0
fi
exec "${CMD[@]}"
