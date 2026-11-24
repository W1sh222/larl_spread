#!/usr/bin/env python3
"""GPU 없이 GiGPO 모듈 배선만 확인 (00_preflight.sh / 06_step0 이 호출).

확인 순서:
  A. 등록 레지스트리: adv_estimator=credit_assignment, credit_assignment method 'gigpo',
     policy_loss registry 는 vanilla 만 쓰고 gigpo_joint/gigpo_geomean 은 존재하지 않음.
  B. 실제 hydra 구성으로 스모크 override 를 해석할 수 있는가
     (spreadsheet_rl_multiturn_grpo + run_gigpo_smoke.sh 와 동일한 오버라이드).
     -> unknown param 키는 GiGPOConfig.from_params 가 여기서 이미 잡는다.
  C. advantage -> vanilla token-level PPO 손실 smoke (가짜 배치, CPU).
     제거된 step 텐서/step 손실에 대한 참조가 config 와 코드 어디에도 없는지 함께 확인.
  D. PolicyLossConfig / AlgoConfig 필드와 스키마 일치 (B 에 통합).
  E. (선택) CPU 수학 테스트 실행 (--with-tests).

usage: check_module.py [--with-tests] --override k=v --override ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

FAILS = []
WARNS = []


def ok(msg):
    print(f"  [ok]   {msg}", flush=True)


def bad(msg):
    print(f"  [FAIL] {msg}", flush=True)
    FAILS.append(msg)


def warn(msg):
    print(f"  [WARN] {msg}", flush=True)
    WARNS.append(msg)


ap = argparse.ArgumentParser()
ap.add_argument("--with-tests", action="store_true", help="CPU 수학 테스트도 실행")
ap.add_argument("--override", action="append", default=[], help="hydra 오버라이드 (k=v)")
args = ap.parse_args()

verl_src = os.environ.get("VERL_SRC") or os.path.join(os.environ.get("REPO", ""), "verl")
if verl_src and os.path.isdir(verl_src):
    sys.path.insert(0, verl_src)

print(f"python = {sys.version.split()[0]}")
print(f"VERL_SRC = {verl_src}")

# ---------------------------------------------------------------- A. 등록 확인
print("\n=== A. 레지스트리 ===")
try:
    import torch

    ok(f"torch {torch.__version__}")
except Exception as e:
    bad(f"torch import 실패: {e}")
    sys.exit(1)

try:
    from verl.trainer.credit_assignment import available_methods
    from verl.trainer.credit_assignment.methods.gigpo.config import GiGPOConfig
    from verl.trainer.ppo import core_algos as C
except Exception as e:
    bad(f"verl import 실패 ({type(e).__name__}: {e})")
    sys.exit(1)

if "credit_assignment" in C.ADV_ESTIMATOR_REGISTRY:
    ok("adv_estimator credit_assignment 등록")
else:
    bad("credit_assignment estimator 미등록")
if "vanilla" in C.POLICY_LOSS_REGISTRY:
    ok("policy loss vanilla 등록 (GiGPO 가 실제로 쓰는 손실)")
else:
    bad("policy loss vanilla 미등록")
# 제거된 step objective 잔존 검사 (있으면 스모크 의미가 사라진다)
for name in ("gigpo_joint", "gigpo_geomean"):
    if name in C.POLICY_LOSS_REGISTRY:
        bad(f"policy loss {name} 가 아직 등록되어 있음 (step objective 미제거)")
    else:
        ok(f"policy loss {name} 없음")
try:
    import verl.trainer.ppo.step_tensors  # noqa: F401

    bad("verl.trainer.ppo.step_tensors 가 아직 존재 (제거 대상)")
except ImportError:
    ok("step_tensors 모듈 없음 (제거 확인)")
if "gigpo" in available_methods():
    ok(f"credit method gigpo (전체 {available_methods()})")
else:
    bad(f"gigpo method 없음: {available_methods()}")

# ---------------------------------------------------------------- B. hydra 구성 해석
print("\n=== B. 스모크 override 로 config 구성 ===")
# configs/qwen3-4b.sh + deploy/run_grpo.sh 이 얹는 필수 기둥을 축약 재현한다 (오버라이드 우선순위 동일).
BASELINE = [
    "actor_rollout_ref.model.path=" + (os.environ.get("MODEL_PATH") or "/dev/null"),
    "actor_rollout_ref.model.use_remove_padding=False",
    "actor_rollout_ref.actor.use_dynamic_bsz=True",
    "actor_rollout_ref.actor.ppo_mini_batch_size=2",
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.mode=async",
    "actor_rollout_ref.rollout.multi_turn.enable=True",
    "data.return_raw_chat=True",
    "algorithm.use_kl_in_reward=False",
    "trainer.n_gpus_per_node=" + os.environ.get("CHECK_NGPUS", "4"),
    "trainer.nnodes=1",
]
overrides = BASELINE + list(args.override)
print(f"  baseline {len(BASELINE)} + smoke overrides {len(args.override)} 개 주입")
cfg = None
try:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.environ["CONFIG_PATH"], version_base=None):
        cfg = compose(config_name="spreadsheet_rl_multiturn_grpo", overrides=overrides)
    ok("hydra compose 성공")
except Exception as e:
    bad(f"hydra compose 실패: {type(e).__name__}: {e}")

if cfg is not None:
    ca = cfg.algorithm.get("credit_assignment", None)
    if ca is None:
        bad("algorithm.credit_assignment 가 없음")
    else:
        print(f"  method={ca.get('method')} collect_step_states={ca.get('collect_step_states')}")
        print(f"  params={dict(ca.get('params') or {})}")
        try:
            g = GiGPOConfig.from_params(dict(ca.get("params") or {}))
            ok(f"GiGPOConfig.from_params 성공: anchor={g.anchor} mode={g.mode} gamma={g.gamma} "
               f"grouping={g.grouping} remove_std={g.remove_std}")
        except Exception as e:
            bad(f"GiGPO params 거부: {e}")
    pl = cfg.actor_rollout_ref.actor.policy_loss
    print(f"  policy_loss keys={sorted(pl.keys())}")
    dead = {"step_gradient", "step_reduction", "include_truncated_steps",
            "exclude_unexecuted_steps", "exclude_invalid_graph_steps", "clip_ratio_step"}
    leftover = [k for k in pl.keys() if k.startswith("step_") or k in dead]
    if leftover:
        bad(f"policy_loss 에 제거된 step 키 잔존: {leftover}")
    else:
        ok("policy_loss 에 step_* 키 없음 (config dangling guard 통과)")
    if pl.loss_mode != "vanilla":
        bad(f"policy_loss.loss_mode={pl.loss_mode} -> vanilla 여야 한다 (GiGPO 는 step objective 를 쓰지 않음)")
    elif pl.loss_mode not in C.POLICY_LOSS_REGISTRY:
        bad(f"loss_mode {pl.loss_mode} 가 registry 에 없음")
    else:
        actor = cfg.actor_rollout_ref.actor
        ok(f"loss_mode=vanilla, loss_agg_mode={actor.loss_agg_mode}, "
           f"clip_ratio={actor.clip_ratio}/{actor.clip_ratio_low}/{actor.clip_ratio_high}, "
           f"use_kl_loss={actor.use_kl_loss} kl_coef={actor.kl_loss_coef}")
    if cfg.algorithm.adv_estimator != "credit_assignment":
        bad(f"adv_estimator={cfg.algorithm.adv_estimator} (credit_assignment 여야 스모크 의미 없음)")

    # rollout 사이드 선택: tool_agent 루프여야 ca_step_* 가 생긴다
    dm = cfg.actor_rollout_ref.rollout.multi_turn
    if not dm.get("enable", False):
        bad("multi_turn.enable=False -> step 이 존재하지 않는다")
    dal = cfg.actor_rollout_ref.rollout.get("agent", {}).get("default_agent_loop", None)
    if dal != "tool_agent":
        warn(f"default_agent_loop={dal} (tool_agent 여야 ca_step_* 메타데이터 나옴)")

    try:
        from verl.utils.config import validate_config

        validate_config(cfg, use_reference_policy=True, use_critic=False)
        ok("validate_config 통과")
    except Exception as e:
        # launcher(qwen3-4b.sh) 가 얹는 키(use_dynamic_bsz/micro_bsz 등)까지 완전 재현하지 않으면
        # 실패할 수 있다 -> 스모크가 의도한 override 자체의 오류와 구분해 WARN 으로 둔다.
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        warn(f"validate_config 미통과(launcher baseline 미포함 가능성): {msg}")

# ---------------------------------------------------------------- C. advantage -> vanilla PPO smoke (순수 torch)
print("\n=== C. step advantage -> 기존 vanilla token-level PPO smoke (가짜 배치) ===")
try:
    import inspect

    import numpy as np

    from verl.trainer.credit_assignment import compute as ca_compute
    from verl.trainer.credit_assignment.interface import INITIAL_STATE_MARKER
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, compute_grpo_outcome_advantage
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    bs, resp_len = 4, 12
    # 2 task(t0/t1) x 2 trajectory. span 밖은 tool observation / padding -> mask 0.
    spans = [[[0, 3], [4, 7], [8, 11]],
             [[0, 3], [4, 7], [8, 11]],
             [[0, 4], [5, 9]],
             [[0, 4], [5, 9]]]
    uids = np.array(["t0", "t0", "t1", "t1"], dtype=object)
    traj_rewards = [1.0, 0.0, 1.0, 0.0]
    obs = [[INITIAL_STATE_MARKER, "oA1", "oA2"],
           [INITIAL_STATE_MARKER, "oA1", "oA2"],
           [INITIAL_STATE_MARKER, "oB1"],
           [INITIAL_STATE_MARKER, "oB1"]]
    # anchor='state' 는 pre-action workbook fingerprint 를 쓴다 (step 0 의 pre == initial state).
    states = [["S0", "A1", "A2"], ["S0", "A1", "A2"], ["S0", "B1"], ["S0", "B1"]]

    response_mask = torch.zeros(bs, resp_len, dtype=torch.long)
    token_level_rewards = torch.zeros(bs, resp_len)
    for row, ss in enumerate(spans):
        for s, e in ss:
            response_mask[row, s:e] = 1
        token_level_rewards[row, ss[-1][1] - 1] = traj_rewards[row]

    nt = {
        "ca_step_spans": np.array(spans, dtype=object),
        "ca_step_observations": np.array(obs, dtype=object),
        "ca_step_transitions": np.array(
            [[{"pre": fp, "post": fp, "executed": True, "status": "executed", "graph_valid": True}
              for fp in ss] for ss in states], dtype=object),
        "ca_initial_state": np.array(["S0"] * bs, dtype=object),
    }
    anchor = os.environ.get("CA_ANCHOR", "state")
    params = {
        "anchor": anchor,
        "step_advantage_w": float(os.environ.get("CA_STEP_W", "1.0")),
        "gamma": float(os.environ.get("CA_GAMMA", "0.95")),
        "mode": os.environ.get("CA_MODE", "mean_norm"),
    }
    out = ca_compute(method=os.environ.get("CA_METHOD", "gigpo"), token_level_rewards=token_level_rewards,
                     response_mask=response_mask, index=uids, non_tensor_batch=nt, params=params)
    adv = out.advantages
    assert adv.shape == (bs, resp_len) and adv.is_floating_point() and adv.device.type == "cpu", adv
    ok(f"advantage tensor {tuple(adv.shape)}/{adv.dtype}/{adv.device} -> 기존 PPO interface 그대로")
    assert int(torch.count_nonzero(adv[response_mask == 0])) == 0, "mask 밖 token 에 advantage 가 남아 있음"
    ok("assistant mask 밖(tool observation/padding) advantage 0")

    step_vals = []
    for row, ss in enumerate(spans):
        vals = []
        for s, e in ss:
            seg = adv[row, s:e][response_mask[row, s:e] > 0]
            assert seg.numel() > 0, (row, s, e)
            assert float((seg - seg[0]).abs().max()) == 0.0, f"row {row} step span [{s},{e}) advantage 가 균일하지 않음: {seg}"
            vals.append(float(seg[0]))
        step_vals.append(vals)
        assert len(set(vals)) > 1, f"row {row}: step advantage 가 모두 같음 {vals}"
    shown = " | ".join(" ".join(f"{v:+.3f}" for v in r) for r in step_vals)
    ok(f"같은 step 은 동일 값, 다른 step 은 다른 값 (Eq.8 broadcast): {shown}")

    # 손실은 기존 vanilla token-level PPO 를 그대로 호출한다. GiGPO 전용 인자/분기 없음 확인.
    actor_cfg = None
    if cfg is not None:
        try:
            from verl.utils.config import omega_conf_to_dataclass

            # 학습 때와 완전히 같은 방법으로 actor config 를 만든다 (worker 의 구성 경로).
            actor_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
            ok(f"omega_conf_to_dataclass(actor) 성공 (실학습과 동일 경로): {type(actor_cfg).__name__}")
        except Exception as e:
            warn(f"ActorConfig 구성 실패({type(e).__name__}: {str(e).splitlines()[0]}) -> dataclass 직접 생성")
            actor_cfg = None
    if actor_cfg is None:
        from verl.workers.config.actor import ActorConfig, PolicyLossConfig

        actor_cfg = ActorConfig(strategy="fsdp2", ppo_mini_batch_size=2, use_dynamic_bsz=True,
                                rollout_n=2, loss_agg_mode="token-mean", clip_ratio=0.2,
                                clip_ratio_low=0.2, clip_ratio_high=0.2,
                                policy_loss=PolicyLossConfig())
    # ActorConfig 는 frozen dataclass -> 필드 재대입은 불가, dict 내용만 갱신한다 (worker 도 동일).
    actor_cfg.global_batch_info.update({"dp_size": 1, "batch_num_tokens": None,
                                        "global_batch_size": None, "loss_scale_factor": None})

    loss_mode = actor_cfg.policy_loss.get("loss_mode", "vanilla") if actor_cfg.policy_loss else "vanilla"
    if loss_mode not in POLICY_LOSS_REGISTRY:
        bad(f"loss_mode {loss_mode} 가 registry 에 없음")
        raise KeyError(loss_mode)
    fn = POLICY_LOSS_REGISTRY[loss_mode]
    if fn is not compute_policy_loss_vanilla:
        warn(f"policy loss={loss_mode} 로 구성됨 (공식 GiGPO 재현은 vanilla 를 쓴다)")
    argnames = list(inspect.signature(fn).parameters)
    if any(a.startswith("step_") for a in argnames):
        bad(f"policy loss 서명에 step_* 인자 잔존: {argnames}")
    else:
        ok(f"policy loss 시그니처 step-free: {argnames}")

    old_lp = torch.zeros(bs, resp_len)
    lp = torch.zeros(bs, resp_len, requires_grad=True)
    loss, metrics = fn(old_log_prob=old_lp, log_prob=lp, advantages=adv, response_mask=response_mask,
                       loss_agg_mode=actor_cfg.loss_agg_mode, config=actor_cfg)
    g = torch.autograd.grad(loss, lp)[0]
    nz = int((g != 0).sum().item())
    nz_in = int((g[response_mask > 0] != 0).sum().item())
    assert torch.isfinite(loss), loss
    n_masked = int((response_mask > 0).sum().item())
    assert nz > 0 and nz == nz_in, f"gradient 가 mask 밖으로도 번짐/전부 0: nz={nz} in_mask={nz_in}"
    if nz == n_masked:
        ok(f"vanilla PPO 실행: loss={float(loss):+.5f}, gradient 는 정확히 assistant token {n_masked}/{bs * resp_len} 개만")
    else:
        bad(f"gradient 위치가 mask 와 어긋남: nonzero={nz}, assistant token={n_masked}")
    stepish = sorted(k for k in metrics if "step" in k)
    if stepish:
        bad(f"vanilla metrics 에 step_* 잔존: {stepish}")
    else:
        ok(f"vanilla 표준 지표만 출력: {sorted(metrics)}")

    # step 항을 끄면 advantage 가 GRPO 와 일치 -> loss/optimizer 는 알고리즘과 무관 (D5/D8).
    adv_off = ca_compute(method=os.environ.get("CA_METHOD", "gigpo"), token_level_rewards=token_level_rewards,
                         response_mask=response_mask, index=uids, non_tensor_batch=nt,
                         params={**params, "step_advantage_w": 0.0, "mode": "mean_norm"}).advantages
    grpo_adv, _ = compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                 response_mask=response_mask, index=uids,
                                                 norm_adv_by_std_in_grpo=False)
    if torch.allclose(adv_off, grpo_adv, atol=1e-6):
        ok("step_advantage_w=0 -> advantage 가 GRPO(mean_std_norm=False) 와 동일 (PPO 측 GiGPO 분기 불필요)")
    else:
        bad(f"step_advantage_w=0 인데 GRPO 와 다름 (max diff {float((adv_off - grpo_adv).abs().max()):.3e})")
except Exception as e:
    bad(f"advantage -> vanilla PPO smoke 실패: {type(e).__name__}: {e}")

# ---------------------------------------------------------------- D. CPU 테스트
if args.with_tests:
    print("\n=== D. CPU 수학 테스트 ===")
    files = ["tests/trainer/ppo/test_credit_assignment_gigpo_on_cpu.py",
             "tests/trainer/ppo/test_gigpo_vanilla_link_on_cpu.py",
             "tests/trainer/ppo/test_core_algos_on_cpu.py"]
    if subprocess.run([sys.executable, "-m", "pytest", "--version"], capture_output=True).returncode != 0:
        # uv venv 에 pytest 가 없는 것이 정상(본 검사는 GPU 학습 전 배선 확인용) -> 건너뛰고 경고만.
        warn(f"pytest 없음 -> CPU 수학 테스트 건너뜀 ({sys.executable} -m pip/uv 로 설치 가능)")
        files = []
    for f in files:
        path = os.path.join(verl_src, f)
        if not os.path.isfile(path):
            warn(f"테스트 파일 없음: {f}")
            continue
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", f], cwd=verl_src)
        (r.returncode == 0) and ok(f"pytest {f}") or bad(f"pytest {f} rc={r.returncode}")

print("\n=== 결론 ===")
for w in WARNS:
    print(f"  WARN: {w}")
if FAILS:
    for f in FAILS:
        print(f"  FAIL: {f}")
    sys.exit(1)
print("  [PASS] GiGPO 모듈 배선 확인 완료 (GPU 스모크 준비 상태 OK)" if not WARNS else
      "  [PASS] 실패 없음 (경고는 위 참조)")
