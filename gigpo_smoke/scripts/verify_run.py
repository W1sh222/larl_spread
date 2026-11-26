#!/usr/bin/env python3
"""run_gigpo_smoke.sh 의 결과 판정기. train.log(console metrics) + rollout dump 를 읽는다.

verl console 로거는 verl/utils/logger/aggregate_logger.py:concat_dict_to_str 로
    step:7 - key:value - key:value ...
형태의 한 줄을 출력한다 (숫자만 포함). 이 형식을 파싱해 GiGPO 모듈이 "실제로" 동작했는지 판정한다:

  (0) 로그에 예외/OOM/레지스트리 실패가 없는가
  (1) credit_assignment/gigpo/* 지표가 존재하는가        -> 모듈 adv 가 실행됨
  (2) step_adv_nonzero_frac > 0, steps_per_traj > 1      -> step 단위 신호가 학습에 기여함 (핵심)
  (3) anchor_state_frac / truncated_step_frac 등 step 메타가 유입됨 -> rollout(tool_agent) 가 ca_step_* 를 내보냄
  (4) actor/pg_loss · actor/ppo_kl · actor/pg_clipfrac 존재/유한   -> 기존 vanilla token-level PPO 가 실행됨
  (5) actor/step_* 지표가 하나도 없음                     -> 제거된 step objective 이 여백 없이 동작 중
  (6) pg_loss / grad_norm 에 nan·inf 없음
  (7) TOTAL_STEPS 만큼 metric 라인이 나왔는가

정책: 공식 GiGPO 는 step-level policy loss 를 쓰지 않는다. advantage 만 모듈이 교체하고
손실은 항상 vanilla 이므로, LOSS_MODE 계열 설정은 이 판정기에 존재하지 않는다.

exit 0 = PASS(WARN 허용), 1 = FAIL, 3 = 로그를 못 읽음.
stdlib 만 사용한다 (아무 python 으로든 돌릴 수 있음).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

METRIC_LINE = re.compile(r"^step:(\d+)\s+-\s+")
PAIR = re.compile(r"([A-Za-z0-9_./-]+):(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|NaN|inf|Inf|-inf|-Inf)")

FATAL_PATTERNS = [
    ("Traceback (most recent call last)", "python 예외"),
    ("CUDA out of memory", "GPU OOM"),
    ("out of memory", "메모리 부족"),
    ("requires metadata", "step 메타데이터 없음 (credit_assignment adapter 거부)"),
    ("Unknown credit assignment method", "credit_assignment method 미등록"),
    ("not in the list of registered", "loss/estimator 미등록"),
    ("Could not override", "hydra override 거부 (struct lock)"),
    ("Key 'credit_assignment' not in node", "algorithm.credit_assignment 미구성"),
]


def parse_log(path):
    """({step: {key: float}}, [(lineno, why, text)], n_lines)."""
    steps = defaultdict(dict)
    errors = []
    n_lines = 0
    with open(path, "r", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            n_lines = lineno
            line = raw.rstrip("\n")
            for pat, why in FATAL_PATTERNS:
                if pat in line:
                    errors.append((lineno, why, line[:200]))
            if not METRIC_LINE.match(line):
                continue
            step = int(re.match(r"^step:(\d+)", line).group(1))
            for k, v in PAIR.findall(line):
                if k == "step":
                    continue
                try:
                    steps[step][k] = float(v)
                except ValueError:
                    pass
    return steps, errors, n_lines


def series(steps, key):
    return [steps[s][key] for s in sorted(steps) if key in steps[s]]


def get_any(steps, keys):
    for k in keys:
        vals = series(steps, k)
        if vals:
            return k, vals
    return None, []


def fmt(vals, n=8):
    if not vals:
        return "n/a"
    s = ", ".join(f"{v:.4g}" for v in vals[:n])
    return s + ("" if len(vals) <= n else f" ...(n={len(vals)})")


class Verdict:
    def __init__(self):
        self.rows = []

    def _add(self, sev, name, detail=""):
        self.rows.append({"sev": sev, "name": name, "detail": detail})
        print(f"  [{sev}] {name}" + (f"  |  {detail}" if detail else ""))

    def ok(self, name, detail=""):
        self._add("PASS", name, detail)

    def warn(self, name, detail=""):
        self._add("WARN", name, detail)

    def fail(self, name, detail=""):
        self._add("FAIL", name, detail)

    def count(self, sev):
        return sum(1 for r in self.rows if r["sev"] == sev)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--loss-mode", default="vanilla",
                    help="기대한 actor.policy_loss.loss_mode (GiGPO 는 vanilla 를 그대로 쓴다)")
    ap.add_argument("--anchor", default="state")
    ap.add_argument("--reward", default="mock")
    ap.add_argument("--method", default="gigpo")
    ap.add_argument("--expect-steps", type=int, default=1, help="기대한 학습 step 수 (TOTAL_STEPS)")
    ap.add_argument("--json", default=None, help="판정 결과를 json 으로도 저장")
    a = ap.parse_args(argv)

    V = Verdict()
    print("=== GiGPO smoke verdict ===")
    print(f"  log        = {a.log}")
    print(f"  loss_mode  = {a.loss_mode}   anchor = {a.anchor}   reward = {a.reward}   method = {a.method}")
    if not os.path.isfile(a.log) or os.path.getsize(a.log) == 0:
        V.fail("로그 없음/비어 있음", f"{a.log} -> 학습이 시작조차 되지 않음")
        return finish(V, a, {})

    steps, errors, n_lines = parse_log(a.log)
    P = f"credit_assignment/{a.method}/"
    print(f"  log lines  = {n_lines}   metric steps = {len(steps)} {sorted(steps)[:12]}")

    # ------------------------------------------------------------ 0. 치명 오류
    if errors:
        seen = set()
        for lineno, why, txt in errors:
            if why in seen:
                continue
            seen.add(why)
            V.fail("로그 내 치명 오류", f"{why} @L{lineno}: {txt[:150]}")
    else:
        V.ok("예외/OOM/레지스트리 실패 문자열 없음")

    # ------------------------------------------------------------ 1. 진행
    if not steps:
        V.fail("metric 라인 0개", "학습 step 이 한 번도 로깅되지 않음 (크래시 또는 trainer.logger 설정 확인)")
        return finish(V, a, {})
    max_step = max(steps)
    if max_step + 1 >= a.expect_steps:
        V.ok("학습 step 진행", f"마지막 step 라인 = {max_step} (기대 >= {a.expect_steps - 1})")
    else:
        V.fail("학습 step 부족", f"max={max_step} < 기대 {a.expect_steps - 1} -> 중간 중단")

    # ------------------------------------------------------------ 2. 모듈 실행 여부
    ca_keys = sorted({k for s in steps.values() for k in s if k.startswith(P)})
    if ca_keys:
        V.ok("모듈 지표 존재", f"{P}* {len(ca_keys)}종")
    else:
        V.fail("credit_assignment 지표 없음", "adv_estimator=credit_assignment 경로가 실행되지 않음")

    # ------------------------------------------------------------ 3. step advantage 기여 (핵심)
    for key, required in (("step_adv_nonzero_frac", True),
                          ("anchor_group_count", False), ("n_anchor_distinct", False),
                          ("baseline_excluded_truncated", False)):
        vals = series(steps, P + key)
        if not vals:
            (V.fail if required else V.warn)(f"{key} 미로깅", "gigpo algorithm.metrics 참고")
            continue
        if required and max(vals) <= 0.0:
            if key == "step_adv_nonzero_frac":
                V.fail(key, f"step advantage 전부 0 -> GiGPO 항이 학습에 기여하지 못함 ({fmt(vals)})")
            else:
                V.fail(key, f"유효 step 이 0개 ({fmt(vals)})")
        else:
            V.ok(key, fmt(vals))

    # ------------------------------------------------------------ 4. rollout -> 모듈 step 메타데이터 유입
    # actor/step_* (제거된 build_step_tensors 지표) 대신 CA 측 진단량을 본다.
    for key, required in (("steps_per_traj", True), ("mean_step_tokens", False),
                          ("truncated_step_frac", False), ("unexecuted_step_frac", False),
                          ("empty_span_step_frac", False), ("final_answer_step_frac", False),
                          ("graph_invalid_step_frac", False)):
        vals = series(steps, P + key)
        if not vals:
            (V.fail if required else V.warn)(f"{key} 미로깅", "state_meta.meta_diagnostics 지표")
            continue
        if key == "steps_per_traj" and max(vals) <= 1.0:
            V.fail(key, f"{fmt(vals)} -> step 분할이 없음 "
                        f"(rollout.agent.default_agent_loop=tool_agent / multi_turn.enable 확인)")
        elif required and max(vals) <= 0.0:
            V.fail(key, f"{fmt(vals)} -> rollout 이 ca_step_* 를 내보내지 않음")
        else:
            V.ok(P + key, fmt(vals))

    # ------------------------------------------------------------ 5. 기존 vanilla PPO 경로 실행 확인
    if a.loss_mode != "vanilla":
        V.warn(f"loss_mode={a.loss_mode} 는 의도된 설정 아님",
               "GiGPO 는 step objective 를 쓰지 않는다. env.sh 에서 policy_loss override 를 제거하라")
    for key, required in (("actor/pg_loss", True), ("actor/ppo_kl", True),
                          ("actor/pg_clipfrac", False), ("actor/loss", True)):
        vals = series(steps, key)
        if not vals:
            (V.fail if required else V.warn)(f"{key} 미로깅", "compute_policy_loss_vanilla 의 표준 지표")
            continue
        if any(not math.isfinite(x) for x in vals):
            V.fail(f"{key} 비유한값", fmt(vals))
        else:
            V.ok(f"vanilla PPO 실행 ({key})", fmt(vals))
    # 제거된 step objective 의 잔존 여부 (dangling guard)
    leftover = sorted({k for s in steps.values() for k in s if k.startswith("actor/step")})
    if leftover:
        V.fail("actor/step_* 지표가 여전히 출력됨", ",".join(leftover[:6]) + " -> step loss 가 남아 있음")
    else:
        V.ok("actor/step_* 지표 없음 (step objective 미사용 확인)")

    # ------------------------------------------------------------ 6. 앵커 설정 일치
    st = series(steps, P + "anchor_state_frac")
    ob = series(steps, P + "anchor_observation_frac")
    iv = series(steps, P + "anchor_invalid_frac")
    if a.anchor == "state":
        if st and max(st) <= 0:
            V.warn("anchor_state_frac == 0", "state 앵커 미사용 -> observation 으로 대체됨 "
                                             "(collect_step_states / ca_step_transitions 수집 확인)")
        else:
            V.ok("anchor_state_frac", fmt(st))
    else:
        if ob and max(ob) <= 0:
            V.warn("anchor_observation_frac == 0", f"anchor={a.anchor} 설정과 불일치")
        else:
            V.ok("anchor_observation_frac", fmt(ob))
    if iv and max(iv) > 0.5:
        V.fail("anchor_invalid_frac > 0.5", f"앵커 지문이 대부분 무효 ({fmt(iv)}) -> fingerprint/transition 확인")
    elif iv:
        V.ok("anchor_invalid_frac", fmt(iv))

    # ------------------------------------------------------------ 7. 수치 건전성
    # 7-1. 모듈 지표 전체를 훑어 nan·inf 검출 (키를 하나씩 나열하지 않음)
    dirty = sorted({k for s in steps.values() for k in s
                    if (k.startswith(P) or k.startswith("actor/step"))
                    and not math.isfinite(s[k])})
    if dirty:
        for k in dirty:
            V.fail(f"{k} 에 nan/inf", fmt(series(steps, k)))
    else:
        V.ok("module/step 지표에 nan/inf 없음")
    # 7-2. 학습 핵심 스칼라
    fin = []
    for k in ("actor/pg_loss", "actor/loss", "actor/grad_norm",
              "critic/score/mean", "critic/rewards/mean"):
        vals = series(steps, k)
        if not vals:
            continue
        if any(not math.isfinite(x) for x in vals):
            V.fail(f"{k} 에 nan/inf", fmt(vals))
        else:
            fin.append(f"{k}={vals[-1]:.4g}")
    if fin:
        V.ok("주요 수치 유한", " | ".join(fin))

    # ------------------------------------------------------------ 8. 보상 분포
    rkey, rvals = get_any(steps, ["critic/score/mean", "critic/rewards/mean"])
    if not rvals:
        V.warn("보상 지표 없음", "critic/score/mean · critic/rewards/mean 둘 다 없음")
    elif max(rvals) <= 0.0:
        V.warn(f"{rkey} 전부 0", f"{fmt(rvals)} -> 보상 없음 (mock 인데 워크스페이스 사본을 못 찾거나 도구 실패). "
                                 f"GiGPO 배선 검증 자체는 성립하나 advantage 가 단조로울 수 있음")
    else:
        V.ok(f"보상 분포 ({rkey})", fmt(rvals))

    # ------------------------------------------------------------ 9. 산출물
    if a.run_dir:
        rd = os.path.join(a.run_dir, "rollout_data")
        dumps = sorted(f for f in os.listdir(rd) if f.endswith(".jsonl")) if os.path.isdir(rd) else []
        if not dumps:
            V.warn("rollout_data/*.jsonl 없음", f"{rd} (trainer.rollout_data_dir 확인)")
        else:
            n_rows, n_multi, scores, mock_keys = 0, 0, [], set()
            for fn in dumps:
                with open(os.path.join(rd, fn), errors="replace") as f:
                    for line in f:
                        try:
                            e = json.loads(line)
                        except Exception:
                            continue
                        n_rows += 1
                        mock_keys |= {k for k in e if k.startswith("mock_")}
                        if isinstance(e.get("score"), (int, float)):
                            scores.append(float(e["score"]))
                        out = str(e.get("output") or "")
                        if "<tool_call>" in out or "<|im_start|>tool" in out or "Ċ" in out:
                            n_multi += 1
            detail = (f"{len(dumps)} 파일 / {n_rows} 행, 멀티턴 흔적 {n_multi} 행, "
                      f"score mean={sum(scores)/len(scores):.3f} max={max(scores):.3f}" if scores else
                      f"{len(dumps)} 파일 / {n_rows} 행")
            if n_rows == 0:
                V.warn("rollout dump 이 비어 있음", detail)
            else:
                V.ok("rollout dump", detail)
            if a.reward == "mock" and mock_keys:
                V.ok("mock 보상 extra_info 컬럼", ",".join(sorted(mock_keys)))
            elif a.reward == "mock":
                V.warn("mock extra_info 컬럼 없음", "reward.custom_reward_function 이 실제로 불렸는지 확인")

    # ------------------------------------------------------------ 10. 참고 값 한 덩어리
    print("\n  --- 마지막 step 의 모듈/손실 지표 ---")
    last = steps[max_step]
    for k in sorted([k for k in last if k.startswith(P) or k.startswith("actor/")][:80]):
        print(f"      {k} = {last[k]:.5g}")
    return finish(V, a, {"max_step": max_step, "n_metric_steps": len(steps),
                         "n_ca_keys": len(ca_keys), "log_lines": n_lines})


def finish(V, a, extra):
    print(f"\n=== 판정: FAIL={V.count('FAIL')} WARN={V.count('WARN')} PASS={V.count('PASS')} ===")
    for r in V.rows:
        if r["sev"] == "FAIL":
            print(f"  FAIL: {r['name']} | {r['detail']}")
    if a.json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
            with open(a.json, "w") as f:
                json.dump({"loss_mode": a.loss_mode, "anchor": a.anchor, "reward": a.reward,
                           "method": a.method, "expect_steps": a.expect_steps,
                           "fail": V.count("FAIL"), "warn": V.count("WARN"),
                           "rows": V.rows, "extra": extra}, f, indent=1, ensure_ascii=False)
            print(f"  json: {a.json}")
        except Exception as e:
            print(f"  [WARN] json 저장 실패: {e}")
    if V.count("FAIL"):
        print("  => 실패. README 의 '흔한 실패' 표와 train.log 마지막 200 줄을 보세요.")
        return 1
    print("  => GiGPO 모듈 end-to-end 경로 확인 완료"
          + (" (경고는 참조)" if V.count("WARN") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
