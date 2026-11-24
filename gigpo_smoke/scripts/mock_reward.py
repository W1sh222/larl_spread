"""GiGPO 스모크용 로컬 의사보상 (REWARD=mock 일 때 reward.custom_reward_function.path 로 로드됨).

Windows 보상 서비스 없이도 GiGPO 모듈 배선을 검증하기 위한 대체 Reward.
의미 있는 "가상의 성공 신호"를 만드는 것이지 스프레드시트 정확도를 재는 것이 아니다:

    score = 0.45*[워크북이 초기본과 달라졌으면] + 0.3*[최종 워크북 해시 jitter]
            + 0.25*[trajectory 텍스트 해시 jitter]

- 그룹(같은 uid 의 rollout_n 개) 내에서 값이 달라지도록 hash 기반 jitter 를 넣는다.
  그래야 episode advantage 와 step advantage 가 0으로 퇴화하지 않는다.
- 최종 상태 해시 jitter 는 상태 앵커와 같은 정보로 결정된다(일관성).
- trajectory 텍스트 해시 jitter 는 "도구가 한 번도 워크북을 못 바꾼" 경우(재계산 서비스 정지)
  에도 그룹 내 분산이 0 이 되지 않게 하는 완충 장치다. 이때 mock_changed=0 이 나오고
  verify_run.py 가 WARN 을 낸다 (배선은 통과, 신호는 단조).
- tool call 횟수는 참고용 메타로만 둔다.

시그니처는 RateLimitedRewardManager._compute_reward 호출과 동일해야 한다:
    compute_score(data_source=, solution_str=, ground_truth=, extra_info=)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Optional


def _data_root() -> Path:
    root = os.environ.get("SPREADSHEET_RL_DATA_ROOT")
    return Path(root).expanduser().resolve() if root else Path.cwd() / "data"


def _thread_dir(ground_truth: Any) -> Optional[str]:
    gt = ground_truth
    if isinstance(gt, list) and len(gt) == 1:
        gt = gt[0]
    if not isinstance(gt, str) or not gt.strip() or ".." in gt or gt.startswith("/"):
        return None
    return gt.strip()


def _find_workbook(directory: Path) -> Optional[Path]:
    for name in ("output.xlsx", "data.xlsx"):
        p = directory / name
        if p.is_file():
            return p
    return None


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_score(data_source: str = "", solution_str: str = "",
                  ground_truth: Any = None, extra_info: Any = None,
                  **_: Any) -> dict:
    extra = dict(extra_info or {})
    root = _data_root()
    td = _thread_dir(ground_truth)
    if td is None:
        return {"score": 0.0, "acc": 0.0, "mock_reason": "bad_thread_dir"}

    # rollout 이 만든 워크북: _workspaces/<id>/ 아래, 없으면 원본 태스크 디렉토리
    ws_id = extra.get("sandbox_workspace_id")
    final_wb = None
    from_workspace = False
    if isinstance(ws_id, str) and ws_id:
        final_wb = _find_workbook(root / "_workspaces" / ws_id)
        from_workspace = final_wb is not None
    if final_wb is None:
        final_wb = _find_workbook(root / td)
    if final_wb is None:
        return {"score": 0.0, "acc": 0.0, "mock_reason": "no_final_workbook"}

    # 기준(초기) 워크북: rollout 은 {thread_dir}/output.xlsx 를 workspace 로 복사해 시작한다.
    # workspace 사본을 못 쓴 경우(시드 실패)에는 "바뀌었는지" 판단할 수 없으므로 changed=0 으로 둔다.
    seed = None
    if from_workspace:
        for name in ("output.xlsx", "input.xlsx"):
            cand = root / td / name
            if cand.is_file() and cand.resolve() != final_wb.resolve():
                seed = cand
                break
    seed_digest = _digest(seed) if seed is not None else ""
    final_digest = _digest(final_wb)
    changed = 1.0 if (seed_digest and seed_digest != final_digest) else 0.0

    # 상태 해시로 결정적 jitter -> 같은 task 의 n 개 trajectory 가 서로 다른 값을 갖고,
    # 같은 상태로는 도달하면 같은 값이 나온다 (anchor grouping 과 reward 가 일관된다).
    jitter = int(final_digest[:8], 16) / float(0xFFFFFFFF)
    # trajectory 텍스트 해시: 도구가 상태를 하나도 못 바꿔도 그룹 내 분산이 0 이 되지 않게 한다.
    traj_digest = hashlib.sha256((solution_str or "").encode("utf-8", "replace")).hexdigest()
    traj_jitter = int(traj_digest[8:16], 16) / float(0xFFFFFFFF)
    score = 0.45 * changed + 0.3 * jitter + 0.25 * traj_jitter

    n_tool_calls = str(solution_str or "").count("<tool_call>")
    msg = (f"mock: changed={changed:.0f} ws={int(from_workspace)} jitter={jitter:.3f} "
           f"traj_jitter={traj_jitter:.3f} calls={n_tool_calls} wb={final_wb.name}")
    return {
        "score": float(score),
        "acc": float(changed),
        "mock_changed": float(changed),
        "mock_from_workspace": float(from_workspace),
        "mock_jitter": float(jitter),
        "mock_traj_jitter": float(traj_jitter),
        "mock_tool_calls": float(n_tool_calls),
        "mock_msg": msg,
    }
