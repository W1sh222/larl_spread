"""Self-Improving 状态的 run-local 持久化与 checkpoint 对齐恢复.

写入 {trainer.default_local_dir}/self_improving/. 除了滚动 state.json, 每次 policy checkpoint
保存成功后都落一份 step snapshot: 恢复 step N 的 policy 时只允许用 step N 的 controller 状态,
避免把未来 step 的 skill / knowledge 混进过去的 checkpoint.
"""

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from typing import Any, Optional

from verl.trainer.credit_assignment.self_improving.config import CONFIG_VERSION

STATE_FILENAME = "state.json"
SNAPSHOT_DIR = "snapshots"
HISTORY_FILENAME = "history.jsonl"
CONTEXTS_FILENAME = "contexts.jsonl"
PREVIOUS_DIR = "previous"

# resume 时允许改变的运营设置 (endpoint / timeout / 落盘开关): 只记录, 不阻断.
OPERATIONAL_PREFIXES = ("llm.", "persistence.")
# 决定实验语义的设置: 变化会写进 history, 但只有 RESEARCH_CRITICAL 不一致才阻断 resume.
RESEARCH_CRITICAL = ("adv_estimator", "bootstrap_method", "collect_step_states", "candidate_skills")


class SelfImprovingStateError(RuntimeError):
    """Resume 状态不一致: 默认 (strict_resume) 直接报错, 不猜恢复内容."""


def _atomic_write_json(path: str, payload: Any) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):  # pragma: no cover
            os.remove(tmp)


def _read_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise SelfImprovingStateError(f"corrupted self_improving state file {path}: {e}") from e


def experiment_identity(config: Any) -> dict:
    """用于判断 resume 是否属于同一实验的粗粒度指纹 (不含 credential)."""
    algo = config.get("algorithm", None) if hasattr(config, "get") else None
    ca = algo.get("credit_assignment", None) if algo is not None else None
    si = config.get("self_improving", None) if hasattr(config, "get") else None
    trainer = config.get("trainer", None) if hasattr(config, "get") else None
    payload = {
        "adv_estimator": algo.get("adv_estimator") if algo is not None else None,
        "bootstrap_method": ca.get("method") if ca is not None else None,
        "collect_step_states": ca.get("collect_step_states") if ca is not None else None,
        "project": trainer.get("project_name") if trainer is not None else None,
        "experiment": trainer.get("experiment_name") if trainer is not None else None,
        "interval_steps": si.get("interval_steps") if si is not None else None,
        "candidate_skills": list(si.get("candidate_skills") or []) if si is not None else [],
        "total_training_steps": trainer.get("total_training_steps") if trainer is not None else None,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    payload["digest"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return payload


def new_run_id() -> str:
    # 同一秒内的多次调用也必须唯一, 因此附加 pid + 随机后缀.
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + f"-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def flatten(node: Any, prefix: str = "") -> dict:
    """把嵌套 config 摊平成 `a.b` 形式的 key, 便于 diff 出改了哪些设置."""
    out: dict = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}{key}."))
        return out
    out[prefix[:-1]] = node
    return out


def config_diff(stored: Optional[dict], current: Optional[dict]) -> list:
    """resume 前后 self_improving 配置的差异, 按运营 / 研究语义分类."""
    if not isinstance(stored, dict) or not isinstance(current, dict):
        return []
    before, after = flatten(stored), flatten(current)
    changes = []
    for key in sorted(set(before) | set(after)):
        if before.get(key, "<absent>") == after.get(key, "<absent>"):
            continue
        changes.append(
            {
                "key": key,
                "stored": before.get(key),
                "current": after.get(key),
                "kind": "operational" if key.startswith(OPERATIONAL_PREFIXES) else "research",
            }
        )
    return changes


class SelfImprovingStore:
    def __init__(self, root: str, strict_resume: bool = True, enabled: bool = True):
        self.root = root
        self.strict_resume = strict_resume
        self.enabled = enabled
        self.run_id = None
        self.lineage = None
        self.prior_lineage = None
        self.archived_to = None

    # ---------- layout ----------
    def prepare(self, run_id: str, resuming: bool) -> None:
        if not self.enabled:
            return
        os.makedirs(self.root, exist_ok=True)
        existing = _read_json(os.path.join(self.root, STATE_FILENAME))
        if existing and not resuming:
            self._archive_existing(existing)
        self.run_id = run_id
        if existing and resuming:
            # 同一实验链: 沿用已存 lineage, 让 resume 前后的记录还能聚在一起.
            self.prior_lineage = str(existing.get("lineage") or existing.get("run_id") or "") or None
            self.lineage = self.prior_lineage or run_id
        else:
            self.lineage = run_id
        os.makedirs(os.path.join(self.root, SNAPSHOT_DIR), exist_ok=True)

    def _archive_existing(self, existing: dict) -> None:
        previous_run = str(existing.get("run_id") or "unknown")
        target = os.path.join(self.root, PREVIOUS_DIR, previous_run)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            target = f"{target}#{int(time.time())}"
        # 整目录搬走, 不删任何历史数据; 新 run 从干净目录开始.
        os.makedirs(target, exist_ok=True)
        for name in os.listdir(self.root):
            if name == PREVIOUS_DIR:
                continue
            shutil.move(os.path.join(self.root, name), os.path.join(target, name))
        self.archived_to = target
        print(f"[self_improving] fresh run in a directory that already holds state for run '{previous_run}'; "
              f"moved the previous run to {target} (nothing was overwritten)", flush=True)

    def start_new_lineage(self, reason: str) -> Optional[str]:
        """恢复不到对齐状态时: 把旧 run 整体归档, 新 lineage 从空目录开始, 不覆盖任何历史."""
        existing = self.load_state() or {}
        if not existing and not os.path.isdir(os.path.join(self.root, SNAPSHOT_DIR)):
            return None
        prior = str(existing.get("lineage") or existing.get("run_id") or "unknown")
        self._archive_existing(existing)
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, SNAPSHOT_DIR), exist_ok=True)
        self.prior_lineage = prior
        self.lineage = self.run_id
        print(f"[self_improving] starting a new self_improving lineage ({reason}); previous state of run "
              f"'{prior}' is kept under {self.archived_to} and is not used as evidence.", flush=True)
        return prior

    # ---------- records ----------
    def write_config(self, si_config: dict, identity: dict, metadata_signature: dict) -> None:
        if not self.enabled:
            return
        _atomic_write_json(
            os.path.join(self.root, "config.json"),
            {"config_version": CONFIG_VERSION, "self_improving": si_config, "experiment_identity": identity, "metadata_bootstrap": metadata_signature},
        )

    def save_state(self, state: dict) -> None:
        if not self.enabled:
            return
        payload = dict(state)
        payload["run_id"] = self.run_id
        payload["lineage"] = self.lineage
        payload["config_version"] = CONFIG_VERSION
        _atomic_write_json(os.path.join(self.root, STATE_FILENAME), payload)

    def load_state(self) -> Optional[dict]:
        return _read_json(os.path.join(self.root, STATE_FILENAME)) if self.enabled else None

    def snapshot_path(self, step: int) -> str:
        return os.path.join(self.root, SNAPSHOT_DIR, f"step_{int(step):06d}.json")

    def save_snapshot(self, step: int, state: dict) -> Optional[str]:
        """policy checkpoint 保存成功后调用 (sidecar snapshot)."""
        if not self.enabled:
            return None
        payload = dict(state)
        payload["snapshot_step"] = int(step)
        payload["run_id"] = self.run_id
        payload["lineage"] = self.lineage
        path = self.snapshot_path(step)
        _atomic_write_json(path, payload)
        return path

    def find_snapshot(self, step: int) -> Optional[dict]:
        if not self.enabled:
            return None
        return _read_json(self.snapshot_path(step))

    def snapshot_steps(self) -> list:
        """已写出的 sidecar snapshot step 列表 (升序), 用于给出 resume 建议."""
        if not self.enabled or not os.path.isdir(os.path.join(self.root, SNAPSHOT_DIR)):
            return []
        steps = []
        for name in os.listdir(os.path.join(self.root, SNAPSHOT_DIR)):
            if name.startswith("step_") and name.endswith(".json"):
                try:
                    steps.append(int(name[len("step_") : -len(".json")]))
                except ValueError:
                    continue
        return sorted(steps)

    def latest_snapshot_step(self, max_step: Optional[int] = None) -> Optional[int]:
        if not self.enabled or not os.path.isdir(os.path.join(self.root, SNAPSHOT_DIR)):
            return None
        steps = []
        for name in os.listdir(os.path.join(self.root, SNAPSHOT_DIR)):
            if name.startswith("step_") and name.endswith(".json"):
                try:
                    value = int(name[len("step_") : -len(".json")])
                except ValueError:
                    continue
                if max_step is None or value <= max_step:
                    steps.append(value)
        return max(steps) if steps else None

    def append_history(self, record: dict) -> None:
        if not self.enabled:
            return
        path = os.path.join(self.root, HISTORY_FILENAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False, default=str) + "\n")

    def append_context(self, record: dict) -> None:
        if not self.enabled:
            return
        path = os.path.join(self.root, CONTEXTS_FILENAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False, default=str) + "\n")

    def read_history(self, limit: Optional[int] = None, max_step: Optional[int] = None, only_run_id: Optional[str] = None) -> list:
        """读 history.jsonl.

        max_step / only_run_id 用于 resume: 同一个目录里可能留着上一个 session 写的、相对被恢复
        checkpoint 而言属于"未来"的记录, 不能当成 controller 的当前证据.
        """
        if not self.enabled:
            return []
        path = os.path.join(self.root, HISTORY_FILENAME)
        if not os.path.exists(path):
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if only_run_id:
            records = [r for r in records if str(r.get("run_id")) == str(only_run_id)]
        if max_step is not None:
            records = [r for r in records if _step_of(r) is None or _step_of(r) <= int(max_step)]
        return records[-limit:] if limit else records

    # ---------- resume ----------
    def state_step(self, state: Optional[dict]) -> Optional[int]:
        return _step_of(state or {})

    def resume_state(self, checkpoint_step: Optional[int]) -> Optional[dict]:
        """按恢复的 policy checkpoint step 选 snapshot; 只使用 step <= checkpoint_step 的状态."""
        state = self.load_state()
        available = self.snapshot_steps()
        if state is None and not available:
            return None
        state = state or {}
        if checkpoint_step is None:
            return {"state": state, "source": STATE_FILENAME, "exact": True}
        snapshot = self.find_snapshot(checkpoint_step)
        if snapshot is not None:
            return {"state": snapshot, "source": f"snapshots/step_{checkpoint_step:06d}.json", "exact": True}
        # state.json 可能来自更晚的 step: 拿它恢复旧 checkpoint 会把未来的 skill / knowledge 带进过去.
        state_step = self.state_step(state)
        state_is_future = bool(state) and state_step is not None and state_step > int(checkpoint_step)
        message = (
            f"no self_improving snapshot for policy checkpoint step {checkpoint_step} "
            f"(available snapshot steps: {available or 'none'}). The policy checkpoint was likely written "
            "before the corresponding Slow Loop snapshot, so the active skill / window / knowledge version "
            "for that step cannot be restored exactly."
        )
        hint = ""
        if self.strict_resume:
            earlier = [s for s in available if s <= int(checkpoint_step)]
            if earlier:
                hint += f" An exactly aligned snapshot exists for step {max(earlier)}; resume that checkpoint instead. "
            if state_is_future:
                hint += (
                    f" {STATE_FILENAME} reports completed_global_step={state_step}, which is later than "
                    f"step {checkpoint_step} and therefore cannot be used for this resume. "
                )
            raise SelfImprovingStateError(message + hint + " Set self_improving.persistence.strict_resume=false to fall back.")
        nearest = self.latest_snapshot_step(max_step=checkpoint_step)
        if nearest is None and (state_is_future or not state):
            # 没有任何可安全使用的旧状态: 交给调用方从 initial template 开新 lineage (或报错).
            return {
                "state": None,
                "source": None,
                "exact": False,
                "start_fresh": True,
                "reason": message + (f" {STATE_FILENAME} holds completed_global_step={state_step}, a later step "
                                     "than the resumed checkpoint." if state_is_future else ""),
                "missing_step": checkpoint_step,
                "restored_step": None,
            }
        fallback = self.find_snapshot(nearest) if nearest is not None else state
        print(f"[self_improving] NON-STRICT RESUME: {message}\n"
              f"[self_improving] falling back to {'snapshot step ' + str(nearest) if nearest is not None else STATE_FILENAME}; "
              "reproducibility from this point on is not guaranteed.", flush=True)
        return {
            "state": fallback or {},
            "source": f"snapshots/step_{nearest:06d}.json" if nearest is not None else STATE_FILENAME,
            "exact": False,
            "reason": message,
            "missing_step": checkpoint_step,
            "restored_step": nearest if nearest is not None else (fallback or {}).get("completed_global_step"),
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _step_of(record: Optional[dict]) -> Optional[int]:
    for key in ("completed_global_step", "step"):
        value = (record or {}).get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return int(value)
    return None
