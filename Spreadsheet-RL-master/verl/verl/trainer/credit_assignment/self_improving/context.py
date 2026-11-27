"""Training Context 采集.

只读取 trainer 已经产出的 batch / metrics, 每 step 压成 CPU 侧轻量摘要; 不跨 step 持有
GPU tensor、完整 DataProto 或整本 workbook 内容. 缺失的量记为 None (不伪装成 0).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from verl.trainer.credit_assignment.self_improving.config import ContextLimits

# 直接透传给 controller 的既有 trainer 指标 (只取真实存在的键).
TRAINING_METRIC_KEYS = (
    "actor/pg_loss",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "actor/grad_norm",
    "actor/entropy",
    "critic/score/mean",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "response_length/mean",
    "response_length/clip_ratio",
    "num_turns/mean",
    "training/global_step",
)
CREDIT_METRIC_PREFIX = "credit_assignment/"
VALID_METRIC_PREFIXES = ("val-core/", "val-aux/")


def jsonable(value: Any) -> Any:
    """递归转成可 JSON 序列化的值; NaN/Inf -> None."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        items = sorted(value) if isinstance(value, set) else value
        return [jsonable(v) for v in items]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if hasattr(value, "detach"):  # torch.Tensor
        try:
            return jsonable(value.detach().cpu().numpy())
        except Exception:  # pragma: no cover - 非数值 tensor
            return None
    return str(value)


def _finite(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _stats(values: np.ndarray) -> dict:
    # 非有限值直接剔除 (不折算成 0), 全部无效时各项为 None.
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": _finite(values.mean()),
        "std": _finite(values.std()),
        "min": _finite(values.min()),
        "max": _finite(values.max()),
    }


def _distribution(values: list) -> dict:
    counts: dict[str, int] = {}
    for v in values:
        key = "unknown" if v is None else str(v)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"p50": None, "p90": None}
    return {"p50": _finite(np.percentile(values, 50)), "p90": _finite(np.percentile(values, 90))}


def _tensor_of(batch: Any, name: str):
    holder = getattr(batch, "batch", None)
    if holder is None:
        return None
    try:
        value = holder.get(name, None)
    except (KeyError, TypeError, RuntimeError):
        return None
    return value


def _row_list(non_tensor: dict, name: str, bs: int) -> Optional[list]:
    arr = non_tensor.get(name)
    if arr is None or len(arr) != bs:
        return None
    return [arr[i] for i in range(bs)]


def _as_float_array(rows: Optional[list]) -> Optional[np.ndarray]:
    if rows is None:
        return None
    out = []
    for r in rows:
        v = _finite(r)
        out.append(np.nan if v is None else v)
    arr = np.asarray(out, dtype=np.float64)
    return arr if np.isfinite(arr).any() else None


def _clip_text(value: Any, limit: int) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\r\n", "\n")
    if limit > 0 and len(text) > limit:
        return text[:limit] + "…[truncated]"
    return text


@dataclass
class TrainingContext:
    """Controller 的输入快照 (JSON 可序列化)."""

    global_step: int
    window_start_step: int
    window_end_step: int
    active_skill: dict
    active_parameters: dict
    task_summary: dict = field(default_factory=dict)
    trajectory_statistics: dict = field(default_factory=dict)
    trajectory_samples: list = field(default_factory=list)
    training_metrics: dict = field(default_factory=dict)
    credit_diagnostics: dict = field(default_factory=dict)
    metadata_health: dict = field(default_factory=dict)
    window_skills: list = field(default_factory=list)
    latest_validation: Optional[dict] = None
    previous_evaluation: Optional[dict] = None
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return jsonable(self.__dict__)


class ContextCollector:
    """把每个 training step 压成轻量摘要, 并按 window 聚合."""

    def __init__(self, limits: ContextLimits, tokenizer: Any = None):
        self.limits = limits
        self.tokenizer = tokenizer
        self.steps: list[dict] = []
        self.samples: list[dict] = []
        self.window_start: Optional[int] = None

    # ---------- lifecycle ----------
    def reset(self, window_start_step: int) -> None:
        self.window_start = window_start_step
        self.steps = []
        self.samples = []

    def observed_steps(self) -> int:
        return len(self.steps)

    def window_skill_ids(self) -> list:
        seen = []
        for s in self.steps:
            sid = s.get("skill")
            if sid and sid not in seen:
                seen.append(sid)
        return seen

    # ---------- per-step ----------
    def observe(self, batch: Any, metrics: dict, global_step: int, skill: str, params: dict) -> dict:
        mask = _tensor_of(batch, "response_mask")
        if mask is None:
            mask = _tensor_of(batch, "loss_mask")
        non_tensor = dict(getattr(batch, "non_tensor_batch", {}) or {})
        bs = int(mask.shape[0]) if mask is not None and len(mask.shape) >= 1 else 0

        shaped = _row_sums(batch, "token_level_rewards")
        raw = _row_sums(batch, "token_level_scores")
        advantages = _tensor_of(batch, "advantages")

        summary = {
            "step": int(global_step),
            "skill": str(skill),
            "params": dict(params or {}),
            "num_trajectories": bs,
            "num_groups": None,
            "raw_reward": _stats(raw) if raw is not None and raw.size else _stats(np.zeros(0)),
            "train_reward": _stats(shaped) if shaped is not None and shaped.size else _stats(np.zeros(0)),
            "raw_reward_hist": _reward_histogram(raw),
            "success_rate": None,
            "within_group_reward_std": None,
            "trajectory_length": {},
            "steps_per_trajectory": {},
            "terminal_reasons": {},
            "trusted_terminal_ratio": None,
            "advantage": {},
            "training_metrics": _pick_metrics(metrics),
            "credit_diagnostics": _pick_prefixed(metrics, CREDIT_METRIC_PREFIX),
            "validation": _pick_prefixed(metrics, VALID_METRIC_PREFIXES),
            "metadata_health": {},
        }
        if bs == 0:
            summary["raw_reward"] = _stats(np.zeros(0))
            summary["train_reward"] = _stats(np.zeros(0))
            self.steps.append(summary)
            return summary

        uids = _row_list(non_tensor, "uid", bs)
        if uids is not None:
            summary["num_groups"] = float(len({str(u) for u in uids}))
            summary["within_group_reward_std"] = _within_group_std(raw, uids)

        acc = _as_float_array(_row_list(non_tensor, "acc", bs))
        if acc is not None:
            summary["success_rate"] = _finite(np.nanmean(acc))

        if mask is not None:
            lengths = (mask != 0).sum(dim=-1).detach().cpu().numpy().astype(np.float64)
            summary["trajectory_length"] = _stats(lengths) | _percentiles(lengths)
            summary["advantage"] = _masked_advantage_stats(advantages, mask)

        spans = _row_list(non_tensor, "ca_step_spans", bs)
        if spans is not None:
            counts = np.asarray([len(s) if s else 0 for s in spans], dtype=np.float64)
            summary["steps_per_trajectory"] = _stats(counts) | _percentiles(counts)

        reasons = _row_list(non_tensor, "ca_terminal_reason", bs)
        if reasons is not None:
            summary["terminal_reasons"] = _distribution(reasons)

        trusted = _row_list(non_tensor, "ca_terminal_trusted", bs)
        if trusted is not None:
            flags = [bool(t) for t in trusted if t is not None]
            summary["trusted_terminal_ratio"] = _finite(sum(flags) / len(flags)) if flags else None

        summary["metadata_health"] = self._metadata_health(non_tensor, bs, spans, reasons, trusted)
        self.samples.extend(
            self._extract_samples(
                batch, non_tensor, bs, raw, shaped, advantages, spans, uids, reasons, trusted, global_step
            )
        )
        # 样本是 window 级预算, 不是 step 级 -> 每 step 后裁剪一次.
        self.samples = self._trim_samples(self.samples)
        summary["sample_rows"] = len(self.samples)
        self.steps.append(summary)
        return summary

    def _metadata_health(self, non_tensor: dict, bs: int, spans, reasons, trusted) -> dict:
        def availability(key: str) -> Optional[float]:
            rows = _row_list(non_tensor, key, bs)
            if rows is None:
                return None
            present = sum(1 for r in rows if r not in (None, "", [], {}))
            return _finite(present / bs)

        health = {
            "step_spans": 1.0 if spans is not None else availability("ca_step_spans"),
            "step_observations": availability("ca_step_observations"),
            "step_transitions": availability("ca_step_transitions"),
            "terminal_trusted": 1.0 if trusted is not None else availability("ca_terminal_trusted"),
            "terminal_reason": 1.0 if reasons is not None else availability("ca_terminal_reason"),
            "eval_workbook_fingerprint": availability("eval_workbook_fingerprint"),
            "structured_state_requested": availability("ca_step_transitions") is not None,
        }
        if trusted is not None:
            health["unknown_state_ratio"] = _finite(sum(1 for t in trusted if t is None) / bs)
        stats_rows = _row_list(non_tensor, "ca_state_stats", bs)
        if stats_rows is not None:
            parse_ms = [_finite((r or {}).get("parse_ms")) for r in stats_rows if isinstance(r, dict)]
            parse_ms = [v for v in parse_ms if v is not None]
            unknown = [
                _finite((r or {}).get("unknown")) for r in stats_rows if isinstance(r, dict)
            ]
            unknown = [v for v in unknown if v is not None]
            if parse_ms:
                health["state_parse_ms_mean"] = _finite(float(np.mean(parse_ms)))
            if unknown:
                health["state_unknown_mean"] = _finite(float(np.mean(unknown)))
        return health

    def _credit_value(self, suffix: str) -> Optional[float]:
        """从已观测 step 的 `credit_assignment/*` 诊断里取最后一个匹配 suffix 的指标."""
        for step in reversed(self.steps):
            for key, value in (step.get("credit_diagnostics") or {}).items():
                if key.endswith(suffix):
                    return _finite(value)
        return None

    # ---------- representative samples ----------
    def _extract_samples(self, batch, non_tensor, bs, raw, shaped, advantages, spans, uids, reasons, trusted, global_step) -> list:
        picks = _select_representative_rows(bs, raw, reasons, global_step)
        if not picks:
            return []
        responses = _tensor_of(batch, "responses")
        obs_rows = _row_list(non_tensor, "ca_step_observations", bs)
        fp_rows = _row_list(non_tensor, "eval_workbook_fingerprint", bs)
        acc_rows = _row_list(non_tensor, "acc", bs)
        limit = self.limits.max_text_chars_per_field
        out = []
        for kind, row, step in picks:
            step_count = len(spans[row]) if spans and spans[row] else 0
            sample = {
                "kind": kind,
                "step": int(step),
                "row": int(row),
                "group_id": str(uids[row]) if uids else None,
                "raw_reward": _finite(raw[row]) if raw is not None else None,
                "train_reward": _finite(shaped[row]) if shaped is not None else None,
                "acc": _finite(acc_rows[row]) if acc_rows else None,
                "terminal_reason": str(reasons[row]) if reasons and reasons[row] is not None else None,
                "terminal_trusted": bool(trusted[row]) if trusted and trusted[row] is not None else None,
                "step_count": int(step_count),
                "trajectory_length": None,
                "has_eval_fingerprint": bool(fp_rows[row]) if fp_rows and fp_rows[row] is not None else False,
                "task_instruction": _clip_text(_task_instruction(batch, non_tensor, row, self.tokenizer), limit),
                "steps": [],
            }
            if responses is not None and spans is not None and step_count:
                sample["steps"] = self._render_steps(responses[row], spans[row], obs_rows[row] if obs_rows else None, advantages, row, limit)
                mask_row = responses[row]
                sample["trajectory_length"] = int((mask_row != 0).sum().item())
            out.append(sample)
        return out

    def _render_steps(self, response_row, span_list, obs_row, advantages, row, limit) -> list:
        steps = []
        max_steps = self.limits.max_steps_per_trajectory
        for k, span in enumerate(list(span_list or [])[:max_steps]):
            try:
                start, end = int(span[0]), int(span[1])
            except (TypeError, ValueError, IndexError):
                continue
            start, end = max(0, start), min(int(response_row.shape[0]), end)
            if end <= start:
                continue
            entry = {"index": k}
            if self.tokenizer is not None:
                try:
                    entry["action"] = _clip_text(
                        self.tokenizer.decode(response_row[start:end].tolist(), skip_special_tokens=True), limit
                    )
                except Exception:  # pragma: no cover - 取决于 tokenizer 行为
                    entry["action"] = None
            obs = obs_row[k] if obs_row is not None and k < len(obs_row) else None
            entry["observation"] = _clip_text(obs, limit)
            if advantages is not None:
                slice_adv = advantages[row, start:end].detach().float().cpu().numpy()
                entry["adv_mean"] = _finite(float(slice_adv.mean())) if slice_adv.size else None
            steps.append(entry)
        if span_list is not None and len(span_list) > max_steps:
            steps.append({"steps_omitted": int(len(span_list) - max_steps)})
        return steps

    def _trim_samples(self, samples: list) -> list:
        if len(samples) <= self.limits.max_trajectory_samples:
            return samples
        order = {"success": 0, "failed": 1, "truncated": 2, "representative": 3}
        ranked = sorted(enumerate(samples), key=lambda pair: (order.get(pair[1].get("kind"), 9), -pair[0]))
        keep = {idx for idx, _ in ranked[: self.limits.max_trajectory_samples]}
        return [s for i, s in enumerate(samples) if i in keep]

    # ---------- window aggregation ----------
    def aggregate(self) -> dict:
        if not self.steps:
            return {"num_steps": 0}
        steps = self.steps

        def series(key: str) -> list:
            return [s[key] for s in steps if s.get(key) is not None]

        def mean_of(key: str, sub: Optional[str] = None):
            vals = []
            for s in steps:
                node = s.get(key)
                value = node.get(sub) if isinstance(node, dict) and sub else node
                v = _finite(value)
                if v is not None:
                    vals.append(v)
            return _finite(float(np.mean(vals))) if vals else None

        reward_stats = _merge_stats(series("raw_reward"))
        terminal: dict[str, int] = {}
        for s in steps:
            for reason, count in (s.get("terminal_reasons") or {}).items():
                terminal[reason] = terminal.get(reason, 0) + int(count)

        training_metrics: dict[str, list] = {}
        credit: dict[str, list] = {}
        for s in steps:
            for key, value in (s.get("training_metrics") or {}).items():
                training_metrics.setdefault(key, []).append(value)
            for key, value in (s.get("credit_diagnostics") or {}).items():
                credit.setdefault(key, []).append(value)

        return {
            "num_steps": len(steps),
            "step_range": [steps[0]["step"], steps[-1]["step"]],
            "skills": self.window_skill_ids(),
            "num_trajectories": float(sum(s.get("num_trajectories", 0) for s in steps)),
            "raw_reward": reward_stats,
            "train_reward": _merge_stats(series("train_reward")),
            "raw_reward_hist": _merge_hist(series("raw_reward_hist")),
            "success_rate": mean_of("success_rate"),
            "within_group_reward_std": mean_of("within_group_reward_std"),
            "trajectory_length": _merge_stats(series("trajectory_length")),
            "steps_per_trajectory": _merge_stats(series("steps_per_trajectory")),
            "terminal_reasons": terminal,
            "trusted_terminal_ratio": mean_of("trusted_terminal_ratio"),
            "advantage": _merge_stats(series("advantage")),
            "num_groups": mean_of("num_groups"),
            "training_metrics": {k: _finite(float(np.mean([v for v in vs if _finite(v) is not None]))) for k, vs in training_metrics.items() if any(_finite(v) is not None for v in vs)},
            "credit_diagnostics": {k: _finite(float(np.mean([v for v in vs if _finite(v) is not None]))) for k, vs in credit.items() if any(_finite(v) is not None for v in vs)},
            "metadata_health": _merge_metadata_health(series("metadata_health")),
        }

    # ---------- context build ----------
    def build_context(
        self,
        global_step: int,
        skill: str,
        params: dict,
        latest_validation: Optional[dict] = None,
        previous_evaluation: Optional[dict] = None,
        extra_notes: Optional[dict] = None,
    ) -> TrainingContext:
        agg = self.aggregate()
        start = self.window_start if self.window_start is not None else global_step
        steps = self.steps
        task_summary = {
            "prompt_example": _first_task_example(self.samples),
            "batch_size_per_step": _finite(float(np.mean([s.get("num_trajectories", 0) for s in steps]))) if steps else None,
        }
        context = TrainingContext(
            global_step=int(global_step),
            window_start_step=int(start),
            window_end_step=int(global_step),
            active_skill={"method": str(skill), "params": dict(params or {})},
            active_parameters=dict(params or {}),
            task_summary=task_summary,
            trajectory_statistics={
                k: v
                for k, v in agg.items()
                if k not in ("training_metrics", "credit_diagnostics", "metadata_health", "skills")
            },
            trajectory_samples=self.samples,
            training_metrics=agg.get("training_metrics", {}),
            credit_diagnostics={
                "window": agg.get("credit_diagnostics", {}),
                "latest_step": (steps[-1].get("credit_diagnostics") if steps else {}),
            },
            metadata_health=agg.get("metadata_health", {}),
            window_skills=agg.get("skills", []),
            latest_validation=latest_validation,
            previous_evaluation=previous_evaluation,
            notes=dict(extra_notes or {}),
        )
        payload = context.to_dict()
        _enforce_budget(payload, self.limits.max_context_chars)
        context.__dict__.update(payload)
        return context


def _first_task_example(samples: list) -> Optional[str]:
    for s in samples or []:
        text = s.get("task_instruction") if isinstance(s, dict) else None
        if text:
            return text[:400]
    return None


def _row_sums(batch, name: str) -> Optional[np.ndarray]:
    tensor = _tensor_of(batch, name)
    if tensor is None:
        return None
    try:
        return tensor.sum(dim=-1).detach().cpu().numpy().astype(np.float64).reshape(-1)
    except Exception:  # pragma: no cover - 非数值 tensor
        return None


def _masked_advantage_stats(advantages, mask) -> dict:
    if advantages is None or mask is None:
        return {"mean": None, "std": None, "zero_ratio": None, "abs_mean": None}
    valid = (mask != 0)
    flat = advantages.detach().float()[valid].cpu().numpy()
    if flat.size == 0:
        return {"mean": None, "std": None, "zero_ratio": None, "abs_mean": None}
    return {
        "mean": _finite(float(flat.mean())),
        "std": _finite(float(flat.std())),
        "abs_mean": _finite(float(np.abs(flat).mean())),
        "zero_ratio": _finite(float((np.abs(flat) < 1e-8).mean())),
    }


def _within_group_std(scores: Optional[np.ndarray], uids: list) -> Optional[float]:
    if scores is None:
        return None
    groups: dict[str, list] = {}
    for i, uid in enumerate(uids):
        value = _finite(scores[i])
        if value is None:
            continue
        groups.setdefault(str(uid), []).append(value)
    stds = [float(np.std(v)) for v in groups.values() if len(v) > 1]
    return _finite(float(np.mean(stds))) if stds else None


def _reward_histogram(scores: Optional[np.ndarray]) -> dict:
    if scores is None or scores.size == 0:
        return {}
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return {}
    edges = [-math.inf, -1e-9, 0.25, 0.5, 0.75, 1.0 - 1e-9, math.inf]
    labels = ["negative", "zero", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1)", "above_1"]
    counts = np.histogram(finite, bins=edges)[0]
    total = float(finite.size)
    return {lab: _finite(float(c) / total) for lab, c in zip(labels, counts) if c}


def _merge_stats(items: list) -> dict:
    """window 内对 per-step 统计取均值 (MVP: 不按 trajectory 数加权)."""
    out: dict[str, list] = {}
    for node in items:
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            v = _finite(value)
            if v is not None:
                out.setdefault(key, []).append(v)
    merged = {key: (_finite(float(np.mean(vals))) if vals else None) for key, vals in out.items()}
    merged["n_steps"] = len(items)
    return merged


def _merge_hist(items: list) -> dict:
    total: dict[str, list] = {}
    for node in items:
        for key, value in (node or {}).items():
            v = _finite(value)
            if v is not None:
                total.setdefault(key, []).append(v)
    return {k: _finite(float(np.mean(v))) for k, v in total.items()}


def _merge_metadata_health(items: list) -> dict:
    merged: dict[str, list] = {}
    for node in items:
        for key, value in (node or {}).items():
            v = _finite(value)
            if v is not None:
                merged.setdefault(key, []).append(v)
    out = {k: _finite(float(np.mean(v))) for k, v in merged.items()}
    flags = [node.get("eval_workbook_fingerprint") for node in items if isinstance(node, dict)]
    out["eval_fingerprint_any"] = any(f is not None and f > 0 for f in flags)
    out["structured_state_any"] = any(
        node.get("step_transitions") is not None and node.get("step_transitions") > 0 for node in items if isinstance(node, dict)
    )
    return out


def _pick_metrics(metrics: dict) -> dict:
    out = {}
    for key in TRAINING_METRIC_KEYS:
        if key in metrics:
            v = _finite(metrics[key])
            if v is not None:
                out[key] = v
    return out


def _pick_prefixed(metrics: dict, prefixes) -> dict:
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    out = {}
    for key, value in (metrics or {}).items():
        if key.startswith(prefixes):
            v = _finite(value)
            if v is not None:
                out[key] = v
    return out


def _select_representative_rows(bs: int, raw: Optional[np.ndarray], reasons, step: int) -> list:
    """确定性挑选: success / failed / truncated / representative."""
    if bs <= 0:
        return []
    scores = raw if raw is not None and raw.size == bs else np.zeros(bs)
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0)
    order = np.argsort(-scores, kind="stable")
    picks: list[tuple[str, int, int]] = []
    used: set[int] = set()

    def add(kind: str, row: Optional[int]):
        if row is not None and row not in used and len(picks) < 4:
            picks.append((kind, int(row), int(step)))
            used.add(int(row))

    add("success", int(order[0]) if scores[int(order[0])] > 0 else None)
    add("failed", int(order[-1]) if scores[int(order[-1])] <= 0 else None)
    truncated = None
    if reasons is not None:
        for i, r in enumerate(reasons):
            if r is not None and str(r) not in ("natural", "final_answer", "None", ""):
                truncated = i
                break
    add("truncated", truncated)
    add("representative", int(order[len(order) // 2]))
    if not picks:
        picks = [("representative", int(order[0]))]
    return picks


def _task_instruction(batch, non_tensor: dict, row: int, tokenizer) -> Optional[str]:
    raw_prompt = non_tensor.get("raw_prompt")
    if raw_prompt is not None and len(raw_prompt) > row:
        value = raw_prompt[row]
        if isinstance(value, (list, tuple)):
            texts = [m.get("content") for m in value if isinstance(m, dict) and m.get("content")]
            return " | ".join(str(t) for t in texts) if texts else None
        return None if value is None else str(value)
    prompts = _tensor_of(batch, "prompts")
    if prompts is not None and tokenizer is not None:
        try:
            return tokenizer.decode(prompts[row].tolist(), skip_special_tokens=True)
        except Exception:  # pragma: no cover
            return None
    return None


def _enforce_budget(payload: dict, max_chars: int) -> None:
    """确定性裁剪: 先砍样本步数, 再砍文本长度, 保证 prompt 不超预算."""
    import json

    def size() -> int:
        return len(json.dumps(payload, ensure_ascii=False, default=str))

    if size() <= max_chars:
        return
    samples = payload.get("trajectory_samples") or []
    for shrink in (0.5, 0.25):
        for sample in samples:
            for step in sample.get("steps") or []:
                for key in ("action", "observation"):
                    text = step.get(key)
                    if isinstance(text, str) and len(text) > 200:
                        step[key] = text[: int(200 * shrink)] + "…"
        if size() <= max_chars:
            return
    while samples and size() > max_chars:
        samples.pop()
    if size() > max_chars:
        for key in ("credit_diagnostics", "training_metrics"):
            node = payload.get(key)
            if isinstance(node, dict) and size() > max_chars:
                payload[key] = {k: v for k, v in list(node.items())[:20]}
