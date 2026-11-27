"""Markdown Skill Knowledge 的加载 / run-local 复制 / 版本化更新.

Repository 内的 skills/*.md 是只读模板. 每次 training run 复制到自己的输出目录后再更新,
不同 run 之间不共享. LLM 只提出"经验解释", 所有数值与 provenance 由 Python 的 evaluation
记录写入, 且 markdown 文本不会反过来扩大可执行参数空间.
"""

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
EXPERIENCE_HEADING = "## Empirical Experience"
GUIDELINE_HEADING = "## Calibration Guidelines"
NO_EXPERIENCE_TEXT = "_No verified experience recorded yet._"
MAX_FILE_CHARS = 200_000
MAX_FIELD_CHARS = 1200
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class KnowledgeError(ValueError):
    pass


def _safe_text(value: Any, limit: int = MAX_FIELD_CHARS) -> str:
    """LLM 文本按不可信输入处理: 去控制字符, 限长, 不允许注入 markdown 标题."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = text.replace("\n", " ").strip()
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def _atomic_write(path: str, content: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):  # pragma: no cover - replace 成功后不存在
            os.remove(tmp)


@dataclass
class KnowledgeVersion:
    version: int
    step: int
    decision_id: Optional[str]
    evaluation_id: Optional[str]
    target_skill: Optional[str]
    previous_version: Optional[int]
    created_at: str
    lineage: str = "initial"


@dataclass
class KnowledgeStore:
    """run-local knowledge 目录: current/*.md + versions/vNNNN/ + index.json."""

    run_dir: str
    candidate_skills: list
    max_prompt_chars: int = 20000
    template_dir: str = TEMPLATE_DIR
    lineage: str = "run"
    current_dir: str = ""
    versions_dir: str = ""
    index_path: str = ""
    version: int = 0
    history: list = field(default_factory=list)
    future_versions: list = field(default_factory=list)
    seen_experience_keys: set = field(default_factory=set)

    def __post_init__(self):
        root = os.path.join(self.run_dir, "knowledge")
        self.current_dir = os.path.join(root, "current")
        self.versions_dir = os.path.join(root, "versions")
        self.index_path = os.path.join(root, "index.json")
        os.makedirs(self.current_dir, exist_ok=True)
        os.makedirs(self.versions_dir, exist_ok=True)

    # ---------- bootstrap ----------
    def bootstrap(self, step: int) -> int:
        """复制模板到 current/; 已有内容时保持不动 (resume / 重启)."""
        created = []
        for skill in self.candidate_skills:
            target = self.path_of(skill)
            if os.path.exists(target):
                continue
            template = os.path.join(self.template_dir, f"{skill}.md")
            if not os.path.exists(template):
                raise KnowledgeError(
                    f"no initial skill knowledge template for '{skill}' at {template}; "
                    "a candidate skill without a knowledge file cannot be bootstrapped"
                )
            with open(template, encoding="utf-8") as f:
                text = f.read()
            _atomic_write(target, text)
            created.append(skill)
        if not os.path.exists(self.index_path) and created:
            self._record_version(
                KnowledgeVersion(
                    version=0,
                    step=int(step),
                    decision_id=None,
                    evaluation_id=None,
                    target_skill=None,
                    previous_version=None,
                    created_at=_utcnow(),
                    lineage=self.lineage,
                ),
                created,
            )
        return self.version

    def adopt(self, version: int, step: int, skills: Optional[list] = None) -> int:
        """Resume: 用指定 snapshot 的 knowledge 覆盖 current/ (避免读到未来版本)."""
        source = os.path.join(self.versions_dir, f"v{version:04d}")
        if not os.path.isdir(source):
            raise KnowledgeError(f"knowledge version {version} not found under {source}")
        recorded = next((v for v in self._load_index() if v.get("version") == int(version)), None)
        written_at = _num((recorded or {}).get("step"))
        if recorded is not None and written_at is not None and written_at > int(step):
            raise KnowledgeError(
                f"knowledge version {version} was written at step {int(written_at)}, which is later than the "
                f"resumed checkpoint step {int(step)}; restoring it would leak experience from the future"
            )
        for skill in skills or self.candidate_skills:
            src = os.path.join(source, f"{skill}.md")
            if os.path.exists(src):
                with open(src, encoding="utf-8") as f:
                    _atomic_write(self.path_of(skill), f.read())
        self.version = int(version)
        return self.version

    def reset_to_templates(self) -> list:
        """版本不可恢复时的兼容兵: 用只读模板重写 current/, 不把未来内容留在手里."""
        reset = []
        for skill in self.candidate_skills:
            template = os.path.join(self.template_dir, f"{skill}.md")
            if not os.path.exists(template):
                continue
            with open(template, encoding="utf-8") as f:
                _atomic_write(self.path_of(skill), f.read())
            reset.append(skill)
        self.version = 0
        return reset

    # ---------- access ----------
    def path_of(self, skill: str) -> str:
        if not _ID_RE.match(str(skill)):
            raise KnowledgeError(f"invalid skill name {skill!r}")
        return os.path.join(self.current_dir, f"{skill}.md")

    def read(self, skill: str) -> str:
        path = self.path_of(skill)
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8") as f:
            return f.read()

    def skills_with_knowledge(self) -> list:
        return [s for s in self.candidate_skills if os.path.exists(self.path_of(s))]

    def prompt_block(self, skills: Optional[list] = None) -> str:
        """给 LLM 的 knowledge 文本, 按预算裁剪 (不做 RAG, 只保留最近若干条经验)."""
        budget = int(self.max_prompt_chars)
        blocks = []
        for skill in skills or self.skills_with_knowledge():
            text = self.read(skill)
            if not text:
                continue
            text = _bounded_experience_text(text, per_skill_budget=max(1200, budget // max(1, len(self.candidate_skills))))
            if len(text) > budget:
                text = text[:budget] + "\n…[knowledge truncated]"
            blocks.append(f"### skill knowledge: {skill}\n\n{text}\n")
            budget -= len(text)
            if budget <= 0:
                blocks.append("### (remaining skill knowledge omitted due to prompt budget)")
                break
        return "\n".join(blocks)

    def experience_entries(self, skill: str) -> list:
        return re.findall(r"^### (exp-\d+)", self.read(skill), flags=re.MULTILINE)

    # ---------- update ----------
    def validate_proposal(self, proposal: dict, evaluation: dict, skill: str) -> tuple[bool, str, dict]:
        """Python 侧门禁: LLM 只能提出解释, 数值一律取 evaluation 记录."""
        if not isinstance(proposal, dict):
            return False, "proposal is not an object", {}
        if not proposal.get("update_required"):
            return False, "llm declined the update", {}
        target = str(proposal.get("target_skill") or skill)
        if target not in self.candidate_skills or not os.path.exists(self.path_of(target)):
            return False, f"target skill '{target}' has no knowledge file", {}
        if not evaluation or not evaluation.get("eligible_for_knowledge_update"):
            return False, "evaluation is not eligible for a knowledge update", {}
        applied = evaluation.get("applied_skill")
        if not applied:
            return False, "evaluation records no executed skill to attach the experience to", {}
        if target != applied:
            # 经验只能写进本 window 实际执行的 skill, LLM 不能把证据搬到另一个 knowledge 文件.
            return False, f"target skill '{target}' is not the skill evaluated in this window ('{applied}')", {}
        decision_id = str((proposal.get("experience") or {}).get("decision_id") or evaluation.get("decision_id") or "")
        if decision_id and decision_id != str(evaluation.get("decision_id")):
            return False, f"decision id {decision_id} does not match the evaluation record", {}

        experience = dict(proposal.get("experience") or {})
        applied_params = evaluation.get("applied_parameters") or {}
        claimed_config = experience.get("configuration")
        if claimed_config is not None and isinstance(claimed_config, dict):
            if set(claimed_config) - set(applied_params) or any(
                _num(claimed_config[k]) != _num(applied_params.get(k))
                for k in claimed_config
                if _num(claimed_config[k]) is not None
            ):
                return False, "proposed configuration differs from the applied one", {}
        key = (target, decision_id)
        if key in self.seen_experience_keys:
            return False, "an identical experience entry already exists", {}
        if decision_id and f"decision {decision_id}" in self.read(target):
            # 进程内 seen_experience_keys 在 resume 后是空的, 因此再对照文件本身.
            return False, "an identical experience entry already exists in the knowledge file", {}

        deltas = evaluation.get("metric_deltas") or {}
        record = {
            "decision_id": decision_id,
            "evaluation_id": evaluation.get("evaluation_id"),
            "outcome": evaluation.get("outcome"),
            "step": evaluation.get("evaluated_at_step"),
            "skill": applied,
            "target_skill": target,
            "configuration": dict(applied_params),
            "metric_deltas": {k: _num(v) for k, v in deltas.items()},
            "evidence_quality": evaluation.get("evidence_quality"),
            "context": _safe_text(experience.get("context")),
            "observation": _safe_text(experience.get("observation")),
            "interpretation": _safe_text(experience.get("interpretation")),
            "limitations": _safe_text(experience.get("limitations"))
            or "Observational evidence only; not a causal effect estimate.",
        }
        return True, "", record

    def apply_update(self, proposal: dict, evaluation: dict, step: int, skill: str) -> tuple[bool, str, Optional[int]]:
        ok, reason, record = self.validate_proposal(proposal, evaluation, skill)
        if not ok:
            return False, reason, None
        target = str(record["target_skill"])
        text = self.read(target)
        entry_id = f"exp-{len(self.experience_entries(target)) + 1:04d}"
        block = _render_experience(entry_id, record)
        text = _insert_under(text, EXPERIENCE_HEADING, block)
        guideline = proposal.get("guideline_proposal")
        if isinstance(guideline, dict) and guideline.get("text"):
            hypothesis = _render_guideline(entry_id, guideline, record)
            text = _insert_under(text, GUIDELINE_HEADING, hypothesis)
        if len(text) > MAX_FILE_CHARS:
            return False, f"knowledge file would exceed {MAX_FILE_CHARS} characters", None
        _atomic_write(self.path_of(target), text)
        self.seen_experience_keys.add((target, record["decision_id"]))
        version = self._next_version()
        self._record_version(
            KnowledgeVersion(
                version=version,
                step=int(step),
                decision_id=record["decision_id"],
                evaluation_id=record["evaluation_id"],
                target_skill=target,
                previous_version=self.version,
                created_at=_utcnow(),
                lineage=self.lineage,
            ),
            [target],
        )
        self.version = version
        return True, entry_id, version

    def _next_version(self) -> int:
        existing = []
        for name in os.listdir(self.versions_dir):
            match = re.fullmatch(r"v(\d{4})", name)
            if match:
                existing.append(int(match.group(1)))
        return max([self.version, *existing]) + 1

    def _record_version(self, meta: KnowledgeVersion, skills: list) -> None:
        target = os.path.join(self.versions_dir, f"v{meta.version:04d}")
        os.makedirs(target, exist_ok=True)
        # 每个 version 目录都是完整快照: 否则 adopt 旧版本时未变更的 skill 会继续读到未来内容.
        for skill in self.candidate_skills:
            src = self.path_of(skill)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(target, f"{skill}.md"))
        record = {
            "version": meta.version,
            "step": meta.step,
            "decision_id": meta.decision_id,
            "evaluation_id": meta.evaluation_id,
            "target_skill": meta.target_skill,
            "previous_version": meta.previous_version,
            "created_at": meta.created_at,
            "lineage": meta.lineage,
            "skills": sorted(skills),
        }
        payload = {"lineage": self.lineage, "versions": self._load_index()}
        payload["versions"] = [v for v in payload["versions"] if v.get("version") != meta.version]
        payload["versions"].append(record)
        payload["versions"].sort(key=lambda v: v["version"])
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.history.append(record)

    def _load_index(self) -> list:
        if not os.path.exists(self.index_path):
            return []
        try:
            with open(self.index_path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            return []
        versions = payload.get("versions") if isinstance(payload, dict) else None
        return list(versions or [])

    def load_index(self, not_after_step: Optional[int] = None) -> dict:
        """载入版本索引.

        not_after_step = 被恢复的 policy checkpoint step 时, 该 step 之后写下的版本只留作诊断,
        不参与 self.version 推导 (不自动采纳最新 index).
        """
        versions = self._load_index()
        allowed = versions
        if not_after_step is not None:
            ceiling = int(not_after_step)
            allowed = [v for v in versions if _num(v.get("step")) is None or _num(v.get("step")) <= ceiling]
        self.future_versions = [v.get("version") for v in versions if v not in allowed]
        self.history = allowed
        if allowed:
            self.version = max(v["version"] for v in allowed)
        return {"lineage": self.lineage, "versions": allowed, "future_versions": self.future_versions}


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return round(float(value), 9)
    except (TypeError, ValueError):
        return None


def _utcnow() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _render_experience(entry_id: str, record: dict) -> str:
    lines = [
        f"### {entry_id} · step {record.get('step')} · decision {record.get('decision_id')}",
        "",
        f"- outcome: {record.get('outcome')}",
        f"- configuration: `{json.dumps(record.get('configuration'), sort_keys=True)}`",
        f"- metric deltas: `{json.dumps(record.get('metric_deltas'), sort_keys=True)}`",
        f"- evidence quality: {record.get('evidence_quality')}",
        f"- context: {record.get('context') or '(none)'}",
        f"- observation: {record.get('observation') or '(none)'}",
        f"- interpretation: {record.get('interpretation') or '(none)'}",
        f"- limitations: {record.get('limitations')}",
    ]
    return "\n".join(lines) + "\n"


def _render_guideline(entry_id: str, guideline: dict, record: dict) -> str:
    scope = _safe_text(guideline.get("applies_when"), 400)
    text = _safe_text(guideline.get("text"), 600)
    return (
        f"- [hypothesis · from {entry_id} · unverified] "
        f"{text}" + (f" (applies when: {scope})" if scope else "")
    ) + "\n"


def _insert_under(text: str, heading: str, block: str) -> str:
    if heading not in text:
        return text.rstrip() + "\n\n" + heading + "\n\n" + block
    if NO_EXPERIENCE_TEXT in text and heading == EXPERIENCE_HEADING:
        text = text.replace(NO_EXPERIENCE_TEXT + "\n", "")
    index = text.index(heading) + len(heading)
    tail = text[index:]
    next_heading = tail.find("\n## ")
    if next_heading < 0:
        return text.rstrip() + "\n\n" + block
    insert_at = index + next_heading
    return text[:insert_at].rstrip() + "\n\n" + block + "\n" + text[insert_at:].lstrip("\n")


def _bounded_experience_text(text: str, per_skill_budget: int) -> str:
    """预算内的 knowledge 文本: 先留空间给最近的经验, 再截静态说明.

    模板本身有 5-7k 字, 若只截尾部则经验永远进不了 prompt, 等于没有知识演化.
    """
    if len(text) <= per_skill_budget:
        return text
    index = text.find(EXPERIENCE_HEADING)
    if index < 0:
        return text[:per_skill_budget] + "\n…[skill description truncated]\n"
    head = text[:index]
    chunks = re.split(r"(?=^### exp-)", text[index:], flags=re.MULTILINE)
    entries = chunks[1:]
    entry_budget = min(sum(len(c) for c in entries), max(400, int(per_skill_budget * 0.6)))
    kept: list[str] = []
    used = 0
    omitted = 0
    for chunk in reversed(entries):
        if used + len(chunk) > entry_budget:
            omitted = len(entries) - len(kept)
            break
        kept.append(chunk)
        used += len(chunk)
    body = chunks[0] + "".join(reversed(kept)) if not entries else "".join(reversed(kept))
    if omitted:
        body += f"\n…[{omitted} older experience entry(ies) omitted from this prompt]\n"
    head_budget = max(0, per_skill_budget - len(EXPERIENCE_HEADING) - len(body) - 4)
    if len(head) > head_budget:
        head = head[:head_budget] + "\n…[skill description truncated]\n\n"
    return head + EXPERIENCE_HEADING + "\n" + body


def validate_skill_markdown(text: str) -> list:
    """检查 knowledge 文件是否含约定的章节标题 (缺失只警告, 不阻断)."""
    required = [
        "## Mechanism",
        "## Required Inputs",
        "## Required Metadata",
        "## Configurable Parameters",
        "## Expected Strengths",
        "## When to Use",
        "## Limitations",
        "## Calibration Guidelines",
        "## Empirical Experience",
    ]
    return [heading for heading in required if heading not in text]
