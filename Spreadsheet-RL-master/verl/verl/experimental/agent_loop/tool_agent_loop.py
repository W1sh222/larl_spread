# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import errno
import json
import logging
import os
import shutil
import stat
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_execution import execute_tool_calls, partition_tool_calls
from verl.experimental.agent_loop.tool_parser import TOOL_PARSER_ERROR_FUNCTION_NAME, FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.trainer.credit_assignment.interface import INITIAL_STATE_MARKER
from verl.utils.paths import get_spreadsheet_rl_data_root, normalize_thread_dir, normalize_workspace_id, resolve_evaluation_artifact
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.workbook_fingerprint import fingerprint_workbook_cached
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_WORKSPACE_MANIFEST_VERSION = 1

# 远程 sandbox 中修改 workbook、但本地 writeback 完成时序无代码级保证的 tool.
# 用于 transition graph validity 排除与 terminal state trust 判定 (两者语义分离).
_UNVERIFIED_WRITEBACK_TOOLS = frozenset({"code_interpreter"})


def _resolve_safe_data_file(root: Path, relpath: Path) -> Optional[Path]:
    if not relpath.parts or relpath.is_absolute() or any(part == ".." for part in relpath.parts):
        return None
    try:
        root_resolved = root.resolve(strict=True)
        candidate = root / relpath
        resolved = candidate.resolve(strict=True)

        # Reject leaf symlink targets.
        if candidate.is_symlink():
            return None

        # Allow only the first path segment to be a symlink (intentional top-level dataset links);
        # reject symlinks on deeper path components.
        first_is_symlink = False
        cursor = root
        for idx, part in enumerate(relpath.parts):
            cursor = cursor / part
            if cursor.is_symlink():
                if idx == 0:
                    first_is_symlink = True
                    continue
                return None

        if resolved.is_relative_to(root_resolved):
            return resolved if resolved.is_file() else None

        if first_is_symlink:
            first_root_resolved = (root / relpath.parts[0]).resolve(strict=True)
            if resolved.is_relative_to(first_root_resolved):
                return resolved if resolved.is_file() else None
    except (OSError, RuntimeError):
        return None
    return None


def _ensure_workspace_writable(workspace_dir: Path, workbook_path: Optional[Path] = None) -> None:
    try:
        os.chmod(workspace_dir, 0o777)
    except OSError:
        pass
    if workbook_path is None:
        return
    try:
        os.chmod(workbook_path, 0o666)
    except OSError:
        pass


def _open_lockfile(lock_path: Path):
    flags = os.O_RDWR | os.O_CREAT
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    else:
        try:
            if lock_path.is_symlink():
                raise RuntimeError(f"lock file path is a symlink: {lock_path}")
        except OSError:
            pass

    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise RuntimeError(f"lock file path is a symlink: {lock_path}") from None
        raise RuntimeError(f"failed to open lock file: {exc}") from None

    try:
        st = os.fstat(fd)
    except OSError as exc:
        os.close(fd)
        raise RuntimeError(f"failed to stat lock file: {exc}") from None
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise RuntimeError(f"lock file is not a regular file: {lock_path}") from None

    return os.fdopen(fd, "r+", encoding="utf-8", errors="replace")


def _atomic_copy_file(src: Path, dest: Path) -> None:
    tmp_dest = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}.{uuid4().hex}")
    try:
        try:
            if tmp_dest.is_symlink() or tmp_dest.is_file():
                tmp_dest.unlink()
            elif tmp_dest.is_dir():
                shutil.rmtree(tmp_dest)
        except OSError:
            pass

        shutil.copy2(src, tmp_dest)
        os.replace(tmp_dest, dest)
    finally:
        try:
            if tmp_dest.is_symlink() or tmp_dest.is_file():
                tmp_dest.unlink()
            elif tmp_dest.is_dir():
                shutil.rmtree(tmp_dest)
        except OSError:
            pass


def _normalize_primary_ext(extra_info: Any) -> str:
    return ".xlsx"


def _resolve_collect_step_states(config: Any) -> tuple[bool, bool]:
    """rollout 侧开关: structured state transition 采集 + eval fingerprint 标记 (generic: 不按具体 method 名分支).

    优先级:
      1. algorithm.adv_estimator != "credit_assignment" 或 credit_assignment 缺失 -> 不采集
         (确保未启用 CA 时, 其他 estimator 不会因残留配置触发 workbook fingerprint 采集).
      2. method 存在 -> 总先解析 required_metadata(params) 判断是否需 structured transitions
         (unknown method 名 -> get_credit_assignment_class 在此初始化阶段即 raise, 不吞).
      3. 显式 collect_step_states=False 但 method 需 structured -> 立即报配置冲突 (rollout 前
         fail-fast: 避免先花完 rollout 成本才在 adapter 处发现无 transitions).
      4. 显式 True -> 采用; null -> 依 method 真实需求自动决定.
    eval fingerprint 标记由 method 需求自动决定, 随样本 extra_fields 传给 reward 侧
    (无进程级环境变量 -> 配置即真相, 不存在开关与 method 不一致的失败态).
    GiGPO observation anchor (不需要 structured) 与 state anchor (需要) 区分保留.
    """
    try:
        algo = config.get("algorithm", None) if hasattr(config, "get") else None
    except Exception:
        return False, False
    if algo is None:
        return False, False
    if algo.get("adv_estimator", None) != "credit_assignment":
        return False, False
    ca = algo.get("credit_assignment", None)
    if not ca:
        return False, False
    explicit = ca.get("collect_step_states", None)
    method = ca.get("method", None)
    needs_structured = False
    needs_eval_fp = False
    if method:
        params = ca.get("params", None) or {}
        # 延迟 import 避免模块加载期依赖; 子包 import 会触发 methods 注册 (registry 就绪).
        from verl.trainer.credit_assignment.interface import needs_eval_fingerprint, needs_structured_state
        from verl.trainer.credit_assignment.registry import get_credit_assignment_class

        cls = get_credit_assignment_class(method)  # unknown method -> raise (不吞)
        required = frozenset(cls.required_metadata(dict(params)))
        needs_structured = needs_structured_state(required)
        needs_eval_fp = needs_eval_fingerprint(required)
    if explicit is not None:
        if not bool(explicit) and needs_structured:
            raise ValueError(
                "credit_assignment configuration conflict: method '%s' requires structured "
                "workbook-state transitions (ca_step_transitions), but "
                "algorithm.credit_assignment.collect_step_states is explicitly false. Set it to "
                "true (or null for auto) before rollout; otherwise the algorithm would silently "
                "degrade to no success anchors." % (method,)
            )
        return bool(explicit), needs_eval_fp
    # null -> auto: 依据 method 真实需求 (未选 method 时不采集; adapter 会按实际 needs 校验).
    return needs_structured, needs_eval_fp


def _workspace_primary_file(*, workspace_id: str, thread_dir: str, primary_ext: str) -> Path:
    """镜像 _prepare_sheet_arena_workspace 的 dest 规则, 解析主 workbook 路径."""
    requested_rel = f"{thread_dir}/output{primary_ext}"
    requested_rel_lower = requested_rel.lower()
    wants_flat_seed = requested_rel_lower.endswith("/output.xlsx") or requested_rel_lower == "output.xlsx"
    dest_rel = Path("data.xlsx") if wants_flat_seed else Path(requested_rel)
    return get_spreadsheet_rl_data_root() / "_workspaces" / workspace_id / dest_rel


def _prepare_sheet_arena_workspace(*, workspace_id: str, thread_dir: str, primary_ext: str) -> None:
    workspace_id = normalize_workspace_id(workspace_id)
    if workspace_id is None:
        return

    data_root = get_spreadsheet_rl_data_root()
    if not data_root.is_dir():
        return

    requested_rel = f"{thread_dir}/output{primary_ext}"
    requested_paths = [requested_rel]
    requested_rel_lower = requested_rel.lower()
    wants_flat_seed = requested_rel_lower.endswith("/output.xlsx") or requested_rel_lower == "output.xlsx"
    dest_rel = Path("data.xlsx") if wants_flat_seed else Path(requested_rel)

    workspace_root = data_root / "_workspaces"
    locks_dir = workspace_root / ".locks"
    workspace_dir = workspace_root / workspace_id

    for path in (locks_dir, workspace_dir):
        try:
            if path.is_symlink():
                raise RuntimeError(f"workspace preparation refused for symlink path: {path}")
        except OSError as exc:
            raise RuntimeError(f"workspace preparation failed checking path: {path}") from exc

    locks_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    lock_path = locks_dir / f"{workspace_id}.lock"
    manifest_path = locks_dir / f"{workspace_id}.manifest.json"

    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("workspace preparation requires fcntl (Unix-only)") from exc

    lock_f = _open_lockfile(lock_path)
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

        dest_path = workspace_dir / dest_rel

        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_paths = manifest.get("workspace_paths")
                manifest_version = manifest.get("version")
                if (
                    manifest_version == _WORKSPACE_MANIFEST_VERSION
                    and isinstance(manifest_paths, list)
                    and all(isinstance(p, str) for p in manifest_paths)
                ):
                    manifest_paths = sorted(manifest_paths)
                    if manifest_paths != sorted(requested_paths):
                        raise RuntimeError(
                            f"workspace_id {workspace_id!r} already prepared with different workspace_paths"
                        )
                    if dest_path.is_file() and not dest_path.is_symlink():
                        _ensure_workspace_writable(workspace_dir, dest_path)
                        return
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        if not dest_path.is_file() or dest_path.is_symlink():
            candidates = [requested_rel]
            try:
                base, ext = os.path.splitext(requested_rel)
                if ext:
                    candidates.append(base + ext.lower())
            except Exception:
                pass

            src_path = None
            for cand in candidates:
                src_path = _resolve_safe_data_file(data_root, Path(cand))
                if src_path is not None:
                    break
            if src_path is None:
                raise FileNotFoundError(f"missing seed workbook (tried: {candidates[:3]})")

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if dest_path.is_symlink() or dest_path.is_file():
                    dest_path.unlink()
                elif dest_path.is_dir():
                    shutil.rmtree(dest_path)
            except OSError:
                pass

            _atomic_copy_file(src_path, dest_path)
            _ensure_workspace_writable(workspace_dir, dest_path)
        else:
            _ensure_workspace_writable(workspace_dir, dest_path)

        tmp_manifest_path = Path(f"{manifest_path}.tmp.{os.getpid()}.{uuid4().hex}")
        try:
            with open(tmp_manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": _WORKSPACE_MANIFEST_VERSION,
                        "workspace_paths": requested_paths,
                    },
                    f,
                    sort_keys=True,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_manifest_path, manifest_path)
        finally:
            try:
                if tmp_manifest_path.exists():
                    tmp_manifest_path.unlink()
            except OSError:
                pass
    finally:
        try:
            lock_f.close()
        except OSError:
            pass


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        data_source: Optional[str] = None,
        ground_truth: Any = None,
        extra_info: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.data_source = data_source
        self.ground_truth = ground_truth
        self.extra_info = extra_info if isinstance(extra_info, dict) else {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Step-level metadata for credit assignment (算法中立): 每个 assistant decision = 1 step.
        # _step_spans[k] = (start, end) 在 response 坐标系中的 assistant token span.
        # _step_observations[k] = 该 step 决策前的环境观测 (pre-action state, 结构化 JSON);
        # step 0 记录显式 INITIAL_STATE_MARKER (与 "观测缺失/为空" 区分, 由下游 method 判 invalid).
        self._step_spans: list[tuple[int, int]] = []
        self._step_observations: list[Optional[str]] = []
        self._cur_obs_text: Optional[str] = None  # 下一次生成时用作 pre-action 观测

        # Structured workbook-state transitions (算法中立; 仅 collect_step_states 启用时填充).
        # _step_transitions[k] 与 _step_spans[k] 一一对应:
        #   {"pre", "post", "executed", "status", "graph_valid", "reason"[, "tool_status"]}
        # pre/post 为 "sha256:..." fingerprint 或 None (=unknown, 不得与其他 unknown 合并).
        self._step_transitions: list[dict[str, Any]] = []
        self._workbook_path: Optional[Path] = None
        self._primary_ext: str = ".xlsx"
        self._cur_state: Optional[str] = None
        self._initial_state: Optional[str] = None
        self._final_state: Optional[str] = None
        self._terminal_reason: Optional[str] = None
        # 本次 trajectory 是否曾 dispatch 过 unverified-writeback tool (e.g. code_interpreter).
        # 单调标志: 一旦置 True 不因后续 read/fingerprint/final-answer/无关 in-process tool 而清除
        # (成功读到 fingerprint 不能证明外部 writeback 已完成).
        self._unresolved_writeback: bool = False
        # 终端 graph state 与实际被评估 artifact 是否一致 (True/False/None=无法判定).
        self._terminal_artifact_consistent: Optional[bool] = None
        self._state_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._state_stats: dict[str, Any] = {
            "compute": 0,
            "cache_hit": 0,
            "unknown": 0,
            "parse_ms": 0.0,
        }

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_serial_calls = self.rollout_config.multi_turn.max_serial_calls
        if self.max_serial_calls is not None and self.max_serial_calls < 1:
            raise ValueError("max_serial_calls must be null or at least 1")
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        if TOOL_PARSER_ERROR_FUNCTION_NAME in self.tools:
            raise ValueError(f"Tool name '{TOOL_PARSER_ERROR_FUNCTION_NAME}' is reserved for parser error feedback.")
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # RewardFlow 等 structured-state method 的采集开关 (默认关, GiGPO/GRPO 零开销);
        # emit_eval_fingerprint 由 method 需求决定, 经 extra_fields 逐样本传给 reward 侧.
        self.collect_step_states, self.emit_eval_fingerprint = _resolve_collect_step_states(self.config)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        data_source = kwargs.get("data_source", None)
        reward_model = kwargs.get("reward_model", {}) if isinstance(kwargs.get("reward_model", {}), dict) else {}
        ground_truth = reward_model.get("ground_truth", None)
        extra_info_raw = kwargs.get("extra_info", {})
        extra_info = dict(extra_info_raw) if isinstance(extra_info_raw, dict) else {}

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            data_source=data_source,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        # Per-sample tool selection: filter global tools by extra_info.tool_selection
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        if data_source == "sheet_arena":
            thread_dir = normalize_thread_dir(ground_truth)
            primary_ext = _normalize_primary_ext(extra_info)
            extra_info["primary_ext"] = primary_ext
            extra_info["sandbox_workspace_id"] = request_id
            if thread_dir is not None:
                try:
                    await asyncio.to_thread(
                        _prepare_sheet_arena_workspace,
                        workspace_id=request_id,
                        thread_dir=thread_dir,
                        primary_ext=primary_ext,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to prepare sheet_arena workspace (workspace_id=%s, thread_dir=%s): %s",
                        request_id,
                        thread_dir,
                        e,
                    )
                if self.collect_step_states:
                    # 初始 state 必须在 rollout 中采集: workspace 会在 reward 评估后被删除.
                    agent_data._workbook_path = _workspace_primary_file(
                        workspace_id=request_id, thread_dir=thread_dir, primary_ext=primary_ext
                    )
                    agent_data._primary_ext = primary_ext
                    state, reason = await self._fingerprint_state(agent_data)
                    agent_data._cur_state = state
                    agent_data._initial_state = state
                    if state is None:
                        logger.warning("initial workbook fingerprint failed (%s): %s", request_id, reason)

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "sandbox_workspace_id": agent_data.request_id,
                # credit assignment step metadata (每 trajectory 一个变长 list, 经 non_tensor_batch 传递).
                # key 名算法中立: rollout 不得知晓具体 credit assignment 算法.
                "ca_step_spans": [[int(s), int(e)] for (s, e) in agent_data._step_spans],
                "ca_step_observations": list(agent_data._step_observations),
            }
        )
        if self.emit_eval_fingerprint:
            # reward 侧逐样本开关: 随 tool_extra_fields 合并进 reward 的 extra_info, sheet_arena
            # 据此计算并回传 canonical eval fingerprint. 未声明需求的 method 不写入 -> 零开销.
            output.extra_fields["emit_eval_fingerprint"] = True
        if self.collect_step_states:
            # terminal final state 只认本次 terminal fingerprint 采集结果; 采集失败 -> None (unknown).
            # 不回退到先前观测 state (_cur_state): 旧 state 可能是中间状态, 把 success anchor 错绑其上.
            # _cur_state 仍作为 last-known live state 保留 (GiGPO step-0 initial fallback / 诊断用),
            # 但绝不用于 RewardFlow terminal success anchor (unknown terminal -> 不锚定).
            final_state, _ = await self._fingerprint_state(agent_data)
            agent_data._final_state = final_state
            if agent_data._terminal_reason is None:
                agent_data._terminal_reason = "natural"
            # 终端 graph state 与实际被评估 artifact (output.xlsx 优先, 否则 data.xlsx) 一致性.
            agent_data._terminal_artifact_consistent = await self._resolve_terminal_artifact_consistency(agent_data)
            # Terminal state trust (与 transition graph validity 分离判定; rollout 侧必要非充分条件):
            #   known   = terminal fingerprint 采集成功 (非 None);
            #   trusted = known 且本 trajectory 无未解决 external writeback 且终端 live state 与
            #             交付评分 artifact 为同一文件/同指纹 (fail-closed: 不一致/无法判定 -> 不建
            #             anchor, trajectory-level advantage 仍保留).
            # 注意: 这里只是 rollout 结束时点的 path/capture 一致性, 不表示实际被评估 bytes 与
            # terminal fingerprint 内容一致 (同一路径 != 同一内容). 评估 bytes 的内容同一性由
            # RewardFlow 侧将 sheet_arena 回传的 eval_workbook_fingerprint 与 ca_final_state 严格
            # 比对单独保证 (TOCTOU 防护); 二者缺一均不锁定 success anchor.
            # in-process / recalc 写入同步且已验证; code_interpreter 远程 writeback 时序不可证.
            terminal_known = agent_data._final_state is not None
            terminal_trusted = (
                terminal_known
                and not agent_data._unresolved_writeback
                and agent_data._terminal_artifact_consistent is True
            )
            output.extra_fields.update(
                {
                    "ca_step_transitions": [dict(t) for t in agent_data._step_transitions],
                    "ca_initial_state": agent_data._initial_state,
                    "ca_final_state": agent_data._final_state,
                    "ca_terminal_trusted": terminal_trusted,
                    "ca_terminal_reason": agent_data._terminal_reason,
                    "ca_terminal_artifact_consistent": agent_data._terminal_artifact_consistent,
                    "ca_unresolved_writeback": agent_data._unresolved_writeback,
                    "ca_state_stats": dict(agent_data._state_stats),
                }
            )
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        # 记录本 step (assistant decision) 的 pre-action 观测与 span 起点.
        # 首次生成 (尚无任何 tool 观测) -> 显式 initial-state marker, 而非 None/空串.
        step_start = len(agent_data.response_mask)
        agent_data._step_observations.append(agent_data._cur_obs_text
            if agent_data._cur_obs_text is not None else INITIAL_STATE_MARKER)
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        # assistant token span [step_start, step_end); 截断后由下游 adapter clamp.
        agent_data._step_spans.append((step_start, len(agent_data.response_mask)))
        if self.collect_step_states:
            # 与 span 一一对应的 transition placeholder; tool 执行完成后在 PROCESSING_TOOLS 中回填.
            agent_data._step_transitions.append(
                {
                    "pre": agent_data._cur_state,
                    "post": None,
                    "executed": False,
                    "status": "pending",
                    "graph_valid": False,
                    "reason": None,
                }
            )
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            agent_data._terminal_reason = "response_length"
            self._close_transition_unexecuted(agent_data, reason="response_length")
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            agent_data._terminal_reason = "max_assistant_turns"
            self._close_transition_unexecuted(agent_data, reason="max_assistant_turns")
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            agent_data._terminal_reason = "max_user_turns"
            self._close_transition_unexecuted(agent_data, reason="max_user_turns")
            return AgentState.TERMINATED

        # Extract tool calls (use per-sample tools if routed)
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            agent_data._terminal_reason = "final_answer"
            # 无 tool call 的 final answer: 不构成 state-changing edge, 但 token span 仍有效.
            self._close_transition_final_answer(agent_data)
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tool_calls, serial_overflow_tool_calls, parallel_overflow_tool_calls = partition_tool_calls(
            agent_data.tool_calls,
            active_tools=active_tools,
            max_parallel_calls=self.max_parallel_calls,
            max_serial_calls=self.max_serial_calls,
        )
        tool_call_names = [tool_call.name for tool_call in agent_data.tool_calls]

        # 记录 dispatch 过的 unverified-writeback tool: 必须在实际执行之前置位, 使 tool execution
        # 抛异常时 uncertainty 仍被保留 (fail-safe, 不再依赖 _execute_tool_calls 正常返回).
        # 判据是实际 dispatch 集 tool_calls (parse 失败 / overflow 未 dispatch 的 call 不在其中),
        # 与 fingerprint 采集成败无关 (§4.2). 单调: 一旦置位不因后续 read/fingerprint/final-answer/
        # 无关 in-process tool 而清除 (§4.3).
        if self.collect_step_states and any(
            tc.name in _UNVERIFIED_WRITEBACK_TOOLS for tc in tool_calls
        ):
            agent_data._unresolved_writeback = True
        with simple_timer("tool_calls", agent_data.metrics):
            responses = await self._execute_tool_calls(tool_calls, agent_data.tools_kwargs, agent_data)
        if serial_overflow_tool_calls:
            message = (
                "Error: tool call not executed because this turn contains write-capable tools and "
                f"max_serial_calls={self.max_serial_calls}. Submit remaining independent calls in a later turn."
            )
            responses.extend(
                (
                    ToolResponse(text=message),
                    0.0,
                    {"status": "error", "error": "serial_call_limit_exceeded", "executed": False},
                )
                for _tool_call in serial_overflow_tool_calls
            )
        if parallel_overflow_tool_calls:
            message = (
                f"Error: tool call not executed because max_parallel_calls={self.max_parallel_calls}. "
                "Submit remaining calls in a later turn."
            )
            responses.extend(
                (
                    ToolResponse(text=message),
                    0.0,
                    {"status": "error", "error": "parallel_call_limit_exceeded", "executed": False},
                )
                for _tool_call in parallel_overflow_tool_calls
            )

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        # 逐条收集 (tool 名 + 响应文本), 保留执行顺序 (serial write 顺序影响后继状态, 不排序).
        obs_parts: list[dict[str, str]] = []
        obs_tool_names = (
            [tc.name for tc in tool_calls]
            + [tc.name for tc in serial_overflow_tool_calls]
            + [tc.name for tc in parallel_overflow_tool_calls]
        )
        call_statuses: list[Optional[str]] = []
        for call_idx, (tool_response, tool_reward, res) in enumerate(responses):
            call_statuses.append(str(res.get("status")) if isinstance(res, dict) and res.get("status") else None)
            obs_parts.append({
                "tool": obs_tool_names[call_idx] if call_idx < len(obs_tool_names) else "",
                "text": tool_response.text or "",
            })
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)
        # 序列化为结构化 JSON 观测 (含并行多 call): 下游 fingerprint 逐条规范化, 避免拼接破坏 JSON 解析.
        agent_data._cur_obs_text = json.dumps(obs_parts, ensure_ascii=False)

        # post-state fingerprint 必须在 observation 是否入 prompt 之前采集:
        # tool 已执行, 下方 response_length 截断路径不能丢失最终 state.
        if self.collect_step_states and agent_data._step_transitions:
            await self._update_last_transition_post_state(agent_data, obs_tool_names, call_statuses)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            agent_data._terminal_reason = "response_length_after_tools"
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    # ---- structured workbook-state collection (算法中立 metadata) ----

    def _fingerprint_sync(self, agent_data: AgentData, path: Optional[Path] = None) -> tuple[Optional[str], Optional[str]]:
        target = path if path is not None else agent_data._workbook_path
        if target is None:
            return None, "no_workspace_file"
        try:
            result = fingerprint_workbook_cached(str(target), agent_data._state_cache)
        except Exception as exc:
            return None, f"fingerprint_error:{type(exc).__name__}"
        stats = agent_data._state_stats
        if result.get("cache_hit"):
            stats["cache_hit"] += 1
        else:
            stats["compute"] += 1
            stats["parse_ms"] = round(float(stats["parse_ms"]) + float(result.get("parse_ms", 0.0)), 3)
        if result.get("status") == "ok":
            return result["fingerprint"], None
        stats["unknown"] += 1
        return None, str(result.get("reason", "unknown"))

    async def _fingerprint_state(self, agent_data: AgentData, path: Optional[Path] = None) -> tuple[Optional[str], Optional[str]]:
        # 避免长时间 block event loop.
        return await asyncio.to_thread(self._fingerprint_sync, agent_data, path)

    async def _resolve_terminal_artifact_consistency(self, agent_data: AgentData) -> Optional[bool]:
        """终端 graph state (live workbook fingerprint) 与交付评分 artifact 在 rollout 时点是否一致.

        - 无 live path 或 artifact 不存在 -> None (无法判定; caller 保守 fail-closed).
        - artifact 与 live workbook 同一文件 (常见: 无 output.xlsx, artifact=data.xlsx) -> True.
        - 否则 fingerprint artifact 与 final live state 比较: 相同 -> True, 不同 -> False.
        注意: same_file 仅代表路径/同一文件对象, 不等于被评估 snapshot 的内容指纹 (路径相同
        不等于内容相同). 这是 capture-time 必要非充分一致性; terminal 采集后、reward 读取
        bytes 之间若文件被外部修改的 TOCTOU 不由本检查保证, 而是由 RewardFlow 侧将 sheet_arena
        回传的评估 bytes canonical fingerprint 与 ca_final_state 严格比对单独保证 (见 README).
        """
        live = agent_data._workbook_path
        if live is None:
            return None
        try:
            artifact = resolve_evaluation_artifact(Path(live).parent, agent_data._primary_ext)
        except Exception:
            return None
        if artifact is None:
            return None
        try:
            same_file = Path(artifact).resolve() == Path(live).resolve()
        except Exception:
            same_file = Path(artifact) == Path(live)
        if same_file:
            return True
        artifact_fp, _ = await self._fingerprint_state(agent_data, artifact)
        if artifact_fp is None or agent_data._final_state is None:
            return None
        return artifact_fp == agent_data._final_state

    def _close_transition_unexecuted(self, agent_data: AgentData, *, reason: str) -> None:
        """assistant decision 产生但未执行 tool (提前终止): 不构成 graph edge."""
        if not (self.collect_step_states and agent_data._step_transitions):
            return
        tr = agent_data._step_transitions[-1]
        if tr.get("status") != "pending":
            return
        tr["post"] = tr.get("pre")
        tr["executed"] = False
        tr["status"] = "terminated_before_tools"
        tr["graph_valid"] = False
        tr["reason"] = reason

    def _close_transition_final_answer(self, agent_data: AgentData) -> None:
        if not (self.collect_step_states and agent_data._step_transitions):
            return
        tr = agent_data._step_transitions[-1]
        if tr.get("status") != "pending":
            return
        tr["post"] = tr.get("pre")
        tr["executed"] = True
        tr["status"] = "final_answer"
        tr["graph_valid"] = False
        tr["reason"] = None

    async def _update_last_transition_post_state(
        self,
        agent_data: AgentData,
        tool_names: list[str],
        call_statuses: list[Optional[str]],
    ) -> None:
        tr = agent_data._step_transitions[-1]
        post, reason = await self._fingerprint_state(agent_data)
        tr["executed"] = True
        tr["status"] = "executed"
        tr["post"] = post
        tr["tool_status"] = [
            {"tool": str(name), "status": status}
            for name, status in zip(tool_names, call_statuses)
        ]
        if post is None:
            tr["graph_valid"] = False
            tr["reason"] = reason or "state_unknown"
            return
        if tr.get("pre") is None:
            tr["graph_valid"] = False
            tr["reason"] = "pre_state_unknown"
            agent_data._cur_state = post
            return
        # sandbox 远程 writeback 完成时序无代码级保证: _unresolved_writeback 是 trajectory 单调
        # 标志 (首个 unverified-writeback dispatch 之前为 False, 之后恒 True). 一旦置位, 该点
        # 之后所有 transition 的 pre/post content 均可能被迟写污染 -> 无依据视为 graph-valid edge,
        # 全部保守排除出 graph (仅计 trajectory advantage; terminal anchor 另由 terminal_trusted 拦).
        # 关键: unsafe 之前已提交的 transition (处理时 flag 仍 False) 完整保留, 不是一刀切删除.
        if agent_data._unresolved_writeback:
            tr["graph_valid"] = False
            tr["reason"] = "external_writeback_unverified"
        else:
            tr["graph_valid"] = True
        agent_data._cur_state = post

    async def _execute_tool_calls(
        self,
        tool_calls: list[FunctionCall],
        tools_kwargs: dict[str, Any],
        agent_data: AgentData,
    ) -> list[tuple[ToolResponse, float | None, dict]]:
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        return await execute_tool_calls(
            tool_calls,
            active_tools=active_tools,
            call_tool=lambda tool_call: self._call_tool(tool_call, tools_kwargs, agent_data),
        )

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float | None, dict]:
        """Call tool and return tool response."""
        active_tools = getattr(agent_data, "_active_tools", self.tools)

        # Validate tool name
        tool_name = tool_call.name
        if tool_name == TOOL_PARSER_ERROR_FUNCTION_NAME and getattr(tool_call, "internal", False):
            try:
                tool_args = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError):
                tool_args = {}
            message = tool_args.get("message") if isinstance(tool_args, dict) else None
            tool_response_text = str(message or "Tool call parse error.")
            if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
                if self.tool_response_truncate_side == "left":
                    tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
                elif self.tool_response_truncate_side == "right":
                    tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
                else:
                    length = self.max_tool_response_length // 2
                    tool_response_text = (
                        tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]
                    )
            return ToolResponse(text=tool_response_text), None, {}

        if tool_name not in active_tools:
            available = list(active_tools.keys())
            msg = f"Unknown function '{tool_name}'. Available tools: {available}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0, {}

        # Validate tool arguments
        try:
            tool_args = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            msg = f"Invalid JSON in arguments for '{tool_name}': {e}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0, {}

        # Execute tool
        tool, instance_id = None, None
        try:
            tool = active_tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id,
                tool_args,
                agent_data=agent_data,
                workspace_id=agent_data.request_id,
                data_source=agent_data.data_source,
                ground_truth=agent_data.ground_truth,
                extra_info=agent_data.extra_info,
            )
        except Exception as e:
            logger.warning(f"Error executing tool '{tool_name}': {e}")
            return ToolResponse(text=f"Error executing tool '{tool_name}': {e}"), 0.0, {}
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res
