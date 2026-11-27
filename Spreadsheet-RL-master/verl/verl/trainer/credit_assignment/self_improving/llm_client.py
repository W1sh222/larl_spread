"""外部 controller LLM 客户端.

只用标准库 (urllib), 不引入新依赖. 面向 OpenAI-compatible Chat Completions:
SPREADSHEET_RL_SI_LLM_URL 指完整 chat completions endpoint (若只填到 /v1 会自动补全).
所有响应都在 Python 侧重新解析与校验; API key 只出现在请求头, 不进 log / prompt / state.
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from verl.trainer.credit_assignment.self_improving.config import LLMConfig


class LLMError(RuntimeError):
    """LLM 调用失败. 调用方应保持当前 Skill 并记录原因 (execution safety fallback)."""


class LLMTimeout(LLMError):
    pass


class LLMNetworkError(LLMError):
    pass


class LLMHTTPError(LLMError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"LLM endpoint returned HTTP {status}: {detail[:400]}")
        self.status = status


class LLMInvalidResponse(LLMError):
    pass


def check_schema(payload: Any, schema: Optional[dict]) -> dict:
    """最小结构化校验: 必填键存在 + 类型匹配. schema 形如 {"action": str} 或 {"action": (str, True)}."""
    if not isinstance(payload, dict):
        raise LLMInvalidResponse(f"expected a JSON object, got {type(payload).__name__}")
    for key, spec in (schema or {}).items():
        expected, required = (spec, True) if not isinstance(spec, tuple) else spec
        if key not in payload:
            if required:
                raise LLMInvalidResponse(f"missing required field '{key}'")
            continue
        value = payload[key]
        if value is None:
            # 可选字段返回 null 等同于没返回; 必填字段仍然报错.
            if required:
                raise LLMInvalidResponse(f"required field '{key}' is null")
            continue
        if expected is not None and not isinstance(value, expected):
            raise LLMInvalidResponse(f"field '{key}' has type {type(value).__name__}, expected {expected}")
    return payload


def parse_json_object(text: str) -> dict:
    """容忍 ```json 围栏与前后说明文字: 取第一个可解析的 JSON 对象."""
    if text is None:
        raise LLMInvalidResponse("empty LLM response")
    candidates = [text.strip()]
    if "```" in text:
        candidates += [block.split("```")[1] if block.startswith("```") else block for block in text.split("```")[1:]]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LLMInvalidResponse(f"no JSON object found in LLM response: {text[:200]!r}")


class LLMClient:
    """Controller / Knowledge Manager 共用的最小 interface."""

    def chat_json(self, messages: list, schema: Optional[dict] = None) -> dict:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    def __init__(self, config: LLMConfig, poster: Optional[Callable[[dict, dict, float], str]] = None):
        self.config = config
        self._poster = poster
        self.last_latency_seconds: Optional[float] = None
        self.last_usage: Optional[dict] = None
        self.call_count = 0
        self.failure_count = 0

    def resolve_settings(self) -> tuple[str, str, Optional[str]]:
        endpoint = os.environ.get(self.config.endpoint_env, "").strip()
        model = os.environ.get(self.config.model_env, "").strip()
        api_key = os.environ.get(self.config.api_key_env, "").strip() or None
        if not endpoint:
            raise LLMError(f"environment variable {self.config.endpoint_env} is not set")
        if not model:
            raise LLMError(f"environment variable {self.config.model_env} is not set")
        if not endpoint.rstrip("/").endswith("chat/completions"):
            endpoint = endpoint.rstrip("/") + "/chat/completions"
        return endpoint, model, api_key

    def chat_json(self, messages: list, schema: Optional[dict] = None) -> dict:
        endpoint, model, api_key = self.resolve_settings()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = {
            "model": model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": False,
        }
        attempts = max(0, int(self.config.max_retries)) + 1
        last_error: Optional[LLMError] = None
        for attempt in range(attempts):
            self.call_count += 1
            started = time.monotonic()
            try:
                content, usage = self._request(endpoint, headers, body)
                self.last_latency_seconds = time.monotonic() - started
                self.last_usage = usage
                payload = parse_json_object(content)
                return check_schema(payload, schema)
            except LLMHTTPError as e:
                # 4xx 是请求本身的问题, 重试无意义.
                self.last_latency_seconds = time.monotonic() - started
                raise
            except LLMInvalidResponse as e:
                self.last_latency_seconds = time.monotonic() - started
                last_error = e
                self.failure_count += 1
            except LLMTimeout as e:
                self.last_latency_seconds = time.monotonic() - started
                last_error = e
                self.failure_count += 1
            except LLMNetworkError as e:
                self.last_latency_seconds = time.monotonic() - started
                last_error = e
                self.failure_count += 1
                if attempt + 1 < attempts:
                    time.sleep(min(2.0, 0.5 * (attempt + 1)))
        raise last_error or LLMError("LLM call failed")

    def _request(self, endpoint: str, headers: dict, body: dict) -> tuple[str, Optional[dict]]:
        if self._poster is not None:
            return extract_content(self._poster(headers, body, self.config.timeout_seconds))
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raise LLMHTTPError(e.code, e.read().decode("utf-8", errors="replace")) from e
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            if "timed out" in str(reason).lower():
                raise LLMTimeout(f"LLM request timed out after {self.config.timeout_seconds}s") from e
            raise LLMNetworkError(f"LLM request failed: {reason}") from e
        except TimeoutError as e:
            raise LLMTimeout(f"LLM request timed out after {self.config.timeout_seconds}s") from e
        return extract_content(raw)


def extract_content(raw: str) -> tuple[str, Optional[dict]]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMInvalidResponse(f"response body is not JSON: {raw[:200]!r}") from e
    if not isinstance(parsed, dict):
        raise LLMInvalidResponse(f"unexpected response envelope: {type(parsed).__name__}")
    choices = parsed.get("choices") or []
    if not choices:
        raise LLMInvalidResponse(f"LLM returned no choices: {str(parsed)[:200]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        # 部分 provider 用 text 字段 / reasoning 模型只返回 refusal.
        content = choices[0].get("text")
    if content is None:
        raise LLMInvalidResponse("LLM returned an empty message content (possibly refused)")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    usage = parsed.get("usage")
    return content, usage if isinstance(usage, dict) else None


class MockLLMClient(LLMClient):
    """离线客户端: 按脚本顺序返回响应; 脚本耗尽后报 invalid, 由 controller 记为 keep fallback.

    供 CPU 集成测试与无外部 API 的 smoke run 使用. 一次 intervention 最多两次调用
    (knowledge update, 然后 selection & calibration), 脚本需按该顺序提供.
    """

    def __init__(self, responses: Optional[list] = None, script_path: Optional[str] = None):
        self.responses: list = list(responses or [])
        if script_path and os.path.exists(script_path):
            with open(script_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        node = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(node, dict) and "response" in node:
                        self.responses.append(node["response"])
                    elif isinstance(node, dict):
                        self.responses.append(node)
        self.calls: list = []
        self.last_latency_seconds = 0.0

    def chat_json(self, messages: list, schema: Optional[dict] = None) -> dict:
        self.calls.append(messages)
        if not self.responses:
            # 不伪造一个缺 selected_skill 的 "keep": 让调用方走显式 fallback 记录路径.
            raise LLMInvalidResponse("mock: no scripted response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, str):
            response = parse_json_object(response)
        return check_schema(dict(response), schema)


def build_llm_client(config: LLMConfig) -> LLMClient:
    if config.client == "mock":
        return MockLLMClient(script_path=config.mock_script)
    return OpenAICompatibleClient(config)
