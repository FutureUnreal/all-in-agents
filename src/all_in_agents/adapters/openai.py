import asyncio
import json
import os
import random
from typing import Any

from .base import (
    ConfigError,
    ErrorClass,
    GenerationOptions,
    LLMAdapter,
    LLMError,
    LLMResponse,
    ToolCall,
    close_async_client,
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


class OpenAIAdapter(LLMAdapter):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_context_tokens: int = 128_000,
        max_retries: int = 3,
        base_delay_ms: int = 250,
        max_delay_ms: int = 8_000,
        backoff_multiplier: float = 2.0,
        api: str = "chat_completions",
        options: GenerationOptions | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        self.model_id = model  # may be None; checked in generate()
        self.max_context_tokens = max_context_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._base_delay_ms = base_delay_ms
        self._max_delay_ms = max_delay_ms
        self._backoff_multiplier = backoff_multiplier
        self._api = _normalize_api(api)
        self._options = GenerationOptions.from_values(
            options,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            extra=model_kwargs,
        )

    async def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 2048,
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("Install openai: pip install 'all-in-agents[openai]'")

        if not self.model_id:
            raise ConfigError("OpenAIAdapter requires an explicit model; e.g. OpenAIAdapter(model='gpt-4o')")

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ConfigError("OPENAI_API_KEY not set")

        client = _openai.AsyncOpenAI(api_key=api_key, base_url=self._base_url)

        try:
            call_options = self._options.merge(options)

            last_err: Exception | None = None
            delay_ms = self._base_delay_ms

            for attempt in range(self._max_retries):
                try:
                    if self._api == "responses":
                        kwargs = self._build_responses_kwargs(messages, tools or [], system, max_tokens, call_options)
                        resp = await client.responses.create(**kwargs)
                        return self._parse_responses_response(resp)

                    kwargs = self._build_chat_kwargs(messages, tools or [], system, max_tokens, call_options)
                    resp = await client.chat.completions.create(**kwargs)
                    return self._parse_chat_response(resp)

                except Exception as e:
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    error_class, retry_after_ms = self._classify_error(e, status)

                    if status not in _RETRYABLE_STATUS and not self._is_transient(e):
                        raise LLMError(str(e), error_class, attempt + 1,
                                       retry_after_ms=retry_after_ms)

                    last_err = e
                    last_retry_after_ms = retry_after_ms
                    if attempt < self._max_retries - 1:
                        if last_retry_after_ms is not None:
                            wait_ms = last_retry_after_ms
                        else:
                            wait_ms = random.random() * delay_ms
                        await asyncio.sleep(wait_ms / 1000)
                        delay_ms = min(int(delay_ms * self._backoff_multiplier), self._max_delay_ms)

            raise LLMError(str(last_err), ErrorClass.TRANSIENT, self._max_retries)
        finally:
            await close_async_client(client)

    @staticmethod
    def _is_transient(e: Exception) -> bool:
        msg = str(e).lower()
        return any(k in msg for k in ("connection", "timeout", "network", "reset"))

    @staticmethod
    def _classify_error(e: Exception, status: int | None) -> tuple[ErrorClass, int | None]:
        """Return (ErrorClass, retry_after_ms) for the given exception."""
        retry_after_ms: int | None = None

        if status == 429:
            response = getattr(e, "response", None)
            headers = getattr(response, "headers", None) or {}
            retry_after = headers.get("retry-after")
            if retry_after is not None:
                try:
                    retry_after_ms = int(float(retry_after) * 1000)
                except (ValueError, TypeError):
                    pass
            return ErrorClass.RATE_LIMITED, retry_after_ms

        if status == 401:
            return ErrorClass.AUTH, None

        if status in (400, 422):
            return ErrorClass.INVALID_REQUEST, None

        if status in (500, 502, 503, 504):
            return ErrorClass.TRANSIENT, None

        # connection / timeout errors
        msg = str(e).lower()
        if any(k in msg for k in ("connection", "timeout", "network", "reset")):
            return ErrorClass.TRANSIENT, None

        return ErrorClass.INTERNAL, None

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Convert minagent tool schemas to OpenAI function tool format."""
        result = []
        for t in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", t.get("parameters", {})),
                },
            })
        return result

    @staticmethod
    def _convert_responses_tools(tools: list[dict]) -> list[dict]:
        """Convert minagent tool schemas to OpenAI Responses function tool format."""
        result = []
        for t in tools:
            result.append({
                "type": "function",
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", t.get("parameters", {})),
            })
        return result

    @staticmethod
    def _convert_messages(messages: list[dict], system: str = "") -> list[dict]:
        """Convert minagent/Anthropic-style messages to OpenAI chat format."""
        result = []
        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")

            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                # Check if this is a user message with tool_result blocks
                if role == "user":
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            result.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": block.get("content", ""),
                            })
                    continue

                # Assistant message with content blocks (tool_use + text)
                if role == "assistant":
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            text_parts.append(block.get("text", ""))
                        elif btype == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            })
                    oai_msg: dict = {"role": "assistant"}
                    if text_parts:
                        oai_msg["content"] = " ".join(text_parts)
                    else:
                        oai_msg["content"] = None
                    if tool_calls:
                        oai_msg["tool_calls"] = tool_calls
                    result.append(oai_msg)
                    continue

            # Fallback: pass through
            result.append({"role": role, "content": content or ""})

        return result

    @staticmethod
    def _convert_responses_input(messages: list[dict]) -> list[dict]:
        result = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")

            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_result":
                        result.append({
                            "type": "function_call_output",
                            "call_id": block.get("tool_use_id", ""),
                            "output": block.get("content", ""),
                        })
                    elif btype == "tool_use":
                        result.append({
                            "type": "function_call",
                            "call_id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        })
                if text_parts:
                    result.append({"role": role, "content": " ".join(text_parts)})
                continue

            result.append({"role": role, "content": content or ""})

        return result

    def _build_chat_kwargs(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
        max_tokens: int,
        options: GenerationOptions,
    ) -> dict:
        kwargs: dict = dict(
            model=self.model_id,
            messages=self._convert_messages(messages, system),
        )
        if "max_tokens" not in options.extra and "max_completion_tokens" not in options.extra:
            kwargs["max_completion_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.top_p is not None:
            kwargs["top_p"] = options.top_p
        if options.response_format is not None:
            kwargs["response_format"] = _response_format_for_chat(options.response_format)
        if options.reasoning_effort is not None:
            kwargs["reasoning_effort"] = options.reasoning_effort
        kwargs.update(options.extra)
        return kwargs

    def _build_responses_kwargs(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
        max_tokens: int,
        options: GenerationOptions,
    ) -> dict:
        kwargs: dict = dict(
            model=self.model_id,
            input=self._convert_responses_input(messages),
        )
        if system:
            kwargs["instructions"] = system
        if "max_output_tokens" not in options.extra:
            kwargs["max_output_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = self._convert_responses_tools(tools)
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.top_p is not None:
            kwargs["top_p"] = options.top_p
        if options.response_format is not None:
            kwargs["text"] = {"format": _response_format_for_responses(options.response_format)}
        if options.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": options.reasoning_effort}
        kwargs.update(options.extra)
        return kwargs

    @staticmethod
    def _map_finish_reason(finish_reason: str | None) -> str:
        return _STOP_REASON_MAP.get(finish_reason or "", "end_turn")

    def _parse_chat_response(self, resp) -> LLMResponse:
        choice = resp.choices[0]
        message = choice.message
        content_text: str | None = message.content

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, Exception):
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            stop_reason=self._map_finish_reason(choice.finish_reason),
        )

    def _parse_response(self, resp) -> LLMResponse:
        return self._parse_chat_response(resp)

    def _parse_responses_response(self, resp) -> LLMResponse:
        content_text = getattr(resp, "output_text", None)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for item in getattr(resp, "output", []) or []:
            item_type = _get_attr(item, "type")
            if item_type == "function_call":
                args = _parse_json_object(_get_attr(item, "arguments") or "{}")
                tool_calls.append(ToolCall(
                    id=_get_attr(item, "call_id") or _get_attr(item, "id") or "",
                    name=_get_attr(item, "name") or "",
                    args=args,
                ))
                continue

            if item_type == "message":
                for block in _get_attr(item, "content") or []:
                    block_type = _get_attr(block, "type")
                    if block_type in ("output_text", "text"):
                        text = _get_attr(block, "text")
                        if text:
                            text_parts.append(text)

        if content_text is None and text_parts:
            content_text = "\n".join(text_parts)

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )


def _normalize_api(api: str) -> str:
    normalized = api.replace("-", "_").lower()
    aliases = {
        "chat": "chat_completions",
        "chat_completion": "chat_completions",
        "chat_completions": "chat_completions",
        "responses": "responses",
        "response": "responses",
    }
    try:
        return aliases[normalized]
    except KeyError:
        raise ConfigError("OpenAIAdapter api must be 'chat_completions' or 'responses'")


def _get_attr(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _parse_json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _response_format_for_chat(response_format: dict[str, Any]) -> dict[str, Any]:
    if response_format.get("type") != "json_schema":
        return response_format
    if "json_schema" in response_format:
        return response_format

    json_schema = {
        key: value
        for key, value in response_format.items()
        if key not in {"type"}
    }
    return {"type": "json_schema", "json_schema": json_schema}


def _response_format_for_responses(response_format: dict[str, Any]) -> dict[str, Any]:
    if response_format.get("type") != "json_schema":
        return response_format
    if "json_schema" not in response_format:
        return response_format

    schema_value = response_format["json_schema"]
    if not isinstance(schema_value, dict):
        return response_format

    json_schema = dict(schema_value)
    json_schema.setdefault("type", "json_schema")
    return json_schema
