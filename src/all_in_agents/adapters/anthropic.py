import asyncio
import os
import random

from .base import ConfigError, ErrorClass, LLMAdapter, LLMError, LLMResponse, ToolCall

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_STOP_REASON_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
}


class AnthropicAdapter(LLMAdapter):
    model_id = "claude-sonnet-4-6"
    max_context_tokens = 200_000

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_retries: int = 3,
        base_delay_ms: int = 250,
        max_delay_ms: int = 8_000,
        backoff_multiplier: float = 2.0,
        enable_cache: bool = False,
    ):
        self.model_id = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._base_delay_ms = base_delay_ms
        self._max_delay_ms = max_delay_ms
        self._backoff_multiplier = backoff_multiplier
        self._enable_cache = enable_cache

    async def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 2048,
    ) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install 'all-in-agents[anthropic]'")

        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigError("ANTHROPIC_API_KEY not set")

        client = anthropic.AsyncAnthropic(api_key=api_key)
        anthropic_tools = [self._convert_tool(t) for t in (tools or [])]

        last_err: Exception | None = None
        delay_ms = self._base_delay_ms

        for attempt in range(self._max_retries):
            try:
                kwargs: dict = dict(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                if system:
                    if self._enable_cache:
                        kwargs["system"] = [{"type": "text", "text": system,
                                             "cache_control": {"type": "ephemeral"}}]
                    else:
                        kwargs["system"] = system
                if anthropic_tools:
                    kwargs["tools"] = anthropic_tools

                resp = await client.messages.create(**kwargs)
                return self._parse_response(resp)

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
    def _convert_tool(tool: dict) -> dict:
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("input_schema", tool.get("parameters", {})),
        }

    @staticmethod
    def _parse_response(resp) -> LLMResponse:
        content_text: str | None = None
        tool_calls: list[ToolCall] = []

        for block in resp.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, args=block.input))

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=_STOP_REASON_MAP.get(resp.stop_reason, resp.stop_reason),
        )
