import asyncio
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
    close_async_client,
)
from .openai_chat import build_chat_kwargs, parse_chat_response
from .openai_responses import build_responses_kwargs, parse_responses_response
from .openai_utils import normalize_api

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


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
        self.model_id = model
        self.max_context_tokens = max_context_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._base_delay_ms = base_delay_ms
        self._max_delay_ms = max_delay_ms
        self._backoff_multiplier = backoff_multiplier
        self._api = normalize_api(api)
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
            return await self._generate_with_client(
                client,
                messages=messages,
                tools=tools or [],
                system=system,
                max_tokens=max_tokens,
                options=self._options.merge(options),
            )
        finally:
            await close_async_client(client)

    async def _generate_with_client(
        self,
        client,
        *,
        messages: list[dict],
        tools: list[dict],
        system: str,
        max_tokens: int,
        options: GenerationOptions,
    ) -> LLMResponse:
        last_err: Exception | None = None
        delay_ms = self._base_delay_ms

        for attempt in range(self._max_retries):
            try:
                if self._api == "responses":
                    kwargs = build_responses_kwargs(self.model_id, messages, tools, system, max_tokens, options)
                    return parse_responses_response(await client.responses.create(**kwargs))

                kwargs = build_chat_kwargs(self.model_id, messages, tools, system, max_tokens, options)
                return parse_chat_response(await client.chat.completions.create(**kwargs))

            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                error_class, retry_after_ms = self._classify_error(e, status)

                if status not in _RETRYABLE_STATUS and not self._is_transient(e):
                    raise LLMError(str(e), error_class, attempt + 1, retry_after_ms=retry_after_ms)

                last_err = e
                if attempt < self._max_retries - 1:
                    wait_ms = retry_after_ms if retry_after_ms is not None else random.random() * delay_ms
                    await asyncio.sleep(wait_ms / 1000)
                    delay_ms = min(int(delay_ms * self._backoff_multiplier), self._max_delay_ms)

        raise LLMError(str(last_err), ErrorClass.TRANSIENT, self._max_retries)

    @staticmethod
    def _is_transient(e: Exception) -> bool:
        msg = str(e).lower()
        return any(k in msg for k in ("connection", "timeout", "network", "reset"))

    @staticmethod
    def _classify_error(e: Exception, status: int | None) -> tuple[ErrorClass, int | None]:
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

        if any(k in str(e).lower() for k in ("connection", "timeout", "network", "reset")):
            return ErrorClass.TRANSIENT, None

        return ErrorClass.INTERNAL, None
