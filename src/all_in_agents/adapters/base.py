from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
import inspect
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens"


@dataclass
class GenerationOptions:
    """Provider-neutral controls for a single model generation call."""

    temperature: float | None = None
    top_p: float | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def merge(self, override: "GenerationOptions | None" = None) -> "GenerationOptions":
        if override is None:
            return replace(self, extra=dict(self.extra))

        extra = dict(self.extra)
        extra.update(override.extra)
        return GenerationOptions(
            temperature=override.temperature if override.temperature is not None else self.temperature,
            top_p=override.top_p if override.top_p is not None else self.top_p,
            response_format=(
                override.response_format if override.response_format is not None else self.response_format
            ),
            reasoning_effort=(
                override.reasoning_effort if override.reasoning_effort is not None else self.reasoning_effort
            ),
            extra=extra,
        )

    @classmethod
    def from_values(
        cls,
        options: "GenerationOptions | None" = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "GenerationOptions":
        base = options.merge() if options is not None else cls()
        merged_extra = dict(base.extra)
        if extra:
            merged_extra.update(extra)
        return cls(
            temperature=temperature if temperature is not None else base.temperature,
            top_p=top_p if top_p is not None else base.top_p,
            response_format=response_format if response_format is not None else base.response_format,
            reasoning_effort=reasoning_effort if reasoning_effort is not None else base.reasoning_effort,
            extra=merged_extra,
        )


class ErrorClass(str, Enum):
    TRANSIENT = "TRANSIENT"
    RATE_LIMITED = "RATE_LIMITED"
    AUTH = "AUTH"
    INVALID_REQUEST = "INVALID_REQUEST"
    INTERNAL = "INTERNAL"


class ConfigError(Exception):
    pass


class LLMError(Exception):
    def __init__(self, message: str, error_class: str | ErrorClass, attempts: int = 1,
                 retry_after_ms: int | None = None):
        if isinstance(error_class, str) and not isinstance(error_class, ErrorClass):
            try:
                error_class = ErrorClass(error_class)
            except ValueError:
                pass  # keep as raw string for backward compat
        self.error_class = error_class
        self.attempts = attempts
        self.retry_after_ms = retry_after_ms
        super().__init__(message)


class LLMAdapter(ABC):
    model_id: str
    max_context_tokens: int

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 2048,
        options: GenerationOptions | None = None,
    ) -> LLMResponse: ...


async def close_async_client(client) -> None:
    close = getattr(client, "close", None) or getattr(client, "aclose", None)
    if close is None:
        return

    result = close()
    if inspect.isawaitable(result):
        await result
