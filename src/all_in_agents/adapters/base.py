from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


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
    ) -> LLMResponse: ...
