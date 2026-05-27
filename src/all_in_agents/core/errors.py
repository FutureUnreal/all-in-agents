from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Awaitable, Callable

from .retry import RetryPolicy

if TYPE_CHECKING:
    from .context import NodeContext


class ErrorAction(str, Enum):
    ABORT = "abort"
    RETRY = "retry"
    SKIP = "skip"
    GOTO = "goto"


@dataclass
class ErrorDecision:
    action: ErrorAction
    next_action: str = "error"
    wait_ms: int = 0


ErrorHandler = Callable[["NodeContext", Exception], ErrorDecision | Awaitable[ErrorDecision]]


@dataclass
class ErrorPolicy:
    """Flow-level error handling policy."""

    action: ErrorAction = ErrorAction.ABORT
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    next_action: str = "error"
    handler: ErrorHandler | None = None

    @classmethod
    def abort(cls) -> "ErrorPolicy":
        return cls(action=ErrorAction.ABORT)

    @classmethod
    def retry(
        cls,
        max_attempts: int = 3,
        *,
        base_delay_ms: int = 0,
        max_delay_ms: int = 30_000,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
        retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
        retry_policy: RetryPolicy | None = None,
    ) -> "ErrorPolicy":
        return cls(
            action=ErrorAction.RETRY,
            retry_policy=retry_policy or RetryPolicy(
                max_attempts=max_attempts,
                base_delay_ms=base_delay_ms,
                max_delay_ms=max_delay_ms,
                backoff_multiplier=backoff_multiplier,
                jitter=jitter,
                retry_exceptions=retry_exceptions,
            ),
        )

    @classmethod
    def skip(cls, next_action: str = "skip") -> "ErrorPolicy":
        return cls(action=ErrorAction.SKIP, next_action=next_action)

    @classmethod
    def goto(cls, next_action: str) -> "ErrorPolicy":
        return cls(action=ErrorAction.GOTO, next_action=next_action)

    async def decide(self, ctx: "NodeContext", error: Exception) -> ErrorDecision:
        if self.handler is not None:
            decision = self.handler(ctx, error)
            if hasattr(decision, "__await__"):
                decision = await decision
            return decision

        if self.action == ErrorAction.RETRY:
            if await self.retry_policy.should_retry(error, ctx.attempt):
                return ErrorDecision(
                    ErrorAction.RETRY,
                    wait_ms=self.retry_policy.delay_ms(ctx.attempt, error),
                )
            return ErrorDecision(ErrorAction.ABORT)

        return ErrorDecision(self.action, next_action=self.next_action)


async def maybe_wait(decision: ErrorDecision) -> None:
    if decision.wait_ms > 0:
        await asyncio.sleep(decision.wait_ms / 1000)
