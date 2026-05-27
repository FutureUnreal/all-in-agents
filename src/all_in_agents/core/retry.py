from __future__ import annotations

import asyncio
import inspect
import random
from dataclasses import dataclass
from typing import Awaitable, Callable


RetryPredicate = Callable[[BaseException], bool | Awaitable[bool]]


@dataclass
class RetryPolicy:
    """Reusable retry policy for node and flow execution."""

    max_attempts: int = 1
    base_delay_ms: int = 0
    max_delay_ms: int = 30_000
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,)
    retry_if: RetryPredicate | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("RetryPolicy.max_attempts must be >= 1")
        if self.base_delay_ms < 0:
            raise ValueError("RetryPolicy.base_delay_ms must be >= 0")
        if self.max_delay_ms < 0:
            raise ValueError("RetryPolicy.max_delay_ms must be >= 0")
        if self.backoff_multiplier < 1:
            raise ValueError("RetryPolicy.backoff_multiplier must be >= 1")

    async def should_retry(self, error: BaseException, failed_attempt_index: int) -> bool:
        if failed_attempt_index >= self.max_attempts - 1:
            return False
        if self.retry_exceptions and not isinstance(error, self.retry_exceptions):
            return False
        if self.retry_if is None:
            return True
        decision = self.retry_if(error)
        if inspect.isawaitable(decision):
            decision = await decision
        return bool(decision)

    def delay_ms(self, failed_attempt_index: int, error: BaseException | None = None) -> int:
        retry_after_ms = getattr(error, "retry_after_ms", None)
        if retry_after_ms is not None:
            try:
                return max(0, int(retry_after_ms))
            except (TypeError, ValueError):
                pass

        if self.base_delay_ms <= 0:
            return 0

        delay = self.base_delay_ms * (self.backoff_multiplier ** failed_attempt_index)
        capped = min(int(delay), self.max_delay_ms)
        if self.jitter and capped > 0:
            return int(random.random() * capped)
        return capped

    async def wait(self, failed_attempt_index: int, error: BaseException | None = None) -> None:
        delay_ms = self.delay_ms(failed_attempt_index, error)
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)
