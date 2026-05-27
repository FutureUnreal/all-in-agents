import asyncio
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

from .context import NodeContext
from .retry import RetryPolicy

_SKIPPED = object()


class BaseNode(ABC):
    def __init__(self, checkpoint_id: str | None = None):
        self.successors: dict[str, "BaseNode"] = {}
        self.checkpoint_id = checkpoint_id

    async def prep(self, ctx: NodeContext) -> Any:
        return None

    @abstractmethod
    async def exec(self, prep_result: Any, ctx: NodeContext) -> Any: ...

    async def exec_with_retry(self, prep_result: Any, ctx: NodeContext) -> Any:
        return await self.exec(prep_result, ctx)

    async def post(self, ctx: NodeContext, exec_result: Any) -> str:
        return "default"

    def __rshift__(self, other: "BaseNode") -> "BaseNode":
        self.successors["default"] = other
        return other

    def __sub__(self, action: str) -> "_ActionBinder":
        return _ActionBinder(self, action)

    def next(self, action: str) -> "BaseNode | None":
        return self.successors.get(action)


class _ActionBinder:
    def __init__(self, node: BaseNode, action: str):
        self.node = node
        self.action = action

    def __rshift__(self, other: BaseNode) -> BaseNode:
        self.node.successors[self.action] = other
        return other


class Node(BaseNode):
    def __init__(
        self,
        retry_policy: RetryPolicy | None = None,
        *,
        max_attempts: int = 1,
        max_retries: int | None = None,
        wait_ms: int = 0,
        checkpoint_id: str | None = None,
    ):
        super().__init__(checkpoint_id=checkpoint_id)
        attempts = max_retries if max_retries is not None else max_attempts
        self.retry_policy = retry_policy or RetryPolicy(
            max_attempts=attempts,
            base_delay_ms=wait_ms,
            max_delay_ms=wait_ms,
            backoff_multiplier=1.0,
            jitter=False,
        )
        self.max_retries = self.retry_policy.max_attempts
        self.wait_ms = wait_ms

    async def exec_with_retry(self, prep_result: Any, ctx: NodeContext) -> Any:
        attempt = 0
        while True:
            try:
                ctx.retry_attempt = attempt
                return await self.exec(prep_result, ctx)
            except Exception as e:
                if not await self.retry_policy.should_retry(e, attempt):
                    raise
                await self.retry_policy.wait(attempt, e)
                attempt += 1


class BatchNode(BaseNode):
    def __init__(self, max_concurrency: int = 4, checkpoint_id: str | None = None):
        super().__init__(checkpoint_id=checkpoint_id)
        self.max_concurrency = max_concurrency

    async def exec(self, items: list[Any], ctx: NodeContext) -> list[Any]:
        sem = asyncio.Semaphore(self.max_concurrency)

        async def run_one(item: Any) -> Any:
            async with sem:
                return await self.exec_item(item, ctx)

        return await asyncio.gather(*[run_one(i) for i in items])

    @abstractmethod
    async def exec_item(self, item: Any, ctx: NodeContext) -> Any: ...


class ConditionalNode(BaseNode):
    """Wrap a node and skip it when ``predicate(ctx)`` is false."""

    def __init__(
        self,
        node: BaseNode,
        predicate: Callable[[NodeContext], bool | Awaitable[bool]],
        *,
        skip_action: str = "skip",
        checkpoint_id: str | None = None,
    ):
        super().__init__(checkpoint_id=checkpoint_id)
        self.node = node
        self.predicate = predicate
        self.skip_action = skip_action
        self.successors.update(node.successors)

    async def prep(self, ctx: NodeContext) -> dict:
        should_run = self.predicate(ctx)
        if hasattr(should_run, "__await__"):
            should_run = await should_run
        if not should_run:
            return {"skipped": True, "prep_result": None}
        return {"skipped": False, "prep_result": await self.node.prep(ctx)}

    async def exec(self, prep_result: dict, ctx: NodeContext) -> Any:
        if prep_result.get("skipped"):
            return _SKIPPED
        return await self.node.exec_with_retry(prep_result["prep_result"], ctx)

    async def post(self, ctx: NodeContext, exec_result: Any) -> str:
        if exec_result is _SKIPPED:
            return self.skip_action
        return await self.node.post(ctx, exec_result)
