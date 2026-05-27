import asyncio
import copy
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

_SKIPPED = object()


class BaseNode(ABC):
    def __init__(self):
        self.successors: dict[str, "BaseNode"] = {}

    async def prep(self, shared: dict) -> Any:
        return None

    @abstractmethod
    async def exec(self, prep_result: Any) -> Any: ...

    async def exec_with_retry(self, prep_result: Any) -> Any:
        return await self.exec(prep_result)

    async def post(self, shared: dict, exec_result: Any) -> str:
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
    def __init__(self, max_retries: int = 1, wait_ms: int = 0):
        super().__init__()
        self.max_retries = max_retries
        self.wait_ms = wait_ms

    async def exec_with_retry(self, prep_result: Any) -> Any:
        for attempt in range(self.max_retries):
            try:
                return await self.exec(prep_result)
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                if self.wait_ms > 0:
                    await asyncio.sleep(self.wait_ms / 1000)


class BatchNode(BaseNode):
    def __init__(self, max_concurrency: int = 4):
        super().__init__()
        self.max_concurrency = max_concurrency

    async def exec(self, items: list[Any]) -> list[Any]:
        sem = asyncio.Semaphore(self.max_concurrency)

        async def run_one(item: Any) -> Any:
            async with sem:
                return await self.exec_item(item)

        return await asyncio.gather(*[run_one(i) for i in items])

    @abstractmethod
    async def exec_item(self, item: Any) -> Any: ...


class ConditionalNode(BaseNode):
    """Wrap a node and skip it when ``predicate(shared)`` is false."""

    def __init__(
        self,
        node: BaseNode,
        predicate: Callable[[dict], bool | Awaitable[bool]],
        *,
        skip_action: str = "skip",
    ):
        super().__init__()
        self.node = node
        self.predicate = predicate
        self.skip_action = skip_action
        self.successors.update(node.successors)

    async def prep(self, shared: dict) -> dict:
        should_run = self.predicate(shared)
        if hasattr(should_run, "__await__"):
            should_run = await should_run
        if not should_run:
            return {"skipped": True, "prep_result": None}
        return {"skipped": False, "prep_result": await self.node.prep(shared)}

    async def exec(self, prep_result: dict) -> Any:
        if prep_result.get("skipped"):
            return _SKIPPED
        return await self.node.exec_with_retry(prep_result["prep_result"])

    async def post(self, shared: dict, exec_result: Any) -> str:
        if exec_result is _SKIPPED:
            return self.skip_action
        return await self.node.post(shared, exec_result)
