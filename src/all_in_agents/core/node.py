import asyncio
import copy
from abc import ABC, abstractmethod
from typing import Any


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
