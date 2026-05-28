from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .checkpoint import to_jsonable

if TYPE_CHECKING:
    from .node import BaseNode
    from .run import Run
    from ..adapters.base import LLMAdapter
    from ..history.manager import HistoryManager
    from ..history.store import FileEventStore
    from ..tools.registry import ToolRegistry


@dataclass
class RunContext:
    """Typed runtime context shared by nodes in a Flow."""

    run: "Run"
    llm: "LLMAdapter"
    tools: "ToolRegistry"
    history: "HistoryManager"
    store: "FileEventStore"
    system: str = ""
    compression_llm: "LLMAdapter | None" = None
    stream_callback: Callable[[Any], Any] | None = None
    on_turn: Callable[[Any], Any] | None = None
    turn_max_retries: int = 3
    final_answer: str = ""
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_compression_llm(self) -> "LLMAdapter":
        return self.compression_llm or self.llm

    def to_checkpoint(self) -> dict:
        return {
            "run": self.run.to_checkpoint(),
            "history": self.history.to_checkpoint(),
            "system": self.system,
            "final_answer": self.final_answer,
            "state": to_jsonable(self.state),
        }

    def restore_checkpoint(self, data: dict) -> None:
        self.run.restore_checkpoint(data.get("run") or {})
        self.history.restore_checkpoint(data.get("history") or {})
        self.system = data.get("system", self.system)
        self.final_answer = data.get("final_answer", self.final_answer)
        self.state = dict(data.get("state") or {})


@dataclass
class NodeContext:
    """Context passed to a node for one Flow step."""

    run_context: RunContext
    node: "BaseNode"
    node_name: str
    step_index: int = 0
    attempt: int = 0
    retry_attempt: int = 0
    prep_result: Any = None
    exec_result: Any = None
    action: str = ""
    error: Exception | None = None

    @property
    def run(self) -> "Run":
        return self.run_context.run

    @property
    def llm(self) -> "LLMAdapter":
        return self.run_context.llm

    @property
    def tools(self) -> "ToolRegistry":
        return self.run_context.tools

    @property
    def history(self) -> "HistoryManager":
        return self.run_context.history

    @property
    def store(self) -> "FileEventStore":
        return self.run_context.store

    @property
    def system(self) -> str:
        return self.run_context.system

    @property
    def state(self) -> dict[str, Any]:
        return self.run_context.state

    @property
    def final_answer(self) -> str:
        return self.run_context.final_answer

    @final_answer.setter
    def final_answer(self, value: str) -> None:
        self.run_context.final_answer = value
