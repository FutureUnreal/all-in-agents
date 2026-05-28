from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from contextlib import suppress
from typing import Any, AsyncIterator, Iterable


@dataclass(frozen=True)
class AgentStreamEvent:
    """Typed event yielded by Agent.stream."""

    type: str
    run_id: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "run_id": self.run_id,
            "data": self.data,
        }


def trace_event_to_stream_event(event: dict) -> AgentStreamEvent | None:
    event_type = event.get("type", "")
    run_id = event.get("run_id", "")
    payload = dict(event.get("payload") or {})
    payload["event_id"] = event.get("event_id", "")
    payload["ts"] = event.get("ts", "")

    mapping = {
        "RUN_CREATED": "run_started",
        "RUN_RESUMED": "run_resumed",
        "CONTROL_DECISION": "control_decision",
        "ASSISTANT_REJECTED": "assistant_rejected",
        "ASSISTANT_MESSAGE": "assistant_message",
        "TOOL_USE": "tool_called",
        "TOOL_RESULT": "tool_result",
        "TOOL_ABORTED": "tool_error",
        "MEMORY_UPDATED": "memory_updated",
        "ARTIFACT_VALIDATION": "artifact_validation",
        "RUN_STOPPED": "run_stopped",
        "RUN_ABORTED": "error",
    }
    stream_type = mapping.get(event_type)
    if stream_type is None:
        return None
    return AgentStreamEvent(type=stream_type, run_id=run_id, data=payload)


class AgentStreamingMixin:
    async def stream(
        self,
        goal: str,
        *,
        initial_messages: Iterable[dict[str, Any]] | None = None,
        parent_run=None,
        checkpoint: bool = False,
        resume_from: str | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        queue: asyncio.Queue[AgentStreamEvent | BaseException | object] = asyncio.Queue()
        sentinel = object()

        async def _emit(event: AgentStreamEvent) -> None:
            await queue.put(event)

        async def _drive() -> None:
            try:
                await self._run(
                    goal,
                    initial_messages=initial_messages,
                    parent_run=parent_run,
                    checkpoint=checkpoint,
                    resume_from=resume_from,
                    stream_callback=_emit,
                )
            except BaseException as e:
                await queue.put(e)
            finally:
                await queue.put(sentinel)

        task = asyncio.create_task(_drive())
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def stream_text(
        self,
        goal: str,
        *,
        initial_messages: Iterable[dict[str, Any]] | None = None,
        parent_run=None,
        checkpoint: bool = False,
        resume_from: str | None = None,
    ) -> AsyncIterator[str]:
        async for event in self.stream(
            goal,
            initial_messages=initial_messages,
            parent_run=parent_run,
            checkpoint=checkpoint,
            resume_from=resume_from,
        ):
            if event.type == "text_delta":
                yield str(event.data.get("delta", ""))
