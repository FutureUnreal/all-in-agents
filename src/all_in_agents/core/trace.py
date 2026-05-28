from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

SCHEMA_VERSION = "1"


@dataclass
class TraceEvent:
    event_id: str
    run_id: str
    ts: str
    type: str
    payload: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "ts": self.ts,
            "type": self.type,
            "schema_version": self.schema_version,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TraceEvent":
        return cls(
            event_id=data.get("event_id", ""),
            run_id=data.get("run_id", ""),
            ts=data.get("ts", ""),
            type=data.get("type", ""),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            payload=data.get("payload") or {},
        )


@dataclass
class RunTrace:
    run_id: str
    events: list[TraceEvent] = field(default_factory=list)

    @property
    def trajectory(self) -> list[dict]:
        event_types = {
            "ASSISTANT_MESSAGE",
            "CONTROL_DECISION",
            "TOOL_USE",
            "TOOL_RESULT",
            "TOOL_ABORTED",
            "ARTIFACT_VALIDATION",
            "RUN_STOPPED",
            "RUN_ABORTED",
        }
        trajectory = []
        for ev in self.events:
            if ev.type not in event_types:
                continue
            trajectory.append({
                "event_id": ev.event_id,
                "ts": ev.ts,
                "type": ev.type,
                **ev.payload,
            })
        return trajectory


class TraceStore(Protocol):
    async def append(self, run_id: str, event_type: str, payload: dict) -> str: ...

    def read_events(self, run_id: str, after_event_id: str | None = None) -> list[TraceEvent]: ...

    def build_trace(self, run_id: str) -> RunTrace: ...
