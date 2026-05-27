from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..utils import iso_now as _iso_now

CHECKPOINT_SCHEMA_VERSION = "1"


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)

    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


@dataclass
class FlowCheckpoint:
    run_id: str
    next_node_id: str | None
    step_index: int
    context: dict[str, Any]
    status: str = "running"
    updated_at: str = ""
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    @classmethod
    def capture(
        cls,
        ctx,
        *,
        next_node_id: str | None,
        step_index: int,
        status: str = "running",
    ) -> "FlowCheckpoint":
        return cls(
            run_id=ctx.run.run_id,
            next_node_id=next_node_id,
            step_index=step_index,
            status=status,
            updated_at=_iso_now(),
            context=ctx.to_checkpoint(),
        )

    def apply_to(self, ctx) -> None:
        ctx.restore_checkpoint(self.context)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "next_node_id": self.next_node_id,
            "step_index": self.step_index,
            "status": self.status,
            "updated_at": self.updated_at,
            "context": to_jsonable(self.context),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowCheckpoint":
        return cls(
            schema_version=data.get("schema_version", CHECKPOINT_SCHEMA_VERSION),
            run_id=data.get("run_id", ""),
            next_node_id=data.get("next_node_id"),
            step_index=int(data.get("step_index", 0) or 0),
            status=data.get("status", "running"),
            updated_at=data.get("updated_at", ""),
            context=data.get("context") or {},
        )


class JsonCheckpointStore:
    def __init__(self, base_dir: str | Path = "./runs"):
        self.base_dir = Path(base_dir)

    def checkpoint_path(self, run_id: str) -> Path:
        return self.base_dir / run_id / "checkpoint.json"

    def resolve(self, ref: str | Path) -> Path:
        path = Path(ref)
        if path.exists():
            if path.is_dir():
                return path / "checkpoint.json"
            return path
        return self.checkpoint_path(str(ref))

    def save(self, checkpoint: FlowCheckpoint) -> str:
        path = self.checkpoint_path(checkpoint.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return str(path)

    def load(self, ref: str | Path) -> FlowCheckpoint:
        path = self.resolve(ref)
        data = json.loads(path.read_text(encoding="utf-8"))
        return FlowCheckpoint.from_dict(data)
