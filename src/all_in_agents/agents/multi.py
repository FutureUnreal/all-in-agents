from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from ..utils import make_ulid as _make_ulid

_LOCK_TIMEOUT = 5.0  # seconds


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_TERMINAL_STATUSES = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED}


@dataclass
class MessageEnvelope:
    msg_id: str
    run_id: str
    from_agent: str
    to_agent: str  # agent_id | "coordinator" | "broadcast"
    msg_type: str
    payload: dict
    ts: str
    ttl_ms: int = 300_000
    idempotency_key: str = ""

    def to_dict(self) -> dict:
        return {
            "msg_id": self.msg_id,
            "run_id": self.run_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "msg_type": self.msg_type,
            "payload": self.payload,
            "ts": self.ts,
            "ttl_ms": self.ttl_ms,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MessageEnvelope":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Task:
    task_id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    assigned_to: str | None = None
    lease_expires_at: int | None = None
    lease_duration_ms: int = 90_000
    renew_every_ms: int = 30_000
    result: dict | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "status": self.status,
            "dependencies": self.dependencies,
            "assigned_to": self.assigned_to,
            "lease_expires_at": self.lease_expires_at,
            "lease_duration_ms": self.lease_duration_ms,
            "renew_every_ms": self.renew_every_ms,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d = dict(d)
        d["status"] = TaskStatus(d.get("status", "PENDING"))
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class MessageBus:
    def __init__(self, run_dir: str | Path):
        self._inbox_dir = Path(run_dir) / "inbox"
        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        self._inbox_locks: dict[str, asyncio.Lock] = {}

    def _inbox_path(self, agent_id: str) -> Path:
        return self._inbox_dir / f"{agent_id}.jsonl"

    def _cursor_path(self, agent_id: str) -> Path:
        return self._inbox_dir / f"{agent_id}.cursor"

    def _read_cursor_sync(self, agent_id: str) -> str | None:
        path = self._cursor_path(agent_id)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return None

    def _write_cursor_sync(self, agent_id: str, last_msg_id: str) -> None:
        path = self._cursor_path(agent_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(last_msg_id, encoding="utf-8")
        tmp.replace(path)

    def _send_to_file_sync(self, path: Path, line: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _read_inbox_sync(self, agent_id: str) -> list[MessageEnvelope]:
        path = self._inbox_path(agent_id)
        if not path.exists():
            return []
        messages = []
        now_ms = int(time.time() * 1000)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    env = MessageEnvelope.from_dict(d)
                    ts_dt = datetime.fromisoformat(env.ts.replace("Z", "+00:00"))
                    ts_ms = int(ts_dt.timestamp() * 1000)
                    if now_ms - ts_ms <= env.ttl_ms:
                        messages.append(env)
                except (json.JSONDecodeError, Exception):
                    continue
        messages.sort(key=lambda m: m.msg_id)
        return messages

    async def send(self, envelope: MessageEnvelope) -> None:
        targets = self._resolve_targets(envelope)
        for target in targets:
            lock = self._inbox_locks.setdefault(target, asyncio.Lock())
            async with lock:
                if envelope.idempotency_key:
                    existing = await asyncio.to_thread(self._read_inbox_sync, target)
                    if any(m.idempotency_key == envelope.idempotency_key for m in existing):
                        continue
                line = json.dumps(envelope.to_dict(), ensure_ascii=False)
                await asyncio.to_thread(self._send_to_file_sync, self._inbox_path(target), line)

    async def read_inbox(self, agent_id: str, *, after_msg_id: str | None = None, limit: int | None = None, use_cursor: bool = False) -> list[MessageEnvelope]:
        if use_cursor and after_msg_id is None:
            after_msg_id = await asyncio.to_thread(self._read_cursor_sync, agent_id)

        messages = await asyncio.to_thread(self._read_inbox_sync, agent_id)

        if after_msg_id:
            messages = [m for m in messages if m.msg_id > after_msg_id]

        if limit is not None:
            messages = messages[:limit]

        return messages

    async def ack(self, agent_id: str, last_msg_id: str) -> None:
        await asyncio.to_thread(self._write_cursor_sync, agent_id, last_msg_id)

    async def sweep_expired(self, agent_id: str | None = None) -> int:
        def _sweep_sync(target_agent: str) -> int:
            messages = self._read_inbox_sync(target_agent)
            cursor = self._read_cursor_sync(target_agent)
            valid_messages = [m for m in messages if not (cursor and m.msg_id <= cursor)]
            deleted_count = len(messages) - len(valid_messages)
            if deleted_count > 0:
                path = self._inbox_path(target_agent)
                tmp = path.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    for m in valid_messages:
                        f.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")
                tmp.replace(path)
            return deleted_count

        async def _sweep_one(target: str) -> int:
            lock = self._inbox_locks.setdefault(target, asyncio.Lock())
            async with lock:
                return await asyncio.to_thread(_sweep_sync, target)

        if agent_id is None:
            total = 0
            for inbox_file in self._inbox_dir.glob("*.jsonl"):
                total += await _sweep_one(inbox_file.stem)
            return total
        return await _sweep_one(agent_id)

    def _resolve_targets(self, envelope: MessageEnvelope) -> list[str]:
        if envelope.to_agent == "broadcast":
            targets = [p.stem for p in self._inbox_dir.glob("*.jsonl")]
            if envelope.from_agent in targets:
                targets.remove(envelope.from_agent)
            return targets
        return [envelope.to_agent]


class TaskManager:
    def __init__(self, run_dir: str | Path):
        self._path = Path(run_dir) / "tasks.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def _events_path(self) -> Path:
        return self._path.parent / "tasks.ndjson"

    def _append_event_sync(self, event: dict) -> None:
        with open(self._events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _materialize_sync(self) -> list[Task]:
        if not self._events_path.exists():
            return []

        tasks_by_id: dict[str, Task] = {}
        with open(self._events_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    event_type = event.get("type")
                    task_data = event.get("task", {})

                    if event_type == "TASK_CREATED":
                        task = Task.from_dict(task_data)
                        tasks_by_id[task.task_id] = task
                    elif event_type in ("TASK_UPDATED", "TASK_CLAIMED", "TASK_RENEWED", "TASK_REAPED"):
                        task_id = task_data.get("task_id")
                        if task_id in tasks_by_id:
                            for k, v in task_data.items():
                                if hasattr(tasks_by_id[task_id], k):
                                    setattr(tasks_by_id[task_id], k, v)
                except (json.JSONDecodeError, Exception):
                    continue

        return list(tasks_by_id.values())

    def _load(self) -> list[Task]:
        if self._events_path.exists():
            return self._materialize_sync()
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [Task.from_dict(d) for d in data]
        except Exception:
            return []

    def _save(self, tasks: list[Task]) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps([t.to_dict() for t in tasks], indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def _lock(self):
        if sys.platform == "win32":
            return _WinFileLock(self._path)
        return _UnixFileLock(self._path)

    def _create_task_sync(self, goal: str, dependencies: list[str] | None) -> Task:
        with self._lock():
            task = Task(task_id=_make_ulid(), goal=goal, dependencies=dependencies or [])
            self._append_event_sync({"type": "TASK_CREATED", "task": task.to_dict()})
        return task

    def _update_task_sync(self, task_id: str, **kwargs) -> Task | None:
        with self._lock():
            tasks = self._load()
            for t in tasks:
                if t.task_id == task_id:
                    for k, v in kwargs.items():
                        if hasattr(t, k):
                            setattr(t, k, v)
                    self._append_event_sync({"type": "TASK_UPDATED", "task": {**{"task_id": task_id}, **kwargs}})
                    status = t.status
                    if not isinstance(status, TaskStatus):
                        try:
                            status = TaskStatus(status)
                        except ValueError:
                            status = None
                    if status in _TERMINAL_STATUSES:
                        self._save(tasks)
                    return t
        return None

    def _claim_task_sync(self, task_id: str, agent_id: str) -> bool:
        with self._lock():
            tasks = self._load()
            now_ms = int(time.time() * 1000)
            for t in tasks:
                if t.task_id == task_id and t.status in (TaskStatus.PENDING, TaskStatus.CLAIMED):
                    if t.status == TaskStatus.CLAIMED and (not t.lease_expires_at or now_ms <= t.lease_expires_at):
                        return False
                    t.status = TaskStatus.CLAIMED
                    t.assigned_to = agent_id
                    t.lease_expires_at = now_ms + t.lease_duration_ms
                    self._append_event_sync({"type": "TASK_CLAIMED", "task": {"task_id": task_id, "status": t.status, "assigned_to": agent_id, "lease_expires_at": t.lease_expires_at}})
                    return True
        return False

    def _get_available_sync(self, agent_id: str) -> list[Task]:
        tasks = self._load()
        done_ids = {t.task_id for t in tasks if t.status == TaskStatus.DONE}
        now_ms = int(time.time() * 1000)
        available = []
        for t in tasks:
            if t.status != TaskStatus.PENDING:
                if t.status == TaskStatus.CLAIMED and t.lease_expires_at and now_ms > t.lease_expires_at:
                    available.append(t)  # expired lease, can reclaim
                continue
            if all(dep in done_ids for dep in t.dependencies):
                available.append(t)
        return available

    def _get_all_sync(self) -> list[Task]:
        return self._load()

    async def create_task(self, goal: str, dependencies: list[str] | None = None) -> Task:
        return await asyncio.to_thread(self._create_task_sync, goal, dependencies)

    async def update_task(self, task_id: str, **kwargs) -> Task | None:
        return await asyncio.to_thread(self._update_task_sync, task_id, **kwargs)

    async def claim_task(self, task_id: str, agent_id: str) -> bool:
        return await asyncio.to_thread(self._claim_task_sync, task_id, agent_id)

    async def get_available(self, agent_id: str) -> list[Task]:
        return await asyncio.to_thread(self._get_available_sync, agent_id)

    async def get_all(self) -> list[Task]:
        return await asyncio.to_thread(self._get_all_sync)

    def _renew_lease_sync(self, task_id: str, agent_id: str, lease_duration_ms: int | None) -> bool:
        with self._lock():
            tasks = self._load()
            now_ms = int(time.time() * 1000)
            for t in tasks:
                if t.task_id == task_id and t.assigned_to == agent_id and t.status in (TaskStatus.CLAIMED, TaskStatus.RUNNING):
                    duration = lease_duration_ms if lease_duration_ms is not None else t.lease_duration_ms
                    t.lease_expires_at = now_ms + duration
                    self._append_event_sync({"type": "TASK_RENEWED", "task": {"task_id": task_id, "lease_expires_at": t.lease_expires_at}})
                    return True
        return False

    def _reap_expired_sync(self) -> list[str]:
        with self._lock():
            tasks = self._load()
            now_ms = int(time.time() * 1000)
            reaped = []
            for t in tasks:
                if t.status in (TaskStatus.CLAIMED, TaskStatus.RUNNING) and t.lease_expires_at and now_ms > t.lease_expires_at:
                    t.status = TaskStatus.PENDING
                    t.assigned_to = None
                    t.lease_expires_at = None
                    self._append_event_sync({"type": "TASK_REAPED", "task": {"task_id": t.task_id, "status": TaskStatus.PENDING, "assigned_to": None, "lease_expires_at": None}})
                    reaped.append(t.task_id)
        return reaped

    async def renew_lease(self, task_id: str, agent_id: str, lease_duration_ms: int | None = None) -> bool:
        return await asyncio.to_thread(self._renew_lease_sync, task_id, agent_id, lease_duration_ms)

    async def reap_expired(self) -> list[str]:
        return await asyncio.to_thread(self._reap_expired_sync)

    async def watch_leases(self, interval_ms: int = 5000, stop_event: asyncio.Event | None = None) -> None:
        while True:
            await asyncio.sleep(interval_ms / 1000)
            await self.reap_expired()
            if stop_event is not None and stop_event.is_set():
                break


class _UnixFileLock:
    def __init__(self, path: Path):
        self._lock_path = path.with_suffix(".lock")

    def __enter__(self):
        import fcntl
        self._f = open(self._lock_path, "w")
        fcntl.flock(self._f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, *_):
        import fcntl
        fcntl.flock(self._f, fcntl.LOCK_UN)
        self._f.close()


class _WinFileLock:
    def __init__(self, path: Path):
        self._lock_path = path.with_suffix(".lock")

    def __enter__(self):
        deadline = time.time() + _LOCK_TIMEOUT
        while time.time() < deadline:
            try:
                self._f = open(self._lock_path, "x")
                return self
            except FileExistsError:
                time.sleep(0.05)
        raise TimeoutError(f"Could not acquire lock on {self._lock_path}")

    def __exit__(self, *_):
        self._f.close()
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass
