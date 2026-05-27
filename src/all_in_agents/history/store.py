from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from ..core.trace import RunTrace, TraceEvent
from ..utils import make_ulid as _make_ulid, iso_now as _iso_now

_SNAPSHOT_EVERY_N = 50
_SNAPSHOT_EVERY_MS = 30_000
_SNAPSHOT_KEEP = 10
_LOCK_TIMEOUT = 5.0

class FileEventStore:
    def __init__(
        self,
        base_dir: str | Path = "./runs",
        redact_tool_result: Callable[[str, Any], Any] | None = None,
        preview_max_chars: int = 2000,
    ):
        self._base = Path(base_dir)
        self._last_snapshot: dict[str, tuple[int, int]] = {}  # run_id -> (event_count, ts_ms)
        self._append_locks: dict[str, asyncio.Lock] = {}
        self._event_counts: dict[str, int] = {}
        self._redact = redact_tool_result
        self._preview_max_chars = preview_max_chars
        self._on_event: Callable[[dict], Any] | None = None

    def _run_dir(self, run_id: str) -> Path:
        d = self._base / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def events_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "events.ndjson"

    def _append_to_file(self, path: Path, line: str) -> None:
        """Sync file write with cross-process lock. Runs inside asyncio.to_thread."""
        import sys
        lock_path = path.with_suffix(".lock")
        if sys.platform == "win32":
            import time as _time
            deadline = _time.time() + _LOCK_TIMEOUT
            while _time.time() < deadline:
                try:
                    lf = open(lock_path, "x")
                    break
                except FileExistsError:
                    _time.sleep(0.05)
            else:
                raise TimeoutError(f"Could not acquire lock: {lock_path}")
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                lf.close()
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
        else:
            import fcntl
            with open(path, "a", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def _snapshots_dir(self, run_id: str) -> Path:
        d = self._run_dir(run_id) / "snapshots"
        d.mkdir(exist_ok=True)
        return d

    async def append(self, run_id: str, event_type: str, payload: dict) -> str:
        event = TraceEvent(
            event_id=_make_ulid(),
            run_id=run_id,
            ts=_iso_now(),
            type=event_type,
            payload=payload,
        )
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        path = self.events_path(run_id)
        async with self._append_locks.setdefault(run_id, asyncio.Lock()):
            await asyncio.to_thread(self._append_to_file, path, line)
        self._event_counts[run_id] = self._event_counts.get(run_id, 0) + 1
        if self._on_event is not None:
            try:
                cb_result = self._on_event(event.to_dict())
                if asyncio.iscoroutine(cb_result):
                    await cb_result
            except Exception:
                pass  # callback errors must not interrupt main flow
        return event.event_id

    def _read_event_dicts(self, run_id: str, after_event_id: str | None = None) -> list[dict]:
        path = self.events_path(run_id)
        if not path.exists():
            return []
        events = []
        found = after_event_id is None
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not found:
                    if ev.get("event_id") == after_event_id:
                        found = True
                    continue
                events.append(ev)
        return events

    def _read_events(self, run_id: str, after_event_id: str | None = None) -> list[TraceEvent]:
        return [TraceEvent.from_dict(ev) for ev in self._read_event_dicts(run_id, after_event_id)]

    def read_events(self, run_id: str, after_event_id: str | None = None) -> list[TraceEvent]:
        """Read typed trace events for a run."""
        return self._read_events(run_id, after_event_id=after_event_id)

    def build_trace(self, run_id: str) -> RunTrace:
        return RunTrace(run_id=run_id, events=self._read_events(run_id))

    def build_trajectory(self, run_id: str) -> list[dict]:
        """Return a compact in-memory trajectory suitable for RunResult."""
        return self.build_trace(run_id).trajectory

    async def replay_all(self, run_id: str, reducer: Callable[[Any, dict], Any]) -> Any:
        state = None
        for ev in self._read_event_dicts(run_id):
            state = reducer(state, ev)
        return state

    async def save_snapshot(self, run_id: str, state: Any | None = None, last_event_id: str = "") -> None:
        if not last_event_id:
            events = self._read_event_dicts(run_id)
            last_event_id = events[-1]["event_id"] if events else "EMPTY"

        snap = {"last_event_id": last_event_id, "state": state}
        snap_path = self._snapshots_dir(run_id) / f"{last_event_id}.json"

        def _write_and_cleanup():
            tmp = snap_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
            tmp.replace(snap_path)
            # 只保留最新 _SNAPSHOT_KEEP 个快照
            snap_dir = self._snapshots_dir(run_id)
            old_snaps = sorted(snap_dir.glob("*.json"), key=lambda p: p.stem)
            for old in old_snaps[:-_SNAPSHOT_KEEP]:
                old.unlink(missing_ok=True)

        await asyncio.to_thread(_write_and_cleanup)

        self._last_snapshot[run_id] = (
            self._event_counts.get(run_id, 0),
            int(time.time() * 1000),
        )

    async def replay_from_snapshot(self, run_id: str, reducer: Callable[[Any, dict], Any]) -> Any:
        snap_dir = self._snapshots_dir(run_id)

        def _load_latest_snapshot():
            snaps = sorted(snap_dir.glob("*.json"), key=lambda p: p.stem)
            if not snaps:
                return None, None
            latest = snaps[-1]
            try:
                data = json.loads(latest.read_text(encoding="utf-8"))
                return data.get("state"), data.get("last_event_id", "")
            except Exception:
                return None, None

        state, last_event_id = await asyncio.to_thread(_load_latest_snapshot)

        if state is None and last_event_id is None:
            return await self.replay_all(run_id, reducer)

        for ev in self._read_event_dicts(run_id, after_event_id=last_event_id):
            state = reducer(state, ev)
        return state

    def should_snapshot(self, run_id: str) -> bool:
        count = self._event_counts.get(run_id)
        if count is None:
            # 冷启动兜底：一次性从文件计数并缓存
            count = len(self._read_event_dicts(run_id))
            self._event_counts[run_id] = count
        last_count, last_ts = self._last_snapshot.get(run_id, (0, 0))
        elapsed_ms = int(time.time() * 1000) - last_ts
        return (count - last_count >= _SNAPSHOT_EVERY_N) or (
            last_ts > 0 and elapsed_ms >= _SNAPSHOT_EVERY_MS
        )

    def redact_tool_response(self, tool_name: str, result: Any) -> Any:
        if self._redact is not None:
            return self._redact(tool_name, result)
        return result

    async def append_tool_use(
        self,
        run_id: str,
        *,
        turn_id: str,
        tool_use_id: str,
        name: str,
        args: dict,
    ) -> str:
        return await self.append(
            run_id,
            "TOOL_USE",
            {"turn_id": turn_id, "tool_use_id": tool_use_id, "name": name, "args": args},
        )

    async def append_tool_result(
        self,
        run_id: str,
        *,
        turn_id: str,
        tool_use_id: str,
        name: str,
        status: str,
        content: str,
    ) -> str:
        return await self.append(
            run_id,
            "TOOL_RESULT",
            {
                "turn_id": turn_id,
                "tool_use_id": tool_use_id,
                "name": name,
                "status": status,
                "content": content[:self._preview_max_chars],
            },
        )

    async def append_tool_aborted(
        self,
        run_id: str,
        *,
        turn_id: str,
        tool_use_id: str,
        name: str,
        reason: str,
        error_class: str | None = None,
    ) -> str:
        return await self.append(
            run_id,
            "TOOL_ABORTED",
            {
                "turn_id": turn_id,
                "tool_use_id": tool_use_id,
                "name": name,
                "reason": reason,
                "error_class": error_class,
            },
        )

    async def append_run_aborted(
        self,
        run_id: str,
        *,
        reason: str,
        error_class: str,
        metrics: dict | None = None,
    ) -> str:
        return await self.append(
            run_id,
            "RUN_ABORTED",
            {"reason": reason, "error_class": error_class, "metrics": metrics or {}},
        )

    async def close_open_tool_uses(
        self,
        run_id: str,
        *,
        reason: str = "process_interrupted",
        error_class: str | None = None,
    ) -> int:
        events = self._read_event_dicts(run_id)
        tool_uses: list[dict] = []
        closed_ids: set[str] = set()
        for ev in events:
            ev_type = ev.get("type")
            payload = ev.get("payload", {})
            tid = payload.get("tool_use_id")
            if ev_type == "TOOL_USE" and tid:
                tool_uses.append(payload)
            elif ev_type in ("TOOL_RESULT", "TOOL_ABORTED") and tid:
                closed_ids.add(tid)
        count = 0
        for payload in tool_uses:
            tid = payload.get("tool_use_id")
            if tid not in closed_ids:
                await self.append_tool_aborted(
                    run_id,
                    turn_id=payload.get("turn_id", ""),
                    tool_use_id=tid,
                    name=payload.get("name", ""),
                    reason=reason,
                    error_class=error_class,
                )
                count += 1
        return count
