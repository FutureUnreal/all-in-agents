from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .compactor import CompactionStrategy
from ..utils import make_ulid as _make_ulid

if TYPE_CHECKING:
    from ..adapters.base import LLMAdapter
    from ..tools.registry import ToolResponse

COMPRESS_THRESHOLD_TOKENS = 14_000
KEEP_RECENT_TURNS = 12
KEEP_RECENT_TOOL_RESULTS = 3
SUMMARY_MAX_TOKENS = 1_200

_INTERNAL_KEYS = {"_turn_id", "_kind", "_source"}

_SUMMARY_PROMPT = (
    "Summarize the conversation history below into structured JSON with keys: "
    '"facts" (list of strings), "decisions" (list of strings), "open_threads" (list of strings). '
    "Be concise. Output only valid JSON.\n\nHistory:\n{history}"
)


def _estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(c) // 4
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict):
                    total += len(str(block.get("text", "") or block.get("content", ""))) // 4
    return total


def _strip_internal(msg: dict) -> dict:
    return {k: v for k, v in msg.items() if k not in _INTERNAL_KEYS}


def _group_by_turn(messages: list[dict]) -> list[list[dict]]:
    """Group messages into turns by _turn_id. Messages without _turn_id form singleton groups."""
    turns: list[list[dict]] = []
    current_id: str | None = None
    current_group: list[dict] = []

    for msg in messages:
        tid = msg.get("_turn_id")
        if tid is None:
            if current_group:
                turns.append(current_group)
                current_group = []
                current_id = None
            turns.append([msg])
        elif tid != current_id:
            if current_group:
                turns.append(current_group)
            current_group = [msg]
            current_id = tid
        else:
            current_group.append(msg)

    if current_group:
        turns.append(current_group)

    return turns


@dataclass
class HistoryManager:
    max_context_tokens: int = 32_000
    compactor: CompactionStrategy | None = None
    compress_threshold_tokens: int = -1
    _messages: list[dict] = field(default_factory=list)
    _summary: str = ""

    def __post_init__(self) -> None:
        self._compactor = self.compactor
        if self.compress_threshold_tokens <= 0:
            self._compress_threshold = int(self.max_context_tokens * 0.7)
        else:
            self._compress_threshold = self.compress_threshold_tokens

    def add(self, role: str, content: str, *, turn_id: str | None = None) -> str:
        if turn_id is None:
            turn_id = _make_ulid()
        self._messages.append({"role": role, "content": content, "_turn_id": turn_id})
        return turn_id

    def add_assistant_tool_calls(
        self, content: str | None, tool_calls: list, *, turn_id: str | None = None
    ) -> str:
        if turn_id is None:
            turn_id = _make_ulid()
        blocks = []
        if content:
            blocks.append({"type": "text", "text": content})
        for tc in tool_calls:
            blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args})
        self._messages.append({"role": "assistant", "content": blocks, "_turn_id": turn_id})
        return turn_id

    def add_tool_result(
        self, tool_use_id: str, result: "ToolResponse", *, turn_id: str | None = None
    ) -> str:
        if turn_id is None:
            turn_id = _make_ulid()
        self._messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": result.content}],
            "_source": "tool_result",
            "_turn_id": turn_id,
        })
        return turn_id

    def get_messages(self, max_tokens: int | None = None) -> list[dict]:
        msgs = self._build_context()
        effective_max = max_tokens or self.max_context_tokens

        if _estimate_tokens(msgs) <= effective_max:
            return msgs

        # Group into turns and drop oldest whole turns until within budget
        # Separate summary preamble (first 2 msgs) from live messages
        preamble: list[dict] = []
        live_msgs = msgs
        if self._summary:
            preamble = msgs[:2]
            live_msgs = msgs[2:]

        turns = _group_by_turn(live_msgs)
        while len(turns) > 1 and _estimate_tokens(preamble + [m for t in turns for m in t]) > effective_max:
            turns.pop(0)

        result = preamble + [m for t in turns for m in t]

        # Last-resort: truncate sole remaining message
        if result and _estimate_tokens(result) > effective_max:
            msg = result[-1]
            if isinstance(msg.get("content"), str):
                max_chars = effective_max * 4
                result[-1] = {**msg, "content": msg["content"][:max_chars]}

        return result

    def needs_compression(self) -> bool:
        return _estimate_tokens(self._messages) > self._compress_threshold

    async def _compact_with_strategy(
        self,
        compactor: CompactionStrategy,
        llm: "LLMAdapter",
        turns: list[list[dict]],
    ):
        """Call custom compactors with target_tokens when their signature supports it."""
        method = compactor.compact_turns
        supports_target = True
        try:
            supports_target = "target_tokens" in inspect.signature(method).parameters
        except (TypeError, ValueError):
            pass

        if supports_target:
            return await method(
                llm,
                turns,
                max_context_tokens=self.max_context_tokens,
                target_tokens=self._compress_threshold,
            )
        return await method(llm, turns, max_context_tokens=self.max_context_tokens)

    async def compress(self, llm: "LLMAdapter") -> bool:
        if not self.needs_compression():
            return True

        if self._compactor is not None:
            turns = _group_by_turn(self._messages)
            result = await self._compact_with_strategy(self._compactor, llm, turns)
            if result.summary:
                self._summary = result.summary
            self._messages = [msg for turn in result.kept_turns for msg in turn]
            return not result.used_fallback

        # Legacy path (no compactor)
        recent = self._split_recent()
        old_msgs = self._messages[: len(self._messages) - len(recent)]
        if not old_msgs:
            return True

        history_text = "\n".join(
            f"{m['role']}: {m['content'] if isinstance(m['content'], str) else str(m['content'])}"
            for m in old_msgs
        )
        prompt = _SUMMARY_PROMPT.format(history=history_text)

        try:
            resp = await llm.generate([{"role": "user", "content": prompt}], max_tokens=512)
            summary_text = resp.content or ""
            if len(summary_text) // 4 > SUMMARY_MAX_TOKENS:
                summary_text = summary_text[: SUMMARY_MAX_TOKENS * 4]
            self._summary = summary_text
            self._messages = recent
            return True
        except Exception:
            # Deterministic fallback: keep most recent turns, don't silently lose all history
            turns = _group_by_turn(self._messages)
            kept = turns[-KEEP_RECENT_TURNS:] if len(turns) > KEEP_RECENT_TURNS else turns
            self._summary = "[deterministic snip]"
            self._messages = [msg for turn in kept for msg in turn]
            return False

    def _split_recent(self) -> list[dict]:
        tool_results = [m for m in self._messages if m.get("_source") == "tool_result"]
        keep_tools = set(id(m) for m in tool_results[-KEEP_RECENT_TOOL_RESULTS:])

        regular = [m for m in self._messages if m.get("_source") != "tool_result"]
        keep_regular = set(id(m) for m in regular[-KEEP_RECENT_TURNS:])

        return [m for m in self._messages if id(m) in keep_regular or id(m) in keep_tools]

    def _build_context(self) -> list[dict]:
        msgs: list[dict] = []
        if self._summary:
            msgs.append({"role": "user", "content": f"[Previous conversation summary]\n{self._summary}"})
            msgs.append({"role": "assistant", "content": "Understood."})
        msgs.extend(_strip_internal(m) for m in self._messages)
        return msgs

    def to_checkpoint(self) -> dict:
        return {
            "messages": self._messages,
            "summary": self._summary,
            "max_context_tokens": self.max_context_tokens,
            "compress_threshold_tokens": self.compress_threshold_tokens,
        }

    def restore_checkpoint(self, data: dict) -> None:
        self._messages = list(data.get("messages") or [])
        self._summary = data.get("summary", "")
        self.max_context_tokens = int(data.get("max_context_tokens", self.max_context_tokens) or self.max_context_tokens)
        self.compress_threshold_tokens = int(
            data.get("compress_threshold_tokens", self.compress_threshold_tokens)
            or self.compress_threshold_tokens
        )
        self.__post_init__()
