"""History compaction with micro-compact, LLM summarization, and deterministic fallback."""
from __future__ import annotations

import copy
import json as _json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..core.content import file_summary, image_summary, is_file_block, is_image_block

if TYPE_CHECKING:
    from ..adapters.base import LLMAdapter


def _middle_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    omitted = max(0, len(text) - max_chars)
    for _ in range(4):
        marker = f"\n...[{omitted} chars truncated]...\n"
        available = max_chars - len(marker)
        if available <= 0:
            return text[:max_chars]

        head_chars = available // 2
        tail_chars = available - head_chars
        next_omitted = len(text) - head_chars - tail_chars
        if next_omitted == omitted:
            tail = text[-tail_chars:] if tail_chars else ""
            return text[:head_chars] + marker + tail
        omitted = next_omitted

    marker = f"\n...[{omitted} chars truncated]...\n"
    available = max_chars - len(marker)
    head_chars = max(0, available // 2)
    tail_chars = max(0, available - head_chars)
    tail = text[-tail_chars:] if tail_chars else ""
    return text[:head_chars] + marker + tail


def _compact_json(value) -> str:
    summary = _json.dumps(value, ensure_ascii=False)
    return _middle_truncate(summary, 200)


def _turn_has_tool_result(turn: list[dict]) -> bool:
    for msg in turn:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return True
    return False


def _tool_result_summary(block: dict, tool_names_by_id: dict[str, str]) -> str:
    tool_id = block.get("tool_use_id", "")
    tool_name = tool_names_by_id.get(tool_id) or tool_id or "unknown_tool"
    content = block.get("content", "")
    content_length = len(content) if isinstance(content, str) else len(str(content))
    status = "failed" if block.get("is_error") else "success"
    return f"[tool_result: {tool_name} -> {status}; {content_length} chars]"


@dataclass
class CompactionResult:
    """Result of history compaction."""
    summary: str
    kept_turns: list[list[dict]]
    used_fallback: bool


class CompactionStrategy(Protocol):
    """Protocol for pluggable history compaction strategies."""

    async def compact_turns(
        self,
        llm: "LLMAdapter",
        turns: list[list[dict]],
        *,
        max_context_tokens: int,
        target_tokens: int | None = None,
    ) -> CompactionResult:
        """Compact grouped conversation turns."""


class HistoryCompactor:
    """Compacts conversation history using micro-compact, LLM summarization, and fallback."""

    def __init__(
        self,
        micro_compact_max_chars: int = 2000,
        keep_recent_turns: int = 12,
        summary_max_tokens: int = 1200,
        summary_keep_recent_tool_results: int = 3,
    ):
        self.micro_compact_max_chars = micro_compact_max_chars
        self.keep_recent_turns = keep_recent_turns
        self.summary_max_tokens = summary_max_tokens
        self.summary_keep_recent_tool_results = max(0, summary_keep_recent_tool_results)

    def micro_compact_turns(self, turns: list[list[dict]]) -> list[list[dict]]:
        """Truncate oversized tool_result content in each turn."""
        new_turns = copy.deepcopy(turns)

        for turn in new_turns:
            for msg in turn:
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_result":
                            block_content = block.get("content", "")
                            if isinstance(block_content, str) and len(block_content) > self.micro_compact_max_chars:
                                block["content"] = _middle_truncate(
                                    block_content,
                                    self.micro_compact_max_chars,
                                )

        return new_turns

    async def summarize_turns(self, llm: "LLMAdapter", turns: list[list[dict]]) -> str:
        """Summarize turns using LLM into structured JSON."""
        history_lines = []
        keep_tool_result_indexes = self._recent_tool_result_turn_indexes(turns)

        for turn_index, turn in enumerate(turns):
            tool_names_by_id: dict[str, str] = {}
            keep_tool_results = turn_index in keep_tool_result_indexes
            for msg in turn:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                content_text = self._summary_content_text(
                    content,
                    tool_names_by_id=tool_names_by_id,
                    keep_tool_results=keep_tool_results,
                )
                history_lines.append(f"{role}: {content_text}")

        history_text = "\n".join(history_lines)

        prompt = (
            'Summarize the conversation history below into structured JSON with keys: '
            '"facts" (list of strings), "decisions" (list of strings), '
            '"open_threads" (list of strings). Be concise. Output only valid JSON.\n\n'
            f"History:\n{history_text}"
        )

        resp = await llm.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=512
        )

        summary = resp.content
        max_chars = self.summary_max_tokens * 4
        if len(summary) > max_chars:
            summary = summary[:max_chars]

        return summary

    def _recent_tool_result_turn_indexes(self, turns: list[list[dict]]) -> set[int]:
        if self.summary_keep_recent_tool_results <= 0:
            return set()

        indexes: list[int] = []
        for index, turn in enumerate(turns):
            if _turn_has_tool_result(turn):
                indexes.append(index)
        return set(indexes[-self.summary_keep_recent_tool_results:])

    def _summary_content_text(
        self,
        content,
        *,
        tool_names_by_id: dict[str, str],
        keep_tool_results: bool,
    ) -> str:
        if not isinstance(content, list):
            return str(content)

        parts = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue

            btype = block.get("type")
            if btype == "tool_use":
                name = block.get("name", "")
                tool_id = block.get("id", "")
                if tool_id:
                    tool_names_by_id[tool_id] = name
                parts.append(f"[tool_use: {name}({_compact_json(block.get('input', {}))})]")
            elif btype == "tool_result":
                if keep_tool_results:
                    parts.append(str(block.get("content", "")))
                else:
                    parts.append(_tool_result_summary(block, tool_names_by_id))
            elif is_image_block(block):
                parts.append(image_summary(block))
            elif is_file_block(block):
                parts.append(file_summary(block))
            else:
                parts.append(str(block.get("text", "") or block.get("content", "")))
        return " ".join(parts)

    def deterministic_snip(self, turns: list[list[dict]]) -> CompactionResult:
        """Fallback: keep recent turns, discard oldest."""
        kept_turns = turns[-self.keep_recent_turns:] if len(turns) > self.keep_recent_turns else turns
        return CompactionResult(
            summary="[deterministic snip: oldest turns removed]",
            kept_turns=kept_turns,
            used_fallback=True
        )

    async def compact_turns(
        self,
        llm: "LLMAdapter",
        turns: list[list[dict]],
        *,
        max_context_tokens: int,
        target_tokens: int | None = None,
    ) -> CompactionResult:
        """Compact turns if they exceed target_tokens.

        ``max_context_tokens`` is the hard model context size. ``target_tokens``
        is the soft budget selected by the framework or caller. When omitted,
        the hard context size is used for backward-compatible behavior.
        """
        def _estimate_tokens(turns_list: list[list[dict]]) -> int:
            return len(str(turns_list)) // 4

        target = target_tokens or max_context_tokens
        target = min(target, max_context_tokens)

        current_tokens = _estimate_tokens(turns)
        if current_tokens <= target:
            return CompactionResult(
                summary="",
                kept_turns=turns,
                used_fallback=False
            )

        # Step 1: micro compact
        micro_turns = self.micro_compact_turns(turns)
        micro_tokens = _estimate_tokens(micro_turns)

        if micro_tokens <= target:
            return CompactionResult(
                summary="",
                kept_turns=micro_turns,
                used_fallback=False
            )

        # Step 2: split old and recent
        if len(micro_turns) <= self.keep_recent_turns:
            # Not enough turns to split, use fallback
            return self.deterministic_snip(micro_turns)

        old_turns = micro_turns[:-self.keep_recent_turns]
        recent_turns = micro_turns[-self.keep_recent_turns:]

        # Step 3: try LLM summarization
        try:
            summary_text = await self.summarize_turns(llm, old_turns)
            return CompactionResult(
                summary=summary_text,
                kept_turns=recent_turns,
                used_fallback=False
            )
        except Exception:
            # Step 4: fallback to deterministic snip
            return self.deterministic_snip(micro_turns)
