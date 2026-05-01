"""History compaction with micro-compact, LLM summarization, and deterministic fallback."""
from __future__ import annotations

import copy
import json as _json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..adapters.base import LLMAdapter


@dataclass
class CompactionResult:
    """Result of history compaction."""
    summary: str
    kept_turns: list[list[dict]]
    used_fallback: bool


class HistoryCompactor:
    """Compacts conversation history using micro-compact, LLM summarization, and fallback."""

    def __init__(
        self,
        micro_compact_max_chars: int = 2000,
        keep_recent_turns: int = 12,
        summary_max_tokens: int = 1200,
    ):
        self.micro_compact_max_chars = micro_compact_max_chars
        self.keep_recent_turns = keep_recent_turns
        self.summary_max_tokens = summary_max_tokens

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
                                block["content"] = (
                                    block_content[:500]
                                    + "\n[TRUNCATED]\n"
                                    + block_content[-500:]
                                )

        return new_turns

    async def summarize_turns(self, llm: "LLMAdapter", turns: list[list[dict]]) -> str:
        """Summarize turns using LLM into structured JSON."""
        # Flatten turns to text
        history_lines = []
        for turn in turns:
            for msg in turn:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if block.get("type") == "tool_use":
                            name = block.get("name", "")
                            input_summary = _json.dumps(
                                block.get("input", {}), ensure_ascii=False
                            )
                            if len(input_summary) > 200:
                                input_summary = input_summary[:200] + "..."
                            parts.append(f"[tool_use: {name}({input_summary})]")
                        else:
                            parts.append(
                                str(block.get("text", "") or block.get("content", ""))
                            )
                    content_text = " ".join(parts)
                else:
                    content_text = str(content)
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
        max_context_tokens: int
    ) -> CompactionResult:
        """Compact turns if they exceed max_context_tokens."""
        def _estimate_tokens(turns_list: list[list[dict]]) -> int:
            return len(str(turns_list)) // 4

        current_tokens = _estimate_tokens(turns)
        if current_tokens <= max_context_tokens:
            return CompactionResult(
                summary="",
                kept_turns=turns,
                used_fallback=False
            )

        # Step 1: micro compact
        micro_turns = self.micro_compact_turns(turns)
        micro_tokens = _estimate_tokens(micro_turns)

        if micro_tokens <= max_context_tokens:
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

