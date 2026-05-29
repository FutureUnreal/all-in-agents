from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ToolSelectionContext:
    """Inputs a tool selector can inspect before each model call."""

    goal: str
    step_index: int
    messages: list[dict[str, Any]]
    system: str = ""
    state: Mapping[str, Any] = field(default_factory=dict)


class ToolSelector(Protocol):
    def select_tools(
        self,
        available_tools: Sequence[dict[str, Any]],
        context: ToolSelectionContext,
    ) -> Iterable[str] | None:
        """Return tool names to expose, or None to expose every available tool."""


class AllToolsSelector:
    """Default selector that preserves the existing behavior."""

    def select_tools(
        self,
        available_tools: Sequence[dict[str, Any]],
        context: ToolSelectionContext,
    ) -> Iterable[str] | None:
        return None


class StaticToolsSelector:
    """Expose a fixed subset of tools by qualified or bare name."""

    def __init__(self, names: Iterable[str]):
        self.names = tuple(names)

    def select_tools(
        self,
        available_tools: Sequence[dict[str, Any]],
        context: ToolSelectionContext,
    ) -> Iterable[str] | None:
        return self.names


class KeywordToolSelector:
    """Expose tools when configured keywords appear in the current context."""

    def __init__(
        self,
        rules: Mapping[str, Iterable[str]],
        *,
        always_include: Iterable[str] = (),
        fallback_names: Iterable[str] | None = None,
    ):
        self.rules = {keyword.lower(): tuple(names) for keyword, names in rules.items()}
        self.always_include = tuple(always_include)
        self.fallback_names = tuple(fallback_names) if fallback_names is not None else None

    def select_tools(
        self,
        available_tools: Sequence[dict[str, Any]],
        context: ToolSelectionContext,
    ) -> Iterable[str] | None:
        selected = list(self.always_include)
        haystack = _context_text(context).lower()

        for keyword, names in self.rules.items():
            if keyword and keyword in haystack:
                selected.extend(names)

        if len(selected) == len(self.always_include) and self.fallback_names is not None:
            selected.extend(self.fallback_names)
        return _dedupe(selected)


def _context_text(context: ToolSelectionContext) -> str:
    parts = [context.system, context.goal]
    for message in context.messages[-6:]:
        parts.append(str(message.get("content", "")))
    return "\n".join(part for part in parts if part)


def _dedupe(names: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return tuple(result)
