"""Project context loader for agent harness.

Loads AGENTS.md, .context/prefs/, and .context/ file index to inject into system prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HarnessContext:
    """Immutable project context snapshot."""

    agents_md: str = ""
    prefs: dict[str, str] = field(default_factory=dict)
    file_index: tuple[str, ...] = ()


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len("[TRUNCATED]"))] + "[TRUNCATED]"


def load_project_context(
    project_root: str | Path = ".",
    max_chars: int = 6000
) -> HarnessContext:
    """Load project context from AGENTS.md and .context/ directory.

    Args:
        project_root: Project root directory
        max_chars: Maximum total characters before truncation
    """
    root = Path(project_root).resolve()

    # Read AGENTS.md
    agents_md = ""
    agents_path = root / "AGENTS.md"
    if agents_path.exists():
        agents_md = agents_path.read_text(encoding="utf-8")

    # Read .context/prefs/*.md
    prefs: dict[str, str] = {}
    prefs_dir = root / ".context" / "prefs"
    if prefs_dir.is_dir():
        for md_file in sorted(prefs_dir.glob("*.md")):
            prefs[md_file.name] = md_file.read_text(encoding="utf-8")

    # Build .context/ file index (relative paths, exclude __pycache__)
    file_index: list[str] = []
    context_dir = root / ".context"
    if context_dir.is_dir():
        for p in sorted(context_dir.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                file_index.append(str(p.relative_to(root / ".context")))

    # 总量预算分配（优先级：agents_md > prefs > file_index）
    agents_budget = max_chars // 2
    prefs_budget = max_chars // 3
    index_budget = max(0, max_chars - agents_budget - prefs_budget)

    # 截断 agents_md
    agents_md = _truncate(agents_md, agents_budget)

    # 截断 prefs（按文件名排序，逐个消耗预算）
    prefs_trimmed: dict[str, str] = {}
    remaining_prefs = prefs_budget
    for name, content in sorted(prefs.items()):
        if remaining_prefs <= 0:
            break
        if len(content) <= remaining_prefs:
            prefs_trimmed[name] = content
            remaining_prefs -= len(content)
        else:
            prefs_trimmed[name] = _truncate(content, remaining_prefs)
            remaining_prefs = 0
    prefs = prefs_trimmed

    # 截断 file_index（超限时追加提示）
    file_index_out: list[str] = []
    remaining_index = index_budget
    for rel in file_index:
        entry = rel + "\n"
        if remaining_index - len(entry) < 0:
            omitted = len(file_index) - len(file_index_out)
            file_index_out.append(f"[... {omitted} more files]")
            break
        file_index_out.append(rel)
        remaining_index -= len(entry)
    file_index = file_index_out

    return HarnessContext(
        agents_md=agents_md,
        prefs=prefs,
        file_index=tuple(file_index),
    )


def build_system_prompt(
    base_system: str = "",
    project_root: str | Path = ".",
    max_chars: int = 6000,
) -> str:
    """Build a system prompt by injecting project context into base_system.

    Args:
        base_system: Base system prompt text
        project_root: Project root directory
        max_chars: Passed to load_project_context for truncation

    Returns:
        Combined system prompt string
    """
    ctx = load_project_context(project_root, max_chars)

    if not ctx.agents_md and not ctx.prefs and not ctx.file_index:
        return base_system

    sections: list[str] = []

    if ctx.agents_md:
        sections.append(f"## Project Guidelines\n{ctx.agents_md}")

    if ctx.prefs:
        prefs_body = "\n\n".join(ctx.prefs.values())
        sections.append(f"## Preferences\n{prefs_body}")

    if ctx.file_index:
        file_list = "\n".join(ctx.file_index)
        sections.append(f"## Project Context Files\n{file_list}")

    return base_system + "\n\n" + "\n\n".join(sections)
