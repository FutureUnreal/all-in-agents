"""Project context and skill loader for agent harness.

Loads AGENTS.md, .context/prefs/, .context/ file indexes, and optional
SKILL.md files to inject into system prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SkillContext:
    """Immutable skill context loaded from a SKILL.md file."""

    name: str
    path: str
    content: str


@dataclass(frozen=True)
class HarnessContext:
    """Immutable project context snapshot."""

    agents_md: str = ""
    prefs: dict[str, str] = field(default_factory=dict)
    file_index: tuple[str, ...] = ()
    skills: tuple[SkillContext, ...] = ()


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len("[TRUNCATED]"))] + "[TRUNCATED]"


def discover_skills(project_root: str | Path = ".") -> dict[str, Path]:
    """Discover project skills.

    Skills are directories containing a SKILL.md file under either
    ``.skills/<name>/SKILL.md`` or ``skills/<name>/SKILL.md``. Hidden
    ``.skills`` entries take precedence when both directories define the
    same skill name.
    """
    root = Path(project_root).resolve()
    found: dict[str, Path] = {}
    for base_name in (".skills", "skills"):
        base = root / base_name
        if not base.is_dir():
            continue
        for skill_file in sorted(base.glob("*/SKILL.md")):
            if skill_file.is_file():
                found.setdefault(skill_file.parent.name, skill_file)
    return found


def load_skills(
    project_root: str | Path = ".",
    skill_names: Iterable[str] | None = None,
    max_chars: int = 12000,
) -> tuple[SkillContext, ...]:
    """Load selected skills from project skill directories.

    Args:
        project_root: Project root directory.
        skill_names: Skill names to load. If omitted, all discovered skills
            are loaded.
        max_chars: Total skill content budget before truncation.
    """
    discovered = discover_skills(project_root)
    if skill_names is None:
        names = sorted(discovered)
    else:
        names = list(skill_names)

    missing = [name for name in names if name not in discovered]
    if missing:
        available = ", ".join(sorted(discovered)) or "none"
        raise FileNotFoundError(
            f"Unknown skill(s): {', '.join(missing)}. Available skills: {available}"
        )

    root = Path(project_root).resolve()
    loaded: list[SkillContext] = []
    remaining = max_chars
    for name in names:
        if remaining <= 0:
            break
        path = discovered[name]
        content = path.read_text(encoding="utf-8")
        content = _truncate(content, remaining)
        remaining -= len(content)
        loaded.append(SkillContext(
            name=name,
            path=str(path.relative_to(root)),
            content=content,
        ))
    return tuple(loaded)


def load_project_context(
    project_root: str | Path = ".",
    max_chars: int = 6000,
    *,
    include_project_context: bool = True,
    include_skills: bool = False,
    skill_names: Iterable[str] | None = None,
    max_skill_chars: int = 12000,
) -> HarnessContext:
    """Load project context from AGENTS.md, .context/, and optional skills.

    Args:
        project_root: Project root directory.
        max_chars: Maximum project-context characters before truncation.
        include_project_context: Whether to load AGENTS.md and .context/.
        include_skills: Whether to load all discovered skills.
        skill_names: Specific skill names to load.
        max_skill_chars: Maximum total skill characters before truncation.
    """
    root = Path(project_root).resolve()

    # Read AGENTS.md
    agents_md = ""
    if include_project_context:
        agents_path = root / "AGENTS.md"
        if agents_path.exists():
            agents_md = agents_path.read_text(encoding="utf-8")

    # Read .context/prefs/*.md
    prefs: dict[str, str] = {}
    if include_project_context:
        prefs_dir = root / ".context" / "prefs"
        if prefs_dir.is_dir():
            for md_file in sorted(prefs_dir.glob("*.md")):
                prefs[md_file.name] = md_file.read_text(encoding="utf-8")

    # Build .context/ file index (relative paths, exclude __pycache__)
    file_index: list[str] = []
    if include_project_context:
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

    skills: tuple[SkillContext, ...] = ()
    if include_skills or skill_names is not None:
        skills = load_skills(root, None if include_skills else skill_names, max_skill_chars)

    return HarnessContext(
        agents_md=agents_md,
        prefs=prefs,
        file_index=tuple(file_index),
        skills=skills,
    )


def build_system_prompt(
    base_system: str = "",
    project_root: str | Path = ".",
    max_chars: int = 6000,
    *,
    include_project_context: bool = True,
    include_skills: bool = False,
    skill_names: Iterable[str] | None = None,
    max_skill_chars: int = 12000,
) -> str:
    """Build a system prompt by injecting project context into base_system.

    Args:
        base_system: Base system prompt text.
        project_root: Project root directory.
        max_chars: Passed to load_project_context for project truncation.
        include_project_context: Whether to include AGENTS.md and .context/.
        include_skills: Whether to include all discovered project skills.
        skill_names: Specific skills to include.
        max_skill_chars: Total skill content budget.

    Returns:
        Combined system prompt string.
    """
    ctx = load_project_context(
        project_root,
        max_chars,
        include_project_context=include_project_context,
        include_skills=include_skills,
        skill_names=skill_names,
        max_skill_chars=max_skill_chars,
    )

    if not ctx.agents_md and not ctx.prefs and not ctx.file_index and not ctx.skills:
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

    if ctx.skills:
        skill_blocks = []
        for skill in ctx.skills:
            skill_blocks.append(
                f"### {skill.name}\nSource: {skill.path}\n\n{skill.content}"
            )
        sections.append("## Skills\n" + "\n\n".join(skill_blocks))

    context = "\n\n".join(sections)
    return f"{base_system}\n\n{context}" if base_system else context
