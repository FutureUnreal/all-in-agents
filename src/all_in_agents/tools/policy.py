from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SideEffectLevel(str, Enum):
    READ_ONLY = "read_only"
    WRITES_LOCAL = "writes_local"
    NETWORK = "network"
    DANGEROUS = "dangerous"


@dataclass(frozen=True)
class ToolPolicy:
    """Policy for controlling tool visibility and approval requirements."""

    require_approval_for: frozenset[SideEffectLevel] = frozenset(
        {SideEffectLevel.DANGEROUS, SideEffectLevel.WRITES_LOCAL}
    )
    tool_allowlist: frozenset[str] | None = None  # None = no filtering
    tool_denylist: frozenset[str] = frozenset()  # Higher priority than allowlist
    workspace_roots: tuple[Path, ...] = ()
    command_denylist: frozenset[str] = frozenset()
    sanitize_env: bool = False
    allowed_env_vars: frozenset[str] = frozenset()

    def is_tool_visible(self, tool_name: str) -> bool:
        """Check if a tool is visible based on allowlist/denylist.

        Denylist takes priority. If allowlist is None, all tools are visible
        (except those in denylist).
        """
        if tool_name in self.tool_denylist:
            return False
        if self.tool_allowlist is None:
            return True
        return tool_name in self.tool_allowlist

    def requires_approval(self, side_effect_level: SideEffectLevel) -> bool:
        """Check if a side effect level requires approval."""
        return side_effect_level in self.require_approval_for

    def resolved_workspace_roots(self, run_workspace_root: str | None = None) -> tuple[Path, ...]:
        """Resolve and deduplicate workspace roots.

        If workspace_roots is empty and run_workspace_root is provided,
        implicitly use run_workspace_root. Otherwise return empty tuple.
        """
        if not self.workspace_roots and run_workspace_root:
            return (Path(run_workspace_root).resolve(),)

        if not self.workspace_roots:
            return ()

        # Resolve and deduplicate
        resolved = {p.resolve() for p in self.workspace_roots}
        return tuple(sorted(resolved))
