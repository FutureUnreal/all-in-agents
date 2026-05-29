import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

from ..core.run import Run
from .coerce import coerce_args
from .policy import ToolPolicy, SideEffectLevel

MAX_CONTENT_LEN = 200_000


@dataclass
class ToolResponse:
    status: str  # "success" | "error"
    content: str
    error_class: str | None = None


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    side_effect_level: SideEffectLevel
    execute: Callable  # async (args: dict, run: Run) -> ToolResponse
    max_calls: int | None = None
    max_wall_ms: int | None = None
    namespace: str = ""

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}:{self.name}" if self.namespace else self.name


async def _default_deny(name: str, args: dict) -> bool:
    return False


def unsafe_defaults() -> Callable[[str, dict], Awaitable[bool]]:
    """Return an always-approve callback for development / testing use.

    Usage::

        registry = ToolRegistry(approval_callback=unsafe_defaults())
    """
    async def _approve_all(name: str, args: dict) -> bool:
        return True
    return _approve_all


class ToolRegistry:
    def __init__(
        self,
        approval_callback: Callable[[str, dict], Awaitable[bool]] | None = None,
        policy: ToolPolicy | None = None,
    ):
        self._tools: dict[str, Tool] = {}
        self._approval_callback = approval_callback or _default_deny
        self._policy = policy

    def register(self, tool: Tool) -> None:
        self._tools[tool.qualified_name] = tool

    def get_tool(self, name: str) -> "Tool | None":
        tool = self._tools.get(name)
        if tool is not None:
            return tool
        for t in self._tools.values():
            if t.name == name:
                return t
        return None

    def _is_visible(self, tool: Tool, policy: ToolPolicy | None = None) -> bool:
        effective_policy = policy or self._policy
        return effective_policy is None or effective_policy.is_tool_visible(tool.qualified_name)

    def list_tool_names(self, policy: ToolPolicy | None = None) -> list[str]:
        return [tool.qualified_name for tool in self._tools.values() if self._is_visible(tool, policy)]

    def get_schemas(
        self,
        policy: ToolPolicy | None = None,
        names: Iterable[str] | None = None,
    ) -> list[dict]:
        if names is None:
            return [_to_schema(t) for t in self._tools.values() if self._is_visible(t, policy)]

        schemas: list[dict] = []
        seen_names: set[str] = set()
        seen_tools: set[str] = set()
        for name in names:
            if name in seen_names:
                continue
            seen_names.add(name)
            tool = self.get_tool(name)
            if (
                tool is not None
                and tool.qualified_name not in seen_tools
                and self._is_visible(tool, policy)
            ):
                seen_tools.add(tool.qualified_name)
                schemas.append(_to_schema(tool))
        return schemas

    async def execute(self, name: str, args: dict, run: Run, *, policy: ToolPolicy | None = None) -> ToolResponse:
        tool = self.get_tool(name)
        if not tool:
            return ToolResponse("error", f"Unknown tool: {name}", "NOT_FOUND")

        if tool.max_calls is not None:
            used = run.tool_mix.get(name, 0)
            if used >= tool.max_calls:
                return ToolResponse("error", f"Tool '{name}' exceeded max_calls={tool.max_calls}", "TOOL_LIMIT")

        effective_policy = policy or self._policy
        if effective_policy is not None:
            if not effective_policy.is_tool_visible(name):
                run.record_policy_block(name)
                return ToolResponse("error", f"Tool '{name}' blocked by policy", "POLICY_BLOCKED")
            needs_approval = effective_policy.requires_approval(tool.side_effect_level)
        else:
            needs_approval = tool.side_effect_level == SideEffectLevel.DANGEROUS

        args = coerce_args(args, tool.input_schema)

        if _HAS_JSONSCHEMA:
            try:
                jsonschema.validate(args, tool.input_schema)
            except jsonschema.ValidationError as e:
                return ToolResponse("error", str(e.message), "VALIDATION")

        if needs_approval:
            try:
                approved = await self._approval_callback(name, args)
            except Exception:
                approved = False
            if not approved:
                return ToolResponse("error", f"Tool '{name}' denied by approval", "POLICY_BLOCKED")

        try:
            if tool.max_wall_ms is not None:
                result = await asyncio.wait_for(
                    tool.execute(args, run),
                    timeout=tool.max_wall_ms / 1000,
                )
            else:
                result = await tool.execute(args, run)
            if len(result.content) > MAX_CONTENT_LEN:
                result = ToolResponse(
                    result.status,
                    result.content[:MAX_CONTENT_LEN] + "[TRUNCATED]",
                    result.error_class,
                )
            return result
        except asyncio.TimeoutError:
            return ToolResponse("error", f"Tool '{name}' exceeded max_wall_ms={tool.max_wall_ms}", "TIMEOUT")
        except Exception as e:
            return ToolResponse("error", str(e), "INTERNAL_BUG")


def _to_schema(tool: Tool) -> dict:
    return {
        "name": tool.qualified_name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }
