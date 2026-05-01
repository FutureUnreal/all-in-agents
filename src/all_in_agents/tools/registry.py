from dataclasses import dataclass
from typing import Any, Awaitable, Callable

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
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> "Tool | None":
        return self._tools.get(name)

    def get_schemas(self, policy: ToolPolicy | None = None) -> list[dict]:
        effective_policy = policy or self._policy
        if effective_policy is not None:
            return [_to_schema(t) for t in self._tools.values() if effective_policy.is_tool_visible(t.name)]
        return [_to_schema(t) for t in self._tools.values()]

    async def execute(self, name: str, args: dict, run: Run, *, policy: ToolPolicy | None = None) -> ToolResponse:
        tool = self._tools.get(name)
        if not tool:
            return ToolResponse("error", f"Unknown tool: {name}", "NOT_FOUND")

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
            result = await tool.execute(args, run)
            if len(result.content) > MAX_CONTENT_LEN:
                result = ToolResponse(
                    result.status,
                    result.content[:MAX_CONTENT_LEN] + "[TRUNCATED]",
                    result.error_class,
                )
            return result
        except Exception as e:
            return ToolResponse("error", str(e), "INTERNAL_BUG")


def _to_schema(tool: Tool) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }
