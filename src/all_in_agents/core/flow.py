import copy
from dataclasses import dataclass
from typing import Any, Callable

from .node import BaseNode


@dataclass
class FlowHooks:
    """Optional lifecycle hooks for Flow execution.

    Hook callbacks receive a mutable context dictionary. They may be sync or
    async callables. Exceptions raised by hooks propagate, making hooks suitable
    for hard gates as well as logging/metrics.
    """

    on_node_start: Callable[[dict], Any] | None = None
    on_node_end: Callable[[dict], Any] | None = None
    on_node_error: Callable[[dict], Any] | None = None


async def _call_hook(callback: Callable[[dict], Any] | None, ctx: dict) -> None:
    if callback is None:
        return
    result = callback(ctx)
    if hasattr(result, "__await__"):
        await result


class Flow:
    def __init__(self, *, hooks: FlowHooks | None = None, copy_nodes: bool = True):
        self.hooks = hooks or FlowHooks()
        # Default keeps the original stateless-node behavior. Set False when a
        # flow intentionally uses persistent node instance state.
        self.copy_nodes = copy_nodes

    def _next_node(self, node: BaseNode | None) -> BaseNode | None:
        if node is None:
            return None
        return copy.copy(node) if self.copy_nodes else node

    async def run(self, shared: dict, start: BaseNode) -> dict:
        node: BaseNode | None = self._next_node(start)
        while node is not None:
            ctx = {"node": node, "node_name": type(node).__name__, "shared": shared}
            await _call_hook(self.hooks.on_node_start, ctx)
            try:
                prep_result = await node.prep(shared)
                exec_result = await node.exec_with_retry(prep_result)
                action = await node.post(shared, exec_result)
            except Exception as e:
                ctx["error"] = e
                await _call_hook(self.hooks.on_node_error, ctx)
                raise

            ctx.update({
                "prep_result": prep_result,
                "exec_result": exec_result,
                "action": action,
            })
            await _call_hook(self.hooks.on_node_end, ctx)
            node = self._next_node(node.successors.get(action))
        return shared
