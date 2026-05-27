import copy
from dataclasses import dataclass
from typing import Any, Callable

from .checkpoint import FlowCheckpoint, JsonCheckpointStore
from .context import NodeContext, RunContext
from .errors import ErrorAction, ErrorPolicy, maybe_wait
from .node import BaseNode


@dataclass
class FlowHooks:
    """Optional lifecycle hooks for Flow execution.

    Hook callbacks receive a NodeContext. They may be sync or
    async callables. Exceptions raised by hooks propagate, making hooks suitable
    for hard gates as well as logging/metrics.
    """

    on_node_start: Callable[[NodeContext], Any] | None = None
    on_node_end: Callable[[NodeContext], Any] | None = None
    on_node_error: Callable[[NodeContext], Any] | None = None


async def _call_hook(callback: Callable[[NodeContext], Any] | None, ctx: NodeContext) -> None:
    if callback is None:
        return
    result = callback(ctx)
    if hasattr(result, "__await__"):
        await result


class Flow:
    def __init__(
        self,
        *,
        hooks: FlowHooks | None = None,
        error_policy: ErrorPolicy | None = None,
        copy_nodes: bool = False,
    ):
        self.hooks = hooks or FlowHooks()
        self.error_policy = error_policy or ErrorPolicy.abort()
        self.copy_nodes = copy_nodes

    def _next_node(self, node: BaseNode | None) -> BaseNode | None:
        if node is None:
            return None
        return copy.copy(node) if self.copy_nodes else node

    def _node_registry(self, start: BaseNode) -> dict[str, BaseNode]:
        registry: dict[str, BaseNode] = {}
        visited: set[int] = set()
        counts: dict[str, int] = {}

        def visit(node: BaseNode | None) -> None:
            if node is None or id(node) in visited:
                return
            visited.add(id(node))

            if node.checkpoint_id:
                node_id = node.checkpoint_id
            else:
                node_type = type(node).__name__
                counts[node_type] = counts.get(node_type, 0) + 1
                node_id = f"{node_type}#{counts[node_type]}"
            if node_id in registry:
                raise ValueError(f"Duplicate flow checkpoint node id: {node_id}")
            setattr(node, "_flow_checkpoint_id", node_id)
            registry[node_id] = node

            for action in sorted(node.successors):
                visit(node.successors[action])

        visit(start)
        return registry

    @staticmethod
    def _checkpoint_id(node: BaseNode | None) -> str | None:
        if node is None:
            return None
        return getattr(node, "_flow_checkpoint_id", None) or node.checkpoint_id

    async def run(
        self,
        ctx: RunContext,
        start: BaseNode,
        *,
        checkpoint_store: JsonCheckpointStore | None = None,
        resume_checkpoint: FlowCheckpoint | None = None,
    ) -> RunContext:
        registry = self._node_registry(start)
        if resume_checkpoint is not None:
            resume_checkpoint.apply_to(ctx)
            step_index = resume_checkpoint.step_index
            if resume_checkpoint.next_node_id is None:
                return ctx
            start_node = registry.get(resume_checkpoint.next_node_id)
            if start_node is None:
                raise ValueError(f"Cannot resume flow; missing node id: {resume_checkpoint.next_node_id}")
            node: BaseNode | None = self._next_node(start_node)
        else:
            node = self._next_node(start)
            step_index = 0

        while node is not None:
            attempt = 0
            while True:
                node_ctx = NodeContext(
                    run_context=ctx,
                    node=node,
                    node_name=type(node).__name__,
                    step_index=step_index,
                    attempt=attempt,
                )
                await _call_hook(self.hooks.on_node_start, node_ctx)
                try:
                    node_ctx.prep_result = await node.prep(node_ctx)
                    node_ctx.exec_result = await node.exec_with_retry(node_ctx.prep_result, node_ctx)
                    node_ctx.action = await node.post(node_ctx, node_ctx.exec_result)
                except Exception as e:
                    node_ctx.error = e
                    await _call_hook(self.hooks.on_node_error, node_ctx)
                    decision = await self.error_policy.decide(node_ctx, e)
                    if decision.action == ErrorAction.RETRY:
                        await maybe_wait(decision)
                        attempt += 1
                        continue
                    if decision.action in (ErrorAction.SKIP, ErrorAction.GOTO):
                        node_ctx.action = decision.next_action
                    else:
                        raise

                await _call_hook(self.hooks.on_node_end, node_ctx)
                step_index += 1
                next_node = node.successors.get(node_ctx.action)
                next_node_id = self._checkpoint_id(next_node)
                if checkpoint_store is not None:
                    checkpoint_store.save(FlowCheckpoint.capture(
                        ctx,
                        next_node_id=next_node_id,
                        step_index=step_index,
                        status="running" if next_node_id else "completed",
                    ))
                node = self._next_node(next_node)
                break
        return ctx
