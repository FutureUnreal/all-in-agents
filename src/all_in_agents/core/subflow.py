from __future__ import annotations

from typing import Any

from .context import NodeContext
from .flow import Flow
from .node import BaseNode


class SubFlowNode(BaseNode):
    """Run a child Flow inside the current RunContext."""

    def __init__(
        self,
        start: BaseNode,
        *,
        flow: Flow | None = None,
        action: str = "default",
        checkpoint_id: str | None = None,
    ):
        super().__init__(checkpoint_id=checkpoint_id)
        self.start = start
        self.flow = flow or Flow()
        self.action = action

    async def exec(self, prep_result: Any, ctx: NodeContext):
        return await self.flow.run(ctx.run_context, self.start)

    async def post(self, ctx: NodeContext, exec_result: Any) -> str:
        return self.action
