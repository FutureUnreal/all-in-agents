import copy

from .node import BaseNode


class Flow:
    async def run(self, shared: dict, start: BaseNode) -> dict:
        node: BaseNode | None = copy.copy(start)
        while node is not None:
            prep_result = await node.prep(shared)
            exec_result = await node.exec_with_retry(prep_result)
            action = await node.post(shared, exec_result)
            next_node = node.successors.get(action)
            node = copy.copy(next_node) if next_node is not None else None
        return shared
