from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Any, Callable

from ..core.node import BaseNode
from ..core.run import Run
from ..history.manager import HistoryManager
from ..history.store import FileEventStore
from ..utils import make_ulid as _make_ulid

if TYPE_CHECKING:
    from ..adapters.base import LLMAdapter, LLMResponse
    from ..tools.registry import ToolRegistry


def _tool_sig(name: str, args: dict) -> str:
    payload = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{name}:{h}"


# Deprecated: use LLMCallNode + ToolDispatchNode
class ReActNode(BaseNode):
    async def exec(self, prep: dict) -> "LLMResponse":
        run: Run = prep["run"]
        run.check_budget("llm_call")
        return await prep["llm"].generate(
            messages=prep["messages"],
            tools=prep["tools"],
            system=prep["system"],
        )

    async def prep(self, shared: dict) -> dict:
        run: Run = shared["run"]
        return {
            "messages": shared["history"].get_messages(),
            "tools": shared["tools"].get_schemas(policy=run.tool_policy),
            "run": run,
            "llm": shared["llm"],
            "system": shared.get("system", ""),
        }

    async def post(self, shared: dict, resp: "LLMResponse") -> str:
        run: Run = shared["run"]
        store: FileEventStore = shared.get("store")
        history: HistoryManager = shared["history"]
        tools: "ToolRegistry" = shared["tools"]
        turn_id = _make_ulid()

        if store:
            await store.append(run.run_id, "ASSISTANT_MESSAGE", {
                "content": resp.content,
                "tool_calls": [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in resp.tool_calls],
                "stop_reason": resp.stop_reason,
            })

        if resp.stop_reason == "end_turn" or not resp.tool_calls:
            shared["final_answer"] = resp.content or ""
            return "done"

        history.add("assistant", resp.content or "")

        for tc in resp.tool_calls:
            run.check_budget("tool_call", _tool_sig(tc.name, tc.args))
            if store:
                await store.append_tool_use(run.run_id, turn_id=turn_id, tool_use_id=tc.id, name=tc.name, args=tc.args)
            try:
                result = await tools.execute(tc.name, tc.args, run)
            except Exception as e:
                if store:
                    await store.append_tool_aborted(run.run_id, turn_id=turn_id, tool_use_id=tc.id, name=tc.name, reason=str(e), error_class=type(e).__name__)
                raise

            if store:
                await store.append_tool_result(run.run_id, turn_id=turn_id, tool_use_id=tc.id, name=tc.name, status=result.status, content=result.content)

            history.add_tool_result(tc.id, result)

        if history.needs_compression():
            await history.compress(shared["llm"])
            if store:
                await store.append(run.run_id, "MEMORY_UPDATED", {"summary": history._summary[:200]})

        return "continue"


class LLMCallNode(BaseNode):
    async def prep(self, shared: dict) -> dict:
        run: Run = shared["run"]
        history: HistoryManager = shared["history"]
        llm: "LLMAdapter" = shared["llm"]
        max_tokens = None
        if run.budget.max_input_tokens_per_call > 0:
            max_tokens = min(llm.max_context_tokens, run.budget.max_input_tokens_per_call)
        max_output_tokens = run.budget.max_output_tokens_per_call
        if max_output_tokens <= 0:
            max_output_tokens = 2048
        return {
            "messages": history.get_messages(max_tokens=max_tokens),
            "tools": shared["tools"].get_schemas(policy=run.tool_policy),
            "run": run,
            "llm": llm,
            "system": shared.get("system", ""),
            "max_output_tokens": max_output_tokens,
        }

    async def exec(self, prep: dict) -> "LLMResponse":
        run: Run = prep["run"]
        run.check_budget("llm_call")
        return await prep["llm"].generate(
            messages=prep["messages"],
            tools=prep["tools"],
            system=prep["system"],
            max_tokens=prep["max_output_tokens"],
        )

    async def post(self, shared: dict, resp: "LLMResponse") -> str:
        run: Run = shared["run"]
        store: FileEventStore = shared.get("store")
        history: HistoryManager = shared["history"]

        run.record_llm_usage(
            getattr(resp, "input_tokens", 0) or 0,
            getattr(resp, "output_tokens", 0) or 0,
        )

        if store:
            await store.append(run.run_id, "ASSISTANT_MESSAGE", {
                "content": resp.content,
                "tool_calls": [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in resp.tool_calls],
                "stop_reason": resp.stop_reason,
            })

        if resp.stop_reason == "end_turn" or not resp.tool_calls:
            shared["final_answer"] = resp.content or ""
            return "done"

        turn_id = _make_ulid()
        shared["turn_id"] = turn_id
        # Write into shared (HC-5: Flow.copy causes node instance fields to not persist)
        shared["llm_response"] = resp
        history.add_assistant_tool_calls(resp.content or "" if resp.content else None, resp.tool_calls, turn_id=turn_id)
        return "dispatch_tools"


class ToolDispatchNode(BaseNode):
    def __init__(self, max_concurrency: int = 4, on_tool_result: Callable | None = None):
        super().__init__()
        self.max_concurrency = max_concurrency
        self.on_tool_result = on_tool_result

    async def prep(self, shared: dict) -> "LLMResponse | None":
        return shared.get("llm_response")

    async def exec(self, resp: "LLMResponse | None") -> "LLMResponse | None":
        # exec cannot access shared; tool execution must happen in post
        return resp

    async def post(self, shared: dict, resp: "LLMResponse | None") -> str:
        if resp is None:
            shared["final_answer"] = shared.get("final_answer", "")
            return "done"

        run: Run = shared["run"]
        store: FileEventStore = shared.get("store")
        history: HistoryManager = shared["history"]
        tools: "ToolRegistry" = shared["tools"]
        turn_id: str = shared.get("turn_id") or _make_ulid()

        from ..tools.policy import SideEffectLevel

        # Write TOOL_USE events for all tool calls first
        for tc in resp.tool_calls:
            run.check_budget("tool_call", _tool_sig(tc.name, tc.args))
            if store:
                await store.append_tool_use(
                    run.run_id, turn_id=turn_id,
                    tool_use_id=tc.id, name=tc.name, args=tc.args,
                )

        # Partition into concurrent-safe vs sequential
        concurrent_safe = []
        sequential = []
        for tc in resp.tool_calls:
            tool = tools.get_tool(tc.name)
            level = tool.side_effect_level if tool else SideEffectLevel.DANGEROUS
            if level in (SideEffectLevel.READ_ONLY, SideEffectLevel.NETWORK):
                concurrent_safe.append(tc)
            else:
                sequential.append(tc)

        # Results dict keyed by tool_use_id to preserve original order
        results: dict[str, Any] = {}

        # Execute concurrent-safe tools in parallel
        if concurrent_safe:
            sem = asyncio.Semaphore(self.max_concurrency)

            async def _run_one(tc):
                async with sem:
                    try:
                        return tc.id, await tools.execute(tc.name, tc.args, run)
                    except Exception as e:
                        return tc.id, e

            gathered = await asyncio.gather(*[_run_one(tc) for tc in concurrent_safe])
            for tc_id, result in gathered:
                results[tc_id] = result

        # Execute sequential tools one by one
        for tc in sequential:
            try:
                results[tc.id] = await tools.execute(tc.name, tc.args, run)
            except Exception as e:
                results[tc.id] = e

        # Write results in original tool_calls order
        for tc in resp.tool_calls:
            result = results.get(tc.id)
            if isinstance(result, Exception):
                if store:
                    await store.append_tool_aborted(
                        run.run_id, turn_id=turn_id, tool_use_id=tc.id,
                        name=tc.name, reason=str(result), error_class=type(result).__name__,
                    )
                # Synthesize an error ToolResponse so history stays consistent
                from ..tools.registry import ToolResponse
                result = ToolResponse("error", str(result), type(result).__name__)
            else:
                # Redact if configured
                if store:
                    result = store.redact_tool_response(tc.name, result)
                if store:
                    await store.append_tool_result(
                        run.run_id, turn_id=turn_id, tool_use_id=tc.id,
                        name=tc.name, status=result.status, content=result.content,
                    )
                run.record_tool_result(tc.name, result.status)

            history.add_tool_result(tc.id, result, turn_id=turn_id)

            if self.on_tool_result is not None:
                try:
                    cb_result = self.on_tool_result({"tool_use_id": tc.id, "name": tc.name, "result": result})
                    if asyncio.iscoroutine(cb_result):
                        await cb_result
                except Exception:
                    pass  # callback errors must not interrupt main flow

        if history.needs_compression():
            await history.compress(shared["llm"])
            run.record_compression()
            if store:
                await store.append(run.run_id, "MEMORY_UPDATED", {"summary": history._summary[:200]})

        return "continue"
