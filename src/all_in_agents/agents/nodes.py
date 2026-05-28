from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Any, Callable

from ..core.context import NodeContext
from ..core.node import BaseNode
from ..utils import make_ulid as _make_ulid

if TYPE_CHECKING:
    from ..adapters.base import LLMResponse, ToolCall


async def _emit_stream_event(ctx: NodeContext, event_type: str, data: dict) -> None:
    callback = ctx.run_context.stream_callback
    if callback is None:
        return
    from .streaming import AgentStreamEvent

    result = callback(AgentStreamEvent(type=event_type, run_id=ctx.run.run_id, data=data))
    if asyncio.iscoroutine(result):
        await result


def _coerce_llm_response(value: Any) -> "LLMResponse | None":
    if value is None:
        return None
    if hasattr(value, "tool_calls"):
        return value
    if not isinstance(value, dict):
        return None

    from ..adapters.base import LLMResponse, ToolCall

    tool_calls = [
        ToolCall(
            id=tc.get("id", ""),
            name=tc.get("name", ""),
            args=tc.get("args") or {},
        )
        for tc in value.get("tool_calls", []) or []
        if isinstance(tc, dict)
    ]
    return LLMResponse(
        content=value.get("content"),
        tool_calls=tool_calls,
        input_tokens=int(value.get("input_tokens", 0) or 0),
        output_tokens=int(value.get("output_tokens", 0) or 0),
        stop_reason=value.get("stop_reason", "end_turn"),
    )


def _tool_sig(name: str, args: dict) -> str:
    payload = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{name}:{h}"


_TURN_RETRY_COUNT_KEY = "_agent_turn_retry_count"
_TURN_RETRY_REASON = "turn_retry_exhausted"


async def _apply_turn_gate(
    ctx: NodeContext,
    resp: "LLMResponse",
) -> tuple["LLMResponse", str]:
    callback = ctx.run_context.on_turn
    if callback is None:
        return resp, "continue"

    from .control import AgentTurn, normalize_turn_decision
    from ..core.run import RunStatus

    prep = ctx.prep_result if isinstance(ctx.prep_result, dict) else {}
    turn = AgentTurn(
        run_id=ctx.run.run_id,
        goal=ctx.run.goal,
        step_index=ctx.step_index,
        response=resp,
        messages=list(prep.get("messages") or []),
        tools=list(prep.get("tools") or []),
        system=ctx.system,
        state=ctx.state,
        metrics=ctx.run.snapshot_metrics(),
    )

    raw_decision = callback(turn)
    if asyncio.iscoroutine(raw_decision):
        raw_decision = await raw_decision
    decision = normalize_turn_decision(raw_decision)
    effective_response = decision.response or resp
    response_replaced = effective_response is not resp
    action = decision.action
    retry_count = int(ctx.state.get(_TURN_RETRY_COUNT_KEY, 0) or 0)
    max_retries = decision.max_retries
    if max_retries is None:
        max_retries = ctx.run_context.turn_max_retries
    else:
        max_retries = int(max_retries)

    if action == "retry":
        if not isinstance(decision.inject_message, str) or not decision.inject_message:
            raise ValueError("AgentTurnDecision.retry requires a non-empty inject_message")
        if max_retries < 0:
            raise ValueError("Agent turn max_retries must be >= 0")
        if retry_count >= max_retries:
            action = "stop"
            decision.stop_reason = _TURN_RETRY_REASON
            decision.status = RunStatus.INCOMPLETE.value
            decision.final_answer = (
                decision.final_answer
                if decision.final_answer is not None
                else f"[stopped: turn retry limit exhausted after {retry_count} retries]"
            )

    if ctx.store:
        await ctx.store.append(ctx.run.run_id, "CONTROL_DECISION", {
            "step_index": ctx.step_index,
            "action": action,
            "requested_action": decision.action,
            "response_replaced": response_replaced,
            "stop_reason": decision.stop_reason if action == "stop" else "",
            "status": decision.status if action == "stop" else "",
            "inject_message": decision.inject_message if decision.action == "retry" else "",
            "retry_count": retry_count + 1 if action == "retry" else retry_count,
            "max_retries": max_retries if decision.action == "retry" else None,
        })

    if decision.action == "retry" and ctx.store:
        await ctx.store.append(ctx.run.run_id, "ASSISTANT_REJECTED", {
            "step_index": ctx.step_index,
            "content": effective_response.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "args": tc.args}
                for tc in effective_response.tool_calls
            ],
            "stop_reason": effective_response.stop_reason,
        })

    if action == "continue":
        ctx.state.pop(_TURN_RETRY_COUNT_KEY, None)
        return effective_response, "continue"
    if action == "retry":
        ctx.state[_TURN_RETRY_COUNT_KEY] = retry_count + 1
        ctx.history.add("user", decision.inject_message)
        return effective_response, "retry"
    if action == "stop":
        ctx.final_answer = (
            decision.final_answer
            if decision.final_answer is not None
            else effective_response.content or ""
        )
        ctx.state["_agent_control_stop"] = {
            "stop_reason": decision.stop_reason,
            "status": decision.status,
        }
        return effective_response, "stop"

    raise ValueError(f"Unknown agent turn decision action: {decision.action}")


class LLMCallNode(BaseNode):
    async def prep(self, ctx: NodeContext) -> dict:
        max_tokens = None
        if ctx.run.budget.max_input_tokens_per_call > 0:
            max_tokens = min(ctx.llm.max_context_tokens, ctx.run.budget.max_input_tokens_per_call)
        max_output_tokens = ctx.run.budget.max_output_tokens_per_call
        if max_output_tokens <= 0:
            max_output_tokens = 2048
        return {
            "messages": ctx.history.get_messages(max_tokens=max_tokens),
            "tools": ctx.tools.get_schemas(policy=ctx.run.tool_policy),
            "max_output_tokens": max_output_tokens,
        }

    async def exec(self, prep: dict, ctx: NodeContext) -> "LLMResponse":
        ctx.run.check_budget("llm_call")
        if ctx.run_context.stream_callback is None:
            return await ctx.llm.generate(
                messages=prep["messages"],
                tools=prep["tools"],
                system=ctx.system,
                max_tokens=prep["max_output_tokens"],
            )

        await _emit_stream_event(ctx, "llm_started", {
            "max_tokens": prep["max_output_tokens"],
            "tool_names": [tool.get("name", "") for tool in prep["tools"]],
        })

        final_response = None
        text_parts: list[str] = []
        tool_calls: list["ToolCall"] = []
        async for event in ctx.llm.stream(
            messages=prep["messages"],
            tools=prep["tools"],
            system=ctx.system,
            max_tokens=prep["max_output_tokens"],
        ):
            if event.type == "text_delta" and event.delta:
                text_parts.append(event.delta)
                await _emit_stream_event(ctx, "text_delta", {"delta": event.delta})
            elif event.type == "tool_call_delta":
                if event.tool_call is not None:
                    tool_calls.append(event.tool_call)
                await _emit_stream_event(ctx, "tool_call_delta", {
                    "delta": event.delta,
                    "tool_call": _tool_call_to_dict(event.tool_call) if event.tool_call else None,
                    "tool_call_delta": event.tool_call_delta,
                })
            elif event.type == "message" and event.response is not None:
                final_response = event.response

        if final_response is not None:
            return final_response

        from ..adapters.base import LLMResponse

        return LLMResponse(
            content="".join(text_parts) or None,
            tool_calls=tool_calls,
            input_tokens=0,
            output_tokens=0,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    async def post(self, ctx: NodeContext, resp: "LLMResponse") -> str:
        ctx.run.record_llm_usage(
            getattr(resp, "input_tokens", 0) or 0,
            getattr(resp, "output_tokens", 0) or 0,
        )

        resp, control_action = await _apply_turn_gate(ctx, resp)

        if control_action == "retry":
            return "retry"

        if ctx.store:
            await ctx.store.append(ctx.run.run_id, "ASSISTANT_MESSAGE", {
                "content": resp.content,
                "tool_calls": [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in resp.tool_calls],
                "stop_reason": resp.stop_reason,
            })

        if control_action == "stop":
            return "done"

        if resp.stop_reason == "end_turn" or not resp.tool_calls:
            ctx.final_answer = resp.content or ""
            return "done"

        turn_id = _make_ulid()
        ctx.state["turn_id"] = turn_id
        ctx.state["llm_response"] = resp
        ctx.history.add_assistant_tool_calls(resp.content or "" if resp.content else None, resp.tool_calls, turn_id=turn_id)
        return "dispatch_tools"


def _tool_call_to_dict(tool_call: "ToolCall | None") -> dict | None:
    if tool_call is None:
        return None
    return {"id": tool_call.id, "name": tool_call.name, "args": tool_call.args}


class ToolDispatchNode(BaseNode):
    def __init__(self, max_concurrency: int = 4, on_tool_result: Callable | None = None):
        super().__init__()
        self.max_concurrency = max_concurrency
        self.on_tool_result = on_tool_result

    async def prep(self, ctx: NodeContext) -> "LLMResponse | None":
        return _coerce_llm_response(ctx.state.get("llm_response"))

    async def exec(self, resp: "LLMResponse | None", ctx: NodeContext) -> "LLMResponse | None":
        return resp

    async def post(self, ctx: NodeContext, resp: "LLMResponse | None") -> str:
        if resp is None:
            return "done"

        turn_id: str = ctx.state.get("turn_id") or _make_ulid()

        from ..tools.policy import SideEffectLevel

        # Write TOOL_USE events for all tool calls first
        for tc in resp.tool_calls:
            ctx.run.check_budget("tool_call", _tool_sig(tc.name, tc.args))
            if ctx.store:
                await ctx.store.append_tool_use(
                    ctx.run.run_id, turn_id=turn_id,
                    tool_use_id=tc.id, name=tc.name, args=tc.args,
                )

        # Partition into concurrent-safe vs sequential
        concurrent_safe = []
        sequential = []
        for tc in resp.tool_calls:
            tool = ctx.tools.get_tool(tc.name)
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
                        return tc.id, await ctx.tools.execute(tc.name, tc.args, ctx.run)
                    except Exception as e:
                        return tc.id, e

            gathered = await asyncio.gather(*[_run_one(tc) for tc in concurrent_safe])
            for tc_id, result in gathered:
                results[tc_id] = result

        # Execute sequential tools one by one
        for tc in sequential:
            try:
                results[tc.id] = await ctx.tools.execute(tc.name, tc.args, ctx.run)
            except Exception as e:
                results[tc.id] = e

        # Write results in original tool_calls order
        for tc in resp.tool_calls:
            result = results.get(tc.id)
            if isinstance(result, Exception):
                if ctx.store:
                    await ctx.store.append_tool_aborted(
                        ctx.run.run_id, turn_id=turn_id, tool_use_id=tc.id,
                        name=tc.name, reason=str(result), error_class=type(result).__name__,
                    )
                # Synthesize an error ToolResponse so history stays consistent
                from ..tools.registry import ToolResponse
                result = ToolResponse("error", str(result), type(result).__name__)
            else:
                # Redact if configured
                if ctx.store:
                    result = ctx.store.redact_tool_response(tc.name, result)
                if ctx.store:
                    await ctx.store.append_tool_result(
                        ctx.run.run_id, turn_id=turn_id, tool_use_id=tc.id,
                        name=tc.name, status=result.status, content=result.content,
                    )
                ctx.run.record_tool_result(tc.name, result.status)

            ctx.history.add_tool_result(tc.id, result, turn_id=turn_id)

            if self.on_tool_result is not None:
                try:
                    cb_result = self.on_tool_result({"tool_use_id": tc.id, "name": tc.name, "result": result})
                    if asyncio.iscoroutine(cb_result):
                        await cb_result
                except Exception:
                    pass  # callback errors must not interrupt main flow

        if ctx.history.needs_compression():
            await ctx.history.compress(ctx.run_context.effective_compression_llm)
            ctx.run.record_compression()
            if ctx.store:
                await ctx.store.append(ctx.run.run_id, "MEMORY_UPDATED", {"summary": ctx.history._summary[:200]})

        ctx.state.pop("llm_response", None)
        ctx.state.pop("turn_id", None)
        return "continue"
