import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from all_in_agents import (
    Agent,
    AgentTurnDecision,
    BatchNode,
    BaseNode,
    ConditionalNode,
    ErrorPolicy,
    FileEventStore,
    Flow,
    FlowHooks,
    JsonCheckpointStore,
    LLMAdapter,
    LLMResponse,
    LLMStreamEvent,
    Node,
    RetryPolicy,
    Run,
    RunContext,
    SideEffectLevel,
    SubFlowNode,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResponse,
)
from all_in_agents import HistoryManager
from all_in_agents.history.compactor import CompactionResult


class StaticNode(BaseNode):
    def __init__(self, value=None, action="default", checkpoint_id=None):
        super().__init__(checkpoint_id=checkpoint_id)
        self.value = value
        self.action = action

    async def prep(self, ctx):
        return self.value

    async def exec(self, prep_result, ctx):
        return prep_result

    async def post(self, ctx, exec_result):
        ctx.state.setdefault("results", []).append(exec_result)
        return self.action


class FailingNode(StaticNode):
    async def exec(self, prep_result, ctx):
        raise RuntimeError("should not run")


class FlakyNode(Node):
    def __init__(self):
        super().__init__(retry_policy=RetryPolicy(max_attempts=3, retry_exceptions=(ValueError,)))
        self.calls = 0

    async def prep(self, ctx):
        return "recovered"

    async def exec(self, prep_result, ctx):
        self.calls += 1
        if self.calls == 1:
            raise ValueError("retryable")
        return prep_result

    async def post(self, ctx, exec_result):
        ctx.state["retry_attempt"] = ctx.retry_attempt
        ctx.state.setdefault("results", []).append(exec_result)
        return "default"


class FlowFlakyNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.calls = 0

    async def exec(self, prep_result, ctx):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("flow retry")
        return "flow-recovered"

    async def post(self, ctx, exec_result):
        ctx.state["flow_attempt"] = ctx.attempt
        ctx.state.setdefault("results", []).append(exec_result)
        return "default"


class EchoBatchNode(BatchNode):
    async def prep(self, ctx):
        return [1, 2, 3]

    async def exec_item(self, item, ctx):
        return item * 2

    async def post(self, ctx, exec_result):
        ctx.state["batch_result"] = exec_result
        return "default"


class FakeLLM(LLMAdapter):
    model_id = "fake"
    max_context_tokens = 1000

    def __init__(self):
        self.calls = 0

    async def generate(self, messages, tools=None, system="", max_tokens=2048, options=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="calling tool",
                tool_calls=[ToolCall(id="tool_1", name="echo", args={"text": "hi"})],
                input_tokens=1,
                output_tokens=1,
                stop_reason="tool_use",
            )
        return LLMResponse(
            content="done",
            tool_calls=[],
            input_tokens=1,
            output_tokens=1,
            stop_reason="end_turn",
        )


class FailingAfterToolLLM(LLMAdapter):
    model_id = "failing"
    max_context_tokens = 1000

    def __init__(self):
        self.calls = 0

    async def generate(self, messages, tools=None, system="", max_tokens=2048, options=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="calling tool",
                tool_calls=[ToolCall(id="tool_1", name="echo", args={"text": "hi"})],
                input_tokens=1,
                output_tokens=1,
                stop_reason="tool_use",
            )
        raise RuntimeError("model stopped")


class FinishLLM(LLMAdapter):
    model_id = "finish"
    max_context_tokens = 1000

    async def generate(self, messages, tools=None, system="", max_tokens=2048, options=None):
        return LLMResponse(
            content="resumed",
            tool_calls=[],
            input_tokens=1,
            output_tokens=1,
            stop_reason="end_turn",
        )


class NoToolThenToolLLM(LLMAdapter):
    model_id = "no-tool-then-tool"
    max_context_tokens = 1000

    def __init__(self):
        self.calls = 0
        self.messages = []

    async def generate(self, messages, tools=None, system="", max_tokens=2048, options=None):
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            return LLMResponse(
                content="text-only answer",
                tool_calls=[],
                input_tokens=1,
                output_tokens=1,
                stop_reason="end_turn",
            )
        if self.calls == 2:
            return LLMResponse(
                content="calling tool",
                tool_calls=[ToolCall(id="tool_1", name="echo", args={"text": "hi"})],
                input_tokens=1,
                output_tokens=1,
                stop_reason="tool_use",
            )
        return LLMResponse(
            content="done",
            tool_calls=[],
            input_tokens=1,
            output_tokens=1,
            stop_reason="end_turn",
        )


class AlwaysNoToolLLM(LLMAdapter):
    model_id = "always-no-tool"
    max_context_tokens = 1000

    def __init__(self):
        self.calls = 0

    async def generate(self, messages, tools=None, system="", max_tokens=2048, options=None):
        self.calls += 1
        return LLMResponse(
            content=f"text-only answer {self.calls}",
            tool_calls=[],
            input_tokens=1,
            output_tokens=1,
            stop_reason="end_turn",
        )


class StreamingLLM(LLMAdapter):
    model_id = "streaming"
    max_context_tokens = 1000

    def __init__(self):
        self.calls = 0

    async def generate(self, messages, tools=None, system="", max_tokens=2048, options=None):
        raise AssertionError("streaming agent should call stream")

    async def stream(self, messages, tools=None, system="", max_tokens=2048, options=None):
        self.calls += 1
        if self.calls == 1:
            tool_call = ToolCall(id="tool_1", name="echo", args={"text": "hi"})
            yield LLMStreamEvent(type="text_delta", delta="calling")
            yield LLMStreamEvent(type="tool_call_delta", tool_call=tool_call)
            yield LLMStreamEvent(
                type="message",
                response=LLMResponse(
                    content="calling",
                    tool_calls=[tool_call],
                    input_tokens=2,
                    output_tokens=3,
                    stop_reason="tool_use",
                ),
            )
            return

        yield LLMStreamEvent(type="text_delta", delta="do")
        yield LLMStreamEvent(type="text_delta", delta="ne")
        yield LLMStreamEvent(
            type="message",
            response=LLMResponse(
                content="done",
                tool_calls=[],
                input_tokens=4,
                output_tokens=5,
                stop_reason="end_turn",
            ),
        )


class RecordingCompactor:
    def __init__(self):
        self.llm = None

    async def compact_turns(self, llm, turns, *, max_context_tokens, target_tokens=None):
        self.llm = llm
        return CompactionResult(summary="summary", kept_turns=turns[-1:], used_fallback=False)


async def echo_tool(args, run):
    return ToolResponse("success", args["text"])


def make_registry():
    registry = ToolRegistry()
    registry.register(Tool(
        name="echo",
        description="Echo text",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        side_effect_level=SideEffectLevel.READ_ONLY,
        execute=echo_tool,
    ))
    return registry


def make_run_context(tmp, *, state=None):
    return RunContext(
        run=Run(run_id="test-run", goal="test"),
        llm=FakeLLM(),
        tools=make_registry(),
        history=HistoryManager(),
        store=FileEventStore(tmp),
        state=state or {},
    )


class RuntimeEnhancementTests(unittest.IsolatedAsyncioTestCase):
    async def test_flow_hooks_observe_node_lifecycle(self):
        seen = []

        hooks = FlowHooks(
            on_node_start=lambda ctx: seen.append(("start", ctx.node_name)),
            on_node_end=lambda ctx: seen.append(("end", ctx.node_name, ctx.action)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            await Flow(hooks=hooks).run(make_run_context(tmp), StaticNode("ok"))

        self.assertEqual(seen, [("start", "StaticNode"), ("end", "StaticNode", "default")])

    async def test_conditional_node_skips_wrapped_node(self):
        skipped = StaticNode("skipped")
        conditional = ConditionalNode(FailingNode(), lambda ctx: False)
        conditional - "skip" >> skipped

        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_run_context(tmp)
            await Flow().run(ctx, conditional)

        self.assertEqual(ctx.state["results"], ["skipped"])

    async def test_error_policy_can_skip_failed_node(self):
        failed = FailingNode()
        recovered = StaticNode("recovered")
        failed - "skip" >> recovered

        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_run_context(tmp)
            await Flow(error_policy=ErrorPolicy.skip()).run(ctx, failed)

        self.assertEqual(ctx.state["results"], ["recovered"])

    async def test_subflow_node_runs_inside_parent_context(self):
        sub_start = StaticNode("sub")
        wrapper = SubFlowNode(sub_start)
        tail = StaticNode("tail")
        wrapper >> tail

        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_run_context(tmp)
            await Flow().run(ctx, wrapper)

        self.assertEqual(ctx.state["results"], ["sub", "tail"])

    async def test_node_retry_policy_can_filter_exceptions(self):
        node = FlakyNode()

        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_run_context(tmp)
            await Flow().run(ctx, node)

        self.assertEqual(node.calls, 2)
        self.assertEqual(ctx.state["results"], ["recovered"])
        self.assertEqual(ctx.state["retry_attempt"], 1)

    async def test_flow_error_policy_can_retry_by_exception_type(self):
        node = FlowFlakyNode()

        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_run_context(tmp)
            await Flow(
                error_policy=ErrorPolicy.retry(
                    max_attempts=2,
                    retry_exceptions=(TimeoutError,),
                    jitter=False,
                )
            ).run(ctx, node)

        self.assertEqual(node.calls, 2)
        self.assertEqual(ctx.state["results"], ["flow-recovered"])
        self.assertEqual(ctx.state["flow_attempt"], 1)

    async def test_batch_node_passes_context_to_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_run_context(tmp)
            await Flow().run(ctx, EchoBatchNode())

        self.assertEqual(ctx.state["batch_result"], [2, 4, 6])

    async def test_flow_can_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonCheckpointStore(tmp)
            first = StaticNode("first")
            pause = FailingNode(checkpoint_id="resume")
            first >> pause
            ctx = make_run_context(tmp)

            with self.assertRaises(RuntimeError):
                await Flow().run(ctx, first, checkpoint_store=store)

            checkpoint = store.load(ctx.run.run_id)
            self.assertEqual(checkpoint.next_node_id, "resume")

            first_again = StaticNode("first")
            resumed = StaticNode("second", checkpoint_id="resume")
            first_again >> resumed
            resumed_ctx = make_run_context(tmp)
            await Flow().run(
                resumed_ctx,
                first_again,
                checkpoint_store=store,
                resume_checkpoint=checkpoint,
            )

        self.assertEqual(resumed_ctx.state["results"], ["first", "second"])

    def test_child_run_consumes_parent_budget_ledger(self):
        parent = Run(run_id="parent", goal="parent")
        child = parent.spawn_child(run_id="child", goal="child")

        child.check_budget("llm_call")
        child.check_budget("tool_call", "echo:{}")
        child.record_llm_usage(3, 5)

        self.assertEqual(child.llm_calls, 1)
        self.assertEqual(parent.llm_calls, 1)
        self.assertEqual(parent.tool_calls, 1)
        self.assertEqual(parent.input_tokens_total, 3)
        self.assertEqual(parent.output_tokens_total, 5)

    async def test_child_agent_consumes_parent_budget_ledger(self):
        parent = Run(run_id="parent", goal="parent")

        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(FakeLLM(), make_registry(), run_dir=tmp)
            await agent.run("use the tool", parent_run=parent)

        self.assertEqual(parent.llm_calls, 2)
        self.assertEqual(parent.tool_calls, 1)

    async def test_agent_can_return_in_memory_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(
                FakeLLM(),
                make_registry(),
                run_dir=tmp,
                include_trajectory=True,
            )
            result = await agent.run("use the tool")

        self.assertEqual(result.final_answer, "done")
        self.assertIsNotNone(result.trajectory)
        event_types = [event["type"] for event in result.trajectory]
        self.assertIn("TOOL_USE", event_types)
        self.assertIn("TOOL_RESULT", event_types)

    async def test_agent_can_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_agent = Agent(
                FailingAfterToolLLM(),
                make_registry(),
                run_dir=tmp,
            )
            failed = await first_agent.run("use the tool", checkpoint=True)

            self.assertEqual(failed.status, "error")
            self.assertTrue(Path(failed.checkpoint_path).exists())

            resumed_agent = Agent(
                FinishLLM(),
                make_registry(),
                run_dir=tmp,
            )
            resumed = await resumed_agent.run("use the tool", resume_from=failed.run_id)

        self.assertEqual(resumed.final_answer, "resumed")
        self.assertEqual(resumed.status, "success")

    async def test_agent_uses_dedicated_compression_llm(self):
        main_llm = FakeLLM()
        compression_llm = FakeLLM()
        compactor = RecordingCompactor()

        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(
                main_llm,
                make_registry(),
                run_dir=tmp,
                history_compactor=compactor,
                history_compress_threshold_tokens=1,
                compression_llm=compression_llm,
            )
            await agent.run("use the tool")

        self.assertIs(compactor.llm, compression_llm)

    async def test_agent_stream_yields_model_and_tool_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(StreamingLLM(), make_registry(), run_dir=tmp)
            events = [event async for event in agent.stream("use the tool")]

        event_types = [event.type for event in events]
        self.assertIn("run_started", event_types)
        self.assertIn("llm_started", event_types)
        self.assertIn("text_delta", event_types)
        self.assertIn("tool_call_delta", event_types)
        self.assertIn("tool_called", event_types)
        self.assertIn("tool_result", event_types)
        self.assertIn("run_stopped", event_types)
        self.assertEqual(
            "".join(event.data["delta"] for event in events if event.type == "text_delta"),
            "callingdone",
        )
        self.assertEqual(events[-1].type, "run_stopped")
        self.assertEqual(events[-1].data["final_answer"], "done")

    async def test_agent_stream_text_filters_text_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(StreamingLLM(), make_registry(), run_dir=tmp)
            text = [chunk async for chunk in agent.stream_text("use the tool")]

        self.assertEqual("".join(text), "callingdone")

    async def test_agent_turn_gate_can_stop_before_tool_dispatch(self):
        seen = []

        async def gate(turn):
            seen.append([tc.name for tc in turn.response.tool_calls])
            return AgentTurnDecision.stop(
                final_answer="waiting for approval",
                stop_reason="human_gate",
            )

        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(
                FakeLLM(),
                make_registry(),
                run_dir=tmp,
                on_turn=gate,
                include_trajectory=True,
            )
            result = await agent.run("use the tool")

        self.assertEqual(seen, [["echo"]])
        self.assertEqual(result.final_answer, "waiting for approval")
        self.assertEqual(result.status, "interrupted")
        self.assertEqual(result.stop_reason, "human_gate")
        event_types = [event["type"] for event in result.trajectory]
        self.assertIn("CONTROL_DECISION", event_types)
        self.assertNotIn("TOOL_USE", event_types)

    async def test_agent_turn_gate_can_replace_response(self):
        def gate(turn):
            return AgentTurnDecision.replace(
                replace(
                    turn.response,
                    content="blocked tool call",
                    tool_calls=[],
                    stop_reason="end_turn",
                )
            )

        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(
                FakeLLM(),
                make_registry(),
                run_dir=tmp,
                on_turn=gate,
                include_trajectory=True,
            )
            result = await agent.run("use the tool")

        self.assertEqual(result.final_answer, "blocked tool call")
        self.assertEqual(result.status, "success")
        event_types = [event["type"] for event in result.trajectory]
        self.assertIn("CONTROL_DECISION", event_types)
        self.assertNotIn("TOOL_USE", event_types)

    async def test_agent_turn_gate_can_retry_with_injected_message(self):
        llm = NoToolThenToolLLM()
        seen_metrics = []

        def gate(turn):
            seen_metrics.append(turn.metrics.copy())
            if turn.metrics["tool_calls"] == 0 and not turn.response.tool_calls:
                return AgentTurnDecision.retry(
                    "Use tools before giving the final answer.",
                    max_retries=2,
                )
            return AgentTurnDecision.continue_()

        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(
                llm,
                make_registry(),
                run_dir=tmp,
                on_turn=gate,
                include_trajectory=True,
            )
            result = await agent.run("use the tool")

        self.assertEqual(result.final_answer, "done")
        self.assertEqual(result.status, "success")
        self.assertEqual(llm.calls, 3)
        self.assertEqual(result.metrics["tool_calls"], 1)
        self.assertEqual(seen_metrics[0]["llm_calls"], 1)
        self.assertEqual(seen_metrics[0]["tool_calls"], 0)
        self.assertEqual(
            llm.messages[1][-1]["content"],
            "Use tools before giving the final answer.",
        )
        event_types = [event["type"] for event in result.trajectory]
        self.assertIn("ASSISTANT_REJECTED", event_types)
        self.assertIn("CONTROL_DECISION", event_types)
        self.assertIn("TOOL_USE", event_types)

    async def test_agent_turn_gate_retry_limit_stops_run(self):
        llm = AlwaysNoToolLLM()

        def gate(turn):
            return AgentTurnDecision.retry(
                "Use tools before giving the final answer.",
                max_retries=1,
            )

        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(
                llm,
                make_registry(),
                run_dir=tmp,
                on_turn=gate,
                include_trajectory=True,
            )
            result = await agent.run("use the tool")

        self.assertEqual(llm.calls, 2)
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.stop_reason, "turn_retry_exhausted")
        self.assertNotIn("TOOL_USE", [event["type"] for event in result.trajectory])
        control_events = [
            event for event in result.trajectory
            if event["type"] == "CONTROL_DECISION"
        ]
        self.assertEqual(control_events[-1]["action"], "stop")
        self.assertEqual(control_events[-1]["requested_action"], "retry")
        self.assertEqual(control_events[-1]["retry_count"], 1)


if __name__ == "__main__":
    unittest.main()
