import tempfile
import unittest
from pathlib import Path

from all_in_agents import (
    Agent,
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


if __name__ == "__main__":
    unittest.main()
