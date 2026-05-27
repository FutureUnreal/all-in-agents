import tempfile
import unittest

from all_in_agents import (
    Agent,
    BatchNode,
    ConditionalNode,
    Flow,
    FlowHooks,
    LLMAdapter,
    LLMResponse,
    SideEffectLevel,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResponse,
)
from all_in_agents.history.compactor import CompactionResult


class StaticNode(BatchNode):
    def __init__(self, value=None, action="default"):
        super().__init__()
        self.value = value
        self.action = action

    async def exec_item(self, item):
        return item

    async def prep(self, shared):
        return self.value

    async def exec(self, prep_result):
        return prep_result

    async def post(self, shared, exec_result):
        shared.setdefault("results", []).append(exec_result)
        return self.action


class FailingNode(StaticNode):
    async def exec(self, prep_result):
        raise RuntimeError("should not run")


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


class RuntimeEnhancementTests(unittest.IsolatedAsyncioTestCase):
    async def test_flow_hooks_observe_node_lifecycle(self):
        seen = []

        hooks = FlowHooks(
            on_node_start=lambda ctx: seen.append(("start", ctx["node_name"])),
            on_node_end=lambda ctx: seen.append(("end", ctx["node_name"], ctx["action"])),
        )
        await Flow(hooks=hooks).run({}, StaticNode("ok"))

        self.assertEqual(seen, [("start", "StaticNode"), ("end", "StaticNode", "default")])

    async def test_conditional_node_skips_wrapped_node(self):
        skipped = StaticNode("skipped")
        conditional = ConditionalNode(FailingNode(), lambda shared: False)
        conditional - "skip" >> skipped

        shared = {}
        await Flow().run(shared, conditional)

        self.assertEqual(shared["results"], ["skipped"])

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
