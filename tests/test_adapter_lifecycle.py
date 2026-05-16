import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from all_in_agents import Agent
from all_in_agents.adapters import AnthropicAdapter, GenerationOptions, OpenAIAdapter
from all_in_agents.adapters.base import LLMError


class FakeOpenAIClient:
    last = None
    response = None
    responses_response = None
    error = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.last_chat_kwargs = None
        self.last_responses_kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.responses = SimpleNamespace(create=self.create_response)
        FakeOpenAIClient.last = self

    async def create(self, **kwargs):
        self.last_chat_kwargs = kwargs
        if FakeOpenAIClient.error:
            raise FakeOpenAIClient.error
        return FakeOpenAIClient.response

    async def create_response(self, **kwargs):
        self.last_responses_kwargs = kwargs
        if FakeOpenAIClient.error:
            raise FakeOpenAIClient.error
        return FakeOpenAIClient.responses_response

    async def close(self):
        self.closed = True


class FakeAnthropicClient:
    last = None
    response = None
    error = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.messages = SimpleNamespace(create=self.create)
        FakeAnthropicClient.last = self

    async def create(self, **kwargs):
        if FakeAnthropicClient.error:
            raise FakeAnthropicClient.error
        return FakeAnthropicClient.response

    async def close(self):
        self.closed = True


def openai_response():
    message = SimpleNamespace(content="done", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2)
    return SimpleNamespace(choices=[choice], usage=usage)


def responses_response():
    usage = SimpleNamespace(input_tokens=3, output_tokens=2)
    return SimpleNamespace(output_text="done", output=[], usage=usage)


def responses_tool_response():
    usage = SimpleNamespace(input_tokens=3, output_tokens=2)
    call = SimpleNamespace(
        type="function_call",
        call_id="call_1",
        name="lookup",
        arguments='{"query":"agent"}',
    )
    return SimpleNamespace(output_text=None, output=[call], usage=usage)


def anthropic_response():
    block = SimpleNamespace(type="text", text="done")
    usage = SimpleNamespace(input_tokens=3, output_tokens=2)
    return SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn")


def bad_request_error():
    error = Exception("bad request")
    error.response = SimpleNamespace(status_code=400)
    return error


class AdapterLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_client_closes_after_success(self):
        FakeOpenAIClient.response = openai_response()
        FakeOpenAIClient.responses_response = responses_response()
        FakeOpenAIClient.error = None

        with patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)}):
            adapter = OpenAIAdapter(model="test-model", api_key="test-key")
            response = await adapter.generate([{"role": "user", "content": "hi"}])

        self.assertEqual(response.content, "done")
        self.assertTrue(FakeOpenAIClient.last.closed)

    async def test_openai_client_closes_after_error(self):
        FakeOpenAIClient.response = None
        FakeOpenAIClient.responses_response = None
        FakeOpenAIClient.error = bad_request_error()

        with patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)}):
            adapter = OpenAIAdapter(model="test-model", api_key="test-key", max_retries=1)
            with self.assertRaises(LLMError):
                await adapter.generate([{"role": "user", "content": "hi"}])

        self.assertTrue(FakeOpenAIClient.last.closed)

    async def test_anthropic_client_closes_after_success(self):
        FakeAnthropicClient.response = anthropic_response()
        FakeAnthropicClient.error = None

        with patch.dict(sys.modules, {"anthropic": SimpleNamespace(AsyncAnthropic=FakeAnthropicClient)}):
            adapter = AnthropicAdapter(model="test-model", api_key="test-key")
            response = await adapter.generate([{"role": "user", "content": "hi"}])

        self.assertEqual(response.content, "done")
        self.assertTrue(FakeAnthropicClient.last.closed)

    async def test_openai_chat_options_map_to_chat_completion_kwargs(self):
        FakeOpenAIClient.response = openai_response()
        FakeOpenAIClient.error = None
        schema = {"type": "json_object"}

        with patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)}):
            adapter = OpenAIAdapter(
                model="test-model",
                api_key="test-key",
                temperature=0.2,
                response_format=schema,
                reasoning_effort="medium",
                model_kwargs={"seed": 7},
            )
            await adapter.generate([{"role": "user", "content": "hi"}], max_tokens=123)

        kwargs = FakeOpenAIClient.last.last_chat_kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["max_completion_tokens"], 123)
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["response_format"], schema)
        self.assertEqual(kwargs["reasoning_effort"], "medium")
        self.assertEqual(kwargs["seed"], 7)

    async def test_openai_call_options_override_adapter_defaults(self):
        FakeOpenAIClient.response = openai_response()
        FakeOpenAIClient.error = None

        with patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)}):
            adapter = OpenAIAdapter(
                model="test-model",
                api_key="test-key",
                options=GenerationOptions(temperature=0.8, extra={"seed": 1}),
            )
            await adapter.generate(
                [{"role": "user", "content": "hi"}],
                options=GenerationOptions(temperature=0.1, extra={"seed": 2}),
            )

        kwargs = FakeOpenAIClient.last.last_chat_kwargs
        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["seed"], 2)

    async def test_openai_responses_options_map_to_responses_kwargs(self):
        FakeOpenAIClient.responses_response = responses_response()
        FakeOpenAIClient.error = None
        schema = {
            "type": "json_schema",
            "json_schema": {"name": "Answer", "schema": {"type": "object"}},
        }

        with patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)}):
            adapter = OpenAIAdapter(
                model="test-model",
                api_key="test-key",
                api="responses",
                response_format=schema,
                reasoning_effort="low",
                model_kwargs={"metadata": {"suite": "test"}},
            )
            response = await adapter.generate(
                [{"role": "user", "content": "hi"}],
                system="You are concise.",
                max_tokens=77,
            )

        kwargs = FakeOpenAIClient.last.last_responses_kwargs
        self.assertEqual(response.content, "done")
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["instructions"], "You are concise.")
        self.assertEqual(kwargs["max_output_tokens"], 77)
        self.assertEqual(kwargs["text"], {"format": {"type": "json_schema", "name": "Answer", "schema": {"type": "object"}}})
        self.assertEqual(kwargs["reasoning"], {"effort": "low"})
        self.assertEqual(kwargs["metadata"], {"suite": "test"})

    async def test_openai_responses_parser_extracts_function_calls(self):
        FakeOpenAIClient.responses_response = responses_tool_response()
        FakeOpenAIClient.error = None

        with patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)}):
            adapter = OpenAIAdapter(model="test-model", api_key="test-key", api="responses")
            response = await adapter.generate([{"role": "user", "content": "hi"}])

        self.assertEqual(response.stop_reason, "tool_use")
        self.assertEqual(response.tool_calls[0].id, "call_1")
        self.assertEqual(response.tool_calls[0].name, "lookup")
        self.assertEqual(response.tool_calls[0].args, {"query": "agent"})

    def test_agent_quick_forwards_generation_options_to_openai_adapter(self):
        schema = {"type": "json_object"}
        agent = Agent.quick(
            model="test-model",
            tools="none",
            api="responses",
            response_format=schema,
            reasoning_effort="medium",
            model_kwargs={"metadata": {"suite": "test"}},
        )

        llm = agent._llm
        self.assertIsInstance(llm, OpenAIAdapter)
        self.assertEqual(llm._api, "responses")
        self.assertEqual(llm._options.response_format, schema)
        self.assertEqual(llm._options.reasoning_effort, "medium")
        self.assertEqual(llm._options.extra["metadata"], {"suite": "test"})

    async def test_anthropic_client_closes_after_error(self):
        FakeAnthropicClient.response = None
        FakeAnthropicClient.error = bad_request_error()

        with patch.dict(sys.modules, {"anthropic": SimpleNamespace(AsyncAnthropic=FakeAnthropicClient)}):
            adapter = AnthropicAdapter(model="test-model", api_key="test-key", max_retries=1)
            with self.assertRaises(LLMError):
                await adapter.generate([{"role": "user", "content": "hi"}])

        self.assertTrue(FakeAnthropicClient.last.closed)


if __name__ == "__main__":
    unittest.main()
