import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from all_in_agents.adapters import AnthropicAdapter, OpenAIAdapter
from all_in_agents.adapters.base import LLMError


class FakeOpenAIClient:
    last = None
    response = None
    error = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        FakeOpenAIClient.last = self

    async def create(self, **kwargs):
        if FakeOpenAIClient.error:
            raise FakeOpenAIClient.error
        return FakeOpenAIClient.response

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
        FakeOpenAIClient.error = None

        with patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)}):
            adapter = OpenAIAdapter(model="test-model", api_key="test-key")
            response = await adapter.generate([{"role": "user", "content": "hi"}])

        self.assertEqual(response.content, "done")
        self.assertTrue(FakeOpenAIClient.last.closed)

    async def test_openai_client_closes_after_error(self):
        FakeOpenAIClient.response = None
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
