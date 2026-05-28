import sys
import unittest
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from all_in_agents import (
    MCPToolProvider,
    Run,
    SSEMCPServer,
    SideEffectLevel,
    StdioMCPServer,
    StreamableHTTPMCPServer,
    ToolRegistry,
)


class FakeSession:
    def __init__(self, *, error=False, structured=None):
        self.initialized = False
        self.error = error
        self.structured = structured
        self.calls = []

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        tool = SimpleNamespace(
            name="echo",
            description="Echo text",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        return SimpleNamespace(tools=[tool])

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        content = [SimpleNamespace(type="text", text=f"{name}:{arguments['text']}")]
        return SimpleNamespace(
            content=content,
            isError=self.error,
            structuredContent=self.structured,
        )


class FakeMCPClientSession(FakeSession):
    def __init__(self, read, write):
        super().__init__()
        self.read = read
        self.write = write

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


def fake_mcp_modules() -> dict[str, ModuleType]:
    mcp_module = ModuleType("mcp")
    mcp_module.__path__ = []
    mcp_module.ClientSession = FakeMCPClientSession

    client_module = ModuleType("mcp.client")
    client_module.__path__ = []
    shared_module = ModuleType("mcp.shared")
    shared_module.__path__ = []

    return {
        "mcp": mcp_module,
        "mcp.client": client_module,
        "mcp.shared": shared_module,
    }


class MCPToolProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_registers_mcp_tools_into_registry(self):
        sessions = []

        @asynccontextmanager
        async def session_factory():
            session = FakeSession()
            sessions.append(session)
            yield session

        provider = MCPToolProvider(
            session_factory=session_factory,
            name_prefix="mcp_",
            side_effect_level=SideEffectLevel.NETWORK,
        )
        registry = ToolRegistry()

        registered = await provider.register_tools(registry)
        result = await registry.execute("mcp_echo", {"text": "hi"}, Run("run", "goal"))

        self.assertEqual([tool.name for tool in registered], ["mcp_echo"])
        self.assertEqual(result.status, "success")
        self.assertEqual(result.content, "echo:hi")
        self.assertTrue(sessions[0].initialized)
        self.assertTrue(sessions[1].initialized)
        self.assertEqual(sessions[1].calls, [("echo", {"text": "hi"})])

    async def test_can_reuse_persistent_session(self):
        session = FakeSession(structured={"ok": True})

        @asynccontextmanager
        async def session_factory():
            yield session

        provider = MCPToolProvider(
            session_factory=session_factory,
            side_effect_level=SideEffectLevel.NETWORK,
        )
        registry = ToolRegistry()

        async with provider:
            await provider.register(registry)
            result = await registry.execute("echo", {"text": "hi"}, Run("run", "goal"))

        self.assertEqual(result.status, "success")
        self.assertIn('"structured": {"ok": true}', result.content)
        self.assertEqual(session.calls, [("echo", {"text": "hi"})])

    async def test_remote_error_maps_to_tool_error(self):
        @asynccontextmanager
        async def session_factory():
            yield FakeSession(error=True)

        provider = MCPToolProvider(
            session_factory=session_factory,
            side_effect_level=SideEffectLevel.NETWORK,
        )
        registry = ToolRegistry()

        await provider.register(registry)
        result = await registry.execute("echo", {"text": "bad"}, Run("run", "goal"))

        self.assertEqual(result.status, "error")
        self.assertEqual(result.content, "echo:bad")

    def test_requires_server_or_session_factory(self):
        with self.assertRaises(ValueError):
            MCPToolProvider()

    def test_stdio_server_config_is_lightweight(self):
        server = StdioMCPServer("python", args=("-m", "server"), env={"A": "B"})
        self.assertEqual(server.command, "python")
        self.assertEqual(server.args, ("-m", "server"))

    def test_http_server_configs_are_lightweight(self):
        sse = SSEMCPServer("http://localhost:8000/sse", headers={"A": "B"})
        streamable = StreamableHTTPMCPServer(
            "http://localhost:8000/mcp",
            terminate_on_close=False,
        )

        self.assertEqual(sse.url, "http://localhost:8000/sse")
        self.assertEqual(sse.headers, {"A": "B"})
        self.assertFalse(streamable.terminate_on_close)

    async def test_mcp_tools_default_to_dangerous(self):
        @asynccontextmanager
        async def session_factory():
            yield FakeSession()

        provider = MCPToolProvider(session_factory=session_factory)
        registry = ToolRegistry()

        registered = await provider.register(registry)

        self.assertEqual(registered[0].side_effect_level, SideEffectLevel.DANGEROUS)

    async def test_sse_transport_creates_client_session(self):
        calls = []
        modules = fake_mcp_modules()
        sse_module = ModuleType("mcp.client.sse")

        @asynccontextmanager
        async def sse_client(**kwargs):
            calls.append(kwargs)
            yield ("sse-read", "sse-write")

        sse_module.sse_client = sse_client
        modules["mcp.client.sse"] = sse_module

        with patch.dict(sys.modules, modules):
            provider = MCPToolProvider(
                SSEMCPServer(
                    "http://example.test/sse",
                    headers={"Authorization": "Bearer token"},
                    timeout=1,
                    sse_read_timeout=2,
                )
            )
            tools = await provider.list_tools()

        self.assertEqual([tool.name for tool in tools], ["echo"])
        self.assertEqual(calls[0]["url"], "http://example.test/sse")
        self.assertEqual(calls[0]["headers"], {"Authorization": "Bearer token"})
        self.assertEqual(calls[0]["timeout"], 1.0)
        self.assertEqual(calls[0]["sse_read_timeout"], 2.0)

    async def test_streamable_http_transport_supports_legacy_sdk_signature(self):
        calls = []
        modules = fake_mcp_modules()
        streamable_module = ModuleType("mcp.client.streamable_http")

        @asynccontextmanager
        async def streamablehttp_client(**kwargs):
            calls.append(kwargs)
            yield ("http-read", "http-write", lambda: "session")

        streamable_module.streamablehttp_client = streamablehttp_client
        modules["mcp.client.streamable_http"] = streamable_module

        with patch.dict(sys.modules, modules):
            provider = MCPToolProvider(
                StreamableHTTPMCPServer(
                    "http://example.test/mcp",
                    headers={"A": "B"},
                    timeout=3,
                    sse_read_timeout=4,
                    terminate_on_close=False,
                )
            )
            tools = await provider.list_tools()

        self.assertEqual([tool.name for tool in tools], ["echo"])
        self.assertEqual(calls[0]["url"], "http://example.test/mcp")
        self.assertEqual(calls[0]["headers"], {"A": "B"})
        self.assertEqual(calls[0]["timeout"], 3)
        self.assertEqual(calls[0]["sse_read_timeout"], 4)
        self.assertFalse(calls[0]["terminate_on_close"])

    async def test_streamable_http_transport_supports_modern_sdk_signature(self):
        client_calls = []
        transport_calls = []
        modules = fake_mcp_modules()
        streamable_module = ModuleType("mcp.client.streamable_http")
        httpx_module = ModuleType("httpx")

        class Timeout:
            def __init__(self, timeout, *, read=None):
                self.timeout = timeout
                self.read = read

        class FakeHTTPClient:
            def __init__(self, **kwargs):
                client_calls.append(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        @asynccontextmanager
        async def streamable_http_client(url, *, http_client=None, terminate_on_close=True):
            transport_calls.append(
                {
                    "url": url,
                    "http_client": http_client,
                    "terminate_on_close": terminate_on_close,
                }
            )
            yield ("http-read", "http-write", lambda: "session")

        httpx_module.Timeout = Timeout
        httpx_module.AsyncClient = FakeHTTPClient
        streamable_module.streamable_http_client = streamable_http_client
        modules["httpx"] = httpx_module
        modules["mcp.client.streamable_http"] = streamable_module

        with patch.dict(sys.modules, modules):
            provider = MCPToolProvider(
                StreamableHTTPMCPServer(
                    "http://example.test/mcp",
                    headers={"A": "B"},
                    timeout=3,
                    sse_read_timeout=4,
                    terminate_on_close=False,
                    auth="auth",
                )
            )
            tools = await provider.list_tools()

        self.assertEqual([tool.name for tool in tools], ["echo"])
        self.assertEqual(client_calls[0]["headers"], {"A": "B"})
        self.assertEqual(client_calls[0]["timeout"].timeout, 3.0)
        self.assertEqual(client_calls[0]["timeout"].read, 4.0)
        self.assertEqual(client_calls[0]["auth"], "auth")
        self.assertTrue(client_calls[0]["follow_redirects"])
        self.assertEqual(transport_calls[0]["url"], "http://example.test/mcp")
        self.assertFalse(transport_calls[0]["terminate_on_close"])
