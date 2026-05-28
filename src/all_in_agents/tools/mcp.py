from __future__ import annotations

import inspect
import json
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncContextManager, Callable

from .policy import SideEffectLevel
from .registry import Tool, ToolRegistry, ToolResponse


@dataclass(frozen=True)
class StdioMCPServer:
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | None = None


@dataclass(frozen=True)
class SSEMCPServer:
    url: str
    headers: dict[str, Any] | None = None
    timeout: float | timedelta = 5
    sse_read_timeout: float | timedelta = 60 * 5
    auth: Any | None = None


@dataclass(frozen=True)
class StreamableHTTPMCPServer:
    url: str
    headers: dict[str, Any] | None = None
    timeout: float | timedelta = 30
    sse_read_timeout: float | timedelta = 60 * 5
    terminate_on_close: bool = True
    auth: Any | None = None


MCPServerConfig = StdioMCPServer | SSEMCPServer | StreamableHTTPMCPServer


def _missing_mcp_error() -> ImportError:
    return ImportError("Install MCP support with: pip install 'all-in-agents[mcp]'")


def _get_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(by_alias=True, exclude_none=True))
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _seconds(value: float | timedelta) -> float:
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(value)


def _has_parameter(fn: Callable[..., Any], name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _content_block_to_dict(block: Any) -> dict:
    data = _jsonable(block)
    if isinstance(data, dict):
        return data
    return {
        "type": _get_attr(block, "type", type(block).__name__),
        "value": data,
    }


def _stringify_call_result(result: Any) -> str:
    structured = (
        _get_attr(result, "structuredContent")
        or _get_attr(result, "structured_content")
    )
    text_parts: list[str] = []
    rich_parts: list[dict] = []

    for block in _get_attr(result, "content", []) or []:
        block_type = _get_attr(block, "type", "")
        text = _get_attr(block, "text")
        if block_type == "text" and text is not None:
            text_parts.append(str(text))
        else:
            rich_parts.append(_content_block_to_dict(block))

    text = "\n".join(part for part in text_parts if part)
    if structured is None and not rich_parts:
        return text

    payload: dict[str, Any] = {}
    if text:
        payload["text"] = text
    if structured is not None:
        payload["structured"] = _jsonable(structured)
    if rich_parts:
        payload["content"] = rich_parts
    return json.dumps(payload, ensure_ascii=False)


@asynccontextmanager
async def _stdio_session(server: StdioMCPServer):
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as e:
        raise _missing_mcp_error() from e

    params_kwargs: dict[str, Any] = {
        "command": server.command,
        "args": list(server.args),
        "env": server.env,
    }
    if server.cwd is not None:
        if not _has_parameter(StdioServerParameters, "cwd"):
            raise ImportError(
                "MCP stdio cwd support requires mcp>=1.12.4. "
                "Reinstall with: pip install 'all-in-agents[mcp]'"
            )
        params_kwargs["cwd"] = server.cwd

    async with stdio_client(StdioServerParameters(**params_kwargs)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def _sse_session(server: SSEMCPServer):
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
    except ImportError as e:
        raise _missing_mcp_error() from e

    async with sse_client(
        url=server.url,
        headers=server.headers,
        timeout=_seconds(server.timeout),
        sse_read_timeout=_seconds(server.sse_read_timeout),
        auth=server.auth,
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def _streamable_http_session(server: StreamableHTTPMCPServer):
    try:
        from mcp import ClientSession
        try:
            from mcp.client.streamable_http import streamable_http_client
        except ImportError:
            from mcp.client.streamable_http import streamablehttp_client as streamable_http_client
    except ImportError as e:
        raise _missing_mcp_error() from e

    if _has_parameter(streamable_http_client, "http_client"):
        try:
            import httpx
        except ImportError as e:
            raise _missing_mcp_error() from e

        http_client = httpx.AsyncClient(
            headers=server.headers,
            timeout=httpx.Timeout(
                _seconds(server.timeout),
                read=_seconds(server.sse_read_timeout),
            ),
            auth=server.auth,
            follow_redirects=True,
        )
        async with http_client:
            async with streamable_http_client(
                url=server.url,
                http_client=http_client,
                terminate_on_close=server.terminate_on_close,
            ) as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        return

    async with streamable_http_client(
        url=server.url,
        headers=server.headers,
        timeout=server.timeout,
        sse_read_timeout=server.sse_read_timeout,
        terminate_on_close=server.terminate_on_close,
        auth=server.auth,
    ) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


class MCPToolProvider:
    """Register tools exposed by an MCP server into a ToolRegistry."""

    def __init__(
        self,
        server: MCPServerConfig | None = None,
        *,
        session_factory: Callable[[], AsyncContextManager[Any]] | None = None,
        name_prefix: str = "",
        namespace: str = "",
        side_effect_level: SideEffectLevel = SideEffectLevel.DANGEROUS,
        max_calls: int | None = None,
        max_wall_ms: int | None = None,
    ):
        if server is None and session_factory is None:
            raise ValueError("MCPToolProvider requires a server or session_factory")
        self.server = server
        self.session_factory = session_factory
        self.name_prefix = name_prefix
        self.namespace = namespace
        self.side_effect_level = side_effect_level
        self.max_calls = max_calls
        self.max_wall_ms = max_wall_ms
        self._stack: AsyncExitStack | None = None
        self._session: Any = None

    async def __aenter__(self) -> "MCPToolProvider":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @asynccontextmanager
    async def _new_session(self):
        if self.session_factory is not None:
            async with self.session_factory() as session:
                initialize = getattr(session, "initialize", None)
                if callable(initialize):
                    await initialize()
                yield session
            return

        assert self.server is not None
        if isinstance(self.server, StdioMCPServer):
            session_cm = _stdio_session(self.server)
        elif isinstance(self.server, SSEMCPServer):
            session_cm = _sse_session(self.server)
        elif isinstance(self.server, StreamableHTTPMCPServer):
            session_cm = _streamable_http_session(self.server)
        else:
            raise TypeError(f"Unsupported MCP server config: {type(self.server).__name__}")

        async with session_cm as session:
            yield session

    async def start(self) -> None:
        if self._session is not None:
            return
        self._stack = AsyncExitStack()
        self._session = await self._stack.enter_async_context(self._new_session())

    async def close(self) -> None:
        if self._stack is None:
            return
        try:
            await self._stack.aclose()
        finally:
            self._stack = None
            self._session = None

    async def _with_session(self, callback: Callable[[Any], Any]) -> Any:
        if self._session is not None:
            return await callback(self._session)
        async with self._new_session() as session:
            return await callback(session)

    async def list_tools(self) -> list[Any]:
        async def _list(session):
            result = await session.list_tools()
            return list(_get_attr(result, "tools", result) or [])

        return await self._with_session(_list)

    async def call_tool(self, name: str, arguments: dict) -> ToolResponse:
        async def _call(session):
            result = await session.call_tool(name, arguments=arguments)
            status = "error" if bool(_get_attr(result, "isError", False)) else "success"
            return ToolResponse(status=status, content=_stringify_call_result(result))

        return await self._with_session(_call)

    async def register_tools(self, registry: ToolRegistry) -> list[Tool]:
        registered: list[Tool] = []
        for remote_tool in await self.list_tools():
            remote_name = str(_get_attr(remote_tool, "name", ""))
            if not remote_name:
                continue
            registered_name = f"{self.name_prefix}{remote_name}"
            description = _get_attr(remote_tool, "description", "") or ""
            input_schema = (
                _get_attr(remote_tool, "inputSchema")
                or _get_attr(remote_tool, "input_schema")
                or {"type": "object", "properties": {}}
            )

            async def _execute(args: dict, run, _remote_name=remote_name) -> ToolResponse:
                return await self.call_tool(_remote_name, args)

            tool = Tool(
                name=registered_name,
                description=description,
                input_schema=_jsonable(input_schema),
                side_effect_level=self.side_effect_level,
                execute=_execute,
                max_calls=self.max_calls,
                max_wall_ms=self.max_wall_ms,
                namespace=self.namespace,
            )
            registry.register(tool)
            registered.append(tool)
        return registered

    async def register(self, registry: ToolRegistry) -> list[Tool]:
        return await self.register_tools(registry)
