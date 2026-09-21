from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Self

from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import TextContent

from .config import ExternalToolConfig, HttpTransport, StdioTransport, read_secret
from .errors import InterventionRequired
from .store import Store
from .types import Role, ToolResult

ToolHandler = Callable[[dict[str, Any]], Awaitable[ToolResult]]
BUDGET_PAUSE_MESSAGE = "irh:budget-pause"
BUDGET_RESUME_MESSAGE = "irh:budget-resume"


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    source: str = "harness"

    def openai_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def _expand(value: str, context: dict[str, str]) -> str:
    try:
        return value.format_map(context)
    except KeyError as error:
        raise ValueError(f"Unknown external-tool template field: {error.args[0]}") from error


class MCPConnection:
    def __init__(self, config: ExternalToolConfig, context: dict[str, str]):
        self.config = config
        self.context = context
        self.client: Client | None = None
        self.lock = asyncio.Lock()
        self._stack = AsyncExitStack()
        self.server_info: dict[str, Any] | None = None

    async def open(self) -> None:
        transport = self.config.transport
        if isinstance(transport, StdioTransport):
            env = {key: _expand(value, self.context) for key, value in transport.env.items()}
            params = StdioServerParameters(
                command=_expand(transport.command, self.context),
                args=[_expand(value, self.context) for value in transport.args],
                env=env,
            )
            client = Client(params, read_timeout_seconds=300)
        elif isinstance(transport, HttpTransport):
            headers = {name: _expand(value, self.context) for name, value in transport.headers.items()}
            headers.update({name: read_secret(path) for name, path in transport.header_files.items()})
            http = await self._stack.enter_async_context(create_mcp_http_client(headers=headers))
            stream = streamable_http_client(_expand(transport.url, self.context), http_client=http)
            client = Client(stream, read_timeout_seconds=300)
        else:  # pragma: no cover - Pydantic makes this unreachable
            raise TypeError("Unknown MCP transport")
        self.client = await self._stack.enter_async_context(client)
        info = self.client.server_info
        self.server_info = info.model_dump(mode="json") if info is not None else None

    async def close(self) -> None:
        await self._stack.aclose()

    async def list_tools(self) -> list[Any]:
        assert self.client is not None
        cursor: str | None = None
        tools: list[Any] = []
        while True:
            async with self.lock:
                page = await self.client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return tools

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        pause_budget: Callable[[], None] | None = None,
        resume_budget: Callable[[], None] | None = None,
    ) -> ToolResult:
        assert self.client is not None
        paused = False

        async def progress(_progress: float, _total: float | None, message: str | None) -> None:
            nonlocal paused
            if message == BUDGET_PAUSE_MESSAGE and not paused and pause_budget is not None:
                pause_budget()
                paused = True
            elif message == BUDGET_RESUME_MESSAGE and paused and resume_budget is not None:
                resume_budget()
                paused = False

        try:
            async with self.lock:
                result = await self.client.call_tool(
                    name,
                    arguments,
                    progress_callback=progress if pause_budget is not None else None,
                )
        finally:
            if paused and resume_budget is not None:
                resume_budget()
        text_parts = [block.text for block in result.content if isinstance(block, TextContent)]
        structured = result.structured_content
        meta = dict(result.meta or {})
        return ToolResult(
            text="\n".join(text_parts)
            or (json.dumps(structured, separators=(",", ":")) if structured is not None else ""),
            structured=structured,
            is_error=bool(result.is_error),
            meta=meta,
        )


class MCPHub:
    """Connect enabled servers and expose their native tools as first-class model tools."""

    def __init__(
        self,
        configs: list[ExternalToolConfig],
        role: Role,
        context: dict[str, str],
        store: Store,
        project_id: str,
        abort: asyncio.Event,
        pause_budget: Callable[[], None] | None = None,
        resume_budget: Callable[[], None] | None = None,
    ):
        self.configs = configs
        self.role = role
        self.context = context
        self.store = store
        self.project_id = project_id
        self.abort = abort
        self.pause_budget = pause_budget
        self.resume_budget = resume_budget
        self.connections: dict[str, MCPConnection] = {}
        self.model_tools: list[AgentTool] = []
        self.monitors: list[asyncio.Task[None]] = []
        self.incident: InterventionRequired | None = None
        self.snapshot: list[dict[str, Any]] = []

    async def __aenter__(self) -> Self:
        try:
            for config in self.configs:
                connection = MCPConnection(config, self.context)
                await connection.open()
                self.connections[config.id] = connection
                all_tools = await connection.list_tools()
                allowed = set(config.roles.get(self.role, []))
                exposed: list[str] = []
                for declared in all_tools:
                    if declared.name not in allowed:
                        continue
                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", declared.name):
                        raise ValueError(
                            f"MCP tool name is not compatible with the configured model API: "
                            f"{config.id}.{declared.name}"
                        )
                    if declared.input_schema.get("type") != "object":
                        raise ValueError(
                            f"MCP tool {config.id}.{declared.name} must use an object input schema"
                        )
                    exposed.append(declared.name)
                    model_name = f"{config.id}__{declared.name}"
                    if len(model_name) > 64:
                        raise ValueError(f"Namespaced MCP tool name exceeds 64 characters: {model_name}")

                    async def handler(
                        arguments: dict[str, Any],
                        *,
                        _connection=connection,
                        _name=declared.name,
                        _config=config,
                    ) -> ToolResult:
                        result = await _connection.call(
                            _name,
                            arguments,
                            self.pause_budget,
                            self.resume_budget,
                        )
                        if _name in _config.pause_on_calls and not result.is_error:
                            raise InterventionRequired(_config.id, result.structured or result.text)
                        return result

                    self.model_tools.append(
                        AgentTool(
                            name=model_name,
                            description=declared.description or declared.title or declared.name,
                            input_schema=declared.input_schema,
                            handler=handler,
                            source=config.id,
                        )
                    )
                self.snapshot.append(
                    {
                        "id": config.id,
                        "configuration_id": config.configuration_id,
                        "server_info": connection.server_info,
                        "exposed_tools": exposed,
                    }
                )
                if config.monitor:
                    available = {tool.name for tool in all_tools}
                    if config.monitor.tool not in available:
                        raise ValueError(f"Monitor tool is absent from {config.id}: {config.monitor.tool}")
                    self.monitors.append(asyncio.create_task(self._monitor(config, connection)))
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def _monitor(self, config: ExternalToolConfig, connection: MCPConnection) -> None:
        assert config.monitor is not None
        while not self.abort.is_set():
            try:
                result = await connection.call(config.monitor.tool, {})
                if result.is_error:
                    raise RuntimeError(result.text)
                payload = result.structured
                if not isinstance(payload, dict):
                    try:
                        payload = json.loads(result.text)
                    except json.JSONDecodeError:
                        payload = {}
                blocked = payload.get(config.monitor.blocked_field) if isinstance(payload, dict) else None
                if blocked:
                    self.incident = InterventionRequired(config.id, blocked)
                    self.abort.set()
                    return
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - any monitor failure may require a safe pause
                if config.monitor.pause_on_error:
                    self.incident = InterventionRequired(config.id, {"monitor_error": str(error)})
                    self.abort.set()
                    return
            try:
                await asyncio.wait_for(self.abort.wait(), timeout=config.monitor.interval_seconds)
            except TimeoutError:
                pass

    async def __aexit__(self, _type: object, _value: object, _traceback: object) -> None:
        for task in self.monitors:
            task.cancel()
        await asyncio.gather(*self.monitors, return_exceptions=True)
        for connection in reversed(list(self.connections.values())):
            await connection.close()

    @property
    def guidance(self) -> str:
        return "\n".join(
            text.strip() for config in self.configs if (text := config.guidance.get(self.role, "")).strip()
        )
