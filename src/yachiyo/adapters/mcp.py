"""Minimal Streamable HTTP MCP client for explicitly trusted servers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from yachiyo.adapters.storage import Database
from yachiyo.config import AppSettings, SecretStore
from yachiyo.config.models import MCPServerSettings
from yachiyo.domain import AccessLevel, RequestPermission
from yachiyo.ports.mcp import RegisteredTool, ToolExecutionContext, ToolResult
from yachiyo.ports.model import ToolDefinition

_TOOL_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_PROTOCOL_VERSION = "2025-11-25"


class MCPError(RuntimeError):
    pass


class _SessionExpired(MCPError):
    pass


@dataclass(slots=True)
class _ServerClient:
    settings: MCPServerSettings
    session: aiohttp.ClientSession
    bearer: str
    session_id: str | None = None
    request_id: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    connected: bool = False
    last_error: str = ""
    protocol_version: str | None = None

    async def initialize(self) -> None:
        async with self.lock:
            await self._initialize_locked()

    async def _initialize_locked(self) -> None:
        if self.connected:
            return
        self.session_id = None
        self.protocol_version = None
        result = await self._request_locked(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "yachiyo", "version": "2.0.0"},
            },
        )
        if not isinstance(result, dict):
            raise MCPError(f"{self.settings.name}: invalid initialize result")
        negotiated = result.get("protocolVersion")
        if negotiated != _PROTOCOL_VERSION:
            raise MCPError(f"{self.settings.name}: unsupported negotiated protocol {negotiated!r}")
        self.protocol_version = negotiated
        await self._notification_locked("notifications/initialized", {})
        self.connected = True
        self.last_error = ""

    async def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        pages = 0
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = await self._request("tools/list", params)
            if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
                raise MCPError(f"{self.settings.name}: invalid tools/list result")
            tools.extend(item for item in result["tools"] if isinstance(item, dict))
            cursor = result.get("nextCursor")
            pages += 1
            if not cursor:
                return tools
            if not isinstance(cursor, str) or pages >= 100 or len(tools) > 10_000:
                raise MCPError(f"{self.settings.name}: invalid tools pagination")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = await self._request("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            raise MCPError(f"{self.settings.name}: invalid tool result")
        content: list[str] = []
        unsupported = False
        for item in result.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                content.append(str(item.get("text", "")))
            elif isinstance(item, dict):
                unsupported = True
                content.append(f"[unsupported MCP content type: {item.get('type', 'unknown')}]")
        structured = result.get("structuredContent")
        if structured is not None:
            content.append(json.dumps(structured, ensure_ascii=False))
        text = "\n".join(part for part in content if part)
        encoded = text.encode()
        if len(encoded) > self.settings.max_result_bytes:
            text = encoded[: self.settings.max_result_bytes].decode("utf-8", errors="ignore")
            text += "\n[tool result truncated]"
        return ToolResult(
            not bool(result.get("isError")) and not unsupported,
            text or "tool returned no text",
        )

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        async with self.lock:
            await self._initialize_locked()
            try:
                return await self._request_locked(method, params)
            except _SessionExpired:
                self.connected = False
                self.session_id = None
                self.protocol_version = None
                await self._initialize_locked()
                return await self._request_locked(method, params)

    async def _request_locked(self, method: str, params: dict[str, Any]) -> Any:
        self.request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params,
        }
        response = await self._post(payload, expected_id=self.request_id)
        if response.get("id") != self.request_id:
            raise MCPError(f"{self.settings.name}: mismatched JSON-RPC response")
        if response.get("error") is not None:
            error = response["error"]
            message = error.get("message", "MCP error") if isinstance(error, dict) else str(error)
            raise MCPError(f"{self.settings.name}: {message}")
        return response.get("result")

    async def _notification_locked(self, method: str, params: dict[str, Any]) -> None:
        await self._post(
            {"jsonrpc": "2.0", "method": method, "params": params},
            response_optional=True,
        )

    async def _post(
        self,
        payload: dict[str, Any],
        *,
        response_optional: bool = False,
        expected_id: int | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self.bearer:
            headers["Authorization"] = f"Bearer {self.bearer}"
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        try:
            async with self.session.post(
                str(self.settings.url),
                json=payload,
                headers=headers,
                allow_redirects=False,
            ) as response:
                if response.status in {202, 204} and response_optional:
                    return {}
                if response.status == 404 and self.session_id:
                    raise _SessionExpired(f"{self.settings.name}: MCP session expired")
                if response.status != 200:
                    self.connected = False
                    self.last_error = f"HTTP {response.status}"
                    raise MCPError(f"{self.settings.name}: HTTP {response.status}")
                returned_session = response.headers.get("Mcp-Session-Id")
                if returned_session:
                    self.session_id = returned_session
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
                if content_type == "text/event-stream":
                    return await self._read_sse(
                        response, expected_id, response_optional=response_optional
                    )
                body = await self._read_limited(response)
        except (aiohttp.ClientError, TimeoutError) as exc:
            self.connected = False
            self.session_id = None
            self.protocol_version = None
            self.last_error = type(exc).__name__
            raise MCPError(f"{self.settings.name}: transport failure") from exc
        if not body and response_optional:
            return {}
        try:
            decoded = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MCPError(f"{self.settings.name}: invalid JSON response") from exc
        if not isinstance(decoded, dict):
            raise MCPError(f"{self.settings.name}: invalid JSON-RPC envelope")
        return decoded

    async def _read_limited(self, response: aiohttp.ClientResponse) -> bytes:
        maximum = self.settings.max_result_bytes + 256 * 1024
        declared = response.content_length
        if declared is not None and declared > maximum:
            raise MCPError(f"{self.settings.name}: response too large")
        body = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            body.extend(chunk)
            if len(body) > maximum:
                raise MCPError(f"{self.settings.name}: response too large")
        return bytes(body)

    async def _read_sse(
        self,
        response: aiohttp.ClientResponse,
        expected_id: int | None,
        *,
        response_optional: bool,
    ) -> dict[str, Any]:
        maximum = self.settings.max_result_bytes + 256 * 1024
        consumed = 0
        data_lines: list[str] = []
        while True:
            raw_line = await response.content.readline()
            if not raw_line:
                break
            consumed += len(raw_line)
            if consumed > maximum:
                raise MCPError(f"{self.settings.name}: response too large")
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue
            if line or not data_lines:
                continue
            try:
                decoded = json.loads("\n".join(data_lines))
            except json.JSONDecodeError as exc:
                raise MCPError(f"{self.settings.name}: invalid SSE JSON") from exc
            data_lines.clear()
            if not isinstance(decoded, dict):
                continue
            if "method" in decoded and "id" in decoded:
                result = (
                    {"jsonrpc": "2.0", "id": decoded["id"], "result": {}}
                    if decoded["method"] == "ping"
                    else {
                        "jsonrpc": "2.0",
                        "id": decoded["id"],
                        "error": {
                            "code": -32601,
                            "message": "server request is not supported by this client",
                        },
                    }
                )
                await self._post(result, response_optional=True)
                continue
            if ("result" in decoded or "error" in decoded) and (
                expected_id is None or decoded.get("id") == expected_id
            ):
                return decoded
        if response_optional:
            return {}
        raise MCPError("MCP SSE response contained no matching JSON-RPC result")

    async def close(self) -> None:
        if self.session_id:
            headers = {
                "Accept": "application/json, text/event-stream",
                "Mcp-Session-Id": self.session_id,
            }
            if self.protocol_version:
                headers["MCP-Protocol-Version"] = self.protocol_version
            if self.bearer:
                headers["Authorization"] = f"Bearer {self.bearer}"
            try:
                async with self.session.delete(
                    str(self.settings.url), headers=headers, allow_redirects=False
                ):
                    pass
            except (aiohttp.ClientError, TimeoutError):
                pass
        self.connected = False
        self.session_id = None
        self.protocol_version = None


class TrustedHTTPToolRegistry:
    def __init__(
        self,
        settings: AppSettings,
        secrets: SecretStore,
        database: Database,
    ):
        self.settings = settings
        self.secrets = secrets
        self.database = database
        self._clients: dict[str, _ServerClient] = {}
        self._tools: dict[str, RegisteredTool] = {}
        self._lifecycle_lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    async def refresh(self) -> None:
        async with self._lifecycle_lock:
            clients: dict[str, _ServerClient] = {}
            tools: dict[str, RegisteredTool] = {}
            try:
                for server in self.settings.mcp.servers:
                    if not server.enabled:
                        continue
                    bearer = (
                        self.secrets.read(server.bearer_secret, required=True)
                        if server.bearer_secret
                        else ""
                    )
                    timeout = aiohttp.ClientTimeout(total=server.timeout_seconds)
                    session = aiohttp.ClientSession(
                        timeout=timeout,
                        cookie_jar=aiohttp.DummyCookieJar(),
                    )
                    client = _ServerClient(server, session, bearer)
                    clients[server.name] = client
                    try:
                        discovered = await client.list_tools()
                        self._register(server, discovered, tools)
                    except MCPError as exc:
                        client.last_error = str(exc)
                        self._logger.warning("MCP discovery failed: %s", exc)
            except BaseException:
                await self._close_clients(clients)
                raise
            previous = self._clients
            self._clients = clients
            self._tools = tools
            await self._close_clients(previous)

    def _register(
        self,
        server: MCPServerSettings,
        discovered: list[dict[str, Any]],
        target: dict[str, RegisteredTool],
    ) -> None:
        allow = set(server.allowed_tools)
        for item in discovered:
            original = item.get("name")
            if not isinstance(original, str) or original not in allow:
                continue
            annotations = item.get("annotations") or {}
            advertised_read_only = annotations.get("readOnlyHint") is True
            locally_read_only = original in server.trusted_read_only_tools and advertised_read_only
            if server.read_only and not locally_read_only:
                continue
            exposed = _TOOL_NAME.sub("_", f"{server.name}__{original}")[:64]
            if not exposed or exposed in target:
                continue
            schema = item.get("inputSchema")
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            elif server.actor_scoped_workspace:
                schema = self._hide_workspace_argument(schema, original)
            target[exposed] = RegisteredTool(
                definition=ToolDefinition(
                    exposed,
                    str(item.get("description") or original)[:2000],
                    schema,
                ),
                server=server.name,
                original_name=original,
                access=AccessLevel(server.access),
                read_only=locally_read_only,
            )

    @staticmethod
    def _hide_workspace_argument(schema: dict[str, Any], original_name: str) -> dict[str, Any]:
        hidden = "name" if original_name == "workspace_create" else "workspace"
        result = dict(schema)
        properties = dict(result.get("properties") or {})
        properties.pop(hidden, None)
        result["properties"] = properties
        required = result.get("required")
        if isinstance(required, list):
            result["required"] = [item for item in required if item != hidden]
        return result

    def _arguments_for_context(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        client = self._clients.get(tool.server)
        if client is None or not client.settings.actor_scoped_workspace:
            return dict(arguments)
        workspace = (
            "user-" + hashlib.blake2b(context.actor_key.encode(), digest_size=16).hexdigest()
        )
        result = dict(arguments)
        if tool.original_name == "workspace_create":
            result["name"] = workspace
        else:
            result["workspace"] = workspace
        return result

    def available(self, permission: RequestPermission) -> tuple[RegisteredTool, ...]:
        return tuple(tool for tool in self._tools.values() if permission.allows(tool.access))

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None or not context.permission.allows(tool.access):
            return ToolResult(False, "tool is unavailable for this request")
        scoped_arguments = self._arguments_for_context(tool, arguments, context)
        if not tool.read_only:
            approval_id = await self.database.create_tool_approval(
                context.conversation_id,
                context.actor_key,
                name,
                scoped_arguments,
            )
            visible_arguments = json.dumps(scoped_arguments, ensure_ascii=False, sort_keys=True)
            return ToolResult(
                False,
                f"Approval required for tool {name} with arguments: {visible_arguments}. "
                f"Use /approve {approval_id} or /deny {approval_id} in the originating "
                "conversation within 10 minutes.",
                {"approval_id": approval_id, "status": "approval_required"},
            )
        return await self._execute(tool, scoped_arguments)

    async def resolve_approval(
        self,
        approval_id: str,
        approve: bool,
        context: ToolExecutionContext,
    ) -> ToolResult:
        pending = await self.database.pending_tool_approval(
            approval_id,
            context.actor_key,
            context.conversation_id,
        )
        if pending is None:
            return ToolResult(False, "Approval is invalid, expired, or already resolved.")
        tool = self._tools.get(pending["tool_name"])
        if tool is None or not context.permission.allows(tool.access):
            return ToolResult(False, "Tool is no longer available for this request.")
        record = await self.database.resolve_tool_approval(
            approval_id,
            context.actor_key,
            context.conversation_id,
            approve=approve,
        )
        if record is None:
            return ToolResult(False, "Approval is invalid, expired, or already resolved.")
        if not record["approved"]:
            return ToolResult(True, "Tool execution was denied.")
        try:
            result = await self._execute(tool, record["arguments"])
        except BaseException:
            await self.database.finish_tool_approval(approval_id, ok=False)
            raise
        else:
            await self.database.finish_tool_approval(approval_id, ok=result.ok)
            return result

    async def _execute(self, tool: RegisteredTool, arguments: dict[str, Any]) -> ToolResult:
        async with self._lifecycle_lock:
            client = self._clients.get(tool.server)
            if client is None:
                return ToolResult(False, "MCP server is disconnected")
            try:
                return await client.call_tool(tool.original_name, arguments)
            except MCPError as exc:
                return ToolResult(False, str(exc))

    def status(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "name": name,
                "state": "connected" if client.connected else "degraded",
                "error": client.last_error,
            }
            for name, client in self._clients.items()
        )

    async def close(self) -> None:
        async with self._lifecycle_lock:
            clients, self._clients = self._clients, {}
            self._tools.clear()
            await self._close_clients(clients)

    @staticmethod
    async def _close_clients(clients: dict[str, _ServerClient]) -> None:
        for client in clients.values():
            await client.close()
            await client.session.close()
