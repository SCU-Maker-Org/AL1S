from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.config import MCPConfig
from src.config import MCPServerConfig as ParsedMCPServerConfig
from src.infra import mcp as mcp_module
from src.infra.mcp import (
    MEDIA_CAPTURE_NONCE_ARGUMENT,
    MEDIA_CAPTURE_OWNER_ARGUMENT,
    MCPServerConfig,
    MCPService,
)


class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


def _install_fake_client(
    monkeypatch,
    *,
    call_result=None,
    listed_tools=None,
    calls=None,
):
    class FakeSession:
        def __init__(self, read, write):
            self.read = read
            self.write = write

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def initialize(self):
            return None

        async def list_tools(self, cursor=None):
            return SimpleNamespace(tools=list(listed_tools or []))

        async def call_tool(self, name, arguments):
            if calls is not None:
                calls.append((name, arguments))
            return call_result

    monkeypatch.setattr(mcp_module, "MCP_AVAILABLE", True)
    monkeypatch.setattr(
        mcp_module,
        "stdio_client",
        lambda server_params: _AsyncContext(("read", "write")),
    )
    monkeypatch.setattr(mcp_module, "ClientSession", FakeSession)


def _tool(name, *, read_only=None):
    annotations = None if read_only is None else {"readOnlyHint": read_only}
    return SimpleNamespace(
        name=name,
        description=f"{name} description",
        inputSchema={"type": "object"},
        annotations=annotations,
    )


def test_mcp_config_rejects_duplicate_server_names():
    with pytest.raises(ValidationError, match="MCP服务器名称重复: duplicate"):
        MCPConfig(
            servers=[
                ParsedMCPServerConfig(name="duplicate", command="one"),
                ParsedMCPServerConfig(name="duplicate", command="two"),
            ]
        )


def test_mcp_server_config_resolves_enabled_environment_reference(monkeypatch):
    monkeypatch.setenv("AL1S_TEST_MCP_TOKEN", "resolved-token")

    server = ParsedMCPServerConfig(
        name="search",
        command="search-server",
        env={
            "TOKEN": "${AL1S_TEST_MCP_TOKEN}",
            "LITERAL": "prefix-${AL1S_TEST_MCP_TOKEN}",
        },
    )

    assert server.env == {
        "TOKEN": "resolved-token",
        "LITERAL": "prefix-${AL1S_TEST_MCP_TOKEN}",
    }


def test_mcp_server_config_requires_enabled_environment_reference(monkeypatch):
    monkeypatch.delenv("AL1S_MISSING_MCP_TOKEN", raising=False)

    with pytest.raises(
        ValidationError, match="MCP环境变量未设置: AL1S_MISSING_MCP_TOKEN"
    ):
        ParsedMCPServerConfig(
            name="search",
            command="search-server",
            env={"TOKEN": "${AL1S_MISSING_MCP_TOKEN}"},
        )

    disabled = ParsedMCPServerConfig(
        name="search",
        command="search-server",
        enabled=False,
        env={"TOKEN": "${AL1S_MISSING_MCP_TOKEN}"},
    )
    assert disabled.env["TOKEN"] == "${AL1S_MISSING_MCP_TOKEN}"


def test_read_only_tool_filter_applies_include_and_exclude_patterns():
    service = MCPService()
    policy = MCPServerConfig(
        name="filesystem",
        command="filesystem-server",
        include_tools=["read_*"],
        exclude_tools=["*_secret"],
        read_only=True,
    )
    readable = _tool("read_file", read_only=True)
    excluded = _tool("read_secret", read_only=True)
    writable = _tool("read_file", read_only=False)
    unannotated = _tool("read_file")
    outside_allowlist = _tool("write_file", read_only=True)

    assert service._include_tool(readable, policy) is True
    assert service._include_tool(excluded, policy) is False
    assert service._include_tool(writable, policy) is False
    assert service._include_tool(unannotated, policy) is False
    assert service._include_tool(outside_allowlist, policy) is False


def test_conflicting_tool_names_receive_stable_server_aliases():
    service = MCPService()
    server = MCPServerConfig(name="docs", command="docs-server")

    first_alias = service._exposed_tool_name(server, "lookup", {"lookup"})
    second_alias = service._exposed_tool_name(
        server, "lookup", {"lookup", "docs__lookup"}
    )

    assert first_alias == "docs__lookup"
    assert second_alias == "docs__lookup_2"


@pytest.mark.asyncio
async def test_remove_server_clears_status_tools_and_errors():
    service = MCPService()
    server = MCPServerConfig(
        name="docs",
        command="docs-server",
        access="public",
    )
    service.servers[server.name] = server
    service.connected_servers.add(server.name)
    service.server_errors[server.name] = "stale error"
    service.tools[server.name] = {
        "lookup": {
            "description": "lookup",
            "schema": {},
            "server": server.name,
            "remote_name": "lookup",
        }
    }

    status = service.get_server_status()[server.name]
    assert status["connected"] is True
    assert status["tools_count"] == 1
    assert status["tools"] == ["lookup"]
    assert status["error"] == "stale error"

    assert await service.remove_server(server.name) is True
    assert service.get_server_status() == {}
    assert service.get_available_tools() == {}
    assert await service.remove_server(server.name) is False


@pytest.mark.asyncio
async def test_call_tool_routes_alias_to_remote_name_and_reports_mcp_error(
    monkeypatch,
):
    result = SimpleNamespace(
        content=[SimpleNamespace(text="upstream rejected request")],
        isError=True,
    )
    calls = []
    _install_fake_client(monkeypatch, call_result=result, calls=calls)
    service = MCPService()
    server = MCPServerConfig(
        name="docs",
        command="docs-server",
        access="public",
    )
    service.servers[server.name] = server
    service.tools[server.name] = {
        "docs__lookup": {
            "description": "lookup",
            "schema": {},
            "server": server.name,
            "remote_name": "lookup",
        }
    }

    response = await service.call_tool("docs__lookup", {"query": "MCP"})

    assert response == "工具调用失败: upstream rejected request"
    assert calls == [("lookup", {"query": "MCP"})]


@pytest.mark.asyncio
async def test_call_tool_truncates_large_result(monkeypatch):
    result = SimpleNamespace(
        content=[SimpleNamespace(text="abcdefghij")],
        isError=False,
    )
    _install_fake_client(monkeypatch, call_result=result)
    service = MCPService()
    server = MCPServerConfig(
        name="docs",
        command="docs-server",
        access="public",
        max_result_chars=5,
    )
    service.servers[server.name] = server
    service.tools[server.name] = {
        "lookup": {
            "description": "lookup",
            "schema": {},
            "server": server.name,
            "remote_name": "lookup",
        }
    }

    response = await service.call_tool("lookup", {})

    assert response == "abcde\n\n[结果已截断，省略 5 个字符]"


@pytest.mark.asyncio
async def test_admin_tool_is_hidden_and_cannot_be_called_by_public_user(
    monkeypatch,
):
    service = MCPService()
    server = MCPServerConfig(
        name="admin-tools",
        command="admin-server",
        access="admin",
    )
    service.servers[server.name] = server
    service.connected_servers.add(server.name)
    service.tools[server.name] = {
        "dangerous_action": {
            "description": "admin only",
            "schema": {},
            "server": server.name,
            "remote_name": "dangerous_action",
            "access": "admin",
        }
    }

    monkeypatch.setattr(mcp_module, "MCP_AVAILABLE", True)

    def fail_if_connected(server_params):
        message = "unauthorized calls must not start an MCP process"
        raise AssertionError(message)

    monkeypatch.setattr(mcp_module, "stdio_client", fail_if_connected)

    assert service.get_available_tools(caller_access="public") == {}
    assert service.get_tools_for_llm(caller_access="public") == []
    assert service.get_server_status(caller_access="public") == {}
    admin_tools = service.get_available_tools(caller_access="admin")
    assert "dangerous_action" in admin_tools

    response = await service.call_tool(
        "dangerous_action",
        {},
        caller_access="public",
    )

    assert response == "工具调用失败: 无权调用 dangerous_action"


@pytest.mark.asyncio
async def test_media_tool_binding_is_injected_and_model_values_are_overridden(
    monkeypatch,
):
    result = SimpleNamespace(
        content=[SimpleNamespace(text="generated without artifact in this unit test")],
        structuredContent=None,
        isError=False,
    )
    calls = []
    _install_fake_client(monkeypatch, call_result=result, calls=calls)
    service = MCPService()
    server = MCPServerConfig(name="media", command="media-server", access="public")
    service.servers[server.name] = server
    service.tools[server.name] = {
        "generate_image": {
            "description": "image",
            "schema": {},
            "server": server.name,
            "remote_name": "generate_image",
            "access": "public",
        }
    }
    token = service.begin_media_capture("telegram:10:10:1:1")
    capture = service._media_capture.get()

    response = await service.call_tool(
        "generate_image",
        {
            "prompt": "diagram",
            MEDIA_CAPTURE_NONCE_ARGUMENT: "model-controlled",
            MEDIA_CAPTURE_OWNER_ARGUMENT: "model-controlled",
        },
    )
    service.finish_media_capture(token)

    assert response == "generated without artifact in this unit test"
    assert calls == [
        (
            "generate_image",
            {
                "prompt": "diagram",
                MEDIA_CAPTURE_NONCE_ARGUMENT: capture.nonce,
                MEDIA_CAPTURE_OWNER_ARGUMENT: capture.owner_tag,
            },
        )
    ]


@pytest.mark.asyncio
async def test_media_tool_call_without_capture_never_starts_server(monkeypatch):
    calls = []
    _install_fake_client(monkeypatch, calls=calls)
    service = MCPService()
    server = MCPServerConfig(name="media", command="media-server", access="public")
    service.servers[server.name] = server
    service.tools[server.name] = {
        "generate_image": {
            "description": "image",
            "schema": {},
            "server": server.name,
            "remote_name": "generate_image",
            "access": "public",
        }
    }

    response = await service.call_tool("generate_image", {"prompt": "diagram"})

    assert response == "工具调用失败: 媒体工具只能响应当前 Telegram 消息"
    assert calls == []


@pytest.mark.asyncio
async def test_media_binding_fields_are_hidden_from_llm_tool_schema(monkeypatch):
    media_tool = SimpleNamespace(
        name="generate_image",
        description="image",
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                MEDIA_CAPTURE_NONCE_ARGUMENT: {"type": "string"},
                MEDIA_CAPTURE_OWNER_ARGUMENT: {"type": "string"},
            },
            "required": [
                "prompt",
                MEDIA_CAPTURE_NONCE_ARGUMENT,
                MEDIA_CAPTURE_OWNER_ARGUMENT,
            ],
        },
        annotations=None,
    )
    _install_fake_client(monkeypatch, listed_tools=[media_tool])
    service = MCPService()

    assert await service.add_server(
        MCPServerConfig(name="media", command="media-server", access="public")
    )
    schema = service.tools["media"]["generate_image"]["schema"]

    assert schema["properties"] == {"prompt": {"type": "string"}}
    assert schema["required"] == ["prompt"]
