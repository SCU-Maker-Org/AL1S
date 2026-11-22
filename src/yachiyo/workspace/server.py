"""Standalone MCP 2.x server for a separately isolated workspace container."""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from yachiyo.config import SecretStore, default_paths, load_config
from yachiyo.config.models import WorkspaceSettings

from .operations import WorkspaceService


class _StaticTokenVerifier:
    def __init__(self) -> None:
        token_path = Path(
            os.getenv("YACHIYO_WORKSPACE_TOKEN_FILE", "/run/secrets/workspace_mcp_token")
        )
        if token_path.parent.resolve() != Path("/run/secrets"):
            raise RuntimeError("workspace bearer must be stored under /run/secrets")
        self.store = SecretStore(token_path.parent, allow_environment=False)
        self.name = token_path.name

    def require(self) -> str:
        return self.store.read(self.name, required=True)

    async def verify_token(self, token: str) -> AccessToken | None:
        if not hmac.compare_digest(token, self.require()):
            return None
        return AccessToken(
            token=token,
            client_id="yachiyo-core",
            scopes=["tools"],
            subject="yachiyo-core",
        )


token_verifier = _StaticTokenVerifier()
public_url = os.getenv("MCP_PUBLIC_URL", "http://yachiyo-workspace:8000/mcp")
server = MCPServer(
    "yachiyo-workspace",
    instructions="Operate only on explicitly mounted workspaces.",
    token_verifier=token_verifier,
    auth=AuthSettings(
        issuer_url="https://yachiyo.invalid/",
        resource_server_url=public_url,
        required_scopes=["tools"],
    ),
)


def _service() -> WorkspaceService:
    config_file = default_paths().config_file
    if config_file.is_file():
        settings = load_config(config_file).settings.workspace
        return WorkspaceService(
            Path(settings.root),
            max_file_bytes=settings.max_file_bytes,
            max_results=settings.max_results,
            command_timeout_seconds=settings.command_timeout_seconds,
        )
    defaults = WorkspaceSettings()
    return WorkspaceService(
        Path(os.getenv("YACHIYO_WORKSPACE_ROOT", defaults.root)),
        max_file_bytes=int(os.getenv("YACHIYO_WORKSPACE_MAX_FILE_BYTES", defaults.max_file_bytes)),
        max_results=int(os.getenv("YACHIYO_WORKSPACE_MAX_RESULTS", defaults.max_results)),
        command_timeout_seconds=int(
            os.getenv("YACHIYO_WORKSPACE_COMMAND_TIMEOUT", defaults.command_timeout_seconds)
        ),
    )


service = _service()


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
async def workspace_create(name: str) -> str:
    return await service.create(name)


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def workspace_list_files(workspace: str, relative: str = ".") -> list[str]:
    return await service.list_files(workspace, relative)


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def workspace_read_file(workspace: str, path: str) -> str:
    return await service.read(workspace, path)


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
async def workspace_write_file(workspace: str, path: str, content: str) -> str:
    return await service.write(workspace, path, content)


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
async def workspace_replace_text(workspace: str, path: str, old: str, new: str) -> str:
    return await service.replace(workspace, path, old, new)


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def workspace_search(workspace: str, query: str) -> list[dict[str, object]]:
    return await service.search(workspace, query)


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
async def workspace_git(workspace: str, args: list[str]) -> str:
    return await service.git(workspace, tuple(args))


@server.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        openWorldHint=False,
    )
)
async def workspace_run_check(workspace: str, profile: str) -> str:
    return await service.run_check(workspace, profile)


def main() -> None:
    token_verifier.require()
    allowed_hosts = os.getenv(
        "MCP_ALLOWED_HOSTS", "yachiyo-workspace:8000,localhost:8000,127.0.0.1:8000"
    ).split(",")
    allowed_origins = [value for value in os.getenv("MCP_ALLOWED_ORIGINS", "").split(",") if value]
    server.run(
        transport=os.getenv("MCP_TRANSPORT", "streamable-http"),
        host=os.getenv("MCP_HOST", "0.0.0.0"),  # noqa: S104 - isolated container listener
        port=int(os.getenv("MCP_PORT", "8000")),
        max_request_body_size=2 * 1024 * 1024,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        ),
    )


if __name__ == "__main__":
    main()
