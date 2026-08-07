from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.bot import AL1SBot
from src.config import (
    AppConfig,
    DevWorkspaceConfig,
    MCPConfig,
    MCPServerConfig,
)
from src.handlers.chat_handler import ChatHandler
from src.infra.mcp import MCPService


class _MCPRecorder:
    def __init__(self):
        self.configs = []

    async def initialize_default_servers(self, configs):
        self.configs = configs

    def get_available_tools(self, caller_access="public"):
        return {}


def test_dev_workspace_defaults_and_app_config_loading(monkeypatch):
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    defaults = DevWorkspaceConfig()

    assert defaults.enabled is False
    assert defaults.root_dir == "data/dev_workspaces"
    assert defaults.max_workspaces == 10
    assert defaults.max_file_bytes == 1_000_000
    assert defaults.max_output_chars == 30_000
    assert defaults.command_timeout == 120
    assert defaults.git_timeout == 60
    assert defaults.git_author_name == "AL1S"
    assert defaults.git_author_email == "al1s@localhost"
    assert defaults.branch_prefix == "al1s/"
    assert defaults.runner_enabled is False
    assert defaults.allowed_github_owners == []
    assert "github_token" not in repr(defaults)

    monkeypatch.setattr(
        AppConfig,
        "_load_unified_config",
        lambda self: {
            "dev_workspace": {
                "enabled": True,
                "root_dir": "custom/workspaces",
                "allowed_github_owners": ["Example", "example", " Octo-Org "],
            }
        },
    )
    loaded = AppConfig()

    assert loaded.dev_workspace.enabled is True
    assert loaded.dev_workspace.root_dir == "custom/workspaces"
    assert loaded.dev_workspace.allowed_github_owners == ["Example", "Octo-Org"]


def test_private_admin_access_is_private_chat_only():
    admin_ids = [42]

    assert MCPService.resolve_caller_access(42, "private", admin_ids) == "private_admin"
    assert MCPService.resolve_caller_access(42, "group", admin_ids) == "admin"
    assert MCPService.resolve_caller_access(7, "private", admin_ids) == "private"
    assert MCPService._access_allowed("admin", "private_admin") is True
    assert MCPService._access_allowed("private_admin", "admin") is False


def test_private_admin_development_prompt_treats_repository_content_as_untrusted():
    instructions = ChatHandler._development_tool_instructions()

    assert "不可信数据" in instructions
    assert "expected_head" in instructions
    assert "不得声称已经" in instructions


def test_dev_workspace_rejects_unsafe_publish_settings():
    with pytest.raises(ValidationError, match="GitHub owner"):
        DevWorkspaceConfig(allowed_github_owners=["owner/repository"])
    with pytest.raises(ValidationError, match="branch_prefix"):
        DevWorkspaceConfig(branch_prefix="main")
    with pytest.raises(ValidationError, match="作者邮箱"):
        DevWorkspaceConfig(git_author_email="not-an-email")


@pytest.mark.asyncio
async def test_bot_injects_dev_environment_and_reuses_github_mcp_token(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    bot = object.__new__(AL1SBot)
    bot.mcp_service = _MCPRecorder()
    bot.config = SimpleNamespace(
        dev_workspace=DevWorkspaceConfig(
            enabled=True,
            root_dir=str(tmp_path / "workspaces"),
            allowed_github_owners=["octo-org"],
        ),
        mcp=MCPConfig(
            servers=[
                MCPServerConfig(
                    name="dev-workspace",
                    command="python",
                    enabled=True,
                    access="private_admin",
                ),
                MCPServerConfig(
                    name="git-publisher",
                    command="python",
                    enabled=True,
                    access="private_admin",
                ),
                MCPServerConfig(
                    name="github",
                    command="github-mcp-server",
                    enabled=False,
                    env={"GITHUB_PERSONAL_ACCESS_TOKEN": "fallback-token"},
                ),
            ]
        ),
    )

    await bot._initialize_mcp_servers()

    configs = {server.name: server for server in bot.mcp_service.configs}
    assert set(configs) == {"dev-workspace", "git-publisher"}
    expected_root = str((tmp_path / "workspaces").resolve())
    assert configs["dev-workspace"].env["AL1S_DEV_WORKSPACE_ROOT"] == expected_root
    assert configs["dev-workspace"].env["AL1S_DEV_MAX_WORKSPACES"] == "10"
    assert configs["dev-workspace"].env["AL1S_DEV_RUNNER_ENABLED"] == "false"
    assert configs["git-publisher"].env["AL1S_DEV_GITHUB_TOKEN"] == "fallback-token"
    assert configs["git-publisher"].env["AL1S_DEV_ALLOWED_GITHUB_OWNERS"] == "octo-org"


@pytest.mark.asyncio
async def test_bot_skips_dev_servers_when_feature_is_disabled(monkeypatch):
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    bot = object.__new__(AL1SBot)
    bot.mcp_service = _MCPRecorder()
    bot.config = SimpleNamespace(
        dev_workspace=DevWorkspaceConfig(enabled=False),
        mcp=MCPConfig(
            servers=[
                MCPServerConfig(name="dev-workspace", command="python", enabled=True),
                MCPServerConfig(name="git-publisher", command="python", enabled=True),
            ]
        ),
    )

    await bot._initialize_mcp_servers()

    assert bot.mcp_service.configs == []
