from __future__ import annotations

import pytest

from src.config import DiscordConfig, config
from src.discord_bot import DiscordBot
from src.services.conversation_service import ConversationService


class FakeAgent:
    def __init__(self, answer: str = "answer"):
        self.answer = answer
        self.calls = []

    async def chat_completion(
        self,
        messages,
        tools=None,
        knowledge_namespace=None,
        knowledge_namespaces=None,
        enable_rag=True,
        tool_access="public",
    ):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "knowledge_namespace": knowledge_namespace,
                "knowledge_namespaces": knowledge_namespaces,
                "enable_rag": enable_rag,
                "tool_access": tool_access,
            }
        )
        return self.answer


class FakeProfileService:
    def __init__(self):
        self.user_ids = []

    async def build_prompt_context(self, user_id):
        self.user_ids.append(user_id)
        return "\nPROFILE-CONTEXT"


class FakeMCPService:
    def __init__(self):
        self.access_levels = []
        self.cleaned = []

    def get_tools_for_llm(self, access):
        self.access_levels.append(access)
        return [{"type": "function", "function": {"name": "repo_status"}}]

    def begin_media_capture(self, owner):
        return owner

    def finish_media_capture(self, token):
        return []

    def cleanup_media_artifacts(self, artifacts, *, owner):
        self.cleaned.append((artifacts, owner))


def make_adapter(discord_config, agent, **kwargs):
    return DiscordBot(
        discord_config,
        agent,
        ConversationService(),
        **kwargs,
    )


def test_discord_config_resolves_token_from_environment(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fresh-token")
    loaded = DiscordConfig(enabled=True, bot_token="${DISCORD_BOT_TOKEN}")

    assert loaded.bot_token == "fresh-token"


def test_discord_uses_non_privileged_intents_and_scoped_storage_ids():
    adapter = make_adapter(DiscordConfig(bot_token="test"), FakeAgent())

    private = adapter.session_key(user_id=10, channel_id=20, guild_id=None)
    guild = adapter.session_key(user_id=10, channel_id=20, guild_id=30)

    assert adapter.client.intents.message_content is False
    assert adapter.client.intents.members is False
    assert adapter.client.intents.presences is False
    assert private.scope == "private"
    assert guild.scope == "topic"
    assert private.chat_id != guild.chat_id
    assert private.user_id == adapter.storage_id("user", 10)
    assert adapter.storage_id("user", 10) != adapter.storage_id("guild", 10)
    assert DiscordBot.strip_bot_mention("<@!42> explain WAL", 42) == "explain WAL"
    assert {command.name for command in adapter.tree.get_commands()} == {
        "ask",
        "ping",
        "reset",
    }


@pytest.mark.asyncio
async def test_guild_request_uses_only_global_technical_rag():
    agent = FakeAgent()
    profile = FakeProfileService()
    adapter = make_adapter(
        DiscordConfig(bot_token="test", enable_group_memory=False),
        agent,
        user_profile_service=profile,
    )
    edited = []
    sent = []

    async def edit_response(text):
        edited.append(text)

    async def send_text(text):
        sent.append(text)

    async def send_media(_file, _caption):
        raise AssertionError("no media expected")

    await adapter._process_request(
        prompt="explain WAL",
        user_id=10,
        username="alice",
        display_name="Alice",
        channel_id=20,
        guild_id=30,
        request_id=40,
        edit_response=edit_response,
        send_text=send_text,
        send_media=send_media,
    )

    assert edited == ["answer"]
    assert sent == []
    assert profile.user_ids == []
    assert agent.calls[0]["knowledge_namespace"] is None
    assert agent.calls[0]["knowledge_namespaces"] == [config.rag.technical_namespace]
    assert agent.calls[0]["tool_access"] == "public"
    system_prompt = agent.calls[0]["messages"][0]["content"]
    assert "Discord" in system_prompt
    assert "不要输出 HTML" in system_prompt


@pytest.mark.asyncio
async def test_admin_dm_gets_profile_private_memory_and_private_admin_tools():
    agent = FakeAgent()
    profile = FakeProfileService()
    mcp = FakeMCPService()
    adapter = make_adapter(
        DiscordConfig(bot_token="test", admin_user_ids=[10]),
        agent,
        mcp_service=mcp,
        user_profile_service=profile,
    )
    edited = []

    async def edit_response(text):
        edited.append(text)

    async def send_text(_text):
        raise AssertionError("single response should be edited in place")

    async def send_media(_file, _caption):
        raise AssertionError("no media expected")

    await adapter._process_request(
        prompt="inspect my repository",
        user_id=10,
        username="admin",
        display_name="Admin",
        channel_id=20,
        guild_id=None,
        request_id=40,
        edit_response=edit_response,
        send_text=send_text,
        send_media=send_media,
    )

    stored_user_id = adapter.storage_id("user", 10)
    assert edited == ["answer"]
    assert profile.user_ids == [stored_user_id]
    assert mcp.access_levels == ["private_admin"]
    assert agent.calls[0]["tool_access"] == "private_admin"
    assert agent.calls[0]["knowledge_namespace"] == f"private:{stored_user_id}"
    assert agent.calls[0]["knowledge_namespaces"] == [
        config.rag.technical_namespace,
        f"private:{stored_user_id}",
    ]
    system_prompt = agent.calls[0]["messages"][0]["content"]
    assert "PROFILE-CONTEXT" in system_prompt
    assert "开发工具规则" in system_prompt
