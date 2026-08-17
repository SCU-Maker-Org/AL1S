from types import SimpleNamespace

import pytest

from src.bot import AL1SBot
from src.config import TelegramConfig


class RecoveringTelegramBot:
    def __init__(self, events):
        self.events = events
        self.identity = None

    @property
    def bot(self):
        if self.identity is None:
            raise RuntimeError("ExtBot is not properly initialized")
        return self.identity

    async def get_me(self):
        self.events.append("telegram")
        self.identity = SimpleNamespace(id=42, username="al1s_bot")
        return self.identity


class InitializableService:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    async def initialize(self):
        self.events.append(self.name)


@pytest.mark.asyncio
async def test_post_init_repairs_identity_before_services():
    events = []
    controller = object.__new__(AL1SBot)
    controller.mcp_service = object()
    controller.unified_agent_service = InitializableService(events, "agent")
    controller.langchain_agent_service = None

    async def initialize_mcp_servers():
        events.append("mcp")

    controller._initialize_mcp_servers = initialize_mcp_servers
    application = SimpleNamespace(bot=RecoveringTelegramBot(events))

    await controller._post_init_callback(application)

    assert events == ["telegram", "mcp", "agent"]


def test_build_application_applies_telegram_timeouts():
    controller = object.__new__(AL1SBot)
    controller.config = SimpleNamespace(
        telegram=TelegramConfig(
            bot_token="123:test",
            connect_timeout=31,
            read_timeout=61,
            write_timeout=62,
            pool_timeout=11,
        )
    )

    application = controller._build_application()
    update_request, request = application.bot._request

    assert request._client.timeout.connect == 31
    assert request._client.timeout.read == 61
    assert request._client.timeout.write == 62
    assert request._client.timeout.pool == 11
    assert update_request._client.timeout.connect == 31
    assert update_request._client.timeout.read == 61
