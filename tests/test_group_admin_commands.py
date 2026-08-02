from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.bot import AL1SBot
from src.config import TelegramConfig, TelegramGroupConfig
from src.services.group_chat_service import GroupChatService


class FakeController:
    def __init__(self, service):
        self.group_chat_service = service
        self.config = SimpleNamespace(telegram=TelegramConfig())
        self.replies = []

    async def _reply(self, update, context, text, **kwargs):
        self.replies.append(text)

    async def _require_group_admin(self, update, context):
        return await AL1SBot._require_group_admin(self, update, context)


class FakeBot:
    def __init__(self, status):
        self.status = status

    async def get_chat_member(self, chat_id, user_id):
        return SimpleNamespace(status=self.status)


@pytest.mark.asyncio
async def test_member_cannot_modify_group_settings(update_factory):
    service = GroupChatService(TelegramGroupConfig())
    controller = FakeController(service)
    update = update_factory()
    context = SimpleNamespace(bot=FakeBot("member"), args=[])
    await AL1SBot._handle_group_disable_command(controller, update, context)
    assert service.is_enabled(update.effective_chat.id)
    assert controller.replies == ["权限不足：仅群管理员可执行该命令。"]


@pytest.mark.asyncio
async def test_administrator_can_view_and_modify_group_settings(update_factory):
    service = GroupChatService(TelegramGroupConfig())
    controller = FakeController(service)
    update = update_factory()
    context = SimpleNamespace(bot=FakeBot("administrator"), args=[])

    await AL1SBot._handle_group_disable_command(controller, update, context)
    assert not service.is_enabled(update.effective_chat.id)

    await AL1SBot._handle_group_status_command(controller, update, context)
    assert any("群聊配置" in reply for reply in controller.replies)
    assert any("停用" in reply for reply in controller.replies)
