from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.config import TelegramGroupConfig, TelegramRateLimitConfig
from src.handlers.chat_handler import ChatHandler
from src.services.conversation_service import ConversationService
from src.services.group_chat_service import GroupChatService
from src.services.rate_limit_service import RateLimitService


class FakeAgent:
    def __init__(self):
        self.calls = []

    async def chat_completion(
        self,
        messages,
        tools=None,
        knowledge_namespace=None,
        enable_rag=True,
    ):
        self.calls.append(
            {
                "messages": messages,
                "namespace": knowledge_namespace,
                "enable_rag": enable_rag,
            }
        )
        return "answer"


class FakeBot:
    username = "AL1SBot"
    id = 99

    def __init__(self):
        self.sent = []
        self.edited = []

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(message_id=100 + len(self.sent))

    async def edit_message_text(self, **kwargs):
        self.edited.append(kwargs)


class FakeApplication:
    def __init__(self):
        self.tasks = []

    def create_task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task


@pytest.mark.asyncio
async def test_unmentioned_observed_then_mention_calls_agent_in_same_topic(
    update_factory,
):
    group = GroupChatService(
        TelegramGroupConfig(
            observe_unmentioned_messages=True,
            session_scope="topic",
            wake_words=[],
        )
    )
    agent = FakeAgent()
    bot = FakeBot()
    application = FakeApplication()
    context = SimpleNamespace(bot=bot, application=application)
    handler = ChatHandler(
        agent,
        ConversationService(),
        group_chat_service=group,
        rate_limit_service=RateLimitService(
            TelegramRateLimitConfig(per_user_requests=100, per_chat_requests=100)
        ),
    )

    ordinary = update_factory("database pool is full", message_id=1, update_id=1)
    assert await handler.handle(ordinary, context)
    assert agent.calls == []
    assert bot.sent == []
    assert await group.context_count(-1001, 7) == 1

    text = "@AL1SBot how should we debug it?"
    entity = SimpleNamespace(type="mention", offset=0, length=len("@AL1SBot"))
    triggered = update_factory(
        text,
        entities=[entity],
        message_id=2,
        update_id=2,
        thread_id=7,
    )
    assert await handler.handle(triggered, context)
    await asyncio.gather(*application.tasks)

    assert len(agent.calls) == 1
    system_prompt = agent.calls[0]["messages"][0]["content"]
    assert "database pool is full" in system_prompt
    assert agent.calls[0]["namespace"] == "topic:-1001:7"
    assert not agent.calls[0]["enable_rag"]
    assert bot.sent[0]["message_thread_id"] == 7
    assert bot.sent[0]["reply_to_message_id"] == 2
    assert bot.edited[0]["chat_id"] == -1001
