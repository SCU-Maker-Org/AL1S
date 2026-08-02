from __future__ import annotations

import asyncio
import time

import pytest

from src.config import TelegramGroupConfig
from src.models import Message, SessionKey
from src.services.conversation_service import ConversationService
from src.services.group_chat_service import GroupChatService


@pytest.mark.parametrize(
    ("scope", "thread_id", "first_user", "second_user", "same"),
    [
        ("per_user", 7, 1, 2, False),
        ("shared", 7, 1, 2, True),
        ("topic", 7, 1, 2, True),
    ],
)
def test_group_session_scope(
    update_factory, scope, thread_id, first_user, second_user, same
):
    service = GroupChatService(TelegramGroupConfig(session_scope=scope))
    first = service.session_key(update_factory(user_id=first_user, thread_id=thread_id))
    second = service.session_key(
        update_factory(user_id=second_user, thread_id=thread_id)
    )
    assert (first == second) is same


def test_private_and_group_topic_isolation(update_factory):
    service = GroupChatService(TelegramGroupConfig(session_scope="topic"))
    private = service.session_key(
        update_factory(chat_id=10, chat_type="private", user_id=10, thread_id=None)
    )
    assert private == SessionKey(10, 0, 10, "private")
    assert service.session_key(
        update_factory(chat_id=-1, thread_id=7)
    ) != service.session_key(update_factory(chat_id=-2, thread_id=7))
    assert service.session_key(
        update_factory(chat_id=-1, thread_id=7)
    ) != service.session_key(update_factory(chat_id=-1, thread_id=8))


def test_reset_only_affects_current_session():
    service = ConversationService()
    first = SessionKey(-1, 7, 0, "topic")
    second = SessionKey(-1, 8, 0, "topic")
    message = Message(role="user", content="hello", timestamp=time.time())
    service.add_message(first, message)
    service.add_message(second, message)
    assert service.reset_conversation(first)
    assert service.get_conversation(first).messages == []
    assert len(service.get_conversation(second).messages) == 1


@pytest.mark.asyncio
async def test_same_session_is_serialized():
    service = ConversationService()
    key = SessionKey(-1, 7, 0, "topic")
    events: list[str] = []

    async def worker(name: str):
        async with service.session_lock(key):
            events.append(f"{name}:start")
            await asyncio.sleep(0.01)
            events.append(f"{name}:end")

    await asyncio.gather(worker("a"), worker("b"))
    assert events in [
        ["a:start", "a:end", "b:start", "b:end"],
        ["b:start", "b:end", "a:start", "a:end"],
    ]


@pytest.mark.asyncio
async def test_different_sessions_can_run_concurrently():
    service = ConversationService()
    first = SessionKey(-1, 7, 0, "topic")
    second = SessionKey(-1, 8, 0, "topic")
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()

    async def worker(key, own, other):
        async with service.session_lock(key):
            own.set()
            await asyncio.wait_for(other.wait(), timeout=0.2)

    await asyncio.gather(
        worker(first, first_entered, second_entered),
        worker(second, second_entered, first_entered),
    )


@pytest.mark.asyncio
async def test_reset_waits_for_inflight_write():
    service = ConversationService()
    key = SessionKey(-1, 7, 0, "topic")
    entered = asyncio.Event()

    async def writer():
        async with service.session_lock(key):
            entered.set()
            await asyncio.sleep(0.02)
            service.add_message(
                key, Message(role="assistant", content="done", timestamp=time.time())
            )

    async def resetter():
        await entered.wait()
        async with service.session_lock(key):
            service.reset_conversation(key)

    await asyncio.gather(writer(), resetter())
    assert service.get_conversation(key).messages == []
