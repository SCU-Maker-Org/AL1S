from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest


class FakeEntity(SimpleNamespace):
    pass


class FakeMessage(SimpleNamespace):
    def parse_entity(self, entity):
        return self.text[entity.offset : entity.offset + entity.length]

    def parse_caption_entity(self, entity):
        return self.caption[entity.offset : entity.offset + entity.length]


def make_update(
    text: str = "hello",
    *,
    chat_id: int = -1001,
    chat_type: str = "supergroup",
    user_id: int = 10,
    is_bot: bool = False,
    thread_id: int | None = 7,
    message_id: int = 1,
    update_id: int | None = None,
    entities=None,
    caption: str | None = None,
    caption_entities=None,
    reply_user_id: int | None = None,
    timestamp: float | None = None,
):
    user = SimpleNamespace(
        id=user_id,
        username=f"user{user_id}",
        first_name=f"User {user_id}",
        last_name=None,
        is_bot=is_bot,
    )
    replied = None
    if reply_user_id is not None:
        replied = SimpleNamespace(
            from_user=SimpleNamespace(id=reply_user_id, is_bot=reply_user_id == 99)
        )
    message = FakeMessage(
        text=text,
        caption=caption,
        entities=list(entities or []),
        caption_entities=list(caption_entities or []),
        reply_to_message=replied,
        message_thread_id=thread_id,
        message_id=message_id,
        date=datetime.fromtimestamp(timestamp or time.time(), tz=timezone.utc),
        from_user=user,
        chat_id=chat_id,
        photo=[],
        document=None,
    )
    chat = SimpleNamespace(id=chat_id, type=chat_type)
    return SimpleNamespace(
        update_id=message_id if update_id is None else update_id,
        effective_chat=chat,
        effective_user=user,
        effective_message=message,
        message=message,
    )


@pytest.fixture
def update_factory():
    return make_update
