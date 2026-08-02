from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.config import TelegramGroupConfig
from src.services.group_chat_service import GroupChatService, TriggerType

BOT_USERNAME = "AL1SBot"
BOT_ID = 99


def mention_entity(text: str, mention: str):
    offset = text.index(mention)
    return SimpleNamespace(type="mention", offset=offset, length=len(mention))


def decide(service, update, **kwargs):
    return service.decide(update, BOT_USERNAME, BOT_ID, **kwargs)


def test_private_message_triggers(update_factory):
    service = GroupChatService(TelegramGroupConfig())
    result = decide(
        service, update_factory(chat_id=10, chat_type="private", thread_id=None)
    )
    assert result.allowed
    assert result.trigger_type == TriggerType.PRIVATE


def test_plain_group_message_does_not_trigger(update_factory):
    result = decide(GroupChatService(TelegramGroupConfig()), update_factory())
    assert not result.allowed
    assert result.denied_reason == "not_triggered"


@pytest.mark.parametrize("mention", ["@AL1SBot", "@al1sbot", "@Al1SbOt"])
def test_current_bot_mention_is_case_insensitive(update_factory, mention):
    text = f"hi {mention} please help"
    update = update_factory(text, entities=[mention_entity(text, mention)])
    result = decide(GroupChatService(TelegramGroupConfig()), update)
    assert result.allowed
    assert result.trigger_type == TriggerType.MENTION
    assert mention.casefold() not in result.cleaned_text.casefold()


def test_reply_to_current_bot_triggers(update_factory):
    result = decide(
        GroupChatService(TelegramGroupConfig()),
        update_factory(reply_user_id=BOT_ID),
    )
    assert result.allowed
    assert result.trigger_type == TriggerType.REPLY


def test_reply_to_other_user_does_not_trigger(update_factory):
    result = decide(
        GroupChatService(TelegramGroupConfig()),
        update_factory(reply_user_id=11),
    )
    assert not result.allowed


def test_wake_word_triggers(update_factory):
    result = decide(
        GroupChatService(TelegramGroupConfig(wake_words=["爱丽丝"])),
        update_factory("爱丽丝，帮我看看"),
    )
    assert result.allowed
    assert result.trigger_type == TriggerType.WAKE_WORD


def test_other_bot_sender_is_ignored(update_factory):
    result = decide(
        GroupChatService(TelegramGroupConfig(ignore_bot_messages=True)),
        update_factory("@AL1SBot hi", is_bot=True),
    )
    assert not result.allowed
    assert result.denied_reason == "bot_sender"


def test_multiple_mentions_only_matches_current_bot(update_factory):
    text = "@OtherBot and @AL1SBot help"
    entities = [
        mention_entity(text, "@OtherBot"),
        mention_entity(text, "@AL1SBot"),
    ]
    result = decide(
        GroupChatService(TelegramGroupConfig()),
        update_factory(text, entities=entities),
    )
    assert result.allowed
    assert result.trigger_type == TriggerType.MENTION

    other_only = "@OtherBot help"
    result = decide(
        GroupChatService(TelegramGroupConfig()),
        update_factory(other_only, entities=[mention_entity(other_only, "@OtherBot")]),
    )
    assert not result.allowed


def test_group_disabled_does_not_trigger(update_factory):
    result = decide(
        GroupChatService(TelegramGroupConfig(enabled=False)),
        update_factory("爱丽丝"),
    )
    assert not result.allowed
    assert result.denied_reason == "group_disabled"


def test_command_targeting(update_factory):
    service = GroupChatService(TelegramGroupConfig())
    own = decide(service, update_factory("/reset@al1sbot"), is_command=True)
    other = decide(service, update_factory("/reset@OtherBot"), is_command=True)
    assert own.allowed and own.trigger_type == TriggerType.COMMAND
    assert not other.allowed and other.denied_reason == "command_for_other_bot"


@pytest.mark.parametrize(
    ("config", "chat_id", "thread_id", "allowed", "reason"),
    [
        ({"allowed_chat_ids": [-1]}, -1, 7, True, None),
        ({"allowed_chat_ids": [-2]}, -1, 7, False, "chat_not_allowed"),
        (
            {"allowed_chat_ids": [-1], "blocked_chat_ids": [-1]},
            -1,
            7,
            False,
            "chat_blocked",
        ),
        ({"allowed_thread_ids": [7]}, -1, 7, True, None),
        ({"allowed_thread_ids": [8]}, -1, 7, False, "thread_not_allowed"),
        ({"ignored_thread_ids": [7]}, -1, 7, False, "thread_ignored"),
    ],
)
def test_chat_and_thread_permissions(
    update_factory, config, chat_id, thread_id, allowed, reason
):
    service = GroupChatService(TelegramGroupConfig(require_mention=False, **config))
    result = decide(service, update_factory(chat_id=chat_id, thread_id=thread_id))
    assert result.allowed is allowed
    assert result.denied_reason == reason


@pytest.mark.asyncio
async def test_admin_status_check():
    service = GroupChatService(TelegramGroupConfig())

    class Bot:
        status = "member"

        async def get_chat_member(self, chat_id, user_id):
            return SimpleNamespace(status=self.status)

    context = SimpleNamespace(bot=Bot())
    assert not await service.is_admin(context, -1, 10, [])
    context.bot.status = "administrator"
    assert await service.is_admin(context, -1, 10, [])
    context.bot.status = "member"
    assert await service.is_admin(context, -1, 10, [10])


@pytest.mark.asyncio
async def test_buffer_is_bounded_ttl_scoped_and_formatted(update_factory):
    service = GroupChatService(
        TelegramGroupConfig(context_buffer_size=2, context_buffer_ttl=10)
    )
    for message_id, text in enumerate(["one", "two", "three"], 1):
        await service.observe(
            update_factory(text, message_id=message_id, update_id=message_id)
        )
    context = await service.get_context(-1001, 7)
    assert [item.text for item in context] == ["two", "three"]
    assert "User 10：two" in await service.format_context(-1001, 7)

    await service.observe(
        update_factory("other topic", thread_id=8, message_id=10, update_id=10)
    )
    assert [item.text for item in await service.get_context(-1001, 8)] == [
        "other topic"
    ]
    assert all(
        item.text != "other topic" for item in await service.get_context(-1001, 7)
    )

    await service.observe(
        update_factory(
            "expired",
            chat_id=-2002,
            timestamp=time.time() - 20,
            message_id=20,
            update_id=20,
        )
    )
    assert await service.get_context(-2002, 7) == []


@pytest.mark.asyncio
async def test_duplicate_update_is_detected():
    service = GroupChatService(TelegramGroupConfig())
    assert not await service.is_duplicate_update(123)
    assert await service.is_duplicate_update(123)


def test_group_config_validation_and_wake_word_normalization():
    config = TelegramGroupConfig(wake_words=[" AL1S ", "al1s", "", "爱丽丝"])
    assert config.wake_words == ["AL1S", "爱丽丝"]
    with pytest.raises(ValidationError):
        TelegramGroupConfig(session_scope="invalid")
    with pytest.raises(ValidationError):
        TelegramGroupConfig(context_buffer_size=0)


def test_runtime_group_admin_settings(update_factory):
    service = GroupChatService(TelegramGroupConfig())
    service.set_enabled(-1001, False)
    assert not service.is_enabled(-1001)
    assert service.set_scope(-1001, "shared")
    assert service.session_scope(-1001) == "shared"
    assert not service.set_scope(-1001, "invalid")
    service.set_wake_words(-1001, ["Alice", "alice", "AL1S"])
    assert service.wake_words(-1001) == ["Alice", "AL1S"]
