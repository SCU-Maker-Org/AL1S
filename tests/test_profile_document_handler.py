from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.bot import AL1SBot
from src.config import ProfileConfig, TelegramGroupConfig, TelegramRateLimitConfig
from src.services.group_chat_service import GroupChatService
from src.services.rate_limit_service import RateLimitService


class _ProfileRecorder:
    def __init__(self):
        self.calls = []

    async def set_profile(self, telegram_user_id, content, *, source):
        self.calls.append((telegram_user_id, content, source))


class _TelegramFile:
    def __init__(self, payload: bytes):
        self.payload = payload

    async def download_as_bytearray(self):
        return bytearray(self.payload)


class _TelegramBot:
    username = "AL1SBot"
    id = 99

    def __init__(self, payload: bytes):
        self.payload = payload
        self.get_file_calls = 0

    async def get_file(self, file_id):
        self.get_file_calls += 1
        return _TelegramFile(self.payload)


def _controller(*, rate_config=None, max_document_bytes=1024):
    controller = object.__new__(AL1SBot)
    controller.config = SimpleNamespace(
        profile=ProfileConfig(max_document_bytes=max_document_bytes)
    )
    controller.group_chat_service = GroupChatService(TelegramGroupConfig())
    controller.rate_limit_service = RateLimitService(
        rate_config
        or TelegramRateLimitConfig(per_user_requests=100, per_chat_requests=100)
    )
    controller.user_profile_service = _ProfileRecorder()
    controller.replies = []

    async def reply(update, context, text, **kwargs):
        controller.replies.append(text)

    controller._reply = reply
    return controller


def _update(*, update_id=1, user_id=10, chat_id=10, file_size=5, filename="profile.md"):
    document = SimpleNamespace(
        file_id=f"file-{update_id}",
        file_name=filename,
        file_size=file_size,
    )
    message = SimpleNamespace(
        document=document,
        caption=None,
        message_thread_id=None,
        message_id=update_id,
    )
    return SimpleNamespace(
        update_id=update_id,
        effective_chat=SimpleNamespace(id=chat_id, type="private"),
        effective_user=SimpleNamespace(id=user_id, is_bot=False),
        effective_message=message,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("declared_size", "expected_reply"),
    [
        (None, "无法确认画像文件大小，未下载。"),
        (1025, "画像文件超过大小上限，未导入。"),
    ],
)
async def test_unknown_or_oversized_profile_document_is_rejected_before_download(
    declared_size, expected_reply
):
    controller = _controller()
    bot = _TelegramBot(b"small")
    context = SimpleNamespace(bot=bot)

    await controller._handle_profile_document(_update(file_size=declared_size), context)

    assert bot.get_file_calls == 0
    assert controller.user_profile_service.calls == []
    assert controller.replies == [expected_reply]


@pytest.mark.asyncio
async def test_profile_document_is_checked_again_after_download():
    controller = _controller()
    bot = _TelegramBot(b"x" * 1025)
    context = SimpleNamespace(bot=bot)

    await controller._handle_profile_document(_update(file_size=1), context)

    assert bot.get_file_calls == 1
    assert controller.user_profile_service.calls == []
    assert controller.replies == ["画像文件超过大小上限，未导入。"]


@pytest.mark.asyncio
async def test_profile_document_guard_deduplicates_updates():
    controller = _controller()
    bot = _TelegramBot("画像".encode())
    context = SimpleNamespace(bot=bot)
    guarded = controller._guard_profile_document(controller._handle_profile_document)
    update = _update(file_size=len(bot.payload))

    await guarded(update, context)
    await guarded(update, context)

    assert bot.get_file_calls == 1
    assert controller.user_profile_service.calls == [(10, "画像", "telegram_document")]


@pytest.mark.asyncio
@pytest.mark.parametrize("limited_dimension", ["user", "chat"])
async def test_profile_document_guard_applies_user_and_chat_rate_limits(
    limited_dimension,
):
    if limited_dimension == "user":
        rate_config = TelegramRateLimitConfig(
            per_user_requests=1, per_chat_requests=100
        )
        first = _update(update_id=1, user_id=10, chat_id=10)
        second = _update(update_id=2, user_id=10, chat_id=10)
    else:
        rate_config = TelegramRateLimitConfig(
            per_user_requests=100, per_chat_requests=1
        )
        first = _update(update_id=1, user_id=10, chat_id=10)
        second = _update(update_id=2, user_id=20, chat_id=10)

    controller = _controller(rate_config=rate_config)
    bot = _TelegramBot(b"hello")
    context = SimpleNamespace(bot=bot)
    guarded = controller._guard_profile_document(controller._handle_profile_document)

    await guarded(first, context)
    await guarded(second, context)

    assert bot.get_file_calls == 1
    assert len(controller.user_profile_service.calls) == 1
    assert controller.replies[-1] == "请求过于频繁，请稍后再试。"
