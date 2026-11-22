"""aiogram transport that only normalizes and renders platform events."""

from __future__ import annotations

import asyncio
import io
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatMemberStatus, MessageEntityType
from aiogram.filters import Command
from aiogram.filters.command import CommandObject
from aiogram.types import (
    BotCommand,
    BufferedInputFile,
    CallbackQuery,
    EphemeralMessageParameters,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from yachiyo.config import AppSettings
from yachiyo.domain import (
    Actor,
    Artifact,
    Attachment,
    ChatKind,
    CommandInvocation,
    ConversationAddress,
    ExternalIdentity,
    InboundEvent,
    RequestPermission,
    ResponsePlan,
    TransportName,
)
from yachiyo.domain.content import ArtifactKind
from yachiyo.ports.transport import CommandDescriptor, EventReceiver

from .renderers import split_text, validated_mime

logger = logging.getLogger(__name__)


class AttachmentError(ValueError):
    pass


class _LimitedBuffer(io.BytesIO):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def write(self, data: bytes) -> int:
        if self.tell() + len(data) > self.limit:
            raise AttachmentError("附件超过允许的大小。")
        return super().write(data)


class TelegramTransport:
    name = TransportName.TELEGRAM

    def __init__(self, token: str, settings: AppSettings):
        self.settings = settings
        session = (
            AiohttpSession(proxy=settings.transports.telegram.proxy_url)
            if settings.transports.telegram.proxy_url
            else AiohttpSession()
        )
        self.bot = Bot(token, session=session, default=DefaultBotProperties(parse_mode=None))
        self.dispatcher = Dispatcher()
        self.router = Router(name="yachiyo")
        self.dispatcher.include_router(self.router)
        self.receiver: EventReceiver | None = None
        self.bot_id = "pending"
        self.username = ""
        self._running = False
        self._handlers_registered = False
        self._admin_cache: dict[tuple[int, int], tuple[float, bool]] = {}
        self._download_slots = asyncio.Semaphore(4)
        self._inflight: set[asyncio.Task[object]] = set()
        self._close_lock = asyncio.Lock()
        self._closed = False

    async def prepare(self, commands: Sequence[CommandDescriptor]) -> None:
        identity = await self.bot.get_me()
        self.bot_id = str(identity.id)
        self.username = identity.username or ""
        if not self._handlers_registered:
            names = [command.name for command in commands]
            self.router.message.register(self._on_command, Command(commands=names))
            self.router.message.register(self._on_message)
            self.router.callback_query.register(self._on_callback)
            self._handlers_registered = True
        try:
            await self.bot.set_my_commands(
                [
                    BotCommand(command=item.name, description=item.description[:100])
                    for item in commands
                ]
            )
        except Exception:
            logger.exception("Telegram command synchronization failed")

    async def run(self, receiver: EventReceiver) -> None:
        self.receiver = receiver
        self._running = True
        try:
            await self.dispatcher.start_polling(
                self.bot,
                handle_signals=False,
                close_bot_session=False,
                allowed_updates=["message", "callback_query"],
                tasks_concurrency_limit=100,
            )
        finally:
            self._running = False

    async def _on_command(self, message: Message, command: CommandObject) -> None:
        invocation = CommandInvocation(command.command.casefold(), command.args or "")
        await self._dispatch(message, invocation)

    async def _on_message(self, message: Message) -> None:
        invocation = self._parse_command(message.text or message.caption or "")
        if invocation is False:
            return
        await self._dispatch(message, invocation)

    def _parse_command(self, text: str) -> CommandInvocation | bool | None:
        if not text.startswith("/"):
            return None
        first, _, arguments = text.partition(" ")
        command_target = first[1:].split("@", 1)
        if len(command_target) == 2 and command_target[1].casefold() != self.username.casefold():
            return False
        return CommandInvocation(command_target[0].casefold(), arguments)

    async def _dispatch(self, message: Message, command: CommandInvocation | None) -> None:
        if self.receiver is None or message.from_user is None:
            return
        event = await self._event(message, command)
        await self._call_receiver(event)

    async def _on_callback(self, query: CallbackQuery) -> None:
        if self.receiver is None or not isinstance(query.message, Message) or not query.data:
            if query.message is not None and not isinstance(query.message, Message):
                await query.answer("操作已过期。")
                return
            await query.answer()
            return
        if not query.data.startswith("approval:"):
            await query.answer("Unknown action")
            return
        message = query.message
        user = query.from_user
        address = self._address(message.chat.id, message.message_thread_id, str(message.chat.type))
        permission = await self._permission(message.chat.id, user.id, address.kind, fresh=True)
        action = query.data.removeprefix("approval:")
        event = InboundEvent(
            event_id=f"callback:{query.id}",
            address=address,
            actor=Actor(
                ExternalIdentity(TransportName.TELEGRAM, str(user.id)),
                user.full_name,
                user.username,
                user.is_bot,
            ),
            permission=permission,
            command=CommandInvocation("_action", action),
        )
        await query.answer()
        await self._call_receiver(event)

    async def _call_receiver(self, event: InboundEvent) -> None:
        receiver = self.receiver
        if receiver is None:
            return
        task = asyncio.current_task()
        if task is not None:
            self._inflight.add(task)
        try:
            await receiver(event)
        finally:
            if task is not None:
                self._inflight.discard(task)

    async def _event(self, message: Message, command: CommandInvocation | None) -> InboundEvent:
        user = message.from_user
        assert user is not None
        address = self._address(message.chat.id, message.message_thread_id, str(message.chat.type))
        raw_text = message.text or message.caption or ""
        text = raw_text.strip()
        mention = self._mentions_bot(message, raw_text)
        reply_user = message.reply_to_message.from_user if message.reply_to_message else None
        replies = bool(reply_user and str(reply_user.id) == self.bot_id)
        attachment_limit = (
            self.settings.profile.max_document_bytes
            if command is not None and command.name in {"profile_set", "profile_add"}
            else self.settings.policies.max_attachment_bytes
        )

        async def load_attachments() -> tuple[Attachment, ...]:
            return await self._attachments(message, limit=attachment_limit)

        return InboundEvent(
            event_id=f"{message.chat.id}:{message.message_id}",
            address=address,
            actor=Actor(
                ExternalIdentity(TransportName.TELEGRAM, str(user.id)),
                user.full_name,
                user.username,
                user.is_bot,
            ),
            permission=await self._permission(
                message.chat.id,
                user.id,
                address.kind,
                fresh=bool(command and command.name in {"_action", "approve", "deny"}),
            ),
            text=text,
            attachment_loader=load_attachments if message.photo or message.document else None,
            command=command,
            mentions_bot=mention,
            replies_to_bot=replies,
            timestamp=message.date,
        )

    def _mentions_bot(self, message: Message, text: str) -> bool:
        username = self.username.casefold()
        entities = message.entities or message.caption_entities or ()
        for entity in entities:
            if entity.type == MessageEntityType.MENTION:
                if entity.extract_from(text).lstrip("@").casefold() == username:
                    return True
            elif entity.type == MessageEntityType.TEXT_MENTION:
                if entity.user is not None and str(entity.user.id) == self.bot_id:
                    return True
        return False

    def _address(self, chat_id: int, thread_id: int | None, chat_type: str) -> ConversationAddress:
        kind = ChatKind.PRIVATE if chat_type == "private" else ChatKind.GROUP
        return ConversationAddress(
            TransportName.TELEGRAM,
            self.bot_id,
            str(chat_id),
            str(thread_id or 0),
            kind,
        )

    async def _permission(
        self, chat_id: int, user_id: int, kind: ChatKind, *, fresh: bool = False
    ) -> RequestPermission:
        configured = str(user_id) in self.settings.transports.telegram.admin_user_ids
        if kind is ChatKind.PRIVATE:
            return RequestPermission(True, configured)
        key = (chat_id, user_id)
        cached = self._admin_cache.get(key)
        now = time.monotonic()
        if not fresh and cached and cached[0] > now:
            return RequestPermission(False, configured or cached[1])
        try:
            member = await self.bot.get_chat_member(chat_id, user_id)
            native_admin = member.status in {
                ChatMemberStatus.ADMINISTRATOR,
                ChatMemberStatus.CREATOR,
            }
        except Exception:
            native_admin = False
        self._admin_cache[key] = (now + 60, native_admin)
        if len(self._admin_cache) > 10_000:
            self._admin_cache = {
                cache_key: value for cache_key, value in self._admin_cache.items() if value[0] > now
            }
            while len(self._admin_cache) > 10_000:
                self._admin_cache.pop(next(iter(self._admin_cache)))
        return RequestPermission(False, configured or native_admin)

    async def _attachments(
        self, message: Message, *, limit: int | None = None
    ) -> tuple[Attachment, ...]:
        candidates: list[tuple[str, str, str, int | None]] = []
        if message.photo:
            photo = message.photo[-1]
            candidates.append((photo.file_id, "image.jpg", "image/jpeg", photo.file_size))
        if message.document:
            document = message.document
            candidates.append(
                (
                    document.file_id,
                    document.file_name or "document",
                    document.mime_type or "application/octet-stream",
                    document.file_size,
                )
            )
        if not candidates:
            return ()
        limit = limit or self.settings.policies.max_attachment_bytes
        declared_total = sum(size or 0 for *_, size in candidates)
        if len(candidates) > 10 or declared_total > limit:
            raise AttachmentError("附件总量超过允许的限制。")
        result: list[Attachment] = []
        async with self._download_slots:
            for file_id, filename, mime_type, declared_size in candidates:
                if declared_size is not None and declared_size > limit:
                    raise AttachmentError("附件超过允许的大小。")
                buffer = _LimitedBuffer(limit - sum(item.size for item in result))
                await self.bot.download(file_id, destination=buffer)
                data = buffer.getvalue()
                try:
                    mime_type = validated_mime(data, mime_type)
                except ValueError as exc:
                    raise AttachmentError("图片格式无效或不受支持。") from exc
                result.append(Attachment(file_id, filename, mime_type, len(data), data))
        return tuple(result)

    async def send(self, event: InboundEvent, response: ResponsePlan) -> None:
        for message in response.messages:
            callback_id = (
                event.event_id.removeprefix("callback:")
                if event.event_id.startswith("callback:")
                else None
            )
            ephemeral = (
                EphemeralMessageParameters(
                    receiver_user_id=int(event.actor.identity.external_id),
                    callback_query_id=callback_id,
                )
                if message.private and event.address.kind is not ChatKind.PRIVATE
                else None
            )
            target = int(event.address.chat_id)
            thread_id = (
                int(event.address.thread_id)
                if target == int(event.address.chat_id) and event.address.thread_id != "0"
                else None
            )
            text_chunks = split_text(message.text, 4096)
            markup = self._markup(message.buttons)
            for index, chunk in enumerate(text_chunks):
                await self.bot.send_message(
                    target,
                    chunk,
                    message_thread_id=thread_id,
                    reply_markup=markup if index == len(text_chunks) - 1 else None,
                    ephemeral_message_parameters=ephemeral,
                )
            for artifact in message.artifacts:
                await self._send_artifact(target, thread_id, artifact, ephemeral)

    @staticmethod
    def _markup(buttons) -> InlineKeyboardMarkup | None:
        if not buttons:
            return None
        rendered = [
            InlineKeyboardButton(
                text=button.label,
                callback_data=button.action_id,
                style=None if button.style == "secondary" else button.style,
            )
            for button in buttons
            if len(button.action_id.encode()) <= 64
        ]
        if not rendered:
            return None
        return InlineKeyboardMarkup(
            inline_keyboard=[rendered[index : index + 2] for index in range(0, len(rendered), 2)]
        )

    async def _send_artifact(
        self,
        target: int,
        thread_id: int | None,
        artifact: Artifact,
        ephemeral: EphemeralMessageParameters | None,
    ) -> None:
        upload = (
            BufferedInputFile(artifact.data, artifact.filename)
            if artifact.data is not None
            else FSInputFile(Path(artifact.path), artifact.filename)
        )
        if artifact.kind is ArtifactKind.IMAGE:
            await self.bot.send_photo(
                target,
                upload,
                message_thread_id=thread_id,
                ephemeral_message_parameters=ephemeral,
            )
        elif artifact.kind is ArtifactKind.AUDIO and artifact.mime_type in {
            "audio/ogg",
            "audio/opus",
        }:
            await self.bot.send_voice(
                target,
                upload,
                message_thread_id=thread_id,
                ephemeral_message_parameters=ephemeral,
            )
        else:
            await self.bot.send_document(
                target,
                upload,
                message_thread_id=thread_id,
                ephemeral_message_parameters=ephemeral,
            )

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self.receiver = None
            if self._running:
                try:
                    await self.dispatcher.stop_polling()
                except RuntimeError:
                    pass
            current = asyncio.current_task()
            pending = [task for task in self._inflight if task is not current]
            if pending:
                _, waiting = await asyncio.wait(pending, timeout=30)
                for task in waiting:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            await self.bot.session.close()
            self._closed = True

    def health(self) -> dict[str, str]:
        return {
            "transport": self.name.value,
            "state": "running" if self._running else "stopped",
            "bot_id": self.bot_id,
        }
