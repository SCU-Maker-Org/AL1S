"""Telegram 群聊触发、权限、作用域和临时上下文策略。"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

from loguru import logger

from ..config import TelegramGroupConfig
from ..models import SessionKey


class TriggerType(str, Enum):
    PRIVATE = "private"
    MENTION = "mention"
    REPLY = "reply"
    WAKE_WORD = "wake_word"
    COMMAND = "command"
    UNRESTRICTED = "unrestricted"
    NOT_TRIGGERED = "not_triggered"


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    allowed: bool
    trigger_type: TriggerType
    denied_reason: Optional[str] = None
    cleaned_text: str = ""
    is_group: bool = False


@dataclass(frozen=True, slots=True)
class BufferedGroupMessage:
    chat_id: int
    thread_id: int
    message_id: int
    user_id: int
    username: Optional[str]
    display_name: str
    text: str
    timestamp: float


class GroupChatService:
    """无网络依赖的群聊策略服务，缓冲接口可替换为 Redis 实现。"""

    def __init__(self, group_config: TelegramGroupConfig):
        self.config = group_config
        self._buffers: dict[tuple[int, int], Deque[BufferedGroupMessage]] = defaultdict(
            lambda: deque(maxlen=self.config.context_buffer_size)
        )
        self._buffer_lock = asyncio.Lock()
        self._seen_updates: dict[int, float] = {}
        self._seen_lock = asyncio.Lock()
        self._enabled_overrides: dict[int, bool] = {}
        self._scope_overrides: dict[int, str] = {}
        self._memory_overrides: dict[int, bool] = {}
        self._wake_word_overrides: dict[int, list[str]] = {}

    @staticmethod
    def _chat_type(update) -> str:
        chat_type = getattr(getattr(update, "effective_chat", None), "type", "")
        return str(getattr(chat_type, "value", chat_type)).lower()

    @staticmethod
    def _thread_id(update) -> int:
        message = getattr(update, "effective_message", None)
        return int(getattr(message, "message_thread_id", None) or 0)

    @staticmethod
    def _message_text(update) -> str:
        message = getattr(update, "effective_message", None)
        return str(
            getattr(message, "text", None) or getattr(message, "caption", None) or ""
        ).strip()

    def is_group(self, update) -> bool:
        return self._chat_type(update) in {"group", "supergroup"}

    def is_enabled(self, chat_id: int) -> bool:
        return self._enabled_overrides.get(chat_id, self.config.enabled)

    def session_scope(self, chat_id: int) -> str:
        return self._scope_overrides.get(chat_id, self.config.session_scope)

    def memory_enabled(self, chat_id: int) -> bool:
        return self._memory_overrides.get(
            chat_id, self.config.memory.enable_long_term_learning
        )

    def wake_words(self, chat_id: int) -> list[str]:
        return self._wake_word_overrides.get(chat_id, self.config.wake_words)

    def set_enabled(self, chat_id: int, enabled: bool) -> None:
        self._enabled_overrides[chat_id] = enabled

    def set_scope(self, chat_id: int, scope: str) -> bool:
        if scope not in {"per_user", "shared", "topic"}:
            return False
        self._scope_overrides[chat_id] = scope
        return True

    def set_memory_enabled(self, chat_id: int, enabled: bool) -> bool:
        if not self.config.memory.allow_admin_toggle:
            return False
        self._memory_overrides[chat_id] = enabled
        return True

    def set_wake_words(self, chat_id: int, words: list[str]) -> None:
        unique: list[str] = []
        seen: set[str] = set()
        for word in words:
            normalized = word.strip()
            folded = normalized.casefold()
            if normalized and folded not in seen:
                unique.append(normalized)
                seen.add(folded)
        self._wake_word_overrides[chat_id] = unique

    def session_key(self, update) -> SessionKey:
        chat_id = int(update.effective_chat.id)
        user_id = int(update.effective_user.id)
        if not self.is_group(update):
            return SessionKey(
                chat_id=chat_id, thread_id=0, user_id=user_id, scope="private"
            )

        scope = self.session_scope(chat_id)
        thread_id = self._thread_id(update)
        if scope == "shared":
            thread_id = 0
        return SessionKey(
            chat_id=chat_id,
            thread_id=thread_id,
            user_id=user_id if scope == "per_user" else 0,
            scope=scope,
        )

    def knowledge_namespace(self, session_key: SessionKey) -> str:
        """按独立的记忆配置计算知识命名空间。"""
        if session_key.scope == "private":
            return session_key.knowledge_namespace
        if self.config.memory.namespace_scope == "group":
            return f"group:{session_key.chat_id}"
        return f"topic:{session_key.chat_id}:{session_key.thread_id}"

    @staticmethod
    def _entity_text(message, entity, text: str) -> str:
        parser_name = (
            "parse_caption_entity"
            if getattr(message, "text", None) is None
            and getattr(message, "caption", None) is not None
            else "parse_entity"
        )
        parser = getattr(message, parser_name, None)
        if callable(parser):
            try:
                return str(parser(entity))
            except (RuntimeError, TypeError, ValueError):
                pass
        offset = int(getattr(entity, "offset", 0))
        length = int(getattr(entity, "length", 0))
        encoded = text.encode("utf-16-le")
        return encoded[offset * 2 : (offset + length) * 2].decode(
            "utf-16-le", errors="ignore"
        )

    def _mentions_current_bot(
        self, message, text: str, bot_username: str, bot_id: int
    ) -> bool:
        username = bot_username.lstrip("@").casefold()
        entities = list(getattr(message, "entities", None) or [])
        if not entities and getattr(message, "caption", None):
            entities = list(getattr(message, "caption_entities", None) or [])
        for entity in entities:
            entity_type = getattr(
                getattr(entity, "type", None), "value", getattr(entity, "type", "")
            )
            normalized_type = str(entity_type).lower()
            if normalized_type == "mention":
                mentioned = (
                    self._entity_text(message, entity, text).lstrip("@").casefold()
                )
                if mentioned == username:
                    return True
            elif normalized_type == "text_mention":
                mentioned_user = getattr(entity, "user", None)
                if int(getattr(mentioned_user, "id", 0)) == int(bot_id):
                    return True

        # 某些客户端或转发路径没有保留 entity，仍识别明确的 @username。
        if not username:
            return False
        pattern = rf"(?<![A-Za-z0-9_])@{re.escape(username)}(?![A-Za-z0-9_])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    @staticmethod
    def _is_reply_to_bot(message, bot_id: int) -> bool:
        replied = getattr(message, "reply_to_message", None)
        author = getattr(replied, "from_user", None)
        return bool(author and int(getattr(author, "id", 0)) == int(bot_id))

    def _clean_mention(
        self, text: str, bot_username: str, message=None, bot_id: int = 0
    ) -> str:
        username = bot_username.lstrip("@")
        if username:
            pattern = rf"(?<![A-Za-z0-9_])@{re.escape(username)}(?![A-Za-z0-9_])"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        entities = list(getattr(message, "entities", None) or [])
        if not entities and getattr(message, "caption", None):
            entities = list(getattr(message, "caption_entities", None) or [])
        for entity in entities:
            entity_type = getattr(
                getattr(entity, "type", None), "value", getattr(entity, "type", "")
            )
            mentioned_user = getattr(entity, "user", None)
            if str(entity_type).lower() == "text_mention" and int(
                getattr(mentioned_user, "id", 0)
            ) == int(bot_id):
                label = self._entity_text(message, entity, text)
                text = text.replace(label, "", 1)
        return " ".join(text.split()).strip()

    def decide(
        self,
        update,
        bot_username: str,
        bot_id: int,
        *,
        is_command: bool = False,
        management_command: bool = False,
    ) -> TriggerDecision:
        """在调用 Agent 前完成权限与触发判断。"""
        text = self._message_text(update)
        if not self.is_group(update):
            return TriggerDecision(True, TriggerType.PRIVATE, cleaned_text=text)

        chat_id = int(update.effective_chat.id)
        thread_id = self._thread_id(update)
        if not self.is_enabled(chat_id) and not management_command:
            return TriggerDecision(
                False, TriggerType.NOT_TRIGGERED, "group_disabled", text, True
            )
        if chat_id in self.config.blocked_chat_ids:
            return TriggerDecision(
                False, TriggerType.NOT_TRIGGERED, "chat_blocked", text, True
            )
        if self.config.allowed_chat_ids and chat_id not in self.config.allowed_chat_ids:
            return TriggerDecision(
                False, TriggerType.NOT_TRIGGERED, "chat_not_allowed", text, True
            )
        if thread_id in self.config.ignored_thread_ids:
            return TriggerDecision(
                False, TriggerType.NOT_TRIGGERED, "thread_ignored", text, True
            )
        if (
            self.config.allowed_thread_ids
            and thread_id not in self.config.allowed_thread_ids
        ):
            return TriggerDecision(
                False, TriggerType.NOT_TRIGGERED, "thread_not_allowed", text, True
            )

        sender = getattr(update, "effective_user", None)
        if self.config.ignore_bot_messages and bool(getattr(sender, "is_bot", False)):
            return TriggerDecision(
                False, TriggerType.NOT_TRIGGERED, "bot_sender", text, True
            )

        if is_command:
            command = text.split(maxsplit=1)[0]
            if "@" in command:
                target = command.rsplit("@", 1)[1]
                if target.casefold() != bot_username.lstrip("@").casefold():
                    return TriggerDecision(
                        False,
                        TriggerType.NOT_TRIGGERED,
                        "command_for_other_bot",
                        text,
                        True,
                    )
            return TriggerDecision(
                True, TriggerType.COMMAND, cleaned_text=text, is_group=True
            )

        message = update.effective_message
        if self._mentions_current_bot(message, text, bot_username, bot_id):
            return TriggerDecision(
                True,
                TriggerType.MENTION,
                cleaned_text=self._clean_mention(
                    text, bot_username, message=message, bot_id=bot_id
                ),
                is_group=True,
            )
        if self.config.allow_reply_trigger and self._is_reply_to_bot(message, bot_id):
            return TriggerDecision(
                True, TriggerType.REPLY, cleaned_text=text, is_group=True
            )
        folded_text = text.casefold()
        if any(word.casefold() in folded_text for word in self.wake_words(chat_id)):
            return TriggerDecision(
                True, TriggerType.WAKE_WORD, cleaned_text=text, is_group=True
            )
        if not self.config.require_mention:
            return TriggerDecision(
                True, TriggerType.UNRESTRICTED, cleaned_text=text, is_group=True
            )
        return TriggerDecision(
            False, TriggerType.NOT_TRIGGERED, "not_triggered", text, True
        )

    async def is_duplicate_update(self, update_id: Optional[int]) -> bool:
        if update_id is None:
            return False
        now = time.monotonic()
        async with self._seen_lock:
            cutoff = now - max(self.config.context_buffer_ttl, 300)
            self._seen_updates = {
                key: seen_at
                for key, seen_at in self._seen_updates.items()
                if seen_at >= cutoff
            }
            if update_id in self._seen_updates:
                return True
            self._seen_updates[update_id] = now
            return False

    async def observe(self, update) -> bool:
        """暂存一条未触发的群消息，不写数据库或长期记忆。"""
        if not self.config.observe_unmentioned_messages or not self.is_group(update):
            return False
        text = self._message_text(update)
        if not text:
            return False
        user = update.effective_user
        display_name = (
            " ".join(
                part
                for part in [
                    getattr(user, "first_name", None),
                    getattr(user, "last_name", None),
                ]
                if part
            )
            or getattr(user, "username", None)
            or str(user.id)
        )
        item = BufferedGroupMessage(
            chat_id=int(update.effective_chat.id),
            thread_id=self._thread_id(update),
            message_id=int(update.effective_message.message_id),
            user_id=int(user.id),
            username=getattr(user, "username", None),
            display_name=display_name,
            text=text,
            timestamp=(
                update.effective_message.date.timestamp()
                if getattr(update.effective_message, "date", None)
                else time.time()
            ),
        )
        async with self._buffer_lock:
            self._prune_locked((item.chat_id, item.thread_id), time.time())
            self._buffers[(item.chat_id, item.thread_id)].append(item)
        return True

    def _prune_locked(self, key: tuple[int, int], now: float) -> None:
        buffer = self._buffers.get(key)
        if not buffer:
            return
        cutoff = now - self.config.context_buffer_ttl
        while buffer and buffer[0].timestamp < cutoff:
            buffer.popleft()
        if not buffer:
            self._buffers.pop(key, None)

    async def get_context(
        self, chat_id: int, thread_id: int
    ) -> list[BufferedGroupMessage]:
        key = (chat_id, thread_id)
        async with self._buffer_lock:
            self._prune_locked(key, time.time())
            return list(self._buffers.get(key, ()))

    async def format_context(self, chat_id: int, thread_id: int) -> str:
        messages = await self.get_context(chat_id, thread_id)
        if not messages:
            return ""
        lines = ["[群聊上下文]"]
        lines.extend(f"{item.display_name}：{item.text}" for item in messages)
        return "\n".join(lines)

    async def context_count(self, chat_id: int, thread_id: int) -> int:
        return len(await self.get_context(chat_id, thread_id))

    async def is_admin(
        self, context, chat_id: int, user_id: int, global_admin_ids: list[int]
    ) -> bool:
        if user_id in global_admin_ids:
            return True
        try:
            member = await context.bot.get_chat_member(chat_id, user_id)
            status = getattr(
                getattr(member, "status", None), "value", getattr(member, "status", "")
            )
            return str(status).lower() in {"creator", "administrator", "owner"}
        except Exception as exc:
            logger.warning(
                "群管理员查询失败 chat_id={} user_id={} error={}", chat_id, user_id, exc
            )
            return False
