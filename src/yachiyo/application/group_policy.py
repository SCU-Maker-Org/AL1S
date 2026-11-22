"""Shared Telegram and Discord group trigger and session policy."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from yachiyo.config import AppSettings
from yachiyo.domain import ChatKind, InboundEvent, SessionKey, SessionScope
from yachiyo.ports.repository import Repository

_MANAGEMENT_COMMANDS = {
    "group_status",
    "group_enable",
    "group_disable",
    "group_scope",
    "group_memory",
    "group_wake_words",
}


@dataclass(frozen=True, slots=True)
class GroupDecision:
    allowed: bool
    session: SessionKey
    chat_row_id: int
    memory_enabled: bool
    reason: str = ""
    memory_consent_at: str | None = None


@dataclass(frozen=True, slots=True)
class BufferedContext:
    author: str
    text: str
    timestamp: float


class GroupContextBuffer:
    def __init__(self, settings: AppSettings):
        self.settings = settings.policies.group
        self._items: dict[str, deque[BufferedContext]] = defaultdict(
            lambda: deque(maxlen=self.settings.context_buffer_size)
        )
        self._lock = asyncio.Lock()
        self._operations = 0

    @staticmethod
    def _key(event: InboundEvent) -> str:
        return (
            f"{event.address.transport.value}:{event.address.bot_id}:"
            f"{event.address.chat_id}:{event.address.thread_id}"
        )

    async def observe(self, event: InboundEvent) -> None:
        text = event.text.strip()
        if not text or event.address.kind is ChatKind.PRIVATE:
            return
        async with self._lock:
            self._prune(self._key(event))
            self._items[self._key(event)].append(
                BufferedContext(event.actor.display_name, text[:4000], time.monotonic())
            )
            self._operations += 1
            if self._operations % 100 == 0 or len(self._items) > 5000:
                self._prune_all()

    async def context(self, event: InboundEvent) -> str:
        if event.address.kind is ChatKind.PRIVATE:
            return ""
        key = self._key(event)
        async with self._lock:
            self._prune(key)
            items = tuple(self._items.get(key, ()))
        return "\n".join(f"{item.author}: {item.text}" for item in items)

    def _prune(self, key: str) -> None:
        items = self._items.get(key)
        if not items:
            return
        cutoff = time.monotonic() - self.settings.context_buffer_ttl_seconds
        while items and items[0].timestamp < cutoff:
            items.popleft()
        if not items:
            self._items.pop(key, None)

    def _prune_all(self) -> None:
        for key in tuple(self._items):
            self._prune(key)
        while len(self._items) > 5000:
            self._items.pop(next(iter(self._items)))


class GroupPolicy:
    def __init__(self, database: Repository, settings: AppSettings):
        self.database = database
        self.settings = settings

    async def decide(self, event: InboundEvent, chat_row_id: int) -> GroupDecision:
        if event.address.kind is ChatKind.PRIVATE:
            return GroupDecision(
                True,
                SessionKey(event.address, event.actor.identity, SessionScope.PRIVATE),
                chat_row_id,
                self.settings.memory.enabled,
            )

        configured = self.settings.policies.group
        stored = await self.database.get_group_settings(chat_row_id)
        enabled = stored["enabled"] if stored else configured.enabled
        scope_value = stored["session_scope"] if stored else configured.session_scope
        consent_at = stored["consent_at"] if stored else None
        memory_enabled = bool(
            self.settings.memory.enabled and stored and stored["memory_enabled"] and consent_at
        )
        wake_words = tuple(stored["wake_words"]) if stored else configured.wake_words
        command_name = event.command.name if event.command else ""
        management = command_name in _MANAGEMENT_COMMANDS

        scope = SessionScope(scope_value)
        session = SessionKey(event.address, event.actor.identity, scope)
        if event.address.chat_id in configured.blocked_chat_ids:
            return GroupDecision(False, session, chat_row_id, memory_enabled, "chat_blocked")
        if configured.allowed_chat_ids and event.address.chat_id not in configured.allowed_chat_ids:
            return GroupDecision(False, session, chat_row_id, memory_enabled, "chat_not_allowed")
        if event.address.thread_id in configured.blocked_thread_ids:
            return GroupDecision(False, session, chat_row_id, memory_enabled, "thread_blocked")
        if (
            configured.allowed_thread_ids
            and event.address.thread_id not in configured.allowed_thread_ids
        ):
            return GroupDecision(False, session, chat_row_id, memory_enabled, "thread_not_allowed")
        if not enabled and not management:
            return GroupDecision(False, session, chat_row_id, memory_enabled, "group_disabled")
        if event.command is not None:
            return GroupDecision(
                True,
                session,
                chat_row_id,
                memory_enabled,
                memory_consent_at=consent_at,
            )
        if event.mentions_bot or (configured.allow_reply_trigger and event.replies_to_bot):
            return GroupDecision(
                True,
                session,
                chat_row_id,
                memory_enabled,
                memory_consent_at=consent_at,
            )
        folded = event.text.casefold()
        if any(word.casefold() in folded for word in wake_words):
            return GroupDecision(
                True,
                session,
                chat_row_id,
                memory_enabled,
                memory_consent_at=consent_at,
            )
        if not configured.require_mention:
            return GroupDecision(
                True,
                session,
                chat_row_id,
                memory_enabled,
                memory_consent_at=consent_at,
            )
        return GroupDecision(False, session, chat_row_id, memory_enabled, "not_triggered")
