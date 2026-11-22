"""Identity and conversation isolation primitives."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum


class TransportName(StrEnum):
    TELEGRAM = "telegram"
    DISCORD = "discord"


class ChatKind(StrEnum):
    PRIVATE = "private"
    GROUP = "group"
    CHANNEL = "channel"


class SessionScope(StrEnum):
    PRIVATE = "private"
    PER_USER = "per_user"
    SHARED = "shared"
    TOPIC = "topic"


@dataclass(frozen=True, slots=True)
class ExternalIdentity:
    provider: TransportName
    external_id: str

    def __post_init__(self) -> None:
        if not self.external_id:
            raise ValueError("external identity must not be empty")

    @property
    def key(self) -> str:
        return f"{self.provider.value}:{self.external_id}"


@dataclass(frozen=True, slots=True)
class ConversationAddress:
    transport: TransportName
    bot_id: str
    chat_id: str
    thread_id: str = "0"
    kind: ChatKind = ChatKind.PRIVATE

    def __post_init__(self) -> None:
        if not self.bot_id or not self.chat_id:
            raise ValueError("bot_id and chat_id must not be empty")


@dataclass(frozen=True, slots=True)
class SessionKey:
    address: ConversationAddress
    actor: ExternalIdentity
    scope: SessionScope

    @property
    def owner(self) -> str:
        if self.scope in {SessionScope.PRIVATE, SessionScope.PER_USER}:
            return self.actor.key
        return "shared"

    @property
    def stable_id(self) -> str:
        thread_id = self.address.thread_id if self.scope is SessionScope.TOPIC else "0"
        raw = "\x1f".join(
            (
                self.address.transport.value,
                self.address.bot_id,
                self.address.chat_id,
                thread_id,
                self.scope.value,
                self.owner,
            )
        ).encode()
        return hashlib.blake2b(raw, digest_size=16).hexdigest()

    @property
    def knowledge_scope(self) -> str:
        prefix = f"{self.address.transport.value}:{self.address.bot_id}"
        if self.scope is SessionScope.PRIVATE:
            return f"private:{prefix}:{self.actor.key}"
        base = (
            f"topic:{prefix}:{self.address.chat_id}:{self.address.thread_id}"
            if self.scope is SessionScope.TOPIC
            else f"group:{prefix}:{self.address.chat_id}"
        )
        if self.scope is SessionScope.PER_USER:
            return f"{base}:user:{self.actor.key}"
        return base
