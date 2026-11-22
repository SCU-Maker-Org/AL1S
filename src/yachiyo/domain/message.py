"""Normalized transport events and stored conversation messages."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from .content import Attachment
from .identity import ConversationAddress, ExternalIdentity, SessionKey
from .permission import RequestPermission

AttachmentLoader = Callable[[], Awaitable[tuple[Attachment, ...]]]


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Actor:
    identity: ExternalIdentity
    display_name: str
    username: str | None = None
    is_bot: bool = False


@dataclass(frozen=True, slots=True)
class CommandInvocation:
    name: str
    arguments: str = ""


@dataclass(frozen=True, slots=True)
class InboundEvent:
    event_id: str
    address: ConversationAddress
    actor: Actor
    permission: RequestPermission
    text: str = ""
    attachments: tuple[Attachment, ...] = ()
    attachment_loader: AttachmentLoader | None = field(default=None, compare=False, repr=False)
    command: CommandInvocation | None = None
    mentions_bot: bool = False
    replies_to_bot: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def idempotency_key(self) -> tuple[str, str, str]:
        return self.address.transport.value, self.address.bot_id, self.event_id


@dataclass(frozen=True, slots=True)
class StoredMessage:
    role: MessageRole
    content: str
    created_at: datetime
    run_id: str | None = None
    model: str | None = None
    status: str = "complete"


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    session: SessionKey
    user_message: StoredMessage
    assistant_message: StoredMessage
    identity_id: int
    input_tokens: int
    output_tokens: int
    memory_scope: str | None = None
    memory_payload: dict[str, str] | None = None
