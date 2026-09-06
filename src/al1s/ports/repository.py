"""Persistence contract used by application use cases."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from al1s.domain import (
    Actor,
    ChatCompletion,
    ConversationAddress,
    ConversationState,
    ExternalIdentity,
    InboundEvent,
    KnowledgeScope,
    ResponsePlan,
    RoleDefinition,
    SessionKey,
    StoredMessage,
)


class Repository(Protocol):
    async def claim_event(self, transport: str, bot_id: str, event_id: str) -> str | None: ...

    async def renew_event(
        self, transport: str, bot_id: str, event_id: str, processing_token: str
    ) -> bool: ...

    async def event_response(
        self, transport: str, bot_id: str, event_id: str
    ) -> ResponsePlan | None: ...

    async def complete_event(
        self,
        event: InboundEvent,
        response: ResponsePlan,
        processing_token: str,
        completion: ChatCompletion | None = None,
    ) -> None: ...

    async def pending_responses(
        self, transport: str, bot_id: str, limit: int = 100
    ) -> Sequence[tuple[InboundEvent, ResponsePlan]]: ...

    async def event_delivery_pending(self, transport: str, bot_id: str, event_id: str) -> bool: ...

    async def mark_event_delivered(self, transport: str, bot_id: str, event_id: str) -> None: ...

    async def mark_event_delivery_failed(
        self, transport: str, bot_id: str, event_id: str, error: str
    ) -> None: ...

    async def discard_event(
        self, transport: str, bot_id: str, event_id: str, processing_token: str
    ) -> None: ...

    async def ensure_identity(self, actor: Actor) -> int: ...

    async def identity_id(self, identity: ExternalIdentity) -> int | None: ...

    async def ensure_chat(self, address: ConversationAddress, title: str | None = None) -> int: ...

    async def ensure_conversation(
        self, session: SessionKey, default_role: str
    ) -> ConversationState: ...

    async def get_conversation(self, session: SessionKey) -> ConversationState | None: ...

    async def set_conversation_role(
        self, session: SessionKey, role_name: str, event_key: str | None = None
    ) -> None: ...

    async def append_message(self, session: SessionKey, message: StoredMessage) -> int: ...

    async def recent_messages(self, session: SessionKey, limit: int) -> Sequence[StoredMessage]: ...

    async def reset_conversation(
        self, session: SessionKey, event_key: str | None = None
    ) -> int: ...

    async def list_roles(self, owner_key: str | None = None) -> Sequence[RoleDefinition]: ...

    async def get_role(self, name: str, owner_key: str | None = None) -> RoleDefinition | None: ...

    async def create_role(self, role: RoleDefinition) -> bool: ...

    async def seed_roles(self, roles: tuple[RoleDefinition, ...]) -> None: ...

    async def get_profile(self, identity_id: int) -> str | None: ...

    async def set_profile(self, identity_id: int, content: str, source: str) -> None: ...

    async def set_profile_once(
        self,
        identity_id: int,
        content: str,
        source: str,
        event_key: str,
        expected_hash: str | None,
    ) -> str: ...

    async def clear_profile(self, identity_id: int, event_key: str | None = None) -> bool: ...

    async def get_group_settings(self, chat_id: int) -> dict[str, Any] | None: ...

    async def set_group_settings(
        self,
        chat_id: int,
        defaults: dict[str, Any],
        *,
        event_key: str | None = None,
        **values: Any,
    ) -> None: ...

    async def list_memories(
        self, scope: KnowledgeScope, limit: int = 20
    ) -> Sequence[dict[str, Any]]: ...

    async def delete_memory(self, entry_id: str, scope: KnowledgeScope) -> bool: ...

    async def memory_ids_beyond(self, scope: KnowledgeScope, keep: int) -> Sequence[str]: ...

    async def database_stats(self) -> dict[str, int]: ...

    async def user_stats(self, identity_id: int) -> dict[str, int]: ...

    async def record_usage(
        self,
        identity_id: int,
        conversation_id: str,
        run_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None: ...

    async def enqueue_job(
        self, kind: str, scope: str, payload: dict[str, Any], *, delay_seconds: int = 0
    ) -> str: ...
