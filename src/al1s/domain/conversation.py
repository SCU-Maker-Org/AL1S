"""Conversation, role and retrieval scope entities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .identity import SessionKey


@dataclass(frozen=True, slots=True)
class KnowledgeScope:
    key: str

    def __post_init__(self) -> None:
        if not self.key or len(self.key) > 512:
            raise ValueError("invalid knowledge scope")


@dataclass(frozen=True, slots=True)
class RoleDefinition:
    name: str
    description: str
    system_prompt: str
    greeting: str
    farewell: str
    owner_key: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationState:
    session: SessionKey
    role_name: str
    created_at: datetime
    updated_at: datetime
