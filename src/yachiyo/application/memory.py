"""Explicit and background long-term memory extraction."""

from __future__ import annotations

import json
import math
import re
import uuid

from yachiyo.config import AppSettings
from yachiyo.domain import KnowledgeScope
from yachiyo.ports.knowledge import KnowledgeIndex
from yachiyo.ports.model import ModelClient, ModelMessage, ModelRequest
from yachiyo.ports.repository import Repository

from .sensitive import contains_secret


class MemoryService:
    def __init__(
        self,
        database: Repository,
        index: KnowledgeIndex,
        model: ModelClient,
        settings: AppSettings,
    ):
        self.database = database
        self.index = index
        self.model = model
        self.settings = settings

    async def remember(
        self,
        scope: KnowledgeScope,
        content: str,
        *,
        category: str = "user_confirmed",
        importance: float = 1.0,
        entry_id: str | None = None,
        consent: tuple[int, str] | None = None,
        effect_event_key: str | None = None,
    ) -> str | None:
        normalized = " ".join(content.split()).strip()
        if not normalized or len(normalized) > 4000:
            raise ValueError("记忆内容必须为 1 到 4000 个字符。")
        if contains_secret(normalized):
            raise ValueError("记忆内容疑似包含密钥或 Token，已拒绝保存。")
        existing = await self.database.list_memories(scope, limit=100)
        for item in existing:
            if " ".join(str(item["content"]).split()).casefold() == normalized.casefold():
                stored = await self.index.store_memory(
                    str(item["id"]),
                    scope,
                    normalized,
                    category,
                    importance,
                    consent=consent,
                    effect_event_key=effect_event_key,
                )
                return str(item["id"]) if stored else None
        entry_id = entry_id or uuid.uuid4().hex
        stored = await self.index.store_memory(
            entry_id,
            scope,
            normalized,
            category,
            importance,
            consent=consent,
            effect_event_key=effect_event_key,
        )
        if not stored:
            return None
        stale = await self.database.memory_ids_beyond(
            scope, self.settings.memory.max_entries_per_scope
        )
        for stale_id in stale:
            await self.index.delete_memory(stale_id, scope)
        return entry_id

    async def forget(
        self, scope: KnowledgeScope, entry_id: str, *, effect_event_key: str | None = None
    ) -> bool:
        if not re.fullmatch(r"[0-9a-f]{32}", entry_id):
            return False
        return await self.index.delete_memory(entry_id, scope, effect_event_key=effect_event_key)

    async def extract_exchange(self, payload: dict[str, object]) -> None:
        if not self.settings.memory.enabled or not await self._consent_is_current(payload):
            return
        scope = KnowledgeScope(str(payload["scope"]))
        user_text = str(payload["user_text"])
        assistant_text = str(payload["assistant_text"])
        request = ModelRequest(
            model=self.settings.models.chat.model,
            instructions=(
                "Extract only durable, useful facts the user intentionally disclosed. "
                "Never store credentials, secrets, sensitive identifiers, guesses, or tool output. "
                'Return strict JSON with this shape: {"memories":[{"content":"...",'
                '"category":"...","importance":0.5}]}. Return an empty array when '
                "nothing is appropriate."
            ),
            messages=(
                ModelMessage(
                    role="user",
                    content=f"User:\n{user_text}\n\nAssistant:\n{assistant_text}",
                ),
            ),
            temperature=0.0,
            max_output_tokens=600,
        )
        result = await self.model.complete(request)
        if not await self._consent_is_current(payload):
            return
        raw = result.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S)
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return
        memories = decoded.get("memories", []) if isinstance(decoded, dict) else []
        if not isinstance(memories, list):
            return
        for item in memories[:5]:
            if not isinstance(item, dict) or not isinstance(item.get("content"), str):
                continue
            try:
                importance = float(item.get("importance", 0.5))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(importance) or contains_secret(item["content"]):
                continue
            try:
                consent = None
                if payload.get("group_chat_id") is not None:
                    consent = (
                        int(str(payload["group_chat_id"])),
                        str(payload.get("group_consent_at", "")),
                    )
                await self.remember(
                    scope,
                    item["content"],
                    category=str(item.get("category", "general"))[:64],
                    importance=max(0.0, min(1.0, importance)),
                    consent=consent,
                )
            except ValueError:
                continue

    async def _consent_is_current(self, payload: dict[str, object]) -> bool:
        chat_id = payload.get("group_chat_id")
        if chat_id is None:
            return True
        try:
            configured = await self.database.get_group_settings(int(str(chat_id)))
        except ValueError:
            return False
        return bool(
            configured
            and configured["memory_enabled"]
            and configured["consent_at"]
            and configured["consent_at"] == payload.get("group_consent_at")
        )
