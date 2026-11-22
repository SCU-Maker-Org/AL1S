"""Explicit private profile management."""

from __future__ import annotations

import hashlib

from yachiyo.config import AppSettings
from yachiyo.ports.repository import Repository

from .sensitive import contains_secret

PROFILE_TEMPLATE = """# Identity and background
- Preferred name:
- Locale / time zone:
- Work or study context:

# Communication preferences
- Default language:
- Preferred response length:
- Preferred explanation style:

# Technical context
- Languages and frameworks:
- Operating systems and infrastructure:
- Current projects:

# Other preferences
- Information yachiyo may remember:
"""


class ProfileError(ValueError):
    pass


class ProfileService:
    def __init__(self, database: Repository, settings: AppSettings):
        self.database = database
        self.settings = settings.profile

    @staticmethod
    def template() -> str:
        return PROFILE_TEMPLATE

    def normalize(self, content: str) -> str:
        normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            raise ProfileError("画像内容不能为空。")
        if len(normalized.encode()) > self.settings.max_document_bytes:
            raise ProfileError("画像内容超过大小限制。")
        if self.settings.reject_secrets and contains_secret(normalized):
            raise ProfileError("画像疑似包含密钥或 Token，请移除后重试。")
        return normalized

    async def set(self, identity_id: int, content: str, *, event_key: str | None = None) -> None:
        normalized = self.normalize(content)
        if event_key is None:
            await self.database.set_profile(identity_id, normalized, "manual")
            return
        await self._set_once(identity_id, normalized, event_key)

    async def append(self, identity_id: int, content: str, *, event_key: str | None = None) -> None:
        if event_key is None:
            current = await self.database.get_profile(identity_id)
            combined = f"{current}\n\n{content}" if current else content
            await self.set(identity_id, combined)
            return
        for _ in range(3):
            current = await self.database.get_profile(identity_id)
            combined = self.normalize(f"{current}\n\n{content}" if current else content)
            result = await self.database.set_profile_once(
                identity_id,
                combined,
                "manual",
                event_key,
                self._hash(current),
            )
            if result != "conflict":
                return
        raise ProfileError("画像同时被其他请求修改，请重试。")

    async def _set_once(self, identity_id: int, content: str, event_key: str) -> None:
        for _ in range(3):
            current = await self.database.get_profile(identity_id)
            result = await self.database.set_profile_once(
                identity_id,
                content,
                "manual",
                event_key,
                self._hash(current),
            )
            if result != "conflict":
                return
        raise ProfileError("画像同时被其他请求修改，请重试。")

    @staticmethod
    def _hash(content: str | None) -> str | None:
        return hashlib.sha256(content.encode()).hexdigest() if content is not None else None

    async def prompt_context(self, identity_id: int) -> str:
        content = await self.database.get_profile(identity_id)
        if not content:
            return ""
        return content[: self.settings.max_prompt_chars]
