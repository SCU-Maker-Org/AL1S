"""内存对话管理与按会话粒度的并发控制。"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger

from ..config import config
from ..models import Conversation, Message, Role, SessionKey


@dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock
    last_used: float


class ConversationService:
    """维护短期对话历史；持久化由 ``DatabaseService`` 负责。"""

    MAX_MESSAGES = 50
    DEFAULT_LOCK_MAX_AGE = 3600

    def __init__(self, database_service=None):
        self.conversations: Dict[SessionKey, Conversation] = {}
        self.users: Dict[int, Dict[str, Any]] = {}
        self.roles: Dict[str, Role] = {}
        self.database_service = database_service
        self._locks: Dict[SessionKey, _LockEntry] = {}
        self._locks_guard = asyncio.Lock()
        self._initialize_roles()
        logger.info("对话服务初始化完成")

    def _initialize_roles(self) -> None:
        try:
            for role_name, role_config in config.roles.items():
                self.roles[role_name] = self._role_from_config(role_config)
                logger.info("加载角色: {}", role_name)
            logger.info("成功加载 {} 个角色", len(self.roles))
        except Exception as exc:
            logger.error("初始化角色失败: {}", exc)

    @staticmethod
    def _role_from_config(role_config) -> Role:
        return Role(
            name=role_config.name,
            english_name=role_config.english_name,
            description=role_config.description,
            personality=role_config.personality,
            greeting=role_config.greeting,
            farewell=role_config.farewell,
        )

    @staticmethod
    def _fallback_role() -> Role:
        return Role(
            name="AI助手",
            english_name="AI Assistant",
            description="智能AI助手",
            personality="你是一个智能、友好的AI助手，能够帮助用户解决各种问题。",
            greeting="您好！我是AI助手，很高兴为您服务。",
            farewell="感谢您的使用！如果还有其他问题，随时可以找我。",
        )

    def get_conversation(self, session_key: SessionKey) -> Conversation:
        """获取或创建明确作用域内的对话。"""
        if session_key not in self.conversations:
            default_role = config.get_default_role()
            role = (
                self._role_from_config(default_role)
                if default_role
                else self._fallback_role()
            )
            now = time.time()
            self.conversations[session_key] = Conversation(
                user_id=session_key.user_id,
                chat_id=session_key.chat_id,
                thread_id=session_key.thread_id,
                session_scope=session_key.scope,
                session_owner_id=session_key.owner_id,
                knowledge_namespace=session_key.knowledge_namespace,
                chat_type="private" if session_key.scope == "private" else "group",
                role=role,
                messages=[],
                created_at=now,
                last_activity=now,
            )
            logger.info(
                "创建对话 chat_id={} thread_id={} scope={} owner_id={}",
                session_key.chat_id,
                session_key.thread_id,
                session_key.scope,
                session_key.owner_id,
            )
        return self.conversations[session_key]

    def add_message(self, session_key: SessionKey, message: Message) -> None:
        conversation = self.get_conversation(session_key)
        conversation.messages.append(message)
        conversation.last_activity = time.time()
        if len(conversation.messages) > self.MAX_MESSAGES:
            conversation.messages = conversation.messages[-self.MAX_MESSAGES :]

    def set_role(self, session_key: SessionKey, role_name: str) -> bool:
        try:
            role_config = config.get_role(role_name)
            if not role_config:
                logger.warning("角色 {} 不存在", role_name)
                return False
            conversation = self.get_conversation(session_key)
            conversation.role = self._role_from_config(role_config)
            conversation.last_activity = time.time()
            if self.database_service:
                try:
                    asyncio.create_task(
                        self.database_service.update_role_stats(role_name)
                    )
                except RuntimeError as exc:
                    logger.warning("更新角色统计任务创建失败: {}", exc)
            return True
        except Exception as exc:
            logger.error("设置角色失败: {}", exc)
            return False

    def get_role(self, session_key: SessionKey) -> Optional[Role]:
        return self.get_conversation(session_key).role

    def list_roles(self) -> List[str]:
        return list(config.roles.keys())

    def create_role(self, session_key: SessionKey, role_data: Dict[str, Any]) -> bool:
        try:
            role = Role(
                name=role_data.get("name", "自定义角色"),
                english_name=role_data.get("english_name", "Custom Role"),
                description=role_data.get("description", "用户自定义角色"),
                personality=role_data.get("personality", "你是一个自定义角色。"),
                greeting=role_data.get("greeting", "你好！"),
                farewell=role_data.get("farewell", "再见！"),
            )
            self.roles[f"custom_{session_key.owner_id}_{int(time.time())}"] = role
            conversation = self.get_conversation(session_key)
            conversation.role = role
            conversation.last_activity = time.time()
            return True
        except Exception as exc:
            logger.error("创建角色失败: {}", exc)
            return False

    def create_custom_role(self, name: str, description: str, personality: str) -> bool:
        """兼容现有命令的运行期自定义角色接口。"""
        if name in self.roles:
            return False
        self.roles[name] = Role(
            name=name,
            english_name=name,
            description=description,
            personality=personality,
            greeting=f"你好，我是{name}。",
            farewell="再见！",
        )
        return True

    def reset_conversation(self, session_key: SessionKey) -> bool:
        conversation = self.conversations.get(session_key)
        if conversation is None:
            return False
        conversation.messages = []
        conversation.last_activity = time.time()
        return True

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        user_conversations = [
            conversation
            for key, conversation in self.conversations.items()
            if key.user_id == user_id and key.scope in {"private", "per_user"}
        ]
        return {
            "user_id": user_id,
            "total_messages": sum(len(item.messages) for item in user_conversations),
            "active_conversations": len(user_conversations),
            "current_role": (
                user_conversations[0].role.name
                if user_conversations and user_conversations[0].role
                else "无"
            ),
        }

    def get_session_stats(self, session_key: SessionKey) -> Dict[str, Any]:
        conversation = self.get_conversation(session_key)
        return {
            "scope": session_key.scope,
            "scope_label": session_key.label,
            "chat_id": session_key.chat_id,
            "thread_id": session_key.thread_id,
            "owner_id": session_key.owner_id,
            "message_count": len(conversation.messages),
            "current_role": conversation.role.name if conversation.role else "无",
        }

    async def _get_lock(self, session_key: SessionKey) -> asyncio.Lock:
        async with self._locks_guard:
            if len(self._locks) > 1024:
                cutoff = time.monotonic() - self.DEFAULT_LOCK_MAX_AGE
                self._locks = {
                    key: entry
                    for key, entry in self._locks.items()
                    if entry.lock.locked() or entry.last_used >= cutoff
                }
            entry = self._locks.get(session_key)
            if entry is None:
                entry = _LockEntry(asyncio.Lock(), time.monotonic())
                self._locks[session_key] = entry
            entry.last_used = time.monotonic()
            return entry.lock

    @asynccontextmanager
    async def session_lock(self, session_key: SessionKey) -> AsyncIterator[None]:
        """串行化同一会话，互不阻塞不同群或 Topic。"""
        lock = await self._get_lock(session_key)
        async with lock:
            try:
                yield
            finally:
                async with self._locks_guard:
                    entry = self._locks.get(session_key)
                    if entry:
                        entry.last_used = time.monotonic()

    def cleanup_expired_conversations(self, max_age: int = 3600) -> int:
        now = time.time()
        expired = [
            key
            for key, conversation in self.conversations.items()
            if now - conversation.last_activity > max_age
        ]
        for key in expired:
            self.conversations.pop(key, None)
        return len(expired)

    async def cleanup_expired_locks(self, max_age: int = DEFAULT_LOCK_MAX_AGE) -> int:
        now = time.monotonic()
        async with self._locks_guard:
            expired = [
                key
                for key, entry in self._locks.items()
                if not entry.lock.locked() and now - entry.last_used > max_age
            ]
            for key in expired:
                self._locks.pop(key, None)
            return len(expired)
