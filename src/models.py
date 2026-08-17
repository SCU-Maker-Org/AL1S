"""
数据模型模块
"""

import time
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SessionScope = Literal["private", "per_user", "shared", "topic"]


@dataclass(frozen=True, slots=True)
class SessionKey:
    """可解释、可哈希的对话作用域键。"""

    chat_id: int
    thread_id: int
    user_id: int
    scope: SessionScope

    @property
    def owner_id(self) -> int:
        """返回当前作用域的拥有者，群共享和 Topic 会话使用 0。"""
        return self.user_id if self.scope in {"private", "per_user"} else 0

    @property
    def knowledge_namespace(self) -> str:
        """返回长期知识隔离命名空间。"""
        if self.scope == "private":
            return f"private:{self.user_id}"
        if self.scope == "topic":
            return f"topic:{self.chat_id}:{self.thread_id}"
        return f"group:{self.chat_id}"

    @property
    def label(self) -> str:
        labels = {
            "private": "私聊",
            "per_user": "群内用户独立",
            "shared": "群共享",
            "topic": "Topic 共享",
        }
        return labels[self.scope]


class Message(BaseModel):
    """聊天消息模型"""

    role: str  # "user" 或 "assistant"
    content: str
    timestamp: float  # Unix时间戳


class Role(BaseModel):
    """角色模型"""

    name: str
    english_name: Optional[str] = None
    description: str
    personality: str  # 角色性格设定
    greeting: str  # 角色问候语
    farewell: str  # 角色告别语
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Conversation(BaseModel):
    """对话模型"""

    user_id: int
    chat_id: int
    thread_id: int = 0
    session_scope: SessionScope = "private"
    session_owner_id: int = 0
    knowledge_namespace: str = ""
    chat_type: str = "private"
    role: Optional[Role] = None
    messages: List[Message] = Field(default_factory=list)
    created_at: float = Field(default_factory=lambda: time.time())
    last_activity: float = Field(default_factory=lambda: time.time())


class ChatResponse(BaseModel):
    """聊天响应模型"""

    text: str
    role: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ImageSearchResult(BaseModel):
    """图片搜索结果模型"""

    source: str
    url: str
    title: Optional[str] = None
    similarity: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class Command(BaseModel):
    """命令模型"""

    name: str
    description: str
    usage: str
    aliases: List[str] = Field(default_factory=list)
    requires_args: bool = False
    admin_only: bool = False


class User(BaseModel):
    """用户模型"""

    id: int
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_bot: bool = False
    language_code: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)
    preferences: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeEntry:
    """知识条目类"""

    def __init__(
        self,
        id: int = None,
        user_id: int = None,
        conversation_id: int = None,
        title: str = "",
        content: str = "",
        summary: str = "",
        keywords: str = "",
        category: str = "general",
        importance_score: float = 0.0,
        embedding_id: str = None,
        source_message_id: int = None,
        knowledge_namespace: str = "",
        created_at=None,
    ):
        self.id = id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.title = title
        self.content = content
        self.summary = summary
        self.keywords = keywords
        self.category = category
        self.importance_score = importance_score
        self.embedding_id = embedding_id
        self.source_message_id = source_message_id
        self.knowledge_namespace = knowledge_namespace
        self.created_at = created_at or datetime.now()

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "keywords": self.keywords,
            "category": self.category,
            "importance_score": self.importance_score,
            "embedding_id": self.embedding_id,
            "source_message_id": self.source_message_id,
            "knowledge_namespace": self.knowledge_namespace,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class UserProfile(BaseModel):
    """用户显式提供、仅用于个性化对话的私有画像。"""

    telegram_user_id: int = Field(gt=0)
    content: str
    content_hash: str = ""
    source: str = "manual"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash:
            self.content_hash = sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MediaArtifact:
    """由 MCP 生成、等待聊天 transport 发送的受控媒体产物。"""

    artifact_id: str
    kind: Literal["photo", "voice"]
    relative_path: str
    mime_type: str
    byte_size: int
    sha256: str
    expires_at: float
    capture_nonce: str
    owner_tag: str
    caption: str = ""
