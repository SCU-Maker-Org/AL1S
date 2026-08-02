"""
数据库服务模块
与SQLite数据库交互，记录用户对话和统计信息
"""

import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ..models import Message, SessionKey

_CJK_RUN_PATTERN = re.compile(r"[\u3400-\u9fff]{2,}")


def _cjk_bigrams(text: str) -> List[str]:
    terms: List[str] = []
    for match in _CJK_RUN_PATTERN.finditer(text):
        value = match.group(0)
        terms.extend(value[index : index + 2] for index in range(len(value) - 1))
    return list(dict.fromkeys(terms))


def _fts_index_text(text: str) -> str:
    """Append searchable CJK bigrams while preserving the original text for ranking."""
    bigrams = _cjk_bigrams(text)
    return f"{text}\n{' '.join(bigrams)}" if bigrams else text


class DatabaseService:
    """数据库服务类"""

    def __init__(self, db_path: str = "data/bot.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)

        # 确保数据库文件存在
        if not self.db_path.exists():
            logger.warning(f"数据库文件不存在: {self.db_path}")
            self._initialize_database()

        self._migrate_database()
        try:
            os.chmod(self.db_path, 0o600)
        except OSError as exc:
            logger.warning("无法收紧数据库文件权限: {}", exc)

        logger.info(f"数据库服务初始化完成: {self.db_path}")

    def _migrate_database(self) -> None:
        """幂等升级旧数据库，并保留已有私聊对话和消息外键。"""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS schema_migrations (
                           version INTEGER PRIMARY KEY,
                           applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
                       )"""
                )
                applied = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = 2"
                ).fetchone()
                if applied:
                    self._migrate_v3(conn)
                    self._migrate_v4(conn)
                    self._migrate_v5(conn)
                    self._migrate_v6(conn)
                    return

                columns = {
                    row["name"]
                    for row in conn.execute(
                        "PRAGMA table_info(conversations)"
                    ).fetchall()
                }
                if "session_scope" not in columns:
                    conn.commit()
                    conn.execute("PRAGMA foreign_keys = OFF")
                    conn.executescript(
                        """
                        BEGIN IMMEDIATE;
                        CREATE TABLE conversations_v2 (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            chat_id INTEGER NOT NULL,
                            thread_id INTEGER NOT NULL DEFAULT 0,
                            session_scope TEXT NOT NULL DEFAULT 'private',
                            session_owner_id INTEGER NOT NULL DEFAULT 0,
                            knowledge_namespace TEXT NOT NULL DEFAULT '',
                            chat_type TEXT NOT NULL DEFAULT 'private',
                            role_name TEXT DEFAULT 'AI助手',
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (user_id) REFERENCES users (id),
                            UNIQUE(chat_id, thread_id, session_scope, session_owner_id)
                        );
                        INSERT INTO conversations_v2 (
                            id, user_id, chat_id, thread_id, session_scope,
                            session_owner_id, knowledge_namespace, chat_type,
                            role_name, created_at, updated_at
                        )
                        SELECT c.id, c.user_id, c.chat_id, 0, 'private',
                               u.telegram_user_id,
                               'private:' || u.telegram_user_id,
                               'private', c.role_name, c.created_at, c.updated_at
                        FROM conversations c
                        JOIN users u ON u.id = c.user_id;
                        DROP TABLE conversations;
                        ALTER TABLE conversations_v2 RENAME TO conversations;
                        CREATE INDEX IF NOT EXISTS idx_conversations_user_chat
                            ON conversations(user_id, chat_id);
                        CREATE INDEX IF NOT EXISTS idx_conversations_session
                            ON conversations(chat_id, thread_id, session_scope, session_owner_id);
                        COMMIT;
                        """
                    )
                    conn.execute("PRAGMA foreign_keys = ON")

                knowledge_columns = {
                    row["name"]
                    for row in conn.execute(
                        "PRAGMA table_info(knowledge_entries)"
                    ).fetchall()
                }
                if knowledge_columns and "knowledge_namespace" not in knowledge_columns:
                    conn.execute(
                        "ALTER TABLE knowledge_entries ADD COLUMN knowledge_namespace TEXT NOT NULL DEFAULT ''"
                    )
                    conn.execute(
                        """UPDATE knowledge_entries
                           SET knowledge_namespace = COALESCE(
                               (SELECT c.knowledge_namespace
                                  FROM conversations c
                                 WHERE c.id = knowledge_entries.conversation_id),
                               'private:' || user_id
                           )
                           WHERE knowledge_namespace = ''"""
                    )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_knowledge_namespace ON knowledge_entries(knowledge_namespace)"
                )
                conn.execute(
                    "INSERT OR IGNORE INTO schema_migrations(version) VALUES (2)"
                )
                conn.commit()
                logger.info("数据库 schema v2 群聊会话迁移完成")
                self._migrate_v3(conn)
                self._migrate_v4(conn)
                self._migrate_v5(conn)
                self._migrate_v6(conn)
        except Exception as exc:
            logger.error("数据库迁移失败: {}", exc)
            raise

    @staticmethod
    def _migrate_v3(conn: sqlite3.Connection) -> None:
        """新增独立文档 RAG、全文索引和用户画像。"""
        if conn.execute("SELECT 1 FROM schema_migrations WHERE version = 3").fetchone():
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rag_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection TEXT NOT NULL,
                knowledge_namespace TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                title TEXT NOT NULL,
                domain TEXT NOT NULL DEFAULT 'sys',
                subdomain TEXT NOT NULL DEFAULT '',
                product TEXT NOT NULL DEFAULT '',
                version TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT '',
                license TEXT NOT NULL DEFAULT '',
                trust_level INTEGER NOT NULL DEFAULT 50,
                content_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(collection, knowledge_namespace, source_uri)
            );
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                heading_path TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES rag_documents (id) ON DELETE CASCADE,
                UNIQUE(document_id, chunk_index)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
                chunk_id UNINDEXED,
                collection UNINDEXED,
                knowledge_namespace UNINDEXED,
                title,
                heading_path,
                content,
                domain,
                product,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            CREATE TABLE IF NOT EXISTS rag_ingestion_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_root TEXT NOT NULL,
                collection TEXT NOT NULL,
                knowledge_namespace TEXT NOT NULL,
                documents_seen INTEGER NOT NULL DEFAULT 0,
                documents_changed INTEGER NOT NULL DEFAULT 0,
                chunks_written INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                error_message TEXT,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                finished_at DATETIME
            );
            CREATE TABLE IF NOT EXISTS user_profiles (
                telegram_user_id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'manual',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_rag_documents_scope
                ON rag_documents(collection, knowledge_namespace, domain, product);
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_document ON rag_chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_hash ON rag_chunks(content_hash);
            INSERT OR IGNORE INTO schema_migrations(version) VALUES (3);
            """
        )
        conn.commit()
        logger.info("数据库 schema v3 文档 RAG 与用户画像迁移完成")

    @staticmethod
    def _migrate_v4(conn: sqlite3.Connection) -> None:
        """为 RAG 文档增加稳定身份，允许来源 URL 原地更新。"""
        if conn.execute("SELECT 1 FROM schema_migrations WHERE version = 4").fetchone():
            return
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(rag_documents)").fetchall()
        }
        if "document_key" not in columns:
            conn.execute(
                "ALTER TABLE rag_documents ADD COLUMN document_key TEXT NOT NULL DEFAULT ''"
            )
        conn.execute(
            "UPDATE rag_documents SET document_key = source_uri WHERE document_key = ''"
        )
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_documents_identity
               ON rag_documents(collection, knowledge_namespace, document_key)"""
        )
        conn.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (4)")
        conn.commit()
        logger.info("数据库 schema v4 RAG 稳定文档身份迁移完成")

    @staticmethod
    def _migrate_v5(conn: sqlite3.Connection) -> None:
        """记录语料根目录，并为现有中文内容补充 FTS bigram。"""
        if conn.execute("SELECT 1 FROM schema_migrations WHERE version = 5").fetchone():
            return
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(rag_documents)").fetchall()
        }
        if "source_root" not in columns:
            conn.execute(
                "ALTER TABLE rag_documents ADD COLUMN source_root TEXT NOT NULL DEFAULT ''"
            )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_rag_documents_source_root
               ON rag_documents(collection, knowledge_namespace, source_root)"""
        )

        rows = conn.execute(
            """SELECT c.id AS chunk_id, c.heading_path, c.content,
                      d.collection, d.knowledge_namespace, d.title, d.domain, d.product
                 FROM rag_chunks c
                 JOIN rag_documents d ON d.id = c.document_id
                 ORDER BY c.id"""
        ).fetchall()
        conn.execute("DELETE FROM rag_chunks_fts")
        for row in rows:
            conn.execute(
                """INSERT INTO rag_chunks_fts (
                       chunk_id, collection, knowledge_namespace, title, heading_path,
                       content, domain, product
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(row["chunk_id"]),
                    row["collection"],
                    row["knowledge_namespace"],
                    _fts_index_text(str(row["title"])),
                    _fts_index_text(str(row["heading_path"])),
                    _fts_index_text(str(row["content"])),
                    row["domain"],
                    row["product"],
                ),
            )
        conn.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (5)")
        conn.commit()
        logger.info("数据库 schema v5 RAG 语料追踪与中文 FTS 迁移完成")

    @staticmethod
    def _migrate_v6(conn: sqlite3.Connection) -> None:
        """隔离旧版群内 per_user 会话及其长期记忆。"""
        if conn.execute("SELECT 1 FROM schema_migrations WHERE version = 6").fetchone():
            return
        conn.execute(
            """UPDATE conversations
                  SET knowledge_namespace = knowledge_namespace || ':user:' || session_owner_id,
                      updated_at = CURRENT_TIMESTAMP
                WHERE session_scope = 'per_user'
                  AND knowledge_namespace NOT LIKE '%:user:%'"""
        )
        conn.execute(
            """UPDATE knowledge_entries
                  SET knowledge_namespace = (
                          SELECT c.knowledge_namespace
                            FROM conversations c
                           WHERE c.id = knowledge_entries.conversation_id
                      ),
                      updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id IN (
                          SELECT id FROM conversations WHERE session_scope = 'per_user'
                      )"""
        )
        conn.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (6)")
        conn.commit()
        logger.info("数据库 schema v6 群内用户长期记忆隔离迁移完成")

    def _initialize_database(self):
        """初始化数据库（如果需要）"""
        try:
            init_sql_path = self.db_path.parent / "init_db.sql"
            if init_sql_path.exists():
                with open(init_sql_path, "r", encoding="utf-8") as f:
                    init_sql = f.read()

                with sqlite3.connect(self.db_path) as conn:
                    conn.executescript(init_sql)
                    conn.commit()

                logger.info("数据库初始化完成")
            else:
                logger.warning("未找到数据库初始化脚本")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")

    def get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使结果可以按列名访问
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    async def ensure_user(
        self,
        telegram_user_id: int,
        username: str = None,
        first_name: str = None,
        last_name: str = None,
    ) -> int:
        """确保用户存在，返回用户ID"""
        try:
            with self.get_connection() as conn:
                # 检查用户是否存在
                cursor = conn.execute(
                    "SELECT id FROM users WHERE telegram_user_id = ?",
                    (telegram_user_id,),
                )
                user = cursor.fetchone()

                if user:
                    # 更新用户信息
                    conn.execute(
                        """UPDATE users SET 
                           username = COALESCE(?, username),
                           first_name = COALESCE(?, first_name),
                           last_name = COALESCE(?, last_name),
                           updated_at = CURRENT_TIMESTAMP
                           WHERE telegram_user_id = ?""",
                        (username, first_name, last_name, telegram_user_id),
                    )
                    return user["id"]
                else:
                    # 创建新用户
                    cursor = conn.execute(
                        """INSERT INTO users (telegram_user_id, username, first_name, last_name)
                           VALUES (?, ?, ?, ?)""",
                        (telegram_user_id, username, first_name, last_name),
                    )
                    logger.info(f"创建新用户: {telegram_user_id}")
                    return cursor.lastrowid

        except Exception as e:
            logger.error(f"确保用户存在失败: {e}")
            return None

    async def ensure_conversation(
        self,
        user_id: int,
        session_key: SessionKey,
        role_name: str = "AI助手",
        chat_type: str = "private",
        knowledge_namespace: Optional[str] = None,
    ) -> Optional[int]:
        """确保对话存在，返回对话ID"""
        try:
            with self.get_connection() as conn:
                # 检查对话是否存在
                cursor = conn.execute(
                    """SELECT id FROM conversations
                       WHERE chat_id = ? AND thread_id = ?
                         AND session_scope = ? AND session_owner_id = ?""",
                    (
                        session_key.chat_id,
                        session_key.thread_id,
                        session_key.scope,
                        session_key.owner_id,
                    ),
                )
                conversation = cursor.fetchone()

                if conversation:
                    # 更新角色信息
                    conn.execute(
                        """UPDATE conversations SET 
                           role_name = ?,
                           knowledge_namespace = ?,
                           updated_at = CURRENT_TIMESTAMP
                           WHERE id = ?""",
                        (
                            role_name,
                            knowledge_namespace or session_key.knowledge_namespace,
                            conversation["id"],
                        ),
                    )
                    return conversation["id"]
                else:
                    # 创建新对话
                    cursor = conn.execute(
                        """INSERT INTO conversations (
                               user_id, chat_id, thread_id, session_scope,
                               session_owner_id, knowledge_namespace, chat_type, role_name
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            user_id,
                            session_key.chat_id,
                            session_key.thread_id,
                            session_key.scope,
                            session_key.owner_id,
                            knowledge_namespace or session_key.knowledge_namespace,
                            chat_type,
                            role_name,
                        ),
                    )
                    logger.info(
                        "创建持久对话 chat_id={} thread_id={} scope={}",
                        session_key.chat_id,
                        session_key.thread_id,
                        session_key.scope,
                    )
                    return cursor.lastrowid

        except Exception as e:
            logger.error(f"确保对话存在失败: {e}")
            return None

    async def save_message(self, conversation_id: int, message: Message) -> bool:
        """保存消息到数据库"""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """INSERT INTO messages (conversation_id, role, content, timestamp, token_count)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        conversation_id,
                        message.role,
                        message.content,
                        datetime.fromtimestamp(message.timestamp),
                        len(message.content.split()),
                    ),  # 简单的token计数
                )
                return True

        except Exception as e:
            logger.error(f"保存消息失败: {e}")
            return False

    async def get_conversation_history(
        self, conversation_id: int, limit: int = 10
    ) -> List[Message]:
        """获取对话历史"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """SELECT role, content, timestamp FROM messages
                       WHERE conversation_id = ?
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (conversation_id, limit),
                )

                messages = []
                for row in cursor.fetchall():
                    timestamp = datetime.fromisoformat(row["timestamp"]).timestamp()
                    messages.append(
                        Message(
                            role=row["role"],
                            content=row["content"],
                            timestamp=timestamp,
                        )
                    )

                return list(reversed(messages))  # 按时间正序返回

        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []

    async def record_tool_call(
        self,
        conversation_id: int,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str = None,
        success: bool = True,
        error_message: str = None,
        execution_time: float = 0.0,
    ) -> bool:
        """记录工具调用"""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """INSERT INTO tool_calls 
                       (conversation_id, tool_name, arguments, result, success, 
                        error_message, execution_time)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        conversation_id,
                        tool_name,
                        json.dumps(arguments),
                        result,
                        success,
                        error_message,
                        execution_time,
                    ),
                )
                return True

        except Exception as e:
            logger.error(f"记录工具调用失败: {e}")
            return False

    async def update_role_stats(self, role_name: str) -> bool:
        """更新角色使用统计"""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO role_stats (role_name, usage_count, last_used)
                       VALUES (?, 
                               COALESCE((SELECT usage_count FROM role_stats WHERE role_name = ?), 0) + 1,
                               CURRENT_TIMESTAMP)""",
                    (role_name, role_name),
                )
                return True

        except Exception as e:
            logger.error(f"更新角色统计失败: {e}")
            return False

    async def get_user_stats(self, telegram_user_id: int) -> Dict[str, Any]:
        """获取用户统计信息"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """SELECT 
                           u.username,
                           COUNT(DISTINCT c.id) as conversation_count,
                           COUNT(m.id) as message_count,
                           c.role_name as current_role,
                           MAX(m.timestamp) as last_activity
                       FROM users u
                       LEFT JOIN conversations c ON u.id = c.user_id
                       LEFT JOIN messages m ON c.id = m.conversation_id
                       WHERE u.telegram_user_id = ?
                       GROUP BY u.id""",
                    (telegram_user_id,),
                )

                row = cursor.fetchone()
                if row:
                    return {
                        "username": row["username"],
                        "conversation_count": row["conversation_count"],
                        "message_count": row["message_count"],
                        "current_role": row["current_role"],
                        "last_activity": row["last_activity"],
                    }
                else:
                    return {}

        except Exception as e:
            logger.error(f"获取用户统计失败: {e}")
            return {}

    async def get_tool_usage_stats(self) -> List[Dict[str, Any]]:
        """获取工具使用统计"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """SELECT * FROM tool_usage_stats
                       ORDER BY usage_count DESC
                       LIMIT 20"""
                )

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"获取工具统计失败: {e}")
            return []

    async def get_role_stats(self) -> List[Dict[str, Any]]:
        """获取角色使用统计"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM role_stats ORDER BY usage_count DESC"
                )

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"获取角色统计失败: {e}")
            return []

    async def cleanup_old_messages(self, days: int = 30) -> int:
        """清理旧消息（保留最近N天）"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """DELETE FROM messages
                       WHERE timestamp < datetime('now', '-{} days')""".format(
                        days
                    )
                )
                deleted_count = cursor.rowcount
                logger.info(f"清理了 {deleted_count} 条旧消息")
                return deleted_count

        except Exception as e:
            logger.error(f"清理旧消息失败: {e}")
            return 0

    def get_all_knowledge_entries(self) -> List[Dict[str, Any]]:
        """获取所有知识条目"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """SELECT id, user_id, conversation_id, title, content, summary,
                              keywords, category, importance_score, knowledge_namespace,
                              created_at
                       FROM knowledge_entries 
                       ORDER BY created_at DESC"""
                )

                entries = []
                for row in cursor.fetchall():
                    entries.append(
                        {
                            "id": row["id"],
                            "user_id": row["user_id"],
                            "conversation_id": row["conversation_id"],
                            "title": row["title"],
                            "content": row["content"],
                            "summary": row["summary"],
                            "keywords": row["keywords"],
                            "category": row["category"],
                            "importance_score": row["importance_score"],
                            "knowledge_namespace": row["knowledge_namespace"],
                            "created_at": row["created_at"],
                        }
                    )

                logger.debug(f"获取了 {len(entries)} 个知识条目")
                return entries

        except Exception as e:
            logger.error(f"获取所有知识条目失败: {e}")
            return []

    def get_knowledge_count(self) -> int:
        """获取知识条目总数"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM knowledge_entries")
                count = cursor.fetchone()[0]
                return count

        except Exception as e:
            logger.error(f"获取知识条目数量失败: {e}")
            return 0

    def save_knowledge_entry(
        self,
        user_id: int,
        conversation_id: int,
        title: str,
        content: str,
        summary: str = "",
        keywords: str = "",
        category: str = "general",
        importance_score: float = 0.5,
        source_message_id: int = None,
        knowledge_namespace: str = "",
    ) -> Optional[int]:
        """保存知识条目"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """INSERT INTO knowledge_entries 
                       (user_id, conversation_id, title, content, summary, keywords,
                        category, importance_score, source_message_id,
                        knowledge_namespace, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                    (
                        user_id,
                        conversation_id,
                        title,
                        content,
                        summary,
                        keywords,
                        category,
                        importance_score,
                        source_message_id,
                        knowledge_namespace,
                    ),
                )
                entry_id = cursor.lastrowid
                logger.debug(f"保存知识条目成功，ID: {entry_id}")
                return entry_id

        except Exception as e:
            logger.error(f"保存知识条目失败: {e}")
            return None

    def delete_knowledge_entry(self, entry_id: int) -> bool:
        """删除知识条目"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM knowledge_entries WHERE id = ?", (entry_id,)
                )
                success = cursor.rowcount > 0
                if success:
                    logger.debug(f"删除知识条目成功，ID: {entry_id}")
                return success

        except Exception as e:
            logger.error(f"删除知识条目失败: {e}")
            return False

    def upsert_rag_document(
        self, document: Dict[str, Any], chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """以文档为事务边界，幂等替换文档分块及 FTS 记录。"""
        required = {
            "collection",
            "knowledge_namespace",
            "document_key",
            "source_uri",
            "title",
            "domain",
            "content_hash",
        }
        missing = sorted(required - set(document))
        if missing:
            raise ValueError(f"RAG 文档缺少字段: {', '.join(missing)}")

        with self.get_connection() as conn:
            existing = conn.execute(
                """SELECT id, content_hash, source_root, source_uri FROM rag_documents
                   WHERE collection = ? AND knowledge_namespace = ? AND document_key = ?""",
                (
                    document["collection"],
                    document["knowledge_namespace"],
                    document["document_key"],
                ),
            ).fetchone()
            if existing and existing["content_hash"] == document["content_hash"]:
                source_root = document.get("source_root", "")
                if (
                    existing["source_root"] != source_root
                    or existing["source_uri"] != document["source_uri"]
                ):
                    conn.execute(
                        """UPDATE rag_documents
                              SET source_root = ?, source_uri = ?,
                                  updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?""",
                        (source_root, document["source_uri"], int(existing["id"])),
                    )
                    conn.commit()
                return {
                    "document_id": int(existing["id"]),
                    "changed": False,
                    "chunks_written": 0,
                }

            values = (
                document["collection"],
                document["knowledge_namespace"],
                document["document_key"],
                document.get("source_root", ""),
                document["source_uri"],
                document["title"],
                document["domain"],
                document.get("subdomain", ""),
                document.get("product", ""),
                document.get("version", ""),
                document.get("language", ""),
                document.get("license", ""),
                int(document.get("trust_level", 50)),
                document["content_hash"],
                json.dumps(document.get("metadata", {}), ensure_ascii=False),
            )
            if existing:
                document_id = int(existing["id"])
                old_chunk_ids = [
                    row["id"]
                    for row in conn.execute(
                        "SELECT id FROM rag_chunks WHERE document_id = ?",
                        (document_id,),
                    )
                ]
                if old_chunk_ids:
                    placeholders = ",".join("?" for _ in old_chunk_ids)
                    conn.execute(
                        f"DELETE FROM rag_chunks_fts WHERE chunk_id IN ({placeholders})",
                        tuple(str(value) for value in old_chunk_ids),
                    )
                conn.execute(
                    "DELETE FROM rag_chunks WHERE document_id = ?", (document_id,)
                )
                conn.execute(
                    """UPDATE rag_documents SET
                           collection = ?, knowledge_namespace = ?, document_key = ?,
                           source_root = ?, source_uri = ?,
                           title = ?, domain = ?, subdomain = ?, product = ?, version = ?,
                           language = ?, license = ?, trust_level = ?, content_hash = ?,
                           metadata_json = ?, updated_at = CURRENT_TIMESTAMP
                       WHERE id = ?""",
                    values + (document_id,),
                )
            else:
                cursor = conn.execute(
                    """INSERT INTO rag_documents (
                           collection, knowledge_namespace, document_key, source_root,
                           source_uri, title, domain, subdomain, product, version,
                           language, license, trust_level, content_hash, metadata_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    values,
                )
                document_id = int(cursor.lastrowid)

            for chunk in chunks:
                cursor = conn.execute(
                    """INSERT INTO rag_chunks (
                           document_id, chunk_index, heading_path, content, content_hash,
                           token_count, metadata_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        document_id,
                        int(chunk["chunk_index"]),
                        chunk.get("heading_path", ""),
                        chunk["content"],
                        chunk["content_hash"],
                        int(chunk.get("token_count", 0)),
                        json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                    ),
                )
                conn.execute(
                    """INSERT INTO rag_chunks_fts (
                           chunk_id, collection, knowledge_namespace, title, heading_path,
                           content, domain, product
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(cursor.lastrowid),
                        document["collection"],
                        document["knowledge_namespace"],
                        _fts_index_text(document["title"]),
                        _fts_index_text(chunk.get("heading_path", "")),
                        _fts_index_text(chunk["content"]),
                        document["domain"],
                        document.get("product", ""),
                    ),
                )
            conn.commit()
            return {
                "document_id": document_id,
                "changed": True,
                "chunks_written": len(chunks),
            }

    def delete_rag_documents_not_seen(
        self,
        *,
        collection: str,
        knowledge_namespace: str,
        source_root: str,
        seen_document_keys: List[str],
    ) -> Dict[str, int]:
        """Delete stale documents owned by one completed ingestion root."""
        conditions = [
            "collection = ?",
            "knowledge_namespace = ?",
            "source_root = ?",
        ]
        parameters: List[Any] = [collection, knowledge_namespace, source_root]
        if seen_document_keys:
            conditions.append(
                f"document_key NOT IN ({','.join('?' for _ in seen_document_keys)})"
            )
            parameters.extend(seen_document_keys)
        with self.get_connection() as conn:
            document_rows = conn.execute(
                f"SELECT id FROM rag_documents WHERE {' AND '.join(conditions)}",
                parameters,
            ).fetchall()
            document_ids = [int(row["id"]) for row in document_rows]
            if not document_ids:
                return {"documents": 0, "chunks": 0}
            placeholders = ",".join("?" for _ in document_ids)
            chunk_rows = conn.execute(
                f"SELECT id FROM rag_chunks WHERE document_id IN ({placeholders})",
                document_ids,
            ).fetchall()
            chunk_ids = [int(row["id"]) for row in chunk_rows]
            if chunk_ids:
                chunk_placeholders = ",".join("?" for _ in chunk_ids)
                conn.execute(
                    f"DELETE FROM rag_chunks_fts WHERE chunk_id IN ({chunk_placeholders})",
                    [str(chunk_id) for chunk_id in chunk_ids],
                )
            conn.execute(
                f"DELETE FROM rag_documents WHERE id IN ({placeholders})", document_ids
            )
            conn.commit()
            return {"documents": len(document_ids), "chunks": len(chunk_ids)}

    def get_all_rag_chunks(self) -> List[Dict[str, Any]]:
        """返回用于重建稠密索引的全部活动文档分块。"""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT c.id AS chunk_id, c.chunk_index, c.heading_path, c.content,
                          c.content_hash, c.token_count, c.metadata_json AS chunk_metadata,
                          d.id AS document_id, d.collection, d.knowledge_namespace,
                          d.source_uri, d.title, d.domain, d.subdomain, d.product,
                          d.version, d.language, d.license, d.trust_level,
                          d.metadata_json AS document_metadata
                   FROM rag_chunks c
                   JOIN rag_documents d ON d.id = c.document_id
                   ORDER BY d.id, c.chunk_index"""
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["chunk_metadata"] = json.loads(item["chunk_metadata"] or "{}")
            item["document_metadata"] = json.loads(item["document_metadata"] or "{}")
            results.append(item)
        return results

    @staticmethod
    def _build_fts_query(query: str) -> str:
        terms = re.findall(r"[A-Za-z0-9_./:+-]{2,}|[\u3400-\u9fff]{2,}", query)
        expanded: List[str] = []
        for term in terms:
            if _CJK_RUN_PATTERN.fullmatch(term):
                expanded.extend(_cjk_bigrams(term))
            else:
                expanded.append(term.casefold())
        unique = list(dict.fromkeys(expanded))[:40]
        return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in unique)

    def search_rag_chunks_lexical(
        self,
        query: str,
        *,
        limit: int = 40,
        collections: Optional[List[str]] = None,
        knowledge_namespaces: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """使用 SQLite FTS5 召回文档分块，过滤在排名前执行。"""
        match_query = self._build_fts_query(query)
        if not match_query:
            return []
        conditions = ["rag_chunks_fts MATCH ?"]
        parameters: List[Any] = [match_query]
        if collections:
            conditions.append(f"d.collection IN ({','.join('?' for _ in collections)})")
            parameters.extend(collections)
        if knowledge_namespaces:
            conditions.append(
                f"d.knowledge_namespace IN ({','.join('?' for _ in knowledge_namespaces)})"
            )
            parameters.extend(knowledge_namespaces)
        parameters.append(int(limit))
        sql = f"""SELECT c.id AS chunk_id, c.chunk_index, c.heading_path, c.content,
                         d.id AS document_id, d.collection, d.knowledge_namespace,
                         d.source_uri, d.title, d.domain, d.subdomain, d.product,
                         d.version, d.language, d.license, d.trust_level,
                         bm25(rag_chunks_fts, 0, 0, 0, 1.5, 1.2, 2.0, 0.8, 1.0)
                             AS lexical_rank
                  FROM rag_chunks_fts
                  JOIN rag_chunks c ON c.id = CAST(rag_chunks_fts.chunk_id AS INTEGER)
                  JOIN rag_documents d ON d.id = c.document_id
                  WHERE {' AND '.join(conditions)}
                  ORDER BY lexical_rank
                  LIMIT ?"""
        with self.get_connection() as conn:
            return [dict(row) for row in conn.execute(sql, parameters).fetchall()]

    def get_rag_document_stats(self) -> Dict[str, int]:
        with self.get_connection() as conn:
            return {
                "documents": int(
                    conn.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0]
                ),
                "chunks": int(
                    conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
                ),
            }

    def get_user_profile(self, telegram_user_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute(
                """SELECT telegram_user_id, content, content_hash, source,
                          created_at, updated_at
                   FROM user_profiles WHERE telegram_user_id = ?""",
                (telegram_user_id,),
            ).fetchone()
            return dict(row) if row else None

    def upsert_user_profile(
        self, telegram_user_id: int, content: str, content_hash: str, source: str
    ) -> None:
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO user_profiles (
                       telegram_user_id, content, content_hash, source
                   ) VALUES (?, ?, ?, ?)
                   ON CONFLICT(telegram_user_id) DO UPDATE SET
                       content = excluded.content,
                       content_hash = excluded.content_hash,
                       source = excluded.source,
                       updated_at = CURRENT_TIMESTAMP""",
                (telegram_user_id, content, content_hash, source),
            )
            conn.commit()

    def delete_user_profile(self, telegram_user_id: int) -> bool:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM user_profiles WHERE telegram_user_id = ?",
                (telegram_user_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
