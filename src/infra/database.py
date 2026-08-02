"""
数据库服务模块
与SQLite数据库交互，记录用户对话和统计信息
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ..models import Message, SessionKey


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
        except Exception as exc:
            logger.error("数据库迁移失败: {}", exc)
            raise

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
                success = cursor.rowchange > 0
                if success:
                    logger.debug(f"删除知识条目成功，ID: {entry_id}")
                return success

        except Exception as e:
            logger.error(f"删除知识条目失败: {e}")
            return False
