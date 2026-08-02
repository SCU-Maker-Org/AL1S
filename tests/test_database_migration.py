from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.infra.database import DatabaseService
from src.models import SessionKey


def create_legacy_database(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_user_id INTEGER UNIQUE NOT NULL,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                role_name TEXT DEFAULT 'AI助手',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, chat_id)
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                token_count INTEGER DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            CREATE TABLE knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                conversation_id INTEGER,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                keywords TEXT,
                category TEXT,
                importance_score REAL,
                embedding_id TEXT,
                source_message_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO users(id, telegram_user_id, username)
                VALUES (1, 12345, 'alice');
            INSERT INTO conversations(id, user_id, chat_id, role_name)
                VALUES (5, 1, 12345, 'AI助手');
            INSERT INTO messages(conversation_id, role, content)
                VALUES (5, 'user', 'legacy message');
            INSERT INTO knowledge_entries(user_id, conversation_id, title, content)
                VALUES (1, 5, 'legacy', 'knowledge');
            """
        )


@pytest.mark.asyncio
async def test_legacy_database_migrates_idempotently(tmp_path):
    path = tmp_path / "legacy.db"
    create_legacy_database(path)
    service = DatabaseService(str(path))
    with service.get_connection() as conn:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(conversations)")
        }
        assert {
            "thread_id",
            "session_scope",
            "session_owner_id",
            "knowledge_namespace",
        } <= columns
        row = conn.execute("SELECT * FROM conversations WHERE id = 5").fetchone()
        assert row["session_scope"] == "private"
        assert row["session_owner_id"] == 12345
        assert row["knowledge_namespace"] == "private:12345"
        assert (
            conn.execute("SELECT content FROM messages").fetchone()[0]
            == "legacy message"
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 2"
            ).fetchone()[0]
            == 1
        )

    # The old unique(user_id, chat_id) constraint is gone: one user can own two topics.
    first = SessionKey(-100, 7, 0, "topic")
    second = SessionKey(-100, 8, 0, "topic")
    assert await service.ensure_conversation(1, first)
    assert await service.ensure_conversation(1, second)

    restarted = DatabaseService(str(path))
    with restarted.get_connection() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 2"
            ).fetchone()[0]
            == 1
        )
        assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 3


def test_fresh_schema_is_already_at_v2(tmp_path):
    path = tmp_path / "fresh.db"
    init_sql = (Path(__file__).parents[1] / "data" / "init_db.sql").read_text(
        encoding="utf-8"
    )
    with sqlite3.connect(path) as conn:
        conn.executescript(init_sql)
    service = DatabaseService(str(path))
    with service.get_connection() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 2"
            ).fetchone()[0]
            == 1
        )
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'conversations'"
        ).fetchone()[0]
        assert "session_owner_id" in table_sql
