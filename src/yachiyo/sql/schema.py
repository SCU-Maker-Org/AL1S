"""Apply numbered migrations and check the database schema at startup."""

import sqlite3
from datetime import UTC, datetime
from importlib import resources

import aiosqlite


def _sql_statements(script: str) -> tuple[str, ...]:
    statements: list[str] = []
    current = ""
    for line in script.splitlines(keepends=True):
        current += line
        if sqlite3.complete_statement(current):
            statement = current.strip()
            if statement:
                statements.append(statement)
            current = ""
    if current.strip():
        raise RuntimeError("migration contains an incomplete SQL statement")
    return tuple(statements)


async def migrate(connection: aiosqlite.Connection) -> None:
    schema_cursor = await connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    )
    has_history = False
    if await schema_cursor.fetchone() is not None:
        history = await connection.execute("SELECT 1 FROM schema_migrations LIMIT 1")
        has_history = await history.fetchone() is not None
    if not has_history:
        existing_cursor = await connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
              AND name != 'schema_migrations'
            LIMIT 1
            """
        )
        if await existing_cursor.fetchone() is not None:
            raise RuntimeError(
                "database predates yachiyo 2.0 and cannot be upgraded in place; "
                "back it up and start with a clean database"
            )
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        await connection.commit()
    migration_root = resources.files("yachiyo.sql.migrations")
    for entry in sorted(migration_root.iterdir(), key=lambda item: item.name):
        if not entry.name.endswith(".sql"):
            continue
        version_text = entry.name.split("_", 1)[0]
        if not version_text.isdigit():
            continue
        sql = entry.read_text(encoding="utf-8")
        version = int(version_text)
        await connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = await connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version=?", (version,)
            )
            if await cursor.fetchone() is not None:
                await connection.rollback()
                continue
            for statement in _sql_statements(sql):
                await connection.execute(statement)
            await connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (version, datetime.now(UTC).isoformat()),
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise


async def validate_schema(connection: aiosqlite.Connection) -> None:
    required = {
        "event_receipts": {
            "updated_at",
            "status",
            "processing_token",
            "response_json",
            "delivered_at",
            "delivery_attempts",
            "next_attempt_at",
            "last_delivery_error",
        },
        "documents": {"source_root"},
        "messages": {"source_event_key"},
        "command_effects": {"event_key", "kind"},
        "jobs": {"lease_token", "source_event_key"},
        "usage_events": {"source_event_key"},
    }
    for table, columns in required.items():
        cursor = await connection.execute(f"PRAGMA table_info({table})")
        present = {str(row["name"]) for row in await cursor.fetchall()}
        missing = columns - present
        if missing:
            raise RuntimeError(
                "database schema predates yachiyo 2.0 and cannot be upgraded in place; "
                f"start with a clean database (missing {table}: {sorted(missing)})"
            )
