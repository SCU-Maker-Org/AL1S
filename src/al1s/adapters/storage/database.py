"""Single-connection asynchronous SQLite repository."""

from __future__ import annotations

import asyncio
import fcntl
import functools
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any

import aiosqlite

from al1s.config import AppPaths, AppSettings
from al1s.domain import (
    ActionButton,
    Actor,
    Artifact,
    ChatCompletion,
    ChatKind,
    ConversationAddress,
    ConversationState,
    ExternalIdentity,
    InboundEvent,
    KnowledgeScope,
    OutboundMessage,
    RequestPermission,
    ResponsePlan,
    RoleDefinition,
    SessionKey,
    SessionScope,
    StoredMessage,
    TransportName,
)
from al1s.domain.content import ArtifactKind
from al1s.domain.message import MessageRole
from al1s.domain.versions import INGESTION_PIPELINE_VERSION, LEXICAL_PIPELINE_VERSION


class _AsyncReentrantLock:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[Any] | None = None
        self._depth = 0

    async def __aenter__(self) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("database operation requires an asyncio task")
        if self._owner is task:
            self._depth += 1
            return
        await self._lock.acquire()
        self._owner = task
        self._depth = 1

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        task = asyncio.current_task()
        if self._owner is not task:
            raise RuntimeError("database operation lock released by a non-owner")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()


def _serialized(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    async def wrapper(self: Database, *args: Any, **kwargs: Any) -> Any:
        async with self.operation():
            try:
                return await method(self, *args, **kwargs)
            except BaseException:
                if self._connection is not None and self._connection.in_transaction:
                    await self._connection.rollback()
                raise

    return wrapper


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


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _as_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


class Database:
    def __init__(self, settings: AppSettings, paths: AppPaths):
        configured = Path(settings.storage.database_path).expanduser()
        self.path = configured if configured.is_absolute() else paths.data_dir / configured
        self.busy_timeout_ms = settings.storage.busy_timeout_ms
        self.event_lease_seconds = settings.policies.request_timeout_seconds + 60
        self.max_output_artifact_bytes = settings.policies.max_output_artifact_bytes
        self.embedding_dimensions = settings.models.embedding.dimensions
        fingerprint_data = {
            "base_url": str(settings.models.embedding.base_url),
            "model": settings.models.embedding.model,
            "dimensions": settings.models.embedding.dimensions,
            "request_dimensions": settings.models.embedding.request_dimensions,
            "chunk_chars": settings.rag.chunk_chars,
            "chunk_overlap_chars": settings.rag.chunk_overlap_chars,
            "max_chunks_per_document": settings.rag.max_chunks_per_document,
            "ingestion_pipeline": INGESTION_PIPELINE_VERSION,
            "lexical_pipeline": LEXICAL_PIPELINE_VERSION,
        }
        self.vector_fingerprint = hashlib.sha256(
            json.dumps(fingerprint_data, sort_keys=True).encode()
        ).hexdigest()
        self._connection: aiosqlite.Connection | None = None
        self._lock_descriptor: int | None = None
        self._owner_lock_descriptor: int | None = None
        self._operation_lock = _AsyncReentrantLock()
        self._vector_rebuild_lock = asyncio.Lock()
        self._vector_extension_loaded = False
        self._vector_enabled = False
        self._vector_rebuild_required = False

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("database is not open")
        return self._connection

    @property
    def vector_enabled(self) -> bool:
        return self._vector_enabled

    @property
    def vector_extension_loaded(self) -> bool:
        return self._vector_extension_loaded

    @property
    def vector_rebuild_required(self) -> bool:
        return self._vector_rebuild_required

    @asynccontextmanager
    async def operation(self) -> AsyncIterator[None]:
        async with self._operation_lock:
            yield

    @asynccontextmanager
    async def vector_rebuild_operation(self) -> AsyncIterator[None]:
        async with self._vector_rebuild_lock:
            yield

    async def open(self) -> None:
        try:
            await self._open()
        except BaseException:
            await self.close()
            raise

    async def _open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink():
            raise RuntimeError("database path must not be a symbolic link")
        self.path = self.path.parent.resolve(strict=True) / self.path.name
        self._validate_single_link_file(self.path, "database")
        owner_lock_path = self.path.with_name(f".{self.path.name}.owner.lock")
        self._owner_lock_descriptor = os.open(
            owner_lock_path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        owner_lock_stat = os.fstat(self._owner_lock_descriptor)
        if not stat.S_ISREG(owner_lock_stat.st_mode) or owner_lock_stat.st_nlink != 1:
            raise RuntimeError("database owner lock must be a single-link regular file")
        try:
            fcntl.flock(
                self._owner_lock_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError("another AL1S process is already using this database") from exc
        self._lock_descriptor = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        lock_stat = os.fstat(self._lock_descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise RuntimeError("database must be a single-link regular file")
        try:
            fcntl.flock(
                self._lock_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None
            raise RuntimeError("another AL1S process is already using this database") from exc
        path_stat = os.stat(self.path, follow_symlinks=False)
        if (path_stat.st_dev, path_stat.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
            raise RuntimeError("database path changed while acquiring its lock")
        descriptor_root = "/proc/self/fd" if sys.platform.startswith("linux") else "/dev/fd"
        self._connection = await aiosqlite.connect(
            f"file:{descriptor_root}/{self._lock_descriptor}?mode=rw",
            uri=True,
        )
        self._validate_single_link_file(self.path, "database")
        connected_path_stat = os.stat(self.path, follow_symlinks=False)
        if (connected_path_stat.st_dev, connected_path_stat.st_ino) != (
            lock_stat.st_dev,
            lock_stat.st_ino,
        ):
            raise RuntimeError("database path changed while opening SQLite")
        self._connection.row_factory = aiosqlite.Row
        await self.connection.execute("PRAGMA foreign_keys = ON")
        cursor = await self.connection.execute("PRAGMA journal_mode = WAL")
        journal_mode = await cursor.fetchone()
        if journal_mode is None or str(journal_mode[0]).casefold() != "wal":
            raise RuntimeError("SQLite WAL mode is required")
        await self.connection.execute("PRAGMA synchronous = FULL")
        await self.connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        await self._migrate()
        await self._validate_schema()
        await self._load_vector_extension()
        await self._seed_vector_generation()
        await self.connection.commit()
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def _validate_single_link_file(path: Path, label: str) -> None:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"{label} must be a single-link regular file")

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
        if self._lock_descriptor is not None:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            os.close(self._lock_descriptor)
            self._lock_descriptor = None
        if self._owner_lock_descriptor is not None:
            fcntl.flock(self._owner_lock_descriptor, fcntl.LOCK_UN)
            os.close(self._owner_lock_descriptor)
            self._owner_lock_descriptor = None

    @_serialized
    async def cleanup_expired_data(self, message_retention_days: int) -> dict[str, int]:
        cutoff = (datetime.now(UTC) - timedelta(days=message_retention_days)).isoformat()
        message_cursor = await self.connection.execute(
            "DELETE FROM messages WHERE created_at<?", (cutoff,)
        )
        receipt_cutoff = (datetime.now(UTC) - timedelta(days=7)).isoformat()
        receipt_count = 0
        while True:
            expired_cursor = await self.connection.execute(
                """
                SELECT transport, bot_id, external_event_id, response_json
                FROM event_receipts
                WHERE status='delivered' AND received_at<?
                ORDER BY received_at LIMIT 100
                """,
                (receipt_cutoff,),
            )
            rows = await expired_cursor.fetchall()
            if not rows:
                break
            for row in rows:
                if row["response_json"]:
                    await asyncio.to_thread(
                        self._remove_response_artifacts,
                        row["response_json"],
                        self._event_outbox_root(
                            row["transport"], row["bot_id"], row["external_event_id"]
                        ),
                    )
            await self.connection.executemany(
                """
                DELETE FROM event_receipts
                WHERE transport=? AND bot_id=? AND external_event_id=?
                  AND status='delivered'
                """,
                [(row["transport"], row["bot_id"], row["external_event_id"]) for row in rows],
            )
            await self.connection.commit()
            receipt_count += len(rows)
        approval_cursor = await self.connection.execute(
            "DELETE FROM tool_approvals WHERE expires_at<?", (_now(),)
        )
        job_cutoff = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        job_cursor = await self.connection.execute(
            "DELETE FROM jobs WHERE status IN ('complete', 'failed') AND updated_at<?",
            (job_cutoff,),
        )
        effect_cursor = await self.connection.execute(
            "DELETE FROM command_effects WHERE created_at<?", (job_cutoff,)
        )
        abandoned_cursor = await self.connection.execute(
            """
            DELETE FROM event_receipts
            WHERE status IN ('processing', 'dead_letter') AND updated_at<?
            """,
            (job_cutoff,),
        )
        await self.connection.commit()
        return {
            "messages": message_cursor.rowcount,
            "event_receipts": receipt_count,
            "tool_approvals": approval_cursor.rowcount,
            "jobs": job_cursor.rowcount,
            "command_effects": effect_cursor.rowcount,
            "abandoned_event_receipts": abandoned_cursor.rowcount,
        }

    @_serialized
    async def cleanup_orphaned_outbox(self) -> int:
        referenced: set[Path] = set()
        active_cursor = await self.connection.execute(
            """
            SELECT transport, bot_id, external_event_id, processing_token
            FROM event_receipts
            WHERE status='processing' AND processing_token IS NOT NULL
            """
        )
        for row in await active_cursor.fetchall():
            referenced.add(
                self._event_outbox_root(row["transport"], row["bot_id"], row["external_event_id"])
                / row["processing_token"]
            )
        cursor = await self.connection.execute(
            "SELECT response_json FROM event_receipts "
            "WHERE status='completed' AND response_json IS NOT NULL"
        )
        while rows := await cursor.fetchmany(100):
            for row in rows:
                try:
                    payload = json.loads(row["response_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                for message in payload.get("messages", []):
                    if not isinstance(message, dict):
                        continue
                    for artifact in message.get("artifacts", []):
                        if not isinstance(artifact, dict) or not artifact.get("path"):
                            continue
                        referenced.add(Path(str(artifact["path"])).parent)
        return await asyncio.to_thread(self._remove_orphaned_outbox, referenced)

    def _remove_orphaned_outbox(self, referenced: set[Path]) -> int:
        root = self.path.parent / "outbox"
        if not root.exists():
            return 0
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError("outbox root is not a safe directory")
        removed = 0
        for event_root in root.iterdir():
            if event_root.is_symlink() or not event_root.is_dir():
                continue
            for token_root in event_root.iterdir():
                if token_root.is_symlink() or not token_root.is_dir():
                    continue
                if token_root.resolve() in referenced:
                    continue
                shutil.rmtree(token_root)
                removed += 1
            try:
                event_root.rmdir()
            except OSError:
                pass
        return removed

    async def _migrate(self) -> None:
        schema_cursor = await self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        )
        if await schema_cursor.fetchone() is None:
            existing_cursor = await self.connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                LIMIT 1
                """
            )
            if await existing_cursor.fetchone() is not None:
                raise RuntimeError(
                    "database predates AL1S 2.0 and cannot be upgraded in place; "
                    "back it up and start with a clean database"
                )
        await self.connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        await self.connection.commit()
        migration_cursor = await self.connection.execute("SELECT COUNT(*) FROM schema_migrations")
        migration_count = int((await migration_cursor.fetchone())[0])
        if migration_count == 0:
            existing_cursor = await self.connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                  AND name != 'schema_migrations'
                LIMIT 1
                """
            )
            if await existing_cursor.fetchone() is not None:
                raise RuntimeError(
                    "database predates AL1S 2.0 and cannot be upgraded in place; "
                    "back it up and start with a clean database"
                )
        migration_root = resources.files("al1s.sql.migrations")
        for entry in sorted(migration_root.iterdir(), key=lambda item: item.name):
            if not entry.name.endswith(".sql"):
                continue
            version_text = entry.name.split("_", 1)[0]
            if not version_text.isdigit():
                continue
            sql = entry.read_text(encoding="utf-8")
            version = int(version_text)
            await self.connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self.connection.execute(
                    "SELECT 1 FROM schema_migrations WHERE version=?", (version,)
                )
                if await cursor.fetchone() is not None:
                    await self.connection.rollback()
                    continue
                for statement in _sql_statements(sql):
                    await self.connection.execute(statement)
                await self.connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, _now()),
                )
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise

    async def _load_vector_extension(self) -> None:
        try:
            import sqlite_vec

            loadable_path = sqlite_vec.loadable_path()
            await self.connection.enable_load_extension(True)
            await self.connection.load_extension(loadable_path)
            await self.connection.enable_load_extension(False)
            await self.connection.execute("SELECT vec_version()")
        except (AttributeError, ImportError, OSError, aiosqlite.Error):
            try:
                await self.connection.enable_load_extension(False)
            except (AttributeError, aiosqlite.Error):
                pass
            self._vector_extension_loaded = False
            self._vector_enabled = False
        else:
            self._vector_extension_loaded = True

    async def _validate_schema(self) -> None:
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
            cursor = await self.connection.execute(f"PRAGMA table_info({table})")
            present = {str(row["name"]) for row in await cursor.fetchall()}
            missing = columns - present
            if missing:
                raise RuntimeError(
                    "database schema predates AL1S 2.0 and cannot be upgraded in place; "
                    f"start with a clean database (missing {table}: {sorted(missing)})"
                )

    async def _seed_vector_generation(self) -> None:
        if not self._vector_extension_loaded:
            return
        await self.connection.execute("BEGIN IMMEDIATE")
        try:
            await self._seed_vector_generation_locked()
            await self.connection.commit()
        except BaseException:
            await self.connection.rollback()
            raise

    async def _seed_vector_generation_locked(self) -> None:
        if not self._vector_extension_loaded:
            return
        dimensions = self.embedding_dimensions
        cursor = await self.connection.execute(
            "SELECT value FROM app_metadata WHERE key = 'vector_dimensions'"
        )
        row = await cursor.fetchone()
        fingerprint_cursor = await self.connection.execute(
            "SELECT value FROM app_metadata WHERE key = 'vector_fingerprint'"
        )
        fingerprint_row = await fingerprint_cursor.fetchone()
        dirty_cursor = await self.connection.execute(
            "SELECT value FROM app_metadata WHERE key = 'vector_dirty'"
        )
        dirty_row = await dirty_cursor.fetchone()
        if row is not None and (
            int(row[0]) != dimensions
            or fingerprint_row is None
            or str(fingerprint_row[0]) != self.vector_fingerprint
            or (dirty_row is not None and dirty_row[0] == "1")
        ):
            self._vector_enabled = False
            self._vector_rebuild_required = True
            return

        generation = 1
        if row is not None:
            generation_cursor = await self.connection.execute(
                "SELECT value FROM app_metadata WHERE key = 'vector_generation'"
            )
            generation_row = await generation_cursor.fetchone()
            generation = int(generation_row[0]) if generation_row else 1

        await self.connection.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec_v{generation} "
            f"USING vec0(entity_id text primary key, scope text, embedding float[{dimensions}])"
        )
        await self.connection.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec_v{generation} "
            f"USING vec0(entity_id text primary key, scope text, embedding float[{dimensions}])"
        )
        now = _now()
        for key, value in (
            ("vector_dimensions", str(dimensions)),
            ("vector_generation", str(generation)),
            ("vector_fingerprint", self.vector_fingerprint),
            ("vector_dirty", "0"),
        ):
            await self.connection.execute(
                """
                INSERT INTO app_metadata(key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, now),
            )
        for kind in ("chunks", "memories"):
            cursor = await self.connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (f"{kind}_vec_v{generation}",),
            )
            schema_row = await cursor.fetchone()
            schema = str(schema_row["sql"] if schema_row else "").casefold()
            if f"float[{dimensions}]" not in schema or "entity_id text primary key" not in schema:
                raise RuntimeError("active sqlite-vec schema does not match configuration")
        chunk_count = await self._count("chunks")
        memory_count = await self._count("memories")
        vector_chunk_count = await self._count(f"chunks_vec_v{generation}")
        vector_memory_count = await self._count(f"memories_vec_v{generation}")
        complete = chunk_count == vector_chunk_count and memory_count == vector_memory_count
        self._vector_enabled = complete
        self._vector_rebuild_required = not complete
        if complete:
            cursor = await self.connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND (name LIKE 'chunks_vec_v%' OR name LIKE 'memories_vec_v%')
                """
            )
            active = {
                f"chunks_vec_v{generation}",
                f"memories_vec_v{generation}",
            }
            for old_row in await cursor.fetchall():
                old_name = str(old_row["name"])
                if old_name in active or not re.fullmatch(
                    r"(?:chunks|memories)_vec_v[1-9][0-9]{0,8}", old_name
                ):
                    continue
                await self.connection.execute(f"DROP TABLE {old_name}")

    async def _count(self, table: str) -> int:
        if table not in {"chunks", "memories"} and not re.fullmatch(
            r"(?:chunks|memories)_vec_v[1-9][0-9]{0,8}", table
        ):
            raise ValueError("invalid count table")
        cursor = await self.connection.execute(
            f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - validated table identifier
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    def mark_vector_ready(self) -> None:
        self._vector_enabled = self._vector_extension_loaded
        self._vector_rebuild_required = False

    @_serialized
    async def verify_vector_state(self) -> bool:
        if not self._vector_extension_loaded:
            return False
        cursor = await self.connection.execute(
            "SELECT key, value FROM app_metadata "
            "WHERE key IN ('vector_fingerprint', 'vector_dirty')"
        )
        metadata = {row["key"]: row["value"] for row in await cursor.fetchall()}
        current = (
            metadata.get("vector_fingerprint") == self.vector_fingerprint
            and metadata.get("vector_dirty", "0") != "1"
        )
        if not current:
            self._vector_enabled = False
            self._vector_rebuild_required = True
        return current and self._vector_enabled

    @_serialized
    async def active_vector_generation(self) -> int:
        cursor = await self.connection.execute(
            "SELECT value FROM app_metadata WHERE key = 'vector_generation'"
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 1

    @_serialized
    async def claim_event(self, transport: str, bot_id: str, event_id: str) -> str | None:
        now = datetime.now(UTC)
        processing_token = uuid.uuid4().hex
        cursor = await self.connection.execute(
            """
            INSERT OR IGNORE INTO event_receipts(
                transport, bot_id, external_event_id, received_at, updated_at,
                status, processing_token
            ) VALUES (?, ?, ?, ?, ?, 'processing', ?)
            """,
            (
                transport,
                bot_id,
                event_id,
                now.isoformat(),
                now.isoformat(),
                processing_token,
            ),
        )
        claimed = cursor.rowcount == 1
        if not claimed:
            stale = (now - timedelta(seconds=self.event_lease_seconds)).isoformat()
            cursor = await self.connection.execute(
                """
                UPDATE event_receipts SET received_at=?, updated_at=?, status='processing',
                    processing_token=?
                WHERE transport=? AND bot_id=? AND external_event_id=?
                  AND status='processing' AND updated_at<?
                """,
                (
                    now.isoformat(),
                    now.isoformat(),
                    processing_token,
                    transport,
                    bot_id,
                    event_id,
                    stale,
                ),
            )
            claimed = cursor.rowcount == 1
        await self.connection.commit()
        return processing_token if claimed else None

    @_serialized
    async def renew_event(
        self, transport: str, bot_id: str, event_id: str, processing_token: str
    ) -> bool:
        cursor = await self.connection.execute(
            """
            UPDATE event_receipts SET updated_at=?
            WHERE transport=? AND bot_id=? AND external_event_id=?
              AND status='processing' AND processing_token=?
            """,
            (_now(), transport, bot_id, event_id, processing_token),
        )
        await self.connection.commit()
        return cursor.rowcount == 1

    @_serialized
    async def event_response(
        self, transport: str, bot_id: str, event_id: str
    ) -> ResponsePlan | None:
        cursor = await self.connection.execute(
            """
            SELECT status, response_json FROM event_receipts
            WHERE transport=? AND bot_id=? AND external_event_id=?
              AND status IN ('completed', 'delivered', 'dead_letter')
            """,
            (transport, bot_id, event_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        if row["status"] == "delivered":
            return ResponsePlan(track_delivery=False)
        if row["status"] == "dead_letter":
            return ResponsePlan(
                messages=(OutboundMessage(text="已生成的回复无法恢复，请重新发送请求。"),),
                track_delivery=False,
            )
        if not row["response_json"]:
            return ResponsePlan()
        try:
            payload = json.loads(row["response_json"])
            self._validate_response_event(payload, (transport, bot_id, event_id))
            return self._response_from_payload(
                payload,
                include_artifacts=row["status"] == "completed",
                artifact_root=self._event_outbox_root(transport, bot_id, event_id),
            )
        except (AttributeError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            await self._quarantine_response(transport, bot_id, event_id, row["response_json"])
            return ResponsePlan(
                messages=(OutboundMessage(text="已生成的回复无法恢复，请重新发送请求。"),),
                track_delivery=False,
            )

    def _response_from_payload(
        self,
        payload: dict[str, Any],
        *,
        include_artifacts: bool,
        artifact_root: Path,
    ) -> ResponsePlan:
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            raise ValueError("response messages must be a list")
        messages: list[OutboundMessage] = []
        for item in payload["messages"]:
            if not isinstance(item, dict):
                raise ValueError("response message must be an object")
            artifacts_payload = item.get("artifacts", [])
            buttons_payload = item.get("buttons", [])
            if not isinstance(artifacts_payload, list) or not isinstance(buttons_payload, list):
                raise ValueError("response artifacts and buttons must be lists")
            artifacts = (
                tuple(
                    self._outbox_artifact(artifact, artifact_root) for artifact in artifacts_payload
                )
                if include_artifacts
                else ()
            )
            if not all(isinstance(button, dict) for button in buttons_payload):
                raise ValueError("response button must be an object")
            buttons = tuple(
                ActionButton(
                    str(button["label"]),
                    str(button["action_id"]),
                    str(button.get("style", "secondary")),
                )
                for button in buttons_payload
            )
            messages.append(
                OutboundMessage(
                    text=str(item.get("text", "")),
                    private=bool(item.get("private", False)),
                    artifacts=artifacts,
                    buttons=buttons,
                )
            )
        return ResponsePlan(messages=tuple(messages))

    @staticmethod
    def _validate_response_event(payload: dict[str, Any], expected: tuple[str, str, str]) -> None:
        event = payload["event"]
        address = event["address"]
        actual = (
            str(address["transport"]),
            str(address["bot_id"]),
            str(event["event_id"]),
        )
        if actual != expected:
            raise ValueError("outbox event does not match its receipt")

    @staticmethod
    def _event_key(transport: str, bot_id: str, event_id: str) -> str:
        return hashlib.blake2b(
            f"{transport}\x1f{bot_id}\x1f{event_id}".encode(), digest_size=16
        ).hexdigest()

    def _event_outbox_root(self, transport: str, bot_id: str, event_id: str) -> Path:
        return self.path.parent / "outbox" / self._event_key(transport, bot_id, event_id)

    def _outbox_artifact(self, payload: dict[str, Any], artifact_root: Path) -> Artifact:
        if not isinstance(payload, dict):
            raise ValueError("response artifact must be an object")
        root = artifact_root.resolve(strict=True)
        candidate = Path(str(payload["path"]))
        if candidate.is_symlink():
            raise ValueError("outbox artifact must not be a symbolic link")
        path = candidate.resolve(strict=True)
        metadata = path.stat()
        if (
            not path.is_relative_to(root)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise ValueError("outbox artifact is not a safe regular file")
        return Artifact(
            kind=ArtifactKind(payload["kind"]),
            filename=str(payload["filename"]),
            mime_type=str(payload["mime_type"]),
            path=path,
        )

    async def complete_event(
        self,
        event: InboundEvent,
        response: ResponsePlan,
        processing_token: str,
        completion: ChatCompletion | None = None,
    ) -> None:
        transport, bot_id, event_id = event.idempotency_key
        async with self.operation():
            cursor = await self.connection.execute(
                """
                SELECT 1 FROM event_receipts
                WHERE transport=? AND bot_id=? AND external_event_id=?
                  AND status='processing' AND processing_token=?
                """,
                (transport, bot_id, event_id, processing_token),
            )
            if await cursor.fetchone() is None:
                raise RuntimeError("event processing lease is no longer valid")
        persistence = asyncio.create_task(
            asyncio.to_thread(
                self._persist_response_artifacts,
                transport,
                bot_id,
                event_id,
                processing_token,
                response,
            )
        )
        try:
            persisted_artifacts = await asyncio.shield(persistence)
        except asyncio.CancelledError:
            try:
                persisted_artifacts = await persistence
            except BaseException:
                raise
            await self._cleanup_persisted_artifacts(persisted_artifacts)
            raise
        payload = {
            "event": {
                "event_id": event.event_id,
                "address": {
                    "transport": event.address.transport.value,
                    "bot_id": event.address.bot_id,
                    "chat_id": event.address.chat_id,
                    "thread_id": event.address.thread_id,
                    "kind": event.address.kind.value,
                },
                "actor": {
                    "provider": event.actor.identity.provider.value,
                    "external_id": event.actor.identity.external_id,
                    "display_name": event.actor.display_name,
                    "username": event.actor.username,
                    "is_bot": event.actor.is_bot,
                },
                "permission": {
                    "is_private": event.permission.is_private,
                    "is_admin": event.permission.is_admin,
                },
            },
            "messages": [
                {
                    "text": message.text,
                    "private": message.private,
                    "artifacts": persisted_artifacts[index],
                    "buttons": [
                        {
                            "label": button.label,
                            "action_id": button.action_id,
                            "style": button.style,
                        }
                        for button in message.buttons
                    ],
                }
                for index, message in enumerate(response.messages)
            ],
        }
        committed = False
        try:
            async with self.operation():
                try:
                    await self.connection.execute("BEGIN IMMEDIATE")
                    cursor = await self.connection.execute(
                        """
                        UPDATE event_receipts SET status='completed', response_json=?, updated_at=?,
                            processing_token=NULL
                        WHERE transport=? AND bot_id=? AND external_event_id=?
                          AND status='processing' AND processing_token=?
                        """,
                        (
                            json.dumps(payload, ensure_ascii=False),
                            _now(),
                            transport,
                            bot_id,
                            event_id,
                            processing_token,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("event processing lease is no longer valid")
                    if completion is not None:
                        await self._store_chat_completion(event, completion)
                    commit = asyncio.create_task(self.connection.commit())
                    try:
                        await asyncio.shield(commit)
                    except asyncio.CancelledError:
                        await commit
                        committed = True
                        raise
                    committed = True
                except BaseException:
                    if not committed and self.connection.in_transaction:
                        rollback = asyncio.create_task(self.connection.rollback())
                        await asyncio.shield(rollback)
                    raise
        except BaseException as exc:
            if not committed and not isinstance(exc, asyncio.CancelledError):
                await self._cleanup_persisted_artifacts(persisted_artifacts)
            raise

    async def _store_chat_completion(self, event: InboundEvent, completion: ChatCompletion) -> None:
        source_event_key = "\x1f".join(event.idempotency_key)
        for message in (completion.user_message, completion.assistant_message):
            await self.connection.execute(
                """
                INSERT INTO messages(
                    conversation_id, role, content, run_id, model, status,
                    created_at, source_event_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id, source_event_key, role) DO UPDATE SET
                    content=excluded.content, run_id=excluded.run_id,
                    model=excluded.model, status=excluded.status,
                    created_at=excluded.created_at
                """,
                (
                    completion.session.stable_id,
                    message.role.value,
                    message.content,
                    message.run_id,
                    message.model,
                    message.status,
                    message.created_at.isoformat(),
                    source_event_key,
                ),
            )
        await self.connection.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (_now(), completion.session.stable_id),
        )
        assistant = completion.assistant_message
        if assistant.run_id is not None and assistant.model is not None:
            await self.connection.execute(
                """
                INSERT INTO usage_events(
                    identity_id, conversation_id, run_id, model,
                    input_tokens, output_tokens, created_at, source_event_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_event_key) DO UPDATE SET
                    run_id=excluded.run_id, model=excluded.model,
                    input_tokens=excluded.input_tokens,
                    output_tokens=excluded.output_tokens,
                    created_at=excluded.created_at
                """,
                (
                    completion.identity_id,
                    completion.session.stable_id,
                    assistant.run_id,
                    assistant.model,
                    completion.input_tokens,
                    completion.output_tokens,
                    assistant.created_at.isoformat(),
                    source_event_key,
                ),
            )
        if completion.memory_scope is not None and completion.memory_payload is not None:
            now = _now()
            await self.connection.execute(
                """
                INSERT INTO jobs(
                    id, kind, scope, payload_json, status, attempts, available_at,
                    source_event_key, created_at, updated_at
                ) VALUES (?, 'memory.extract', ?, ?, 'pending', 0, ?, ?, ?, ?)
                ON CONFLICT(source_event_key) DO NOTHING
                """,
                (
                    uuid.uuid4().hex,
                    completion.memory_scope,
                    json.dumps(completion.memory_payload, ensure_ascii=False),
                    now,
                    source_event_key,
                    now,
                    now,
                ),
            )

    @_serialized
    async def pending_responses(
        self, transport: str, bot_id: str, limit: int = 100
    ) -> list[tuple[InboundEvent, ResponsePlan]]:
        cursor = await self.connection.execute(
            """
            SELECT transport, bot_id, external_event_id, response_json
            FROM event_receipts
            WHERE transport=? AND bot_id=? AND status='completed'
              AND response_json IS NOT NULL
              AND (next_attempt_at IS NULL OR next_attempt_at<=?)
            ORDER BY updated_at LIMIT ?
            """,
            (transport, bot_id, _now(), limit),
        )
        result = []
        for row in await cursor.fetchall():
            try:
                payload = json.loads(row["response_json"])
                event_payload = payload["event"]
                address_payload = event_payload["address"]
                actor_payload = event_payload["actor"]
                permission_payload = event_payload["permission"]
                address = ConversationAddress(
                    transport=TransportName(address_payload["transport"]),
                    bot_id=address_payload["bot_id"],
                    chat_id=address_payload["chat_id"],
                    thread_id=address_payload["thread_id"],
                    kind=ChatKind(address_payload["kind"]),
                )
                actor = Actor(
                    identity=ExternalIdentity(
                        TransportName(actor_payload["provider"]),
                        actor_payload["external_id"],
                    ),
                    display_name=actor_payload["display_name"],
                    username=actor_payload.get("username"),
                    is_bot=bool(actor_payload.get("is_bot", False)),
                )
                event = InboundEvent(
                    event_id=event_payload["event_id"],
                    address=address,
                    actor=actor,
                    permission=RequestPermission(
                        bool(permission_payload["is_private"]),
                        bool(permission_payload["is_admin"]),
                    ),
                )
                if event.idempotency_key != (
                    row["transport"],
                    row["bot_id"],
                    row["external_event_id"],
                ):
                    raise ValueError("outbox event does not match its receipt")
                response = self._response_from_payload(
                    payload,
                    include_artifacts=True,
                    artifact_root=self._event_outbox_root(
                        row["transport"], row["bot_id"], row["external_event_id"]
                    ),
                )
            except (
                AttributeError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                await self._quarantine_response(
                    row["transport"],
                    row["bot_id"],
                    row["external_event_id"],
                    row["response_json"],
                    error=f"invalid outbox payload: {exc}",
                )
                continue
            result.append((event, response))
        await self.connection.commit()
        return result

    async def _quarantine_response(
        self,
        transport: str,
        bot_id: str,
        event_id: str,
        response_json: str,
        *,
        error: str = "invalid outbox payload",
    ) -> None:
        await self.connection.execute(
            """
            UPDATE event_receipts SET status='dead_letter', response_json=NULL,
                last_delivery_error=?, updated_at=?
            WHERE transport=? AND bot_id=? AND external_event_id=?
              AND status='completed'
            """,
            (error[:2000], _now(), transport, bot_id, event_id),
        )
        await self.connection.commit()
        await asyncio.to_thread(
            self._remove_response_artifacts,
            response_json,
            self._event_outbox_root(transport, bot_id, event_id),
        )

    @_serialized
    async def event_delivery_pending(self, transport: str, bot_id: str, event_id: str) -> bool:
        cursor = await self.connection.execute(
            """
            SELECT 1 FROM event_receipts
            WHERE transport=? AND bot_id=? AND external_event_id=? AND status='completed'
            """,
            (transport, bot_id, event_id),
        )
        return await cursor.fetchone() is not None

    def _persist_response_artifacts(
        self,
        transport: str,
        bot_id: str,
        event_id: str,
        processing_token: str,
        response: ResponsePlan,
    ) -> list[list[dict[str, str]]]:
        event_key = self._event_key(transport, bot_id, event_id)
        root = self.path.parent / "outbox" / event_key / processing_token
        result: list[list[dict[str, str]]] = []
        total_bytes = 0
        try:
            if any(message.artifacts for message in response.messages):
                for directory in (root.parent.parent, root.parent, root):
                    directory.mkdir(exist_ok=True, mode=0o700)
                    if directory.is_symlink() or not directory.is_dir():
                        raise ValueError("outbox directory is not safe")
                    directory.chmod(0o700)
            for message_index, message in enumerate(response.messages):
                artifacts: list[dict[str, str]] = []
                for artifact_index, artifact in enumerate(message.artifacts):
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact.filename)[:128]
                    safe_name = safe_name or "artifact"
                    destination = root / f"{message_index}-{artifact_index}-{safe_name}"
                    data = self._artifact_bytes(artifact)
                    total_bytes += len(data)
                    if total_bytes > self.max_output_artifact_bytes:
                        raise ValueError("artifacts exceed the output byte budget")
                    temporary = destination.with_name(f".{destination.name}.tmp")
                    descriptor = os.open(
                        temporary,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_CLOEXEC
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                    )
                    try:
                        with os.fdopen(descriptor, "wb") as handle:
                            descriptor = -1
                            handle.write(data)
                            handle.flush()
                            os.fsync(handle.fileno())
                    finally:
                        if descriptor >= 0:
                            os.close(descriptor)
                    temporary.replace(destination)
                    artifacts.append(
                        {
                            "kind": artifact.kind.value,
                            "filename": artifact.filename,
                            "mime_type": artifact.mime_type,
                            "path": str(destination),
                        }
                    )
                result.append(artifacts)
            if root.exists():
                for directory in (root, root.parent, root.parent.parent, self.path.parent):
                    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        return result

    def _artifact_bytes(self, artifact: Artifact) -> bytes:
        if artifact.data is not None:
            data = artifact.data
        else:
            descriptor = os.open(
                Path(artifact.path),
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ValueError("artifact path is not a safe regular file")
                if metadata.st_size > self.max_output_artifact_bytes:
                    raise ValueError("artifact exceeds the output byte budget")
                with os.fdopen(descriptor, "rb") as handle:
                    descriptor = -1
                    data = handle.read(self.max_output_artifact_bytes + 1)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        if len(data) > self.max_output_artifact_bytes:
            raise ValueError("artifact exceeds the output byte budget")
        return data

    def _remove_persisted_artifacts(self, messages: list[list[dict[str, str]]]) -> None:
        payload = json.dumps({"messages": [{"artifacts": item} for item in messages]})
        self._remove_response_artifacts(payload)

    async def _cleanup_persisted_artifacts(self, messages: list[list[dict[str, str]]]) -> None:
        cleanup = asyncio.create_task(asyncio.to_thread(self._remove_persisted_artifacts, messages))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    @_serialized
    async def mark_event_delivered(self, transport: str, bot_id: str, event_id: str) -> None:
        cursor = await self.connection.execute(
            """
            SELECT response_json FROM event_receipts
            WHERE transport=? AND bot_id=? AND external_event_id=?
              AND status='completed'
            """,
            (transport, bot_id, event_id),
        )
        row = await cursor.fetchone()
        now = _now()
        update_cursor = await self.connection.execute(
            """
            UPDATE event_receipts SET status='delivered', delivered_at=?, updated_at=?,
                next_attempt_at=NULL, last_delivery_error=NULL
            WHERE transport=? AND bot_id=? AND external_event_id=?
              AND status='completed'
            """,
            (now, now, transport, bot_id, event_id),
        )
        await self.connection.commit()
        if update_cursor.rowcount == 1 and row is not None and row["response_json"]:
            await asyncio.to_thread(
                self._remove_response_artifacts,
                row["response_json"],
                self._event_outbox_root(transport, bot_id, event_id),
            )

    @_serialized
    async def mark_event_delivery_failed(
        self, transport: str, bot_id: str, event_id: str, error: str
    ) -> None:
        cursor = await self.connection.execute(
            """
            SELECT delivery_attempts, response_json FROM event_receipts
            WHERE transport=? AND bot_id=? AND external_event_id=?
              AND status='completed'
            """,
            (transport, bot_id, event_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return
        attempts = int(row["delivery_attempts"]) + 1
        dead_letter = attempts >= 8
        retry_at = None
        if not dead_letter:
            retry_at = (
                datetime.now(UTC) + timedelta(seconds=min(5 * (2 ** (attempts - 1)), 3600))
            ).isoformat()
        await self.connection.execute(
            """
            UPDATE event_receipts SET status=?, delivery_attempts=?, next_attempt_at=?,
                last_delivery_error=?, updated_at=?,
                response_json=CASE WHEN ? THEN NULL ELSE response_json END
            WHERE transport=? AND bot_id=? AND external_event_id=?
              AND status='completed'
            """,
            (
                "dead_letter" if dead_letter else "completed",
                attempts,
                retry_at,
                error[:2000],
                _now(),
                dead_letter,
                transport,
                bot_id,
                event_id,
            ),
        )
        await self.connection.commit()
        if dead_letter and row["response_json"]:
            await asyncio.to_thread(
                self._remove_response_artifacts,
                row["response_json"],
                self._event_outbox_root(transport, bot_id, event_id),
            )

    def _remove_response_artifacts(
        self, response_json: str, artifact_root: Path | None = None
    ) -> None:
        try:
            payload = json.loads(response_json)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("messages", []), list):
            return
        parents: set[Path] = set()
        for message in payload.get("messages", []):
            if not isinstance(message, dict) or not isinstance(message.get("artifacts", []), list):
                continue
            for artifact in message.get("artifacts", []):
                if not isinstance(artifact, dict):
                    continue
                candidate = Path(str(artifact.get("path", "")))
                root = (
                    artifact_root if artifact_root is not None else self.path.parent / "outbox"
                ).resolve()
                if not candidate.is_absolute() or candidate.is_symlink():
                    continue
                path = candidate.resolve()
                if path.name and path.is_relative_to(root):
                    path.unlink(missing_ok=True)
                    parents.add(path.parent)
        for parent in parents:
            while parent != root and parent.is_relative_to(root):
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

    @_serialized
    async def discard_event(
        self, transport: str, bot_id: str, event_id: str, processing_token: str
    ) -> None:
        now = _now()
        await self.connection.execute(
            """
            UPDATE event_receipts SET status='delivered', delivered_at=?, updated_at=?,
                processing_token=NULL
            WHERE transport=? AND bot_id=? AND external_event_id=?
              AND status='processing' AND processing_token=?
            """,
            (now, now, transport, bot_id, event_id, processing_token),
        )
        await self.connection.commit()

    @_serialized
    async def ensure_identity(self, actor: Actor) -> int:
        now = _now()
        await self.connection.execute(
            """
            INSERT INTO identities(
                provider, external_id, display_name, username, is_bot, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, external_id) DO UPDATE SET
                display_name=excluded.display_name,
                username=excluded.username,
                is_bot=excluded.is_bot,
                updated_at=excluded.updated_at
            """,
            (
                actor.identity.provider.value,
                actor.identity.external_id,
                actor.display_name,
                actor.username,
                int(actor.is_bot),
                now,
                now,
            ),
        )
        cursor = await self.connection.execute(
            "SELECT id FROM identities WHERE provider=? AND external_id=?",
            (actor.identity.provider.value, actor.identity.external_id),
        )
        row = await cursor.fetchone()
        await self.connection.commit()
        if row is None:
            raise RuntimeError("identity upsert did not return a row")
        return int(row[0])

    @_serialized
    async def identity_id(self, identity: ExternalIdentity) -> int | None:
        cursor = await self.connection.execute(
            "SELECT id FROM identities WHERE provider=? AND external_id=?",
            (identity.provider.value, identity.external_id),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else None

    @_serialized
    async def ensure_chat(self, address: ConversationAddress, title: str | None = None) -> int:
        now = _now()
        await self.connection.execute(
            """
            INSERT INTO chats(transport, bot_id, external_id, kind, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(transport, bot_id, external_id) DO UPDATE SET
                kind=excluded.kind,
                title=COALESCE(excluded.title, chats.title),
                updated_at=excluded.updated_at
            """,
            (
                address.transport.value,
                address.bot_id,
                address.chat_id,
                address.kind.value,
                title,
                now,
                now,
            ),
        )
        cursor = await self.connection.execute(
            "SELECT id FROM chats WHERE transport=? AND bot_id=? AND external_id=?",
            (address.transport.value, address.bot_id, address.chat_id),
        )
        row = await cursor.fetchone()
        await self.connection.commit()
        if row is None:
            raise RuntimeError("chat upsert did not return a row")
        return int(row[0])

    @_serialized
    async def ensure_conversation(
        self, session: SessionKey, default_role: str
    ) -> ConversationState:
        now = _now()
        await self.connection.execute(
            """
            INSERT OR IGNORE INTO conversations(
                id, transport, bot_id, chat_external_id, thread_id, scope, owner_key,
                knowledge_scope, role_name, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.stable_id,
                session.address.transport.value,
                session.address.bot_id,
                session.address.chat_id,
                session.address.thread_id,
                session.scope.value,
                session.owner,
                session.knowledge_scope,
                default_role,
                now,
                now,
            ),
        )
        await self.connection.commit()
        conversation = await self.get_conversation(session)
        if conversation is None:
            raise RuntimeError("conversation upsert did not return a row")
        return conversation

    @_serialized
    async def get_conversation(self, session: SessionKey) -> ConversationState | None:
        cursor = await self.connection.execute(
            "SELECT role_name, created_at, updated_at FROM conversations WHERE id=?",
            (session.stable_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return ConversationState(
            session=session,
            role_name=str(row["role_name"]),
            created_at=_as_datetime(row["created_at"]),
            updated_at=_as_datetime(row["updated_at"]),
        )

    @_serialized
    async def set_conversation_role(
        self, session: SessionKey, role_name: str, event_key: str | None = None
    ) -> None:
        if event_key:
            await self.connection.execute("BEGIN IMMEDIATE")
        try:
            if event_key and await self._command_effect_exists(event_key):
                await self.connection.rollback()
                return
            await self.connection.execute(
                "UPDATE conversations SET role_name=?, updated_at=? WHERE id=?",
                (role_name, _now(), session.stable_id),
            )
            if event_key:
                await self._record_command_effect(event_key, "role")
            await self.connection.commit()
        except BaseException:
            await self.connection.rollback()
            raise

    @_serialized
    async def append_message(self, session: SessionKey, message: StoredMessage) -> int:
        cursor = await self.connection.execute(
            """
            INSERT INTO messages(conversation_id, role, content, run_id, model, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.stable_id,
                message.role.value,
                message.content,
                message.run_id,
                message.model,
                message.status,
                message.created_at.isoformat(),
            ),
        )
        await self.connection.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?", (_now(), session.stable_id)
        )
        await self.connection.commit()
        return int(cursor.lastrowid or 0)

    @_serialized
    async def recent_messages(self, session: SessionKey, limit: int) -> list[StoredMessage]:
        cursor = await self.connection.execute(
            """
            SELECT role, content, run_id, model, status, created_at
            FROM messages WHERE conversation_id=?
            ORDER BY created_at DESC, id DESC LIMIT ?
            """,
            (session.stable_id, limit),
        )
        rows = list(reversed(await cursor.fetchall()))
        return [
            StoredMessage(
                role=MessageRole(row["role"]),
                content=row["content"],
                run_id=row["run_id"],
                model=row["model"],
                status=row["status"],
                created_at=_as_datetime(row["created_at"]),
            )
            for row in rows
        ]

    @_serialized
    async def reset_conversation(self, session: SessionKey, event_key: str | None = None) -> int:
        if event_key:
            await self.connection.execute("BEGIN IMMEDIATE")
        try:
            if event_key and await self._command_effect_exists(event_key):
                await self.connection.rollback()
                return 0
            cursor = await self.connection.execute(
                "DELETE FROM messages WHERE conversation_id=?", (session.stable_id,)
            )
            if event_key:
                await self._record_command_effect(event_key, "reset")
            await self.connection.commit()
            return cursor.rowcount
        except BaseException:
            await self.connection.rollback()
            raise

    @_serialized
    async def seed_roles(self, roles: tuple[RoleDefinition, ...]) -> None:
        now = _now()
        for role in roles:
            await self.connection.execute(
                """
                INSERT INTO roles(
                    owner_key, name, description, system_prompt, greeting, farewell,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_key, name) DO UPDATE SET
                    description=excluded.description, system_prompt=excluded.system_prompt,
                    greeting=excluded.greeting, farewell=excluded.farewell,
                    updated_at=excluded.updated_at
                """,
                (
                    role.owner_key or "",
                    role.name,
                    role.description,
                    role.system_prompt,
                    role.greeting,
                    role.farewell,
                    now,
                    now,
                ),
            )
        await self.connection.commit()

    @_serialized
    async def list_roles(self, owner_key: str | None = None) -> list[RoleDefinition]:
        owner = owner_key or ""
        cursor = await self.connection.execute(
            """
            SELECT owner_key, name, description, system_prompt, greeting, farewell
            FROM roles WHERE owner_key IN ('', ?) ORDER BY owner_key, name
            """,
            (owner,),
        )
        return [self._role_from_row(row) for row in await cursor.fetchall()]

    @_serialized
    async def get_role(self, name: str, owner_key: str | None = None) -> RoleDefinition | None:
        owner = owner_key or ""
        cursor = await self.connection.execute(
            """
            SELECT owner_key, name, description, system_prompt, greeting, farewell
            FROM roles WHERE name=? AND owner_key IN ('', ?)
            ORDER BY CASE WHEN owner_key=? THEN 0 ELSE 1 END LIMIT 1
            """,
            (name, owner, owner),
        )
        row = await cursor.fetchone()
        return self._role_from_row(row) if row else None

    @staticmethod
    def _role_from_row(row: aiosqlite.Row) -> RoleDefinition:
        return RoleDefinition(
            owner_key=row["owner_key"] or None,
            name=row["name"],
            description=row["description"],
            system_prompt=row["system_prompt"],
            greeting=row["greeting"],
            farewell=row["farewell"],
        )

    @_serialized
    async def create_role(self, role: RoleDefinition) -> bool:
        now = _now()
        cursor = await self.connection.execute(
            """
            INSERT OR IGNORE INTO roles(
                owner_key, name, description, system_prompt, greeting, farewell,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                role.owner_key or "",
                role.name,
                role.description,
                role.system_prompt,
                role.greeting,
                role.farewell,
                now,
                now,
            ),
        )
        await self.connection.commit()
        return cursor.rowcount == 1

    @_serialized
    async def get_profile(self, identity_id: int) -> str | None:
        cursor = await self.connection.execute(
            "SELECT content FROM profiles WHERE identity_id=?", (identity_id,)
        )
        row = await cursor.fetchone()
        return str(row[0]) if row else None

    @_serialized
    async def set_profile(self, identity_id: int, content: str, source: str) -> None:
        digest = hashlib.sha256(content.encode()).hexdigest()
        await self.connection.execute(
            """
            INSERT INTO profiles(identity_id, content, content_hash, source, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(identity_id) DO UPDATE SET
                content=excluded.content, content_hash=excluded.content_hash,
                source=excluded.source, updated_at=excluded.updated_at
            """,
            (identity_id, content, digest, source, _now()),
        )
        await self.connection.commit()

    @_serialized
    async def set_profile_once(
        self,
        identity_id: int,
        content: str,
        source: str,
        event_key: str,
        expected_hash: str | None,
    ) -> str:
        await self.connection.execute("BEGIN IMMEDIATE")
        try:
            if await self._command_effect_exists(event_key):
                await self.connection.rollback()
                return "duplicate"
            cursor = await self.connection.execute(
                "SELECT content_hash FROM profiles WHERE identity_id=?", (identity_id,)
            )
            row = await cursor.fetchone()
            current_hash = str(row["content_hash"]) if row else None
            if current_hash != expected_hash:
                await self.connection.rollback()
                return "conflict"
            digest = hashlib.sha256(content.encode()).hexdigest()
            await self.connection.execute(
                """
                INSERT INTO profiles(identity_id, content, content_hash, source, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(identity_id) DO UPDATE SET
                    content=excluded.content, content_hash=excluded.content_hash,
                    source=excluded.source, updated_at=excluded.updated_at
                """,
                (identity_id, content, digest, source, _now()),
            )
            await self._record_command_effect(event_key, "profile")
            await self.connection.commit()
            return "applied"
        except BaseException:
            await self.connection.rollback()
            raise

    async def _command_effect_exists(self, event_key: str) -> bool:
        cursor = await self.connection.execute(
            "SELECT 1 FROM command_effects WHERE event_key=?", (event_key,)
        )
        return await cursor.fetchone() is not None

    async def _record_command_effect(self, event_key: str, kind: str) -> None:
        await self.connection.execute(
            "INSERT INTO command_effects(event_key, kind, created_at) VALUES (?, ?, ?)",
            (event_key, kind, _now()),
        )

    @_serialized
    async def clear_profile(self, identity_id: int, event_key: str | None = None) -> bool:
        if event_key:
            await self.connection.execute("BEGIN IMMEDIATE")
        try:
            if event_key and await self._command_effect_exists(event_key):
                await self.connection.rollback()
                return False
            cursor = await self.connection.execute(
                "DELETE FROM profiles WHERE identity_id=?", (identity_id,)
            )
            if event_key:
                await self._record_command_effect(event_key, "profile_clear")
            await self.connection.commit()
            return cursor.rowcount == 1
        except BaseException:
            await self.connection.rollback()
            raise

    @_serialized
    async def get_group_settings(self, chat_id: int) -> dict[str, Any] | None:
        cursor = await self.connection.execute(
            "SELECT * FROM group_settings WHERE chat_id=?", (chat_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "enabled": bool(row["enabled"]),
            "session_scope": row["session_scope"],
            "memory_enabled": bool(row["memory_enabled"]),
            "wake_words": json.loads(row["wake_words_json"]),
            "consent_at": row["consent_at"],
        }

    @_serialized
    async def set_group_settings(
        self,
        chat_id: int,
        defaults: dict[str, Any],
        *,
        event_key: str | None = None,
        **values: Any,
    ) -> None:
        if event_key:
            await self.connection.execute("BEGIN IMMEDIATE")
        try:
            if event_key and await self._command_effect_exists(event_key):
                await self.connection.rollback()
                return
            current = await self.get_group_settings(chat_id) or dict(defaults)
            current.update(values)
            await self.connection.execute(
                """
                INSERT INTO group_settings(
                    chat_id, enabled, session_scope, memory_enabled, wake_words_json,
                    consent_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    enabled=excluded.enabled, session_scope=excluded.session_scope,
                    memory_enabled=excluded.memory_enabled,
                    wake_words_json=excluded.wake_words_json,
                    consent_at=excluded.consent_at, updated_at=excluded.updated_at
                """,
                (
                    chat_id,
                    int(current["enabled"]),
                    current["session_scope"],
                    int(current["memory_enabled"]),
                    json.dumps(current["wake_words"], ensure_ascii=False),
                    current.get("consent_at"),
                    _now(),
                ),
            )
            if event_key:
                await self._record_command_effect(event_key, "group_settings")
            await self.connection.commit()
        except BaseException:
            await self.connection.rollback()
            raise

    @_serialized
    async def list_memories(self, scope: KnowledgeScope, limit: int = 20) -> list[dict[str, Any]]:
        cursor = await self.connection.execute(
            """
            SELECT id, content, category, importance, created_at, updated_at
            FROM memories WHERE scope=? ORDER BY updated_at DESC LIMIT ?
            """,
            (scope.key, limit),
        )
        return [dict(row) for row in await cursor.fetchall()]

    @_serialized
    async def delete_memory(self, entry_id: str, scope: KnowledgeScope) -> bool:
        await self.connection.execute(
            "DELETE FROM memories_fts WHERE memory_id=? AND scope=?", (entry_id, scope.key)
        )
        cursor = await self.connection.execute(
            "DELETE FROM memories WHERE id=? AND scope=?", (entry_id, scope.key)
        )
        await self.connection.commit()
        return cursor.rowcount == 1

    @_serialized
    async def memory_ids_beyond(self, scope: KnowledgeScope, keep: int) -> list[str]:
        cursor = await self.connection.execute(
            """
            SELECT id FROM memories WHERE scope=?
            ORDER BY updated_at DESC, id DESC LIMIT -1 OFFSET ?
            """,
            (scope.key, keep),
        )
        return [str(row[0]) for row in await cursor.fetchall()]

    @_serialized
    async def database_stats(self) -> dict[str, int]:
        result: dict[str, int] = {}
        tables = (
            "identities",
            "chats",
            "conversations",
            "messages",
            "memories",
            "documents",
            "chunks",
        )
        for table in tables:
            cursor = await self.connection.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - table is a fixed allowlist
            )
            row = await cursor.fetchone()
            result[table] = int(row[0]) if row else 0
        database_bytes = self.path.stat().st_size if self.path.exists() else 0
        wal = self.path.with_name(f"{self.path.name}-wal")
        shm = self.path.with_name(f"{self.path.name}-shm")
        wal_bytes = wal.stat().st_size if wal.exists() else 0
        shm_bytes = shm.stat().st_size if shm.exists() else 0
        result["database_bytes"] = database_bytes
        result["wal_bytes"] = wal_bytes
        result["shm_bytes"] = shm_bytes
        result["storage_bytes"] = database_bytes + wal_bytes + shm_bytes
        return result

    @_serialized
    async def user_stats(self, identity_id: int) -> dict[str, int]:
        cursor = await self.connection.execute(
            "SELECT provider, external_id FROM identities WHERE id=?", (identity_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return {"conversations": 0, "messages": 0}
        owner_key = f"{row['provider']}:{row['external_id']}"
        cursor = await self.connection.execute(
            "SELECT COUNT(*) FROM conversations WHERE owner_key=?", (owner_key,)
        )
        conversations = int((await cursor.fetchone())[0])
        cursor = await self.connection.execute(
            """
            SELECT COUNT(*) FROM messages m
            JOIN conversations c ON c.id=m.conversation_id WHERE c.owner_key=?
            """,
            (owner_key,),
        )
        messages = int((await cursor.fetchone())[0])
        return {"conversations": conversations, "messages": messages}

    @_serialized
    async def conversation_from_id(self, conversation_id: str) -> ConversationState | None:
        cursor = await self.connection.execute(
            "SELECT * FROM conversations WHERE id=?", (conversation_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        owner_key = str(row["owner_key"])
        if owner_key == "shared":
            actor = ExternalIdentity(TransportName(row["transport"]), "shared")
        else:
            provider, external_id = owner_key.split(":", 1)
            actor = ExternalIdentity(TransportName(provider), external_id)
        address = ConversationAddress(
            transport=TransportName(row["transport"]),
            bot_id=row["bot_id"],
            chat_id=row["chat_external_id"],
            thread_id=row["thread_id"],
            kind=ChatKind.PRIVATE if row["scope"] == "private" else ChatKind.GROUP,
        )
        return ConversationState(
            session=SessionKey(address, actor, SessionScope(row["scope"])),
            role_name=row["role_name"],
            created_at=_as_datetime(row["created_at"]),
            updated_at=_as_datetime(row["updated_at"]),
        )

    @_serialized
    async def record_usage(
        self,
        identity_id: int,
        conversation_id: str,
        run_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        await self.connection.execute(
            """
            INSERT INTO usage_events(
                identity_id, conversation_id, run_id, model,
                input_tokens, output_tokens, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identity_id,
                conversation_id,
                run_id,
                model,
                input_tokens,
                output_tokens,
                _now(),
            ),
        )
        await self.connection.commit()

    @_serialized
    async def enqueue_job(
        self, kind: str, scope: str, payload: dict[str, Any], *, delay_seconds: int = 0
    ) -> str:
        job_id = uuid.uuid4().hex
        now = datetime.now(UTC)
        await self.connection.execute(
            """
            INSERT INTO jobs(
                id, kind, scope, payload_json, status, attempts, available_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?)
            """,
            (
                job_id,
                kind,
                scope,
                json.dumps(payload, ensure_ascii=False),
                (now + timedelta(seconds=delay_seconds)).isoformat(),
                now.isoformat(),
                now.isoformat(),
            ),
        )
        await self.connection.commit()
        return job_id

    @_serialized
    async def lease_jobs(self, *, limit: int) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        lease_until = (now + timedelta(minutes=5)).isoformat()
        await self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self.connection.execute(
                """
                SELECT id, kind, payload_json, attempts FROM jobs
                WHERE (status='pending' OR (status='running' AND lease_until < ?))
                  AND available_at <= ?
                ORDER BY created_at LIMIT ?
                """,
                (now.isoformat(), now.isoformat(), limit),
            )
            rows = await cursor.fetchall()
            lease_tokens: dict[str, str] = {}
            for row in rows:
                lease_token = uuid.uuid4().hex
                lease_tokens[str(row["id"])] = lease_token
                await self.connection.execute(
                    """
                    UPDATE jobs SET status='running', attempts=attempts+1,
                        lease_until=?, lease_token=?, updated_at=? WHERE id=?
                    """,
                    (lease_until, lease_token, now.isoformat(), row["id"]),
                )
            await self.connection.commit()
        except BaseException:
            await self.connection.rollback()
            raise
        return [
            {
                "id": row["id"],
                "kind": row["kind"],
                "payload": json.loads(row["payload_json"]),
                "attempts": int(row["attempts"]) + 1,
                "lease_token": lease_tokens[str(row["id"])],
            }
            for row in rows
        ]

    @_serialized
    async def complete_job(self, job_id: str, lease_token: str) -> None:
        await self.connection.execute(
            """
            UPDATE jobs SET status='complete', lease_until=NULL, lease_token=NULL, updated_at=?
            WHERE id=? AND status='running' AND lease_token=?
            """,
            (_now(), job_id, lease_token),
        )
        await self.connection.commit()

    @_serialized
    async def fail_job(self, job_id: str, lease_token: str, error: str, *, retry: bool) -> None:
        status = "pending" if retry else "failed"
        available = datetime.now(UTC) + timedelta(seconds=30)
        await self.connection.execute(
            """
            UPDATE jobs SET status=?, available_at=?, lease_until=NULL, lease_token=NULL,
                error=?, updated_at=?
            WHERE id=? AND status='running' AND lease_token=?
            """,
            (
                status,
                available.isoformat(),
                error[:2000],
                _now(),
                job_id,
                lease_token,
            ),
        )
        await self.connection.commit()

    @_serialized
    async def create_tool_approval(
        self,
        conversation_id: str,
        actor_key: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        ttl_seconds: int = 600,
    ) -> str:
        approval_id = uuid.uuid4().hex
        now = datetime.now(UTC)
        await self.connection.execute(
            """
            INSERT INTO tool_approvals(
                id, conversation_id, actor_key, tool_name, arguments_json,
                status, expires_at, created_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                approval_id,
                conversation_id,
                actor_key,
                tool_name,
                json.dumps(arguments, ensure_ascii=False),
                (now + timedelta(seconds=ttl_seconds)).isoformat(),
                now.isoformat(),
            ),
        )
        await self.connection.commit()
        return approval_id

    @_serialized
    async def pending_tool_approval(
        self, approval_id: str, actor_key: str, conversation_id: str
    ) -> dict[str, Any] | None:
        cursor = await self.connection.execute(
            """
            SELECT tool_name, arguments_json FROM tool_approvals
            WHERE id=? AND actor_key=? AND conversation_id=?
              AND status='pending' AND expires_at>?
            """,
            (approval_id, actor_key, conversation_id, _now()),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "tool_name": row["tool_name"],
            "arguments": json.loads(row["arguments_json"]),
        }

    @_serialized
    async def resolve_tool_approval(
        self,
        approval_id: str,
        actor_key: str,
        conversation_id: str,
        *,
        approve: bool,
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        await self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self.connection.execute(
                """
                SELECT tool_name, arguments_json FROM tool_approvals
                WHERE id=? AND actor_key=? AND conversation_id=?
                  AND status='pending' AND expires_at>?
                """,
                (approval_id, actor_key, conversation_id, now.isoformat()),
            )
            row = await cursor.fetchone()
            if row is None:
                await self.connection.rollback()
                return None
            status = "approved" if approve else "denied"
            await self.connection.execute(
                "UPDATE tool_approvals SET status=?, resolved_at=? WHERE id=?",
                (status, now.isoformat(), approval_id),
            )
            await self.connection.commit()
        except BaseException:
            await self.connection.rollback()
            raise
        return {
            "tool_name": row["tool_name"],
            "arguments": json.loads(row["arguments_json"]),
            "approved": approve,
        }

    @_serialized
    async def finish_tool_approval(self, approval_id: str, *, ok: bool) -> None:
        await self.connection.execute(
            """
            UPDATE tool_approvals SET status=?, resolved_at=?
            WHERE id=? AND status='approved'
            """,
            ("complete" if ok else "failed", _now(), approval_id),
        )
        await self.connection.commit()
