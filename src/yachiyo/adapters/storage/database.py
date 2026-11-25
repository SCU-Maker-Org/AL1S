"""Single-connection asynchronous SQLite repository."""

from __future__ import annotations

import asyncio
import fcntl
import functools
import hashlib
import json
import os
import stat
import sys
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from yachiyo.config import AppPaths, AppSettings
from yachiyo.domain import (
    ActionButton,
    Actor,
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
    StoredMessage,
    TransportName,
)
from yachiyo.domain.message import MessageRole
from yachiyo.sql.schema import migrate, validate_schema

from .outbox import OutboxArtifacts


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


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, settings: AppSettings, paths: AppPaths):
        configured = Path(settings.storage.database_path).expanduser()
        self.path = configured if configured.is_absolute() else paths.data_dir / configured
        self.busy_timeout_ms = settings.storage.busy_timeout_ms
        self.event_lease_seconds = settings.policies.request_timeout_seconds + 60
        self.max_output_artifact_bytes = settings.policies.max_output_artifact_bytes
        self._connection: aiosqlite.Connection | None = None
        self._lock_descriptor: int | None = None
        self._owner_lock_descriptor: int | None = None
        self._operation_lock = _AsyncReentrantLock()

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("database is not open")
        return self._connection

    @asynccontextmanager
    async def operation(self) -> AsyncIterator[None]:
        async with self._operation_lock:
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
        self._outbox = OutboxArtifacts(self.path.parent / "outbox", self.max_output_artifact_bytes)
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
            raise RuntimeError("another yachiyo process is already using this database") from exc
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
            raise RuntimeError("another yachiyo process is already using this database") from exc
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
        await migrate(self.connection)
        await validate_schema(self.connection)
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
                        self._outbox.remove,
                        row["response_json"],
                        self._outbox.event_dir(
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
                self._outbox.event_dir(row["transport"], row["bot_id"], row["external_event_id"])
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
        return await asyncio.to_thread(self._outbox.remove_orphans, referenced)

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
                artifact_root=self._outbox.event_dir(transport, bot_id, event_id),
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
                    self._outbox.restore(artifact, artifact_root) for artifact in artifacts_payload
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
                self._outbox.persist,
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
            await self._outbox.discard(persisted_artifacts)
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
                await self._outbox.discard(persisted_artifacts)
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
                    artifact_root=self._outbox.event_dir(
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
            self._outbox.remove,
            response_json,
            self._outbox.event_dir(transport, bot_id, event_id),
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
                self._outbox.remove,
                row["response_json"],
                self._outbox.event_dir(transport, bot_id, event_id),
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
                self._outbox.remove,
                row["response_json"],
                self._outbox.event_dir(transport, bot_id, event_id),
            )

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
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
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
    async def recent_messages(self, session: SessionKey, limit: int) -> list[StoredMessage]:
        rows = await self.connection.execute_fetchall(
            """
            SELECT role, content, run_id, model, status, created_at
            FROM messages WHERE conversation_id=?
            ORDER BY created_at DESC, id DESC LIMIT ?
            """,
            (session.stable_id, limit),
        )
        return [
            StoredMessage(
                role=MessageRole(row["role"]),
                content=row["content"],
                run_id=row["run_id"],
                model=row["model"],
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in reversed(rows)
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
        rows = await self.connection.execute_fetchall(
            """
            SELECT owner_key, name, description, system_prompt, greeting, farewell
            FROM roles WHERE owner_key IN ('', ?) ORDER BY owner_key, name
            """,
            (owner,),
        )
        return [self._role_from_row(row) for row in rows]

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
        await self._upsert_profile(identity_id, content, source)
        await self.connection.commit()

    async def _upsert_profile(self, identity_id: int, content: str, source: str) -> None:
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
            await self._upsert_profile(identity_id, content, source)
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
        rows = await self.connection.execute_fetchall(
            """
            SELECT id, content, category, importance, created_at, updated_at
            FROM memories WHERE scope=? ORDER BY updated_at DESC LIMIT ?
            """,
            (scope.key, limit),
        )
        return [dict(row) for row in rows]

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
        rows = await self.connection.execute_fetchall(
            """
            SELECT id FROM memories WHERE scope=?
            ORDER BY updated_at DESC, id DESC LIMIT -1 OFFSET ?
            """,
            (scope.key, keep),
        )
        return [str(row[0]) for row in rows]

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
