"""Bounded request locks keyed by isolated conversation ID."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock
    used_at: float
    users: int = 0


class SessionBusyError(RuntimeError):
    pass


class SessionLockPool:
    def __init__(
        self,
        *,
        max_entries: int = 10_000,
        idle_seconds: float = 3600,
        max_waiters: int = 3,
        acquire_timeout: float = 5.0,
    ):
        self.max_entries = max_entries
        self.idle_seconds = idle_seconds
        self.max_waiters = max_waiters
        self.acquire_timeout = acquire_timeout
        self._entries: dict[str, _LockEntry] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        async with self._guard:
            self._prune_locked()
            entry = self._entries.get(key)
            if entry is None:
                entry = _LockEntry(asyncio.Lock(), time.monotonic())
                self._entries[key] = entry
            if entry.users >= self.max_waiters + 1:
                raise SessionBusyError("session queue is full")
            entry.users += 1
            entry.used_at = time.monotonic()
        try:
            try:
                await asyncio.wait_for(entry.lock.acquire(), timeout=self.acquire_timeout)
            except TimeoutError as exc:
                raise SessionBusyError("session is busy") from exc
            try:
                yield
            finally:
                entry.lock.release()
        finally:
            async with self._guard:
                entry.users -= 1
                entry.used_at = time.monotonic()
                self._prune_locked()

    def _prune_locked(self) -> None:
        now = time.monotonic()
        stale = [
            key
            for key, entry in self._entries.items()
            if entry.users == 0
            and not entry.lock.locked()
            and now - entry.used_at >= self.idle_seconds
        ]
        for key in stale:
            self._entries.pop(key, None)
        if len(self._entries) <= self.max_entries:
            return
        available = sorted(
            (
                (entry.used_at, key)
                for key, entry in self._entries.items()
                if entry.users == 0 and not entry.lock.locked()
            )
        )
        for _, key in available[: len(self._entries) - self.max_entries]:
            self._entries.pop(key, None)
