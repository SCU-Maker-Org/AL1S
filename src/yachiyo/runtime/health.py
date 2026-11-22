"""Atomic filesystem heartbeat used by the CLI and OCI healthcheck."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HealthProvider = Callable[[], dict[str, Any]]


class Heartbeat:
    def __init__(self, path: Path, provider: HealthProvider, *, interval: float = 10):
        self.path = path
        self.provider = provider
        self.interval = interval
        self._stop = asyncio.Event()

    async def run(self) -> None:
        await asyncio.to_thread(self.path.parent.mkdir, parents=True, exist_ok=True, mode=0o700)
        while not self._stop.is_set():
            await self.write()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except TimeoutError:
                continue

    async def write(self) -> None:
        payload = {
            "updated_at": datetime.now(UTC).isoformat(),
            **self.provider(),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)

        def commit() -> None:
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            os.chmod(temporary, 0o600)
            temporary.replace(self.path)

        await asyncio.to_thread(commit)

    def stop(self) -> None:
        self._stop.set()


def read_heartbeat(path: Path, *, max_age_seconds: float = 35) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"state": "unavailable"}
    if not isinstance(decoded, dict):
        return {"state": "invalid"}
    try:
        updated_at = datetime.fromisoformat(str(decoded["updated_at"]))
        age = (datetime.now(UTC) - updated_at.astimezone(UTC)).total_seconds()
    except (KeyError, TypeError, ValueError):
        return {"state": "invalid"}
    if age < 0 or age > max_age_seconds:
        return {**decoded, "state": "stale", "age_seconds": max(0.0, age)}
    return decoded
