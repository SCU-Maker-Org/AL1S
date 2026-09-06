"""Durable SQLite-backed background jobs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from al1s.adapters.storage import Database

JobHandler = Callable[[dict[str, Any]], Awaitable[None]]


class PersistentJobWorker:
    def __init__(
        self,
        database: Database,
        handlers: dict[str, JobHandler],
        *,
        poll_seconds: float = 1.0,
        concurrency: int = 2,
        handler_timeout_seconds: float = 90.0,
    ):
        self.database = database
        self.handlers = handlers
        self.poll_seconds = poll_seconds
        self.concurrency = concurrency
        self.handler_timeout_seconds = handler_timeout_seconds
        self._stop = asyncio.Event()
        self._logger = logging.getLogger(__name__)
        self._error = ""

    async def run(self) -> None:
        async with asyncio.TaskGroup() as group:
            for worker_id in range(self.concurrency):
                group.create_task(self._worker(), name=f"job-worker:{worker_id}")

    async def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                jobs = await self.database.lease_jobs(limit=1)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception("failed to lease a background job")
                self._error = "database job leasing failed"
                await self._sleep()
                continue
            self._error = ""
            if not jobs:
                await self._sleep()
                continue
            await self._execute(jobs[0])

    async def _sleep(self) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
        except TimeoutError:
            pass

    async def _execute(self, job: dict[str, Any]) -> None:
        handler = self.handlers.get(job["kind"])
        if handler is None:
            try:
                await self.database.fail_job(
                    job["id"], job["lease_token"], "unknown job kind", retry=False
                )
            except Exception:
                self._logger.exception("failed to reject unknown background job")
            return
        try:
            async with asyncio.timeout(self.handler_timeout_seconds):
                await handler(job["payload"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.exception("background job failed", extra={"job_id": job["id"]})
            try:
                await self.database.fail_job(
                    job["id"],
                    job["lease_token"],
                    str(exc),
                    retry=job["attempts"] < 3,
                )
            except Exception:
                self._logger.exception("failed to update failed background job")
        else:
            try:
                await self.database.complete_job(job["id"], job["lease_token"])
            except Exception:
                self._logger.exception("failed to complete background job")

    def stop(self) -> None:
        self._stop.set()

    def health(self) -> dict[str, str]:
        return {
            "state": "degraded" if self._error else "running",
            "error": self._error,
        }
