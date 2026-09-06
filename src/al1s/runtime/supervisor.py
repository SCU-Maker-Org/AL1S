"""Independent transport supervision and response dispatch."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence

from al1s.application.chat import ChatApplication
from al1s.domain import ResponsePlan
from al1s.ports.transport import CommandDescriptor, Transport

TransportFactory = Callable[[], Transport]


class TransportSupervisor:
    def __init__(
        self,
        factory: TransportFactory,
        commands: Sequence[CommandDescriptor],
        application: ChatApplication,
        *,
        name: str,
    ):
        self.factory = factory
        self.commands = commands
        self.application = application
        self.name = name
        self._stop = asyncio.Event()
        self._transport: Transport | None = None
        self._state = "created"
        self._error = ""
        self._delivery_lock = asyncio.Lock()
        self._logger = logging.getLogger(f"al1s.transport.{name}")

    async def run(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            transport: Transport | None = None
            started_at: float | None = None
            stable = False
            try:
                transport = self.factory()
                self._transport = transport
                self._state = "starting"
                await transport.prepare(self.commands)
                self._state = "running"
                self._error = ""
                started_at = time.monotonic()

                async def receive(event, current_transport=transport) -> None:
                    try:
                        response = await self.application.handle(event)
                    except Exception:
                        self._logger.exception("inbound event failed")
                        return
                    if response is not None:
                        async with self._delivery_lock:
                            if (
                                not response.track_delivery
                                or await self.application.delivery_pending(event)
                            ):
                                await self._deliver(
                                    current_transport,
                                    event,
                                    response,
                                    attempts=1,
                                )

                delivery_task = asyncio.create_task(
                    self._replay_outbox(transport), name=f"outbox:{self.name}"
                )
                try:
                    await transport.run(receive)
                finally:
                    delivery_task.cancel()
                    await asyncio.gather(delivery_task, return_exceptions=True)
                stable = time.monotonic() - started_at >= 30
                if not self._stop.is_set():
                    self._state = "degraded"
                    self._error = "transport stopped unexpectedly"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                stable = bool(started_at is not None and time.monotonic() - started_at >= 30)
                self._state = "degraded"
                self._error = f"{type(exc).__name__}: {exc}"
                self._logger.exception("transport failed; retrying")
            finally:
                if transport is not None:
                    try:
                        await transport.close()
                    except Exception:
                        self._logger.exception("transport close failed")
                self._transport = None
            if not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except TimeoutError:
                    delay = 1.0 if stable else min(delay * 2, 60.0)
        self._state = "stopped"

    async def _deliver(
        self,
        transport: Transport,
        event,
        response: ResponsePlan,
        *,
        attempts: int,
    ) -> bool:
        last_error = "response delivery failed"
        for attempt in range(attempts):
            try:
                await transport.send(event, response)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 == attempts:
                    self._logger.exception("response delivery failed")
                    break
                await asyncio.sleep(2**attempt)
            else:
                if response.track_delivery:
                    try:
                        await self.application.mark_delivered(event)
                    except Exception:
                        self._logger.exception("response was delivered but acknowledgement failed")
                return True
        if response.track_delivery:
            try:
                await self.application.mark_delivery_failed(event, last_error)
            except Exception:
                self._logger.exception("failed to record response delivery failure")
        return False

    async def _replay_outbox(self, transport: Transport) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(5)
            if transport.health().get("state") != "running":
                continue
            try:
                async with self._delivery_lock:
                    bot_id = transport.health().get("bot_id", "")
                    if not bot_id or bot_id == "pending":
                        continue
                    pending = await self.application.pending_responses(self.name, bot_id)
                    for event, response in pending:
                        try:
                            if not await self.application.delivery_pending(event):
                                continue
                            await self._deliver(transport, event, response, attempts=1)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            self._logger.exception("outbox item replay failed")
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception("outbox replay cycle failed")

    async def stop(self) -> None:
        self._stop.set()
        if self._transport is not None:
            await self._transport.close()

    def health(self) -> dict[str, str]:
        result = {"transport": self.name, "state": self._state, "error": self._error}
        if self._transport is not None:
            result.update(self._transport.health())
        return result
