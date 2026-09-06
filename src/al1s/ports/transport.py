"""Chat transport contract."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from al1s.domain import InboundEvent, ResponsePlan, TransportName

EventReceiver = Callable[[InboundEvent], Awaitable[None]]


class CommandDescriptor(Protocol):
    name: str
    description: str


class Transport(Protocol):
    name: TransportName

    async def prepare(self, commands: Sequence[CommandDescriptor]) -> None: ...

    async def run(self, receiver: EventReceiver) -> None: ...

    async def send(self, event: InboundEvent, response: ResponsePlan) -> None: ...

    async def close(self) -> None: ...

    def health(self) -> dict[str, str]: ...
