"""MCP tool registry contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from al1s.domain import AccessLevel, Artifact, RequestPermission
from al1s.ports.model import ToolDefinition


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[Artifact, ...] = ()


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    definition: ToolDefinition
    server: str
    original_name: str
    access: AccessLevel
    read_only: bool


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    conversation_id: str
    actor_key: str
    permission: RequestPermission


class ToolRegistry(Protocol):
    async def refresh(self) -> None: ...

    def available(self, permission: RequestPermission) -> tuple[RegisteredTool, ...]: ...

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult: ...

    async def resolve_approval(
        self,
        approval_id: str,
        approve: bool,
        context: ToolExecutionContext,
    ) -> ToolResult: ...

    def status(self) -> tuple[dict[str, str], ...]: ...

    async def close(self) -> None: ...
