"""Model provider contract and neutral request types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ModelImage:
    mime_type: str
    data: bytes


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str = ""
    images: tuple[ModelImage, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    model: str
    messages: tuple[ModelMessage, ...]
    instructions: str
    tools: tuple[ToolDefinition, ...] = ()
    temperature: float = 0.7
    max_output_tokens: int = 2000
    previous_response_id: str | None = None
    continuation_endpoint: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ModelResult:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: ModelUsage = ModelUsage()
    provider_request_id: str | None = None
    model: str | None = None
    continuation_id: str | None = None
    continuation_endpoint: str | None = None


class ModelClient(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResult: ...

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...

    async def close(self) -> None: ...
