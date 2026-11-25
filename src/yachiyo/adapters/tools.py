"""Composition of remote MCP tools and lightweight direct API tools."""

from __future__ import annotations

from typing import Any

from yachiyo.domain import AccessLevel, RequestPermission
from yachiyo.integrations.media import MediaError, MediaService
from yachiyo.ports.mcp import RegisteredTool, ToolExecutionContext, ToolRegistry, ToolResult
from yachiyo.ports.model import ToolDefinition


class MediaToolRegistry:
    def __init__(self, media: MediaService):
        self.media = media
        self._tools = {
            "media_generate_image": RegisteredTool(
                ToolDefinition(
                    "media_generate_image",
                    "Generate one PNG image from a text prompt.",
                    {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "maxLength": 4000},
                            "size": {
                                "type": "string",
                                "enum": ["1024x1024", "1536x1024", "1024x1536"],
                            },
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                ),
                "builtin",
                "media_generate_image",
                AccessLevel.PRIVATE_ADMIN,
                requires_approval=False,
            ),
            "media_synthesize_speech": RegisteredTool(
                ToolDefinition(
                    "media_synthesize_speech",
                    "Synthesize speech as Opus or MP3 audio.",
                    {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "maxLength": 10_000},
                            "voice": {"type": "string"},
                            "format": {"type": "string", "enum": ["opus", "mp3"]},
                            "speed": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                        },
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                ),
                "builtin",
                "media_synthesize_speech",
                AccessLevel.PRIVATE_ADMIN,
                requires_approval=False,
            ),
        }

    async def refresh(self) -> None:
        return None

    def available(self, permission: RequestPermission) -> tuple[RegisteredTool, ...]:
        return tuple(tool for tool in self._tools.values() if permission.allows(tool.access))

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None or not context.permission.allows(tool.access):
            return ToolResult(False, "media tool is unavailable")
        try:
            if name == "media_generate_image":
                artifact = await self.media.generate_image(
                    str(arguments.get("prompt", "")),
                    size=str(arguments.get("size", "1024x1024")),
                )
            else:
                artifact = await self.media.synthesize_speech(
                    str(arguments.get("text", "")),
                    voice=str(arguments["voice"]) if arguments.get("voice") else None,
                    response_format=str(arguments.get("format", "opus")),
                    speed=float(arguments.get("speed", 1.0)),
                )
        except (MediaError, TypeError, ValueError) as exc:
            return ToolResult(False, str(exc))
        return ToolResult(True, f"Generated {artifact.filename}", artifacts=(artifact,))

    async def resolve_approval(
        self,
        approval_id: str,
        approve: bool,
        context: ToolExecutionContext,
    ) -> ToolResult:
        return ToolResult(False, "approval does not belong to a media tool")

    def status(self) -> tuple[dict[str, str], ...]:
        return ({"name": "media", "state": "connected", "error": ""},)

    async def close(self) -> None:
        await self.media.close()


class CompositeToolRegistry:
    def __init__(
        self,
        registries: tuple[ToolRegistry, ...],
        *,
        approval_registry: ToolRegistry | None = None,
    ):
        self.registries = registries
        self.approval_registry = approval_registry

    async def refresh(self) -> None:
        for registry in self.registries:
            await registry.refresh()

    def available(self, permission: RequestPermission) -> tuple[RegisteredTool, ...]:
        result: dict[str, RegisteredTool] = {}
        for registry in self.registries:
            for tool in registry.available(permission):
                result.setdefault(tool.definition.name, tool)
        return tuple(result.values())

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        for registry in self.registries:
            if any(tool.definition.name == name for tool in registry.available(context.permission)):
                return await registry.call(name, arguments, context)
        return ToolResult(False, "tool is unavailable for this request")

    async def resolve_approval(
        self,
        approval_id: str,
        approve: bool,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if self.approval_registry is None:
            return ToolResult(False, "tool approvals are unavailable")
        return await self.approval_registry.resolve_approval(approval_id, approve, context)

    def status(self) -> tuple[dict[str, str], ...]:
        return tuple(status for registry in self.registries for status in registry.status())

    async def close(self) -> None:
        for registry in reversed(self.registries):
            await registry.close()
