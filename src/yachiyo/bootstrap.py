"""The only composition root for the yachiyo process."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from yachiyo.adapters.mcp import TrustedHTTPToolRegistry
from yachiyo.adapters.model import OpenAICompatibleModel
from yachiyo.adapters.rag import SQLiteKnowledgeIndex
from yachiyo.adapters.storage import Database
from yachiyo.adapters.tools import CompositeToolRegistry, MediaToolRegistry
from yachiyo.adapters.transports import DiscordTransport, TelegramTransport
from yachiyo.application import (
    COMMANDS,
    AgentRunner,
    ChatApplication,
    CommandBus,
    EmptyToolRegistry,
    MemoryService,
    ProfileService,
    RateLimiter,
    SessionLockPool,
)
from yachiyo.config import LoadedConfig
from yachiyo.domain import RoleDefinition
from yachiyo.integrations import MediaService, ReverseImageService
from yachiyo.ports.mcp import ToolRegistry
from yachiyo.runtime import PersistentJobWorker
from yachiyo.runtime.health import Heartbeat
from yachiyo.runtime.supervisor import TransportSupervisor

logger = logging.getLogger(__name__)


class Runtime:
    def __init__(self, loaded: LoadedConfig):
        self.loaded = loaded
        self.settings = loaded.settings
        self.database = Database(self.settings, loaded.paths)
        self.model = OpenAICompatibleModel(self.settings, loaded.secrets)
        self.knowledge: SQLiteKnowledgeIndex | None = None
        self.memory: MemoryService | None = None
        self.tools: ToolRegistry = EmptyToolRegistry()
        self.reverse_image: ReverseImageService | None = None
        self.supervisors: list[TransportSupervisor] = []
        self.worker: PersistentJobWorker | None = None
        self.heartbeat: Heartbeat | None = None
        self._stop = asyncio.Event()
        self._built = False

    async def build(self) -> None:
        if self._built:
            return
        try:
            await self._build()
        except BaseException:
            await self.close()
            raise

    async def _build(self) -> None:
        self.model.validate_credentials()
        await self.database.open()
        await self.database.cleanup_expired_data(self.settings.storage.message_retention_days)
        await self.database.cleanup_orphaned_outbox()
        await self.database.seed_roles(
            tuple(
                RoleDefinition(
                    role.name,
                    role.description,
                    role.system_prompt,
                    role.greeting,
                    role.farewell,
                )
                for role in self.settings.roles
            )
        )
        self.knowledge = SQLiteKnowledgeIndex(self.database, self.settings)
        self.memory = MemoryService(
            self.database,
            self.knowledge,
            self.model,
            self.settings,
        )
        self.reverse_image = (
            ReverseImageService(self.settings, self.loaded.secrets)
            if self.settings.reverse_image.enabled
            else None
        )
        self.tools = await self._build_tools()
        agent = AgentRunner(self.model, self.tools, self.settings)
        profiles = ProfileService(self.database, self.settings)
        commands = CommandBus(
            self.database,
            self.settings,
            profiles,
            self.memory,
            self.knowledge,
            self.tools,
            self.reverse_image,
        )
        application = ChatApplication(
            self.database,
            self.settings,
            agent,
            self.knowledge,
            commands,
            profiles,
            SessionLockPool(),
            RateLimiter(self.settings.policies.rate_limit),
            self.reverse_image,
        )
        self.supervisors = self._build_supervisors(application)
        self.worker = PersistentJobWorker(
            self.database,
            (
                {"memory.extract": self.memory.extract_exchange}
                if self.settings.memory.enabled
                else {}
            ),
            handler_timeout_seconds=min(90.0, self.settings.policies.request_timeout_seconds),
        )
        self.heartbeat = Heartbeat(
            self.loaded.paths.heartbeat_file,
            self.health,
        )
        self._built = True

    async def _build_tools(self) -> ToolRegistry:
        registries: list[ToolRegistry] = []
        approval_registry: ToolRegistry | None = None
        try:
            if self.settings.mcp.enabled:
                mcp = TrustedHTTPToolRegistry(self.settings, self.loaded.secrets, self.database)
                registries.append(mcp)
                approval_registry = mcp
            if self.settings.media.enabled:
                media = MediaService(self.settings, self.loaded.secrets)
                registries.append(MediaToolRegistry(media))
            if not registries:
                return EmptyToolRegistry()
            if len(registries) == 1:
                result = registries[0]
            else:
                result = CompositeToolRegistry(
                    tuple(registries), approval_registry=approval_registry
                )
            await result.refresh()
            return result
        except BaseException:
            for registry in reversed(registries):
                try:
                    await registry.close()
                except Exception:
                    logger.exception("failed to close a partially initialized tool registry")
            raise

    def _build_supervisors(self, application: ChatApplication) -> list[TransportSupervisor]:
        result: list[TransportSupervisor] = []
        telegram = self.settings.transports.telegram
        if telegram.enabled:
            result.append(
                TransportSupervisor(
                    lambda: TelegramTransport(
                        self.loaded.secrets.read(telegram.token_secret, required=True),
                        self.settings,
                    ),
                    COMMANDS,
                    application,
                    name="telegram",
                )
            )
        discord = self.settings.transports.discord
        if discord.enabled:
            result.append(
                TransportSupervisor(
                    lambda: DiscordTransport(
                        self.loaded.secrets.read(discord.token_secret, required=True),
                        self.settings,
                    ),
                    COMMANDS,
                    application,
                    name="discord",
                )
            )
        return result

    async def run(self) -> None:
        try:
            await self.build()
            assert self.worker is not None
            assert self.heartbeat is not None
            async with asyncio.TaskGroup() as group:
                group.create_task(self.worker.run(), name="jobs")
                group.create_task(self.heartbeat.run(), name="heartbeat")
                group.create_task(self._maintenance(), name="maintenance")
                group.create_task(self._reconnect_tools(), name="tool-reconnect")
                for supervisor in self.supervisors:
                    group.create_task(supervisor.run(), name=f"transport:{supervisor.name}")
                await self._stop.wait()
                self.worker.stop()
                self.heartbeat.stop()
                await asyncio.gather(
                    *(supervisor.stop() for supervisor in self.supervisors),
                    return_exceptions=True,
                )
        finally:
            await self.close()

    async def _maintenance(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=6 * 60 * 60)
                return
            except TimeoutError:
                try:
                    await self.database.cleanup_expired_data(
                        self.settings.storage.message_retention_days
                    )
                    await self.database.cleanup_orphaned_outbox()
                except Exception:
                    logger.exception("periodic data cleanup failed")

    async def _reconnect_tools(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=30)
                return
            except TimeoutError:
                statuses = self.tools.status()
                if not any(status.get("state") == "degraded" for status in statuses):
                    continue
                try:
                    await self.tools.refresh()
                except Exception:
                    logger.exception("tool discovery retry failed")

    def stop(self) -> None:
        self._stop.set()

    async def close(self) -> None:
        if self.reverse_image is not None:
            try:
                await self.reverse_image.close()
            except Exception:
                logger.exception("failed to close reverse image service")
            self.reverse_image = None
        try:
            await self.tools.close()
        except Exception:
            logger.exception("failed to close tool registry")
        self.tools = EmptyToolRegistry()
        try:
            await self.model.close()
        except Exception:
            logger.exception("failed to close model clients")
        try:
            await self.database.close()
        except Exception:
            logger.exception("failed to close database")
        self._built = False

    def health(self) -> dict[str, Any]:
        transport_health = [supervisor.health() for supervisor in self.supervisors]
        worker_health = (
            self.worker.health()
            if self.worker is not None
            else {
                "state": "unavailable",
                "error": "worker is not initialized",
            }
        )
        running = sum(item.get("state") == "running" for item in transport_health)
        if running == len(transport_health) and worker_health["state"] == "running":
            state = "running"
        elif running:
            state = "degraded"
        else:
            state = "unavailable"
        return {
            "state": state,
            "database": str(self.database.path),
            "transports": transport_health,
            "jobs": worker_health,
            "tools": self.tools.status(),
        }
