"""discord.py transport sharing the exact same application workflow as Telegram."""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Sequence
from pathlib import Path

import discord
from discord import app_commands

from al1s.config import AppSettings
from al1s.domain import (
    Actor,
    Artifact,
    Attachment,
    ChatKind,
    CommandInvocation,
    ConversationAddress,
    ExternalIdentity,
    InboundEvent,
    RequestPermission,
    ResponsePlan,
    TransportName,
)
from al1s.ports.transport import CommandDescriptor, EventReceiver

from .renderers import split_text, validated_mime

logger = logging.getLogger(__name__)


class DiscordTransport:
    name = TransportName.DISCORD

    def __init__(self, token: str, settings: AppSettings):
        self.token = token
        self.settings = settings
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        self.client = discord.Client(
            intents=intents,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.tree = app_commands.CommandTree(self.client)
        self.receiver: EventReceiver | None = None
        self.bot_id = "pending"
        self._running = False
        self._synced = False
        self._commands_registered = False
        self._pending: dict[str, tuple[discord.Interaction, bool]] = {}
        self._download_slots = asyncio.Semaphore(4)
        self._inflight: set[asyncio.Task[object]] = set()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self.client.event(self.on_ready)
        self.client.event(self.on_message)

    async def prepare(self, commands: Sequence[CommandDescriptor]) -> None:
        if self._commands_registered:
            return
        for descriptor in commands:
            callback = self._make_command_callback(
                descriptor.name,
                bool(
                    getattr(descriptor, "private_only", False)
                    or getattr(descriptor, "private_response", False)
                ),
                descriptor.name in {"profile_set", "profile_add"},
            )
            self.tree.add_command(
                app_commands.Command(
                    name=descriptor.name,
                    description=descriptor.description[:100],
                    callback=callback,
                )
            )
        self._commands_registered = True

    def _make_command_callback(self, name: str, private_response: bool, accepts_attachment: bool):
        if accepts_attachment:

            async def callback(
                interaction: discord.Interaction,
                arguments: str = "",
                attachment: discord.Attachment | None = None,
            ) -> None:
                await self._on_command(interaction, name, arguments, private_response, attachment)

        else:

            async def callback(
                interaction: discord.Interaction,
                arguments: str = "",
            ) -> None:
                await self._on_command(interaction, name, arguments, private_response)

        callback.__name__ = f"command_{name}"
        return callback

    async def run(self, receiver: EventReceiver) -> None:
        self.receiver = receiver
        self._running = True
        try:
            await self.client.login(self.token)
            if self.client.user is not None:
                self.bot_id = str(self.client.user.id)
            if self.settings.transports.discord.sync_commands:
                try:
                    await self._sync_commands()
                except Exception:
                    logger.exception("Discord command synchronization failed")
            await self.client.connect(reconnect=True)
        finally:
            self._running = False

    async def on_ready(self) -> None:
        if self.client.user is None:
            return
        self.bot_id = str(self.client.user.id)

    async def _sync_commands(self) -> None:
        guild_ids = self.settings.transports.discord.allowed_guild_ids
        if guild_ids:
            for guild_id in guild_ids:
                guild = discord.Object(id=int(guild_id))
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
        allow_direct = self.settings.transports.discord.allow_direct_messages
        if guild_ids and not allow_direct:
            self.tree.clear_commands(guild=None)
        else:
            contexts = app_commands.AppCommandContext(
                guild=not bool(guild_ids),
                dm_channel=allow_direct,
                private_channel=allow_direct,
            )
            for command in self.tree.get_commands(guild=None):
                command.allowed_contexts = contexts
        await self.tree.sync()
        self._synced = True

    async def on_message(self, message: discord.Message) -> None:
        if self.receiver is None or message.author.bot:
            return
        if not self._guild_allowed(message.guild):
            return
        event = await self._message_event(message)
        await self._call_receiver(event)

    async def _on_command(
        self,
        interaction: discord.Interaction,
        name: str,
        arguments: str,
        private_response: bool,
        attachment: discord.Attachment | None = None,
    ) -> None:
        if self.receiver is None or interaction.user is None or interaction.channel_id is None:
            return
        if not self._guild_allowed(interaction.guild):
            await interaction.response.send_message(
                "This server is not allowed to use AL1S.", ephemeral=True
            )
            return
        await interaction.response.defer(thinking=True, ephemeral=private_response)
        event_id = f"interaction:{interaction.id}"
        self._pending[event_id] = (interaction, private_response)
        event = await self._interaction_event(interaction, event_id, name, arguments, attachment)
        try:
            await self._call_receiver(event)
        finally:
            pending = self._pending.pop(event_id, None)
            if pending is not None:
                try:
                    await interaction.edit_original_response(
                        content="The request produced no response."
                    )
                except discord.HTTPException:
                    pass

    async def _interaction_event(
        self,
        interaction: discord.Interaction,
        event_id: str,
        name: str,
        arguments: str,
        attachment: discord.Attachment | None = None,
    ) -> InboundEvent:
        user = interaction.user
        guild = interaction.guild
        private = guild is None
        chat_id, thread_id = self._channel_coordinates(interaction.channel, interaction.channel_id)
        address = ConversationAddress(
            TransportName.DISCORD,
            self.bot_id,
            chat_id,
            thread_id if not private else "0",
            ChatKind.PRIVATE if private else ChatKind.GROUP,
        )
        attachment_limit = (
            self.settings.profile.max_document_bytes
            if name in {"profile_set", "profile_add"}
            else self.settings.policies.max_attachment_bytes
        )

        async def load_attachments() -> tuple[Attachment, ...]:
            return await self._attachments(
                [attachment] if attachment else [], limit=attachment_limit
            )

        return InboundEvent(
            event_id=event_id,
            address=address,
            actor=self._actor(user),
            permission=self._permission(user, private),
            text=f"/{name} {arguments}".strip(),
            attachment_loader=load_attachments if attachment else None,
            command=CommandInvocation(name, arguments),
        )

    async def _message_event(self, message: discord.Message) -> InboundEvent:
        private = message.guild is None
        chat_id, thread_id = self._channel_coordinates(message.channel, message.channel.id)
        address = ConversationAddress(
            TransportName.DISCORD,
            self.bot_id,
            chat_id,
            thread_id if not private else "0",
            ChatKind.PRIVATE if private else ChatKind.GROUP,
        )
        replies = False
        if message.reference and message.reference.message_id:
            resolved = message.reference.resolved
            replies = (
                isinstance(resolved, discord.Message) and str(resolved.author.id) == self.bot_id
            )

        async def load_attachments() -> tuple[Attachment, ...]:
            return await self._attachments(message.attachments)

        text = message.content.strip()
        if self.client.user is not None:
            text = text.replace(f"<@{self.client.user.id}>", "")
            text = text.replace(f"<@!{self.client.user.id}>", "").strip()
        return InboundEvent(
            event_id=f"message:{message.id}",
            address=address,
            actor=self._actor(message.author),
            permission=self._permission(message.author, private),
            text=text,
            attachment_loader=load_attachments if message.attachments else None,
            mentions_bot=bool(self.client.user and self.client.user in message.mentions),
            replies_to_bot=replies,
            timestamp=message.created_at,
        )

    @staticmethod
    def _channel_coordinates(
        channel: discord.abc.Messageable | None, fallback_id: int | None
    ) -> tuple[str, str]:
        if isinstance(channel, discord.Thread):
            return str(channel.parent_id or channel.id), str(channel.id)
        if fallback_id is None:
            raise ValueError("Discord event has no channel")
        return str(fallback_id), "0"

    def _guild_allowed(self, guild: discord.Guild | None) -> bool:
        allowed = self.settings.transports.discord.allowed_guild_ids
        if guild is None:
            return self.settings.transports.discord.allow_direct_messages
        return not allowed or str(guild.id) in allowed

    async def _call_receiver(self, event: InboundEvent) -> None:
        receiver = self.receiver
        if receiver is None:
            return
        task = asyncio.current_task()
        if task is not None:
            self._inflight.add(task)
        try:
            await receiver(event)
        finally:
            if task is not None:
                self._inflight.discard(task)

    @staticmethod
    def _actor(user: discord.abc.User) -> Actor:
        display_name = getattr(user, "display_name", None) or user.name
        return Actor(
            ExternalIdentity(TransportName.DISCORD, str(user.id)),
            display_name,
            user.name,
            user.bot,
        )

    def _permission(self, user: discord.abc.User, private: bool) -> RequestPermission:
        configured = str(user.id) in self.settings.transports.discord.admin_user_ids
        native_admin = isinstance(user, discord.Member) and user.guild_permissions.administrator
        return RequestPermission(private, configured or native_admin)

    async def _attachments(
        self, attachments: list[discord.Attachment], *, limit: int | None = None
    ) -> tuple[Attachment, ...]:
        result: list[Attachment] = []
        limit = limit or self.settings.policies.max_attachment_bytes
        if len(attachments) > 10 or sum(item.size for item in attachments) > limit:
            raise ValueError("附件总量超过允许的限制。")
        async with self._download_slots:
            for item in attachments:
                if item.size > limit:
                    raise ValueError("附件超过允许的大小。")
                data = await item.read(use_cached=False)
                if len(data) > limit:
                    raise ValueError("附件超过允许的大小。")
                mime_type = item.content_type or "application/octet-stream"
                try:
                    mime_type = validated_mime(data, mime_type)
                except ValueError as exc:
                    raise ValueError("图片格式无效或不受支持。") from exc
                result.append(
                    Attachment(
                        str(item.id),
                        item.filename,
                        mime_type,
                        len(data),
                        data,
                    )
                )
        return tuple(result)

    async def send(self, event: InboundEvent, response: ResponsePlan) -> None:
        pending = self._pending.get(event.event_id)
        if pending is not None and not response.messages:
            await pending[0].edit_original_response(content="This request was already completed.")
            self._pending.pop(event.event_id, None)
            return
        file_limit = pending[0].filesize_limit if pending is not None else 10 * 1024 * 1024
        used_original = False
        for message in response.messages:
            view = (
                None
                if message.private
                and event.address.kind is not ChatKind.PRIVATE
                and pending is None
                else self._view(message.buttons)
            )
            chunks = split_text(message.text, 2000) or ("",)
            if any(self._artifact_size(artifact) > file_limit for artifact in message.artifacts):
                raise ValueError("generated artifact exceeds the Discord upload limit")
            files = [self._file(artifact) for artifact in message.artifacts]
            if pending is not None:
                interaction, deferred_private = pending
                for index, chunk in enumerate(chunks):
                    attachments = files if index == len(chunks) - 1 else []
                    final_view = view if index == len(chunks) - 1 else None
                    if not used_original:
                        await interaction.edit_original_response(
                            content=chunk or None,
                            attachments=attachments,
                            view=final_view,
                            allowed_mentions=discord.AllowedMentions.none(),
                        )
                        used_original = True
                    else:
                        await interaction.followup.send(
                            content=chunk or None,
                            files=attachments,
                            view=final_view,
                            ephemeral=deferred_private or message.private,
                            allowed_mentions=discord.AllowedMentions.none(),
                        )
                continue
            destination = await self._destination(event, message.private)
            for index, chunk in enumerate(chunks):
                await destination.send(
                    content=chunk or None,
                    files=files if index == len(chunks) - 1 else [],
                    view=view if index == len(chunks) - 1 else None,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        if pending is not None:
            self._pending.pop(event.event_id, None)

    async def _destination(self, event: InboundEvent, private: bool):
        if private and event.address.kind is not ChatKind.PRIVATE:
            return await self.client.fetch_user(int(event.actor.identity.external_id))
        destination_id = (
            event.address.thread_id
            if event.address.kind is not ChatKind.PRIVATE and event.address.thread_id != "0"
            else event.address.chat_id
        )
        channel = self.client.get_channel(int(destination_id))
        return channel or await self.client.fetch_channel(int(destination_id))

    def _view(self, buttons) -> discord.ui.View | None:
        if not buttons:
            return None
        view = discord.ui.View(timeout=600)
        styles = {
            "primary": discord.ButtonStyle.primary,
            "danger": discord.ButtonStyle.danger,
            "secondary": discord.ButtonStyle.secondary,
        }
        for action in buttons:
            button = discord.ui.Button(
                label=action.label,
                custom_id=action.action_id,
                style=styles.get(action.style, discord.ButtonStyle.secondary),
            )

            async def callback(
                interaction: discord.Interaction,
                *,
                action_id: str = action.action_id,
            ) -> None:
                if not self._guild_allowed(interaction.guild):
                    await interaction.response.send_message(
                        "This server is not allowed to use AL1S.", ephemeral=True
                    )
                    return
                await interaction.response.defer(thinking=True, ephemeral=True)
                event_id = f"interaction:{interaction.id}"
                self._pending[event_id] = (interaction, True)
                action_value = action_id.removeprefix("approval:")
                event = await self._interaction_event(
                    interaction,
                    event_id,
                    "_action",
                    action_value,
                )
                try:
                    await self._call_receiver(event)
                finally:
                    pending = self._pending.pop(event_id, None)
                    if pending is not None:
                        try:
                            await interaction.edit_original_response(
                                content="The request produced no response."
                            )
                        except discord.HTTPException:
                            pass

            button.callback = callback
            view.add_item(button)
        return view

    @staticmethod
    def _file(artifact: Artifact) -> discord.File:
        source = (
            io.BytesIO(artifact.data)
            if artifact.data is not None
            else Path(artifact.path).open("rb")
        )
        return discord.File(source, filename=artifact.filename)

    @staticmethod
    def _artifact_size(artifact: Artifact) -> int:
        if artifact.data is not None:
            return len(artifact.data)
        return Path(artifact.path).stat().st_size

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self.receiver = None
            current = asyncio.current_task()
            pending = [task for task in self._inflight if task is not current]
            if pending:
                _, waiting = await asyncio.wait(pending, timeout=30)
                for task in waiting:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            if not self.client.is_closed():
                await self.client.close()
            self._pending.clear()
            self._closed = True

    def health(self) -> dict[str, str]:
        return {
            "transport": self.name.value,
            "state": "running" if self._running and self.client.is_ready() else "stopped",
            "bot_id": self.bot_id,
        }
