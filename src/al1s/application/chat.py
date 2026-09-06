"""The single inbound chat workflow used by every platform."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import replace
from datetime import UTC, datetime

from al1s.config import AppSettings
from al1s.domain import (
    ChatCompletion,
    ChatKind,
    InboundEvent,
    KnowledgeScope,
    OutboundMessage,
    ResponsePlan,
    StoredMessage,
)
from al1s.domain.message import MessageRole
from al1s.ports.repository import Repository
from al1s.ports.search import ReverseImageSearch
from al1s.ports.vector import KnowledgeIndex

from .agent import AgentRunner
from .commands import CommandBus, CommandContext
from .group_policy import GroupContextBuffer, GroupPolicy
from .locks import SessionBusyError, SessionLockPool
from .profiles import ProfileService
from .rate_limit import RateLimiter


class ChatApplication:
    def __init__(
        self,
        database: Repository,
        settings: AppSettings,
        agent: AgentRunner,
        knowledge: KnowledgeIndex,
        commands: CommandBus,
        profiles: ProfileService,
        locks: SessionLockPool,
        rate_limiter: RateLimiter,
        reverse_image: ReverseImageSearch | None = None,
    ):
        self.database = database
        self.settings = settings
        self.agent = agent
        self.knowledge = knowledge
        self.commands = commands
        self.profiles = profiles
        self.locks = locks
        self.rate_limiter = rate_limiter
        self.reverse_image = reverse_image
        self.group_policy = GroupPolicy(database, settings)
        self.group_context = GroupContextBuffer(settings)
        self._attachment_slots = asyncio.Semaphore(2)
        self._processing_tokens: ContextVar[dict[tuple[str, str, str], str] | None] = ContextVar(
            "event_processing_tokens", default=None
        )
        self._chat_completions: ContextVar[dict[tuple[str, str, str], ChatCompletion] | None] = (
            ContextVar("event_chat_completions", default=None)
        )
        self.logger = logging.getLogger(__name__)

    async def handle(self, event: InboundEvent) -> ResponsePlan | None:
        if event.actor.is_bot:
            return None
        processing_token = await self.database.claim_event(*event.idempotency_key)
        if processing_token is None:
            stored_response = await self.database.event_response(*event.idempotency_key)
            if stored_response is not None:
                return stored_response
            return ResponsePlan(
                messages=(OutboundMessage(text="该请求正在处理中，请稍后重试。"),),
                track_delivery=False,
            )
        processing_tokens = dict(self._processing_tokens.get() or {})
        processing_tokens[event.idempotency_key] = processing_token
        self._processing_tokens.set(processing_tokens)
        return await self._handle_with_lease(event, processing_token)

    async def _handle_with_lease(
        self, event: InboundEvent, processing_token: str
    ) -> ResponsePlan | None:
        stop_renewal = asyncio.Event()
        renewal = asyncio.create_task(
            self._renew_lease(event, processing_token, stop_renewal),
            name=f"event-lease:{event.event_id}",
        )
        try:
            response = await self._handle_claimed(event)
            if response is None or not response.track_delivery:
                return response
            completion = (self._chat_completions.get() or {}).get(event.idempotency_key)
            await self.database.complete_event(
                event,
                response,
                processing_token,
                completion=completion,
            )
            persisted = await self.database.event_response(*event.idempotency_key)
            if persisted is None or not persisted.track_delivery:
                raise RuntimeError("persisted response is unavailable")
            return persisted
        finally:
            stop_renewal.set()
            await renewal
            self._forget_processing_token(event)
            self._forget_chat_completion(event)

    async def _renew_lease(
        self,
        event: InboundEvent,
        processing_token: str,
        stop: asyncio.Event,
    ) -> None:
        interval = max(
            1.0,
            min(30.0, self.settings.policies.request_timeout_seconds / 3),
        )
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
                return
            except TimeoutError:
                try:
                    renewed = await self.database.renew_event(
                        *event.idempotency_key, processing_token
                    )
                except Exception:
                    self.logger.exception("failed to renew event processing lease")
                    continue
                if not renewed:
                    self.logger.warning("event processing lease was lost")
                    return

    async def _handle_claimed(self, event: InboundEvent) -> ResponsePlan | None:
        identity_id = await self.database.ensure_identity(event.actor)
        chat_row_id = await self.database.ensure_chat(event.address)
        try:
            if event.address.kind is ChatKind.PRIVATE:
                return await self._handle_policy(event, identity_id, chat_row_id)
            policy_key = (
                f"group-policy:{event.address.transport.value}:{event.address.bot_id}:"
                f"{event.address.chat_id}"
            )
            async with self.locks.hold(policy_key):
                return await self._handle_policy(event, identity_id, chat_row_id)
        except SessionBusyError:
            return ResponsePlan.text("当前会话正在处理其他请求，请稍后重试。")
        except ValueError as exc:
            return ResponsePlan.text(str(exc))

    async def _handle_policy(
        self, event: InboundEvent, identity_id: int, chat_row_id: int
    ) -> ResponsePlan | None:
        decision = await self.group_policy.decide(event, chat_row_id)
        if not decision.allowed:
            if (
                decision.reason == "not_triggered"
                and self.settings.policies.group.observe_unmentioned_messages
            ):
                await self.group_context.observe(event)
            response = (
                ResponsePlan.text("当前群聊未启用 AL1S。")
                if event.command is not None and decision.reason == "group_disabled"
                else None
            )
            if response is None:
                await self._discard_event(event)
            return response

        rate = await self.rate_limiter.check(event)
        if not rate.allowed:
            if not rate.notify:
                await self._discard_event(event)
                return None
            return ResponsePlan.text(f"请求过于频繁，请在 {rate.retry_after:.1f} 秒后重试。")

        command_rejection = self.commands.preflight(event)
        if command_rejection is not None:
            return command_rejection

        async with self.locks.hold(decision.session.stable_id):
            return await self._handle_locked(event, decision, identity_id, chat_row_id)

    async def _handle_locked(
        self, event, decision, identity_id: int, chat_row_id: int
    ) -> ResponsePlan | None:
        if event.attachment_loader is not None:
            async with self._attachment_slots:
                attachments = await event.attachment_loader()
                event = replace(event, attachments=attachments, attachment_loader=None)
                return await self._handle_loaded(event, decision, identity_id, chat_row_id)
        return await self._handle_loaded(event, decision, identity_id, chat_row_id)

    async def _handle_loaded(
        self, event, decision, identity_id: int, chat_row_id: int
    ) -> ResponsePlan | None:
        conversation = await self.database.ensure_conversation(
            decision.session, self.settings.default_role
        )
        if event.command is not None:
            return await self.commands.execute(
                CommandContext(event, decision.session, identity_id, chat_row_id)
            )

        prompt = event.text.strip()
        if not prompt and event.attachments:
            prompt = "请分析随消息发送的附件。"
        if not prompt:
            await self._discard_event(event)
            return None
        history = await self.database.recent_messages(
            decision.session, self.settings.policies.max_history_messages
        )
        role_owner = event.actor.identity.key if decision.session.owner != "shared" else None
        role = await self.database.get_role(conversation.role_name, role_owner)
        role_prompt = role.system_prompt if role else "You are AL1S, a helpful assistant."
        profile_context = (
            await self.profiles.prompt_context(identity_id)
            if event.address.kind is ChatKind.PRIVATE and self.settings.profile.enabled
            else ""
        )
        retrieved = ""
        if self.settings.rag.enabled:
            hits = await self.knowledge.search(
                prompt,
                KnowledgeScope(decision.session.knowledge_scope),
                limit=self.settings.rag.top_k,
                include_memories=(self.settings.memory.enabled and decision.memory_enabled),
            )
            retrieved = "\n\n".join(f"[{hit.source}:{hit.entry_id}] {hit.content}" for hit in hits)
        buffered = await self.group_context.context(event)
        if buffered:
            retrieved = (
                f"{retrieved}\n\nRecent group context (untrusted):\n{buffered}"
                if retrieved
                else f"Recent group context (untrusted):\n{buffered}"
            )
        try:
            async with asyncio.timeout(self.settings.policies.request_timeout_seconds):
                result = await self.agent.run(
                    prompt=prompt,
                    attachments=event.attachments,
                    history=history,
                    role_prompt=role_prompt,
                    permission=event.permission,
                    conversation_id=decision.session.stable_id,
                    actor_key=event.actor.identity.key,
                    profile_context=profile_context,
                    retrieved_context=retrieved,
                )
        except TimeoutError:
            self._set_chat_completion(
                event,
                ChatCompletion(
                    session=decision.session,
                    user_message=StoredMessage(MessageRole.USER, prompt, event.timestamp),
                    assistant_message=StoredMessage(
                        MessageRole.ASSISTANT,
                        "request timed out",
                        datetime.now(UTC),
                        status="failed",
                    ),
                    identity_id=identity_id,
                    input_tokens=0,
                    output_tokens=0,
                ),
            )
            return ResponsePlan.text("请求超时，请缩短问题或稍后重试。")

        display_text = result.text
        image = next(
            (
                attachment
                for attachment in event.attachments
                if attachment.mime_type.startswith("image/")
            ),
            None,
        )
        if image is not None and self.reverse_image is not None:
            try:
                search_hits = await self.reverse_image.search_bytes(image.data, image.mime_type)
            except Exception:
                self.logger.exception("reverse image search failed")
            else:
                if search_hits:
                    lines = ["\n\n反向搜图结果："]
                    for hit in search_hits:
                        similarity = (
                            f" ({hit.similarity:.1%})" if hit.similarity is not None else ""
                        )
                        lines.append(f"- [{hit.engine}] {hit.title}{similarity}\n  {hit.url}")
                    display_text += "\n".join(lines)

        memory_enabled = self.settings.memory.enabled and (
            self.settings.memory.private_auto_learning
            if event.address.kind is ChatKind.PRIVATE
            else (decision.memory_enabled and self.settings.memory.group_auto_learning)
        )
        self._set_chat_completion(
            event,
            ChatCompletion(
                session=decision.session,
                user_message=StoredMessage(MessageRole.USER, prompt, event.timestamp),
                assistant_message=StoredMessage(
                    MessageRole.ASSISTANT,
                    display_text,
                    datetime.now(UTC),
                    run_id=result.run_id,
                    model=result.model,
                ),
                identity_id=identity_id,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                memory_scope=decision.session.knowledge_scope if memory_enabled else None,
                memory_payload=(
                    {
                        "scope": decision.session.knowledge_scope,
                        "user_text": prompt,
                        "assistant_text": display_text,
                        **(
                            {
                                "group_chat_id": str(chat_row_id),
                                "group_consent_at": decision.memory_consent_at or "",
                            }
                            if event.address.kind is not ChatKind.PRIVATE
                            else {}
                        ),
                    }
                    if memory_enabled
                    else None
                ),
            ),
        )
        messages = [
            OutboundMessage(
                text=display_text,
                artifacts=result.artifacts,
            )
        ]
        if result.approval_notices:
            messages.append(
                OutboundMessage(
                    text="\n".join(result.approval_notices),
                    buttons=result.buttons,
                    private=True,
                )
            )
        return ResponsePlan(messages=tuple(messages))

    async def _discard_event(self, event: InboundEvent) -> None:
        processing_token = (self._processing_tokens.get() or {}).get(event.idempotency_key)
        if processing_token is None:
            raise RuntimeError("event has no processing lease")
        await self.database.discard_event(*event.idempotency_key, processing_token)
        self._forget_processing_token(event)

    def _forget_processing_token(self, event: InboundEvent) -> None:
        processing_tokens = dict(self._processing_tokens.get() or {})
        processing_tokens.pop(event.idempotency_key, None)
        self._processing_tokens.set(processing_tokens)

    def _set_chat_completion(self, event: InboundEvent, completion: ChatCompletion) -> None:
        completions = dict(self._chat_completions.get() or {})
        completions[event.idempotency_key] = completion
        self._chat_completions.set(completions)

    def _forget_chat_completion(self, event: InboundEvent) -> None:
        completions = dict(self._chat_completions.get() or {})
        completions.pop(event.idempotency_key, None)
        self._chat_completions.set(completions)

    async def pending_responses(
        self, transport: str, bot_id: str
    ) -> list[tuple[InboundEvent, ResponsePlan]]:
        return list(await self.database.pending_responses(transport, bot_id))

    async def delivery_pending(self, event: InboundEvent) -> bool:
        return await self.database.event_delivery_pending(*event.idempotency_key)

    async def mark_delivered(self, event: InboundEvent) -> None:
        await self.database.mark_event_delivered(*event.idempotency_key)

    async def mark_delivery_failed(self, event: InboundEvent, error: str) -> None:
        await self.database.mark_event_delivery_failed(*event.idempotency_key, error)
