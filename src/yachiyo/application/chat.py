"""The single inbound chat workflow used by every platform."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import UTC, datetime

from yachiyo.config import AppSettings
from yachiyo.domain import (
    ChatCompletion,
    ChatKind,
    InboundEvent,
    KnowledgeScope,
    OutboundMessage,
    ResponsePlan,
    StoredMessage,
)
from yachiyo.domain.message import MessageRole
from yachiyo.ports.knowledge import KnowledgeIndex
from yachiyo.ports.repository import Repository
from yachiyo.ports.search import ReverseImageSearch

from .agent import AgentRunner
from .commands import CommandBus, CommandContext
from .group_policy import GroupContextBuffer, GroupDecision, GroupPolicy
from .locks import SessionBusyError, SessionLockPool
from .profiles import ProfileService
from .rate_limit import RateLimiter

type _TurnResult = tuple[ResponsePlan | None, ChatCompletion | None]


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

        stop_renewal = asyncio.Event()
        renewal = asyncio.create_task(
            self._renew_lease(event, processing_token, stop_renewal),
            name=f"event-lease:{event.event_id}",
        )
        try:
            response, completion = await self._handle_claimed(event)
            if response is None:
                await self.database.discard_event(*event.idempotency_key, processing_token)
                return None
            if not response.track_delivery:
                return response
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

    async def _handle_claimed(self, event: InboundEvent) -> _TurnResult:
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
            return ResponsePlan.text("当前会话正在处理其他请求，请稍后重试。"), None
        except ValueError as exc:
            return ResponsePlan.text(str(exc)), None

    async def _handle_policy(
        self, event: InboundEvent, identity_id: int, chat_row_id: int
    ) -> _TurnResult:
        decision = await self.group_policy.decide(event, chat_row_id)
        if not decision.allowed:
            if (
                decision.reason == "not_triggered"
                and self.settings.policies.group.observe_unmentioned_messages
            ):
                await self.group_context.observe(event)
            if event.command is not None and decision.reason == "group_disabled":
                return ResponsePlan.text("当前群聊未启用 yachiyo。"), None
            return None, None

        rate = await self.rate_limiter.check(event)
        if not rate.allowed:
            if not rate.notify:
                return None, None
            return ResponsePlan.text(f"请求过于频繁，请在 {rate.retry_after:.1f} 秒后重试。"), None

        command_rejection = self.commands.preflight(event)
        if command_rejection is not None:
            return command_rejection, None

        async with self.locks.hold(decision.session.stable_id):
            return await self._handle_locked(event, decision, identity_id)

    async def _handle_locked(
        self, event: InboundEvent, decision: GroupDecision, identity_id: int
    ) -> _TurnResult:
        if event.attachment_loader is not None:
            async with self._attachment_slots:
                attachments = await event.attachment_loader()
                event = replace(event, attachments=attachments, attachment_loader=None)
                return await self._handle_loaded(event, decision, identity_id)
        return await self._handle_loaded(event, decision, identity_id)

    async def _handle_loaded(
        self, event: InboundEvent, decision: GroupDecision, identity_id: int
    ) -> _TurnResult:
        conversation = await self.database.ensure_conversation(
            decision.session, self.settings.default_role
        )
        if event.command is not None:
            context = CommandContext(event, decision.session, identity_id, decision.chat_row_id)
            return await self.commands.execute(context), None

        prompt = event.text.strip()
        if not prompt and event.attachments:
            prompt = "请分析随消息发送的附件。"
        if not prompt:
            return None, None
        history = await self.database.recent_messages(
            decision.session, self.settings.policies.max_history_messages
        )
        role_owner = event.actor.identity.key if decision.session.owner != "shared" else None
        role = await self.database.get_role(conversation.role_name, role_owner)
        role_prompt = role.system_prompt if role else "You are yachiyo, a helpful assistant."
        profile_context = (
            await self.profiles.prompt_context(identity_id)
            if event.address.kind is ChatKind.PRIVATE and self.settings.profile.enabled
            else ""
        )
        references = []
        if self.settings.rag.enabled:
            hits = await self.knowledge.search(
                prompt,
                KnowledgeScope(decision.session.knowledge_scope),
                limit=self.settings.rag.top_k,
                include_memories=(self.settings.memory.enabled and decision.memory_enabled),
            )
            references.extend(f"[{hit.source}:{hit.entry_id}] {hit.content}" for hit in hits)
        buffered = await self.group_context.context(event)
        if buffered:
            references.append(f"Recent group context (untrusted):\n{buffered}")
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
                    retrieved_context="\n\n".join(references),
                )
        except TimeoutError:
            completion = ChatCompletion(
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
            )
            return ResponsePlan.text("请求超时，请缩短问题或稍后重试。"), completion

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
        memory_payload = None
        if memory_enabled:
            memory_payload = {
                "scope": decision.session.knowledge_scope,
                "user_text": prompt,
                "assistant_text": display_text,
            }
            if event.address.kind is not ChatKind.PRIVATE:
                memory_payload["group_chat_id"] = str(decision.chat_row_id)
                memory_payload["group_consent_at"] = decision.memory_consent_at or ""
        completion = ChatCompletion(
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
            memory_payload=memory_payload,
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
        return ResponsePlan(messages=tuple(messages)), completion

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
