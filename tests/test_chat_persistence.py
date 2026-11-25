import asyncio
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock

from yachiyo.adapters.storage import Database
from yachiyo.application.agent import AgentRunner, EmptyToolRegistry
from yachiyo.application.chat import ChatApplication
from yachiyo.application.commands import CommandBus
from yachiyo.application.locks import SessionLockPool
from yachiyo.application.profiles import ProfileService
from yachiyo.application.rate_limit import RateLimiter
from yachiyo.config import AppSettings, default_paths
from yachiyo.domain import (
    Actor,
    ConversationAddress,
    ExternalIdentity,
    InboundEvent,
    RequestPermission,
    SessionKey,
    SessionScope,
    TransportName,
)
from yachiyo.ports.model import ModelRequest, ModelResult


class ChatPersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_chats_and_redelivery(self):
        directory = self.enterContext(TemporaryDirectory())
        settings = AppSettings()
        database = Database(settings, replace(default_paths(), data_dir=Path(directory)))
        await database.open()
        self.addAsyncCleanup(database.close)

        # Both requests must reach the model before either response can be saved.
        barrier = asyncio.Barrier(2)

        async def reply(request: ModelRequest) -> ModelResult:
            await asyncio.wait_for(barrier.wait(), timeout=5)
            return ModelResult(text=f"Reply to {request.messages[-1].content}")

        model = AsyncMock()
        model.complete.side_effect = reply
        tools = EmptyToolRegistry()
        knowledge = AsyncMock()
        knowledge.search.return_value = ()
        profiles = ProfileService(database, settings)
        app = ChatApplication(
            database,
            settings,
            AgentRunner(model, tools, settings),
            knowledge,
            CommandBus(database, settings, profiles, Mock(), knowledge, tools, None),
            profiles,
            SessionLockPool(),
            RateLimiter(settings.policies.rate_limit),
        )
        events = [
            InboundEvent(
                event_id=user,
                address=ConversationAddress(TransportName.TELEGRAM, "test-bot", user),
                actor=Actor(ExternalIdentity(TransportName.TELEGRAM, user), user),
                permission=RequestPermission(is_private=True, is_admin=False),
                text=f"Message from {user}",
            )
            for user in ("alice", "bob")
        ]
        responses = await asyncio.gather(*(app.handle(event) for event in events))
        for event, response in zip(events, responses, strict=True):
            expected = f"Reply to {event.text}"
            self.assertEqual(response.messages[0].text, expected)
            self.assertEqual(await app.handle(event), response)
            session = SessionKey(event.address, event.actor.identity, SessionScope.PRIVATE)
            messages = await database.recent_messages(session, limit=10)
            self.assertEqual([item.content for item in messages], [event.text, expected])
        self.assertEqual(model.complete.await_count, 2)
