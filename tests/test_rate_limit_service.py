from __future__ import annotations

import pytest

from src.config import TelegramRateLimitConfig
from src.services.rate_limit_service import RateLimitService


@pytest.mark.asyncio
async def test_user_rate_limit_blocks_without_consuming_agent_request():
    service = RateLimitService(
        TelegramRateLimitConfig(
            per_user_requests=1,
            per_user_window_seconds=60,
            per_chat_requests=10,
            per_chat_window_seconds=60,
        )
    )
    assert (await service.check(1, -1, 7)).allowed
    denied = await service.check(1, -2, 8)
    assert not denied.allowed
    assert denied.dimension == "user"
    assert denied.notify
    assert not (await service.check(1, -2, 8)).notify


@pytest.mark.asyncio
async def test_chat_rate_limit_isolated_by_topic():
    service = RateLimitService(
        TelegramRateLimitConfig(
            per_user_requests=10,
            per_user_window_seconds=60,
            per_chat_requests=1,
            per_chat_window_seconds=60,
        )
    )
    assert (await service.check(1, -1, 7)).allowed
    assert not (await service.check(2, -1, 7)).allowed
    assert (await service.check(2, -1, 8)).allowed
