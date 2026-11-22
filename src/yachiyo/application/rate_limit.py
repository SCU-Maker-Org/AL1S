"""Bounded in-process sliding-window rate limiting."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass

from yachiyo.config.models import RateLimitSettings
from yachiyo.domain import InboundEvent


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    retry_after: float = 0.0
    notify: bool = True


class RateLimiter:
    def __init__(self, settings: RateLimitSettings, *, max_keys: int = 20_000):
        self.settings = settings
        self.max_keys = max_keys
        self._user: dict[str, deque[float]] = {}
        self._chat: dict[str, deque[float]] = {}
        self._notified: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def check(self, event: InboundEvent) -> RateLimitDecision:
        now = time.monotonic()
        user_key = event.actor.identity.key
        chat_key = (
            f"{event.address.transport.value}:{event.address.bot_id}:"
            f"{event.address.chat_id}:{event.address.thread_id}"
        )
        async with self._lock:
            user_bucket, user_wait = self._check(
                self._user,
                user_key,
                now,
                self.settings.per_user_requests,
                self.settings.per_user_window_seconds,
            )
            chat_bucket, chat_wait = self._check(
                self._chat,
                chat_key,
                now,
                self.settings.per_chat_requests,
                self.settings.per_chat_window_seconds,
            )
            retry_after = max(user_wait, chat_wait)
            if retry_after <= 0:
                user_bucket.append(now)
                chat_bucket.append(now)
                notify = True
            else:
                notification_key = f"{user_key}|{chat_key}"
                last_notified = self._notified.get(notification_key, 0.0)
                notify = now - last_notified >= min(
                    self.settings.per_user_window_seconds,
                    self.settings.per_chat_window_seconds,
                )
                if notify:
                    self._notified[notification_key] = now
            self._bound(self._user)
            self._bound(self._chat)
            if len(self._notified) > self.max_keys:
                cutoff = now - max(
                    self.settings.per_user_window_seconds,
                    self.settings.per_chat_window_seconds,
                )
                self._notified = {
                    key: timestamp
                    for key, timestamp in self._notified.items()
                    if timestamp >= cutoff
                }
        return RateLimitDecision(retry_after <= 0, retry_after, notify)

    @staticmethod
    def _check(
        buckets: dict[str, deque[float]],
        key: str,
        now: float,
        limit: int,
        window: int,
    ) -> tuple[deque[float], float]:
        bucket = buckets.setdefault(key, deque())
        cutoff = now - window
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            return bucket, max(0.0, window - (now - bucket[0]))
        return bucket, 0.0

    def _bound(self, buckets: dict[str, deque[float]]) -> None:
        if len(buckets) <= self.max_keys:
            return
        empty = [key for key, values in buckets.items() if not values]
        for key in empty:
            buckets.pop(key, None)
        while len(buckets) > self.max_keys:
            buckets.pop(next(iter(buckets)))
