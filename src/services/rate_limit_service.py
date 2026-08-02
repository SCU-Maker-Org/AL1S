"""按用户和聊天维度实现的异步滑动窗口限流。"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

from ..config import TelegramRateLimitConfig


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    retry_after: float = 0.0
    dimension: str = ""
    notify: bool = True


class RateLimitService:
    def __init__(self, rate_config: TelegramRateLimitConfig):
        self.config = rate_config
        self._user_requests: dict[int, Deque[float]] = defaultdict(deque)
        self._chat_requests: dict[tuple[int, int], Deque[float]] = defaultdict(deque)
        self._last_notice: dict[tuple[str, object], float] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _prune(values: Deque[float], cutoff: float) -> None:
        while values and values[0] <= cutoff:
            values.popleft()

    async def check(
        self, user_id: int, chat_id: int, thread_id: int = 0
    ) -> RateLimitDecision:
        if not self.config.enabled:
            return RateLimitDecision(True)
        now = time.monotonic()
        async with self._lock:
            user_values = self._user_requests[user_id]
            chat_key = (chat_id, thread_id)
            chat_values = self._chat_requests[chat_key]
            self._prune(user_values, now - self.config.per_user_window_seconds)
            self._prune(chat_values, now - self.config.per_chat_window_seconds)

            if len(user_values) >= self.config.per_user_requests:
                retry = self.config.per_user_window_seconds - (now - user_values[0])
                return self._denied("user", user_id, now, retry)
            if len(chat_values) >= self.config.per_chat_requests:
                retry = self.config.per_chat_window_seconds - (now - chat_values[0])
                return self._denied("chat", chat_key, now, retry)

            user_values.append(now)
            chat_values.append(now)
            self._cleanup_empty()
            return RateLimitDecision(True)

    def _denied(
        self, dimension: str, key: object, now: float, retry: float
    ) -> RateLimitDecision:
        notice_key = (dimension, key)
        notify = now - self._last_notice.get(notice_key, 0.0) >= 5.0
        if notify:
            self._last_notice[notice_key] = now
        return RateLimitDecision(False, max(retry, 0.0), dimension, notify)

    def _cleanup_empty(self) -> None:
        self._user_requests = defaultdict(
            deque, {key: value for key, value in self._user_requests.items() if value}
        )
        self._chat_requests = defaultdict(
            deque, {key: value for key, value in self._chat_requests.items() if value}
        )
