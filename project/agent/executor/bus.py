"""进程内事件总线：Executor 发布，WebSocket / 录制器订阅。每会话一个队列组。"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger("agent.bus")


class EventBus:
    def __init__(self, max_queue: int = 512):
        self._subs: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._max_queue = max_queue

    def subscribe(self, session_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._subs[session_id].add(q)
        return q

    def unsubscribe(self, session_id: str, q: asyncio.Queue) -> None:
        self._subs[session_id].discard(q)
        if not self._subs[session_id]:
            self._subs.pop(session_id, None)

    def publish(self, session_id: str, event: dict[str, Any]) -> None:
        for q in list(self._subs.get(session_id, ())):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("session %s 事件队列已满，丢弃 %s 事件", session_id, event.get("type"))

    def has_subscribers(self, session_id: str) -> bool:
        return bool(self._subs.get(session_id))
