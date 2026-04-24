"""Thread-safe shared context store for multi-agent requests.

Uses asyncio.Lock — correct for a single event-loop process.
Each request is keyed by trace_id to allow future concurrent request support.
"""
from __future__ import annotations

import asyncio
from typing import Any


class SharedContext:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._store: dict[str, dict[str, Any]] = {}

    async def set(self, trace_id: str, key: str, value: Any) -> None:
        async with self._lock:
            if trace_id not in self._store:
                self._store[trace_id] = {}
            self._store[trace_id][key] = value

    async def get(self, trace_id: str, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._store.get(trace_id, {}).get(key, default)

    async def get_all(self, trace_id: str) -> dict[str, Any]:
        async with self._lock:
            return dict(self._store.get(trace_id, {}))

    async def clear(self, trace_id: str) -> None:
        async with self._lock:
            self._store.pop(trace_id, None)
