"""Per-agent two-tier memory with token-based compaction.

Short-term: ring buffer of recent chat messages, capped at SHORT_TERM_TOKEN_LIMIT.
Long-term: list of MemoryFact produced by LLM summarization of evicted messages.

Compaction trigger: when adding a new message would push short-term over the limit,
summarize the oldest 60% into MemoryFacts and keep the newest 40% verbatim.

vs exercise-2: that version compacted every 5 turns (count-based, token-blind).
This version compacts when the actual token budget is exhausted (precise, cost-aware).
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.messages import MemoryFact
from src.memory.token_counter import count_message_tokens, count_tokens

if TYPE_CHECKING:
    from src.llm.protocol import LLMClient

SHORT_TERM_TOKEN_LIMIT = int(os.environ.get("SHORT_TERM_TOKEN_LIMIT", "8000"))
COMPACT_KEEP_RATIO = 0.40  # keep newest 40% verbatim after compaction


class AgentMemory:
    """Two-tier token-aware memory for a single agent instance."""

    def __init__(
        self,
        agent_name: str,
        llm_client: LLMClient,
        summarizer_model: str,
        token_limit: int = SHORT_TERM_TOKEN_LIMIT,
    ) -> None:
        self._name = agent_name
        self._llm = llm_client
        self._model = summarizer_model
        self._limit = token_limit
        self._short_term: list[dict[str, Any]] = []
        self._long_term: list[MemoryFact] = []
        self._short_term_tokens: int = 0
        self._lock = asyncio.Lock()

    async def add_message(self, message: dict[str, Any]) -> None:
        tokens = count_message_tokens(message)
        async with self._lock:
            if self._short_term_tokens + tokens > self._limit:
                await self._compact()
            self._short_term.append(message)
            self._short_term_tokens += tokens

    async def get_messages(self) -> list[dict[str, Any]]:
        """Return messages for the next LLM call.

        If long-term facts exist, prepends a synthetic context injection so
        the agent sees prior accumulated context without burning short-term budget.
        """
        async with self._lock:
            if not self._long_term:
                return list(self._short_term)
            fact_block = "\n".join(f"- {f.content}" for f in self._long_term)
            injection = {
                "role": "user",
                "content": f"[Prior context from this session]\n{fact_block}",
            }
            ack = {
                "role": "assistant",
                "content": "Understood. I have noted the prior context.",
            }
            return [injection, ack, *self._short_term]

    @property
    def short_term_token_count(self) -> int:
        return self._short_term_tokens

    @property
    def long_term_fact_count(self) -> int:
        return len(self._long_term)

    async def _compact(self) -> None:
        """Summarize oldest 60% into MemoryFacts; keep newest 40% in short-term."""
        n = len(self._short_term)
        keep_n = max(1, int(n * COMPACT_KEEP_RATIO))
        to_summarize = self._short_term[: n - keep_n]
        to_keep = self._short_term[n - keep_n :]

        facts_text = await self._summarize(to_summarize)
        fact = MemoryFact(
            content=facts_text,
            source_agent=self._name,
            token_count=count_tokens(facts_text),
            created_at=datetime.now(timezone.utc),
        )
        self._long_term.append(fact)
        self._short_term = to_keep
        self._short_term_tokens = sum(count_message_tokens(m) for m in to_keep)

    async def _summarize(self, messages: list[dict[str, Any]]) -> str:
        convo = "\n".join(
            f"{m['role'].upper()}: "
            + (m["content"] if isinstance(m["content"], str) else str(m["content"]))
            for m in messages
        )
        system = (
            "You are a memory compactor. Extract the 3-5 most important facts "
            "from this conversation excerpt. Be concise. Output plain text, one fact per line."
        )
        response = await self._llm.complete(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": f"Summarize:\n{convo}"}],
            max_tokens=300,
        )
        return response.content or ""
