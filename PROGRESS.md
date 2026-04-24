# Production Upgrade — Progress Summary

This document captures all enhancements made across the four exercises beyond the original assessment baseline.

---

## Exercise 1 — MCP Server

### Hybrid RAG (BM25 + pgvector)
- **`src/database.py`** — pgvector extension enabled, HNSW indexes on `tickets.embedding` and `knowledge_base.embedding`.
- **`src/tools/embeddings.py`** — `text-embedding-3-small` via OpenAI API, async-safe wrapper.
- **`src/tools/knowledge.py`** + **`src/tools/tickets.py`** — BM25 keyword search fused with cosine similarity via Reciprocal Rank Fusion (k=60). Falls back to BM25 alone for rows without embeddings.
- **`src/tools/tickets.py`** — `create_ticket` / `update_ticket` fire embedding generation in background threads (non-blocking).

### Cross-Request Customer Memory
- **`src/models.py`** — `CustomerMemory` table (customer_id, fact, created_at).
- **`src/tools/memory_tools.py`** — `get_customer_memories` and `save_customer_memory` MCP tools.
- **`src/server.py`** — both tools registered and exposed to agents.

---

## Exercise 3 — Multi-Agent Orchestration

### Provider Abstraction (LLM Protocol)
- **`src/llm/protocol.py`** — `LLMClient` Protocol, `LLMResponse`, `ToolDefinition`, `ToolCall`. All agents depend on this interface, not on any specific SDK.
- **`src/llm/openai_client.py`** — concrete adapter: OpenAI SDK pointed at `api.anthropic.com/v1/`. Passes `cache_control: ephemeral` for Anthropic prompt caching via `extra_body`.

### Pydantic Structured Outputs
- **`src/messages.py`** — all inter-agent types (`AgentOutput`, `AgentMessage`, `RouterDecision`, `SynthesizedResult`, `OrchestratorResult`, `MemoryFact`) are Pydantic models. No regex parsing, no bare `json.loads`.

### Token-Based Memory with Compaction
- **`src/memory/token_counter.py`** — `tiktoken`-based token counting.
- **`src/memory/agent_memory.py`** — short-term window capped at 8K tokens; oldest 60% summarized into a `MemoryFact` via cheap model when limit is reached.

### LLM Router (replaces static routing table)
- **`src/routing/llm_router.py`** — fast path (confidence ≥ 0.85 trusts triage), slow path (cheap model call for borderline cases), safe fallback to escalation. Every decision is logged with reasoning.

### OpenTelemetry Tracing
- **`src/trace.py`** — OTel spans alongside stderr logging. Root span per request, hop spans per agent, tool spans per MCP call. Console exporter by default; OTLP via `OTEL_EXPORTER_OTLP_ENDPOINT`.

### Prompt Caching
- **`src/llm/openai_client.py`** — system prompt sent with `cache_control: ephemeral` when Anthropic endpoint is detected. Cache hit/write token counts recorded in each trace.

### Flint Workflow DAG (replaces hand-rolled Orchestrator)
- **`src/flint_adapter.py`** — `SpecialistAgentAdapter(FlintAdapter)` wraps any `BaseSpecialistAgent` subclass; opens a fresh MCP session per invocation so it works inside Flint's embedded event loop. `SynthesisAdapter(FlintAdapter)` calls `complete_structured(SynthesizedResult)` for the fan-in merge.
- **`src/flint_orchestrator.py`** — `FlintOrchestrator` builds and runs a Flint Workflow DAG:
  ```
  triage ──┬── resolution ──┐
           └── escalation ──┴── synthesis
  ```
  Flint handles parallel fan-out, per-node retries, dead-letter queue, and prompt interpolation (`{triage}`, `{resolution}`, `{escalation}`). `Workflow.run()` is synchronous; called via `run_in_executor` from the async entry point.
- **`src/main.py`** — rewritten to construct adapters and `FlintOrchestrator`; manual `stdio_client` setup removed.
- **`pyproject.toml`** — `flint-ai` added via `[tool.uv.sources]` pointing at local clone.
- **Dashboard** — `http://localhost:5160/ui/` (embedded, alive for the duration of each request).
- **`src/orchestrator.py`** — kept unchanged for reference.

---

## Exercise 4 — Evaluation Harness

### Extended Metrics
- **`src/metrics.py`** — four new fields on `ScenarioMetrics`: `routing_used_llm` (bool), `routing_confidence` (float), `routing_correct` (bool), `parallel_speedup_ms` (int).

### Retrieval Eval
- **`src/retrieval_eval.py`** — compares BM25 vs hybrid search using NDCG@5, Recall@10, and MRR.

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Anthropic key (OpenAI-compat endpoint) |
| `OPENAI_BASE_URL` | `https://api.anthropic.com/v1/` | Provider base URL |
| `AGENT_MODEL` | `claude-sonnet-4-6` | Model for specialist agents |
| `ROUTER_MODEL` | `claude-haiku-4-5-20251001` | Model for router, synthesis, memory compaction |
| `SHORT_TERM_TOKEN_LIMIT` | `8000` | Agent memory window size |
| `EMBEDDING_API_KEY` | — | OpenAI key for `text-embedding-3-small` (optional, BM25 fallback if absent) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | OTLP endpoint; console exporter used if unset |
| `OTEL_DISABLED` | — | Set to `true` to suppress all OTel output |
| `DATABASE_URL` | `postgresql://helpdesk:helpdesk@localhost:5432/helpdesk` | PostgreSQL connection |
