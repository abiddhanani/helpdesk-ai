# IT Help Desk — AI Engineering Assessment

Four progressive exercises building an agentic IT support system on top of the Model Context Protocol (MCP).

| Directory | What it is |
|----------|-----------|
| [mcp-server](mcp-server/) | MCP server — 10 tools over a PostgreSQL help desk database |
| [helpdesk-agent](helpdesk-agent/) | Single AI agent with Reason/Act/Observe loop, multi-turn memory |
| [multi-agent](multi-agent/) | Multi-agent orchestration — Triage, Resolution, Escalation (Flint DAG) |
| [eval-harness](eval-harness/) | Evaluation harness — automated metrics + LLM-as-judge |

---

## LLM Provider

**Anthropic Claude** (`claude-sonnet-4-6` by default, configurable via `ANTHROPIC_MODEL`).

Reasons:
- Native tool-use support in the messages API: the SDK returns `tool_use` content blocks that map directly to MCP tool calls without extra parsing.
- Reliable instruction-following for structured JSON outputs required by the agent communication protocol in Exercise 3.
- `claude-haiku-4-5` provides a cost-effective alternative for the LLM-as-judge in Exercise 4.

---

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- Docker (for PostgreSQL)
- An Anthropic API key

---

## Quick Start

```bash
# 1. Start PostgreSQL
docker compose up -d

# 2. Configure environment
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY at minimum

# 3. Install and seed the MCP server (required by all other components)
cd mcp-server
uv sync
uv run seed_data.py
cd ..
```

Each exercise is then runnable independently — see its own `README.md` for details.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key (also accepted as `OPENAI_API_KEY`) |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-6` | Model for specialist agents (also `AGENT_MODEL`) |
| `ROUTER_MODEL` | No | `claude-haiku-4-5-20251001` | Cheap model for routing, synthesis, memory compaction |
| `JUDGE_MODEL` | No | `claude-haiku-4-5-20251001` | Model for LLM-as-judge in eval harness |
| `DATABASE_URL` | Yes | `postgresql://helpdesk:helpdesk@localhost:5432/helpdesk` | PostgreSQL connection string |
| `OPENAI_BASE_URL` | No | `https://api.anthropic.com/v1/` | Provider base URL for OpenAI-compat client |
| `EMBEDDING_API_KEY` | No | — | OpenAI key for `text-embedding-3-small`; falls back to BM25-only if absent |

Never commit `.env`. The `.env.example` file documents all variables.

---

## Assumptions and Trade-offs

**PostgreSQL over SQLite** — Exercise 3 runs three agents concurrently, each with its own MCP client session. SQLite's single-writer lock would serialize all writes. PostgreSQL handles concurrent connections with MVCC and provides native full-text search (`tsvector`) for better keyword matching quality.

**In-process orchestration** — Exercise 3 agents run as coroutines within a single Python process rather than as separate services. This avoids network complexity for a demonstration system while still modelling distinct agent responsibilities. A production deployment would run each agent as an independent service behind a message queue.

**Shared seed data** — all exercises use the same PostgreSQL database seeded by Exercise 1. Exercises 2-4 do not re-seed; they read and mutate the same data. Running scenarios in Exercise 4 may leave tickets in modified states; re-run `seed_data.py` to reset.
