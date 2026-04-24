# IT Help Desk — AI Engineering Assessment

Four progressive exercises building an agentic IT support system on top of the Model Context Protocol (MCP).

| Exercise | What it is |
|----------|-----------|
| [exercise-1](exercise-1/) | MCP server — 10 tools over a PostgreSQL help desk database |
| [exercise-2](exercise-2/) | Single AI agent with Reason/Act/Observe loop, multi-turn memory |
| [exercise-3](exercise-3/) | Multi-agent orchestration — Triage, Resolution, Escalation |
| [exercise-4](exercise-4/) | Evaluation harness — automated metrics + LLM-as-judge |

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
# Edit .env and set ANTHROPIC_API_KEY

# 3. Install and seed Exercise 1 (required by all exercises)
cd exercise-1
uv sync
uv run seed_data.py
cd ..
```

Each exercise is then runnable independently — see its own `README.md` for details.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes (ex. 2-4) | — | Anthropic API key |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-6` | Model for agent calls |
| `DATABASE_URL` | Yes | `postgresql://helpdesk:helpdesk@localhost:5432/helpdesk` | PostgreSQL connection string |
| `JUDGE_MODEL` | No (ex. 4) | `claude-haiku-4-5` | Model for LLM-as-judge evaluation |

Never commit `.env`. The `.env.example` file documents all required variables.

---

## Assumptions and Trade-offs

**PostgreSQL over SQLite** — Exercise 3 runs three agents concurrently, each with its own MCP client session. SQLite's single-writer lock would serialize all writes. PostgreSQL handles concurrent connections with MVCC and provides native full-text search (`tsvector`) for better keyword matching quality.

**In-process orchestration** — Exercise 3 agents run as coroutines within a single Python process rather than as separate services. This avoids network complexity for a demonstration system while still modelling distinct agent responsibilities. A production deployment would run each agent as an independent service behind a message queue.

**Shared seed data** — all exercises use the same PostgreSQL database seeded by Exercise 1. Exercises 2-4 do not re-seed; they read and mutate the same data. Running scenarios in Exercise 4 may leave tickets in modified states; re-run `seed_data.py` to reset.
