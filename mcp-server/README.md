# Exercise 1 — IT Help Desk MCP Server

An MCP (Model Context Protocol) server exposing 10 tools for managing an IT help desk system.
Built with FastMCP + PostgreSQL. Communicates over stdio transport.

---

## Quick Start

**Prerequisites:** Docker, Python 3.12+, `uv`

```bash
# 1. Start PostgreSQL
docker compose up -d          # from repo root

# 2. Install dependencies
cd mcp-server
uv sync

# 3. Configure environment
cp ../.env.example ../.env
# Edit .env — set DATABASE_URL if non-default

# 4. Seed the database
uv run seed_data.py

# 5. Verify the server starts
uv run python -m src.server
```

The server communicates via stdio. You will not see any output on stdout (that is correct — stdout is reserved for JSON-RPC). Startup logs appear on stderr.

---

## Verification

**Option A — MCP Inspector:**

```bash
npx @modelcontextprotocol/inspector uv run python -m src.server
```

Open the URL printed in the terminal, select a tool, and call it interactively.

**Option B — Claude Code local MCP:**

This repo's `.claude/settings.json` already has `enableAllProjectMcpServers: true`. Add the server entry to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "helpdesk": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.server"],
      "cwd": "/absolute/path/to/mcp-server",
      "env": {
        "DATABASE_URL": "postgresql://helpdesk:helpdesk@localhost:5432/helpdesk"
      }
    }
  }
}
```

---

## Tools Reference

| Tool | Parameters | Description |
|---|---|---|
| `search_tickets` | `status?`, `priority?`, `category?`, `assigned_agent_id?`, `customer_id?`, `keyword?`, `sla_overdue?` | Full-text + filter search across tickets |
| `get_ticket_details` | `ticket_id` | Full ticket including all comments |
| `create_ticket` | `title`, `description`, `priority`, `category`, `customer_id` | Creates ticket, auto-calculates SLA deadline |
| `update_ticket` | `ticket_id`, any fields | Updates any ticket fields |
| `assign_ticket` | `ticket_id`, `agent_id` | Assigns ticket; enforces availability and capacity |
| `add_comment` | `ticket_id`, `author_type`, `author_id`, `content` | Adds comment to a ticket |
| `search_knowledge_base` | `keyword?`, `category?`, `tags?` | Full-text search across KB articles |
| `get_agent_workload` | `agent_id` | Agent details + active ticket count and capacity |
| `list_agents` | `specialty?`, `is_available?` | Lists agents with live workload counts |
| `get_sla_report` | `category?`, `priority?` | SLA breach rates overall and by category/priority |

**Enums:**
- `status`: `open`, `in_progress`, `waiting_on_customer`, `resolved`, `closed`
- `priority`: `low`, `medium`, `high`, `critical`
- `category`: `hardware`, `software`, `network`, `access`, `other`
- `specialty`: `hardware`, `software`, `network`, `security`
- `author_type`: `agent`, `customer`, `system`

**Error format:** Errors raise exceptions which FastMCP converts to `isError: true` MCP responses. Error messages are prefixed with their type: `NOT_FOUND:`, `INVALID_INPUT:`, `BUSINESS_RULE:`.

---

## Seed Data

| Entity | Count | Notes |
|---|---|---|
| Agents | 8 | Specialties: network (2), software (2), hardware (2), security (2). One agent is unavailable. |
| Customers | 15 | Across Engineering, Marketing, Finance, HR, Sales, Operations, Legal |
| Tickets | 35 | ~10 overdue SLA deadlines across critical and high priorities |
| KB Articles | 10 | Covers VPN, passwords, workstation setup, printers, ransomware, Wi-Fi, SSL, macOS, storage, access |
| Comments | 25 | Mix of agent, customer, and system comments |

---

## Architecture Review

### Why FastMCP?

FastMCP is the official Python MCP server framework from the MCP SDK authors. The `@mcp.tool()` decorator auto-generates JSON schemas from Python type hints and docstrings — the LLM receives accurate tool descriptions without any manual schema maintenance. Supports stdio transport natively.

### Why PostgreSQL over SQLite?

Exercise 3 runs three agents concurrently, each with its own MCP client session. SQLite has a single-writer lock — three concurrent agents writing tickets and comments would serialize all writes and create lock contention. PostgreSQL handles concurrent connections natively with MVCC. Additionally, PostgreSQL's `tsvector` full-text search provides stemming and ranking that `LIKE` queries cannot, improving the quality of `search_tickets` and `search_knowledge_base` for LLM-driven queries.

### Tool registration pattern

Tools are grouped into submodules (`tools/tickets.py`, `tools/knowledge.py`, `tools/agents_tools.py`). Each exports a `register_tools(mcp: FastMCP)` function. In `server.py`, tools are registered after the FastMCP instance is created:

```python
mcp = FastMCP("IT Help Desk")
register_ticket_tools(mcp)
register_knowledge_tools(mcp)
register_agent_tools(mcp)
```

This avoids circular imports (the `mcp` instance is not module-level in a submodule) and avoids the namespace prefixing that `mcp.mount()` introduces.

### Session lifecycle

Each tool call opens a database session via `with get_db() as db:`, performs its work, and commits/rolls back on exit. Sessions are never shared across tool calls. `expire_on_commit=False` prevents `DetachedInstanceError` when converting ORM objects to dicts after commit. ORM objects are always converted to plain dicts inside the session block before returning.

### Full-text search

`tsvector` generated columns are added to `tickets` and `knowledge_articles` via DDL after `create_all()` in `init_db()`. GIN indexes are created on those columns.

Ticket search uses `plainto_tsquery` (AND logic, safe for raw user input). Knowledge base search uses `websearch_to_tsquery` with OR between terms so that any matching keyword finds an article — this gives higher recall for natural-language queries like "VPN connection error" where the article uses "issue" not "error".

```sql
-- Added by init_db():
ALTER TABLE tickets ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(description,''))) STORED;
CREATE INDEX idx_tickets_search ON tickets USING GIN(search_vector);
```

### SLA deadline calculation

SLA deadlines are calculated in the `create_ticket` tool handler using a constant dict:

```python
SLA_HOURS = {
    "critical": timedelta(hours=4),
    "high":     timedelta(hours=8),
    "medium":   timedelta(hours=24),
    "low":      timedelta(hours=72),
}
sla_deadline = datetime.utcnow() + SLA_HOURS[priority]
```

This is explicit and testable. Not in the model `__init__` (bypassed by bulk inserts), not in a DB trigger (opaque, hard to test).

### Error handling

Three custom exception classes in `src/errors.py`:

- `NotFoundError` — prefixed `NOT_FOUND:`
- `InvalidInputError` — prefixed `INVALID_INPUT:`
- `BusinessRuleError` — prefixed `BUSINESS_RULE:`

FastMCP catches these and sets `isError: true` in the `CallToolResult`. The LLM client receives a clearly marked error it can reason about and retry with corrected parameters.

### Enum fields defined as String, not PostgreSQL ENUM

`status`, `priority`, `category`, `specialty`, and `author_type` are all defined as `Column(String)` in the SQLAlchemy models — not as PostgreSQL `ENUM` types. Reasons:

- **No migration cost to add values** — PostgreSQL `ENUM` is a schema object; adding a new value requires `ALTER TYPE ... ADD VALUE`. A `VARCHAR` column just accepts new values without any DDL change.
- **Simpler ORM code** — no need to define Python `enum.Enum` classes or keep them in sync with the DB type.

Validation is enforced at the application layer in three stages:

1. **FastMCP schema** — tool signatures use `Literal["open", "in_progress", ...]` type hints, so FastMCP generates a JSON schema `enum` constraint. The LLM client will not pass invalid values.
2. **Tool handler** — explicit set membership checks (`VALID_STATUSES`, `VALID_PRIORITIES`, etc.) raise `InvalidInputError` for any value that slips through.
3. **Database** — permissive `VARCHAR` column; the two layers above ensure nothing invalid reaches it.

---

### assign_ticket business rules (checked in order)

1. Ticket exists — `NotFoundError` if not
2. Agent exists — `NotFoundError` if not
3. Agent `is_available` — `BusinessRuleError` if false
4. Agent active ticket count < `max_tickets` — `BusinessRuleError` if at capacity

Order matters: `NotFoundError` before `BusinessRuleError` gives the LLM the most actionable error message.

### Forward compatibility for Exercise 3

- `pool_size=10`, `pool_pre_ping=True` in the SQLAlchemy engine — handles 3 concurrent agents with headroom; `pool_pre_ping` detects stale connections in a long-running server process
- `version` column on `Ticket` for optimistic concurrency — incremented on every update, available to agents in Exercise 3 for safe concurrent writes
- Partial index on `sla_deadline` filtered to non-closed tickets — efficient `sla_overdue` queries as ticket volume grows

### stdout discipline

`logging.basicConfig(stream=sys.stderr)` is configured in `server.py` before any tool registration. All print statements in the server use `file=sys.stderr`. Writing to stdout in the server process corrupts the JSON-RPC stream and breaks the MCP client connection.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | `postgresql://helpdesk:helpdesk@localhost:5432/helpdesk` | PostgreSQL connection string |
| `ANTHROPIC_API_KEY` | No (Ex. 1) | — | Required from Exercise 2 onward |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-6` | Configurable model for LLM calls |
