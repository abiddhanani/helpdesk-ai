# Exercise 2 — IT Help Desk Agent

An AI agent that connects to the Exercise 1 MCP server and handles multi-step IT support tasks via a Reason → Act → Observe loop.

## Prerequisites

- Python 3.12+
- `uv` package manager
- Exercise 1 MCP server dependencies installed (the agent spawns it as a subprocess)
- PostgreSQL running with the Exercise 1 database seeded

## Setup

```bash
cd exercise-2
uv sync
```

Copy the root `.env.example` to `.env` and fill in your values:

```bash
cp ../.env.example .env
```

Required environment variables:

```
ANTHROPIC_API_KEY=<your key>
DATABASE_URL=postgresql://helpdesk:helpdesk@localhost:5432/helpdesk
```

Optional:

```
ANTHROPIC_MODEL=claude-sonnet-4-6   # default; swap to claude-haiku-4-5-20251001 for cheaper runs
```

## Running

```bash
uv run helpdesk-agent
# or
uv run python -m src
```

The agent spawns the Exercise 1 MCP server as a subprocess automatically — no separate server process needed.

Agent responses go to stdout. Activity logs (reasoning, tool calls, retries, errors) go to stderr.

### REPL commands

- Type any natural language request and press Enter
- `reset` — clear conversation memory and start a fresh session
- `exit` / `quit` / Ctrl-D — exit

## Architecture

```
User (stdin)
    |
    v
AgentController.run_turn()
    |
    +-- anthropic.messages.create()   (REASON: LLM decides next action)
    |
    +-- session.call_tool()           (ACT: execute MCP tool)
    |
    +-- append tool_result to messages (OBSERVE: feed result back to LLM)
    |
    +-- loop until stop_reason=end_turn or max 20 iterations
    |
    v
Answer printed to stdout
```

### Key design decisions

**Async lifecycle** — the entire session runs inside a single `asyncio.run(run())` call. The MCP subprocess and event loop are created once and stay alive across all REPL turns. `input()` is dispatched via `loop.run_in_executor(None, sys.stdin.readline)` so it never blocks the event loop. Calling `asyncio.run()` inside the loop would tear down and recreate the subprocess on every turn — avoided by design.

**Tool discovery** — `session.list_tools()` is called once at startup. The resulting schemas are cached in `AgentController.tools` and passed to every `anthropic.messages.create()` call. No per-turn refetch.

**Conversation memory format** — `self.messages` is a `list[MessageParam]` that conforms exactly to the Anthropic messages API contract:
- When the LLM returns `stop_reason="tool_use"`, the full `response.content` (which may contain `tool_use` blocks) is appended as an `assistant` message.
- A single `user` message follows immediately, containing one `tool_result` block per `tool_use` block, matched by `tool_use_id`.
- Raw strings are never appended; only properly typed content blocks are used. Violating this format causes a 400 from the API on the second tool-using turn.

**Memory summarization** — after every completed turn, if the conversation exceeds 5 user turns the agent compacts old messages. The oldest messages (everything except the last 2 turns) are summarized into 3-5 bullet points via a separate Claude API call, then replaced with a synthetic user/assistant exchange containing that summary. The last 2 turns are kept verbatim so the model retains fine-grained context for the current task. This prevents unbounded context window growth while preserving conversational continuity. Compaction is logged to stderr as `REASON | Memory compacted: summarized N messages`.

**Logging boundary** — all activity logs (reasoning, tool calls, results, retries, errors) go to stderr. Only the final user-facing answer goes to stdout. This mirrors the stdout/stderr discipline established in Exercise 1 (where stdout was reserved for JSON-RPC).

### Error recovery

If a tool call returns an error, the error string is returned as the `tool_result` content and the LLM retries with corrected parameters. Maximum 3 attempts per tool call. After exhausting retries, the error is surfaced in the `tool_result` so the LLM can explain the failure to the user rather than silently stalling.

Ambiguous requests and impossible requests are handled via the system prompt, which instructs the LLM to ask one clarifying question when a request is unclear, and to explain concisely when a request cannot be fulfilled. Both cases produce a normal `end_turn` response — no special code path required.

## Example interactions

See `examples/` for transcripts of all 5 required scenarios plus a multi-turn memory demo:

- `01_sla_breach_search.txt` — critical tickets past SLA deadline
- `02_kb_article_comment.txt` — find KB article for a ticket and add it as a comment
- `03_create_assign_ticket.txt` — create ticket, find best agent, assign
- `04_redistribute_tickets.txt` — redistribute an agent's tickets across available agents
- `05_sla_report_kb_lookup.txt` — SLA compliance report + KB gap analysis
- `06_multi_turn_memory.txt` — three-turn conversation demonstrating memory
