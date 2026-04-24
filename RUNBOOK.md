# Runbook — How to Run and Validate the Solution

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12+ | |
| uv | latest | `pip install uv` |
| Docker | any | runs PostgreSQL |
| Anthropic API key | — | for agents and router |
| OpenAI API key | — | optional, only needed for `text-embedding-3-small` (hybrid search); falls back to BM25 if absent |

---

## One-Time Setup

### 1. Start PostgreSQL

```bash
cd /Users/shama/projects/helpdesk-ai
docker compose up -d
```

This starts PostgreSQL on port 5432 with database `helpdesk`, user `helpdesk`, password `helpdesk`.

### 2. Configure environment

Create `.env` in the repo root:

```env
# Required — Anthropic key used as the OpenAI-compatible provider
OPENAI_API_KEY=sk-ant-...
OPENAI_BASE_URL=https://api.anthropic.com/v1/

# Optional — override models
AGENT_MODEL=claude-sonnet-4-6
ROUTER_MODEL=claude-haiku-4-5-20251001

# Optional — for semantic (vector) search; BM25-only fallback if absent
EMBEDDING_API_KEY=sk-...   # OpenAI key for text-embedding-3-small

# Optional — OTel export; console exporter used otherwise
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
# Set to suppress all OTel output entirely
# OTEL_DISABLED=true

DATABASE_URL=postgresql://helpdesk:helpdesk@localhost:5432/helpdesk
```

### 3. Seed the database

```bash
cd mcp-server
uv sync
uv run python seed_data.py
cd ..
```

This creates all tables, inserts sample agents, tickets, customers, and knowledge base articles. All subsequent exercises read/write this same database.

### 4. Install dependencies

```bash
# Exercise 1 (MCP server — already done above via uv sync)

# Exercise 3 (multi-agent + Flint DAG)
cd multi-agent
uv sync
cd ..

# Exercise 4 (eval harness)
cd eval-harness
uv sync
cd ..
```

---

## Running Exercise 3 — Multi-Agent Helpdesk (Flint DAG)

### Single request

```bash
cd multi-agent
uv run python -m src.main "My VPN is not working"
```

Expected output:

```
Multi-Agent IT Help Desk (Flint DAG). Model: claude-sonnet-4-6 | Router: claude-haiku-4-5-20251001
Dashboard: http://localhost:5160/ui/ (available during each request)

Flint — running workflow 'helpdesk-<trace_id>' (4 nodes)
   Dashboard → http://localhost:5160/ui/

  triage
  resolution
  escalation
  synthesis

<final summary printed here>
```

### Interactive REPL

```bash
cd multi-agent
uv run python -m src.main
# > My laptop won't connect to Wi-Fi
# > exit
```

### Run all 4 canonical scenarios

```bash
cd multi-agent
uv run python run_scenarios.py
```

Scenario outputs and full traces are written to `multi-agent/examples/` as JSON files named `<label>_<trace_id>.json`.

### Run Exercise 4 evaluation harness

```bash
cd eval-harness
uv run python src/harness.py
# or with a custom scenario file:
uv run python src/harness.py --scenarios test_scenarios/scenarios.yaml
```

Results are written to `eval-harness/results/` as JSON + Markdown reports.

---

## Validating the Solution

### Smoke test (fastest)

```bash
cd multi-agent
uv run python -c "
from src.flint_adapter import SpecialistAgentAdapter, SynthesisAdapter
from src.flint_orchestrator import FlintOrchestrator
from src.agents.triage import TriageAgent
from src.agents.resolution import ResolutionAgent
from src.agents.escalation import EscalationAgent
print('All imports OK')
"
```

### Live end-to-end test

Run a request and check:

1. **4 Flint nodes complete** — you see `triage`, `resolution`, `escalation`, `synthesis` lines in stdout (with or without emoji depending on verbose mode).
2. **Trace JSON written** — a file appears in `multi-agent/examples/` with the full request/result/trace.
3. **Dashboard** — open `http://localhost:5160/ui/` while a request is in flight to inspect live node states.
4. **Final summary** — a coherent one-sentence summary is printed at the end.

### Checking the trace file

Each trace JSON (`multi-agent/examples/<id>.json`) has this structure:

```json
{
  "request": "original user request",
  "result": {
    "trace_id": "...",
    "final_intent": "resolved | escalated | waiting_on_customer | error",
    "final_summary": "...",
    "hops": [ ... ]
  },
  "trace": { ... }
}
```

Verify:
- `final_intent` is one of the valid `Intent` values, not `"error"`.
- `hops` contains entries for `triage`, `resolution`, and `escalation`.
- Each hop has non-zero `input_tokens` / `output_tokens`.

### Validating hybrid RAG (Exercise 1)

To confirm vector search is active:

```bash
cd mcp-server
uv run python -c "
from src.database import get_engine
from sqlalchemy import text
with get_engine().connect() as conn:
    result = conn.execute(text(\"SELECT COUNT(*) FROM knowledge_base WHERE embedding IS NOT NULL\"))
    print('KB rows with embeddings:', result.scalar())
    result = conn.execute(text(\"SELECT COUNT(*) FROM tickets WHERE embedding IS NOT NULL\"))
    print('Ticket rows with embeddings:', result.scalar())
"
```

If counts are 0, the `EMBEDDING_API_KEY` was not set when tickets/KB articles were created. Re-seed or trigger embedding via the MCP tools.

---

## How the Flow Works

### Architecture overview

```
User request
    │
    ▼
src/main.py  ─── constructs SpecialistAgentAdapter × 3 + SynthesisAdapter
    │
    ▼
FlintOrchestrator.run()  ──► Workflow("helpdesk-<id>").run()
    │                              [embedded Flint server, port 5160]
    │
    ├─ Node: triage ──────────────────────────────────────────────────┐
    │     SpecialistAgentAdapter("triage", TriageAgent)              │
    │     Opens MCP session → TriageAgent.run(request)               │
    │                                                                 │
    ├─ Node: resolution ◄─── depends_on("triage") ──────────────────┤ (parallel fan-out)
    │     prompt = "Resolve this ticket.\n\nTriage context:\n{triage}"│
    │     SpecialistAgentAdapter("resolution", ResolutionAgent)       │
    │                                                                 │
    ├─ Node: escalation ◄─── depends_on("triage") ──────────────────┘
    │     prompt = "Evaluate for escalation.\n\nTriage context:\n{triage}"
    │     SpecialistAgentAdapter("escalation", EscalationAgent)
    │
    └─ Node: synthesis ◄─── depends_on("resolution", "escalation")
          SynthesisAdapter  →  LLMClient.complete_structured(SynthesizedResult)
          Returns: { final_intent, final_summary }
```

### Step-by-step flow

#### Step 1 — Triage

`TriageAgent` receives the raw user request. It:

1. Loads prior customer memory (`get_customer_memories`) for context.
2. Searches for an existing ticket or creates one (`search_tickets` / `create_ticket`).
3. Classifies priority (low / medium / high / critical) and category (hardware / software / network / access / other).
4. Searches the knowledge base for a known self-service solution.
5. Adds a triage comment to the ticket.
6. Produces a structured `AgentOutput` with an `intent`:
   - `resolved` — KB article fully addresses the issue; no further agents needed.
   - `route_to_resolution` — needs KB-driven resolution or agent assignment.
   - `route_to_escalation` — pure operational/management task.
   - `waiting_on_customer` — waiting for more info.

The triage output JSON is stored by Flint and injected into the downstream prompts via `{triage}` interpolation.

#### Step 2 — Parallel fan-out (Resolution + Escalation)

Both agents run **simultaneously** once triage completes. Each receives the full triage context in its prompt.

**ResolutionAgent** handles three patterns:
- *Batch KB resolution* — loops over multiple tickets, adds KB comments, escalates those with no match.
- *Single-ticket resolution* — searches KB, assigns a human agent if no KB match, escalates critical tickets.
- *Analytics requests* — gathers KB coverage data and hands off to Escalation.

**EscalationAgent** handles:
- *Single-ticket escalation* — checks SLA deadline, bumps priority if needed, assigns the best available specialist.
- *Operational/management* — redistributes tickets (e.g. agent going on leave).
- *Analytics* — adds SLA breach rates and agent utilization to the Resolution agent's KB gap analysis.

#### Step 3 — Synthesis (fan-in)

`SynthesisAdapter` receives both branch outputs (interpolated as `{resolution}` and `{escalation}`) and calls `LLMClient.complete_structured(SynthesizedResult)` with the rule:

- escalated > resolved (escalation is an audit requirement)
- If one branch errored, use the healthy branch
- Produces a single `final_intent` + `final_summary`

#### Step 4 — Result assembly

`FlintOrchestrator._parse_results()` parses the JSON strings from each Flint node back into `AgentMessage` Pydantic models and wraps them in an `OrchestratorResult`. The result is written to `examples/<trace_id>.json` and the `final_summary` is printed to stdout.

---

### Memory and state across agents

| Mechanism | Scope | Where |
|-----------|-------|-------|
| `AgentMemory` | Within a single agent run (short-term, 8K token window) | `src/memory/agent_memory.py` |
| `CustomerMemory` (DB) | Across requests for the same customer | `mcp-server/src/models.py` + `memory_tools.py` |
| Flint `WorkflowContext` | Across nodes within one workflow run (via `{node_id}` interpolation) | flint-ai internals |
| `SharedContext` | Per-agent invocation (in-memory dict, keyed by trace_id) | `src/context.py` |

### Prompt caching

When `OPENAI_BASE_URL` points to `api.anthropic.com`, each LLM call sends the system prompt with `cache_control: ephemeral`. Cache hit/write token counts are logged per hop in the trace file. For long multi-turn agent runs this significantly reduces latency and cost on repeated calls with the same system prompt.

### Retry and fault tolerance

Flint handles retries and dead-letter queueing at the node level:
- Each node retries up to 3 times on failure (configurable via `Node.with_retries(n)`).
- Persistent failures are moved to the dead-letter queue rather than crashing the workflow.
- If synthesis fails, `FlintOrchestrator._parse_results()` falls back to the triage hop's intent and summary.

---

## Common Issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `OPENAI_API_KEY not set` | Missing env var | Add to `.env` and re-source |
| `connection refused` on DB | PostgreSQL not running | `docker compose up -d` |
| All embeddings are 0 | No `EMBEDDING_API_KEY` | Set key and re-seed, or accept BM25-only mode |
| Port 5160 already in use | Previous Flint run crashed | `lsof -ti:5160 \| xargs kill` |
| `synthesis` node output is `""` | LLM synthesis call failed | Check ROUTER_MODEL is accessible; `_parse_results` falls back to triage |
| `run_scenarios.py` fails | Script uses old `AsyncAnthropic` client | Use `src/main.py` directly instead |
