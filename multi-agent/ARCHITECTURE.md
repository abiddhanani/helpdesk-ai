# Exercise 3 — Multi-Agent System Architecture

## Overview

A sequential multi-agent pipeline for IT help desk ticket handling. An Orchestrator
routes incoming requests through up to three specialist agents, each operating its own
independent MCP ClientSession subprocess against the Exercise 1 database.

---

## System Diagram

```
User request
     |
     v
Orchestrator.run(request, trace_id)
     |
     |-- SharedContext: stores ticket_id, agent summaries, visited list
     |-- TraceLogger:   writes structured JSON trace to examples/<trace_id>.json
     |
     v
TriageAgent  [MCP subprocess 1]
  tools: search_tickets, get_ticket_details, create_ticket,
         update_ticket, add_comment, search_knowledge_base
  - finds or creates ticket
  - classifies priority + category
  - quick KB check
  - returns intent: route_to_resolution | route_to_escalation | resolved
     |
     v (if route_to_resolution)
ResolutionAgent  [MCP subprocess 2]
  tools: search_tickets, get_ticket_details, search_knowledge_base,
         update_ticket, assign_ticket, add_comment, list_agents,
         get_agent_workload
  - deep KB search for solution
  - if found: documents solution, marks resolved
  - if not found: finds best available agent by specialty + workload, assigns
  - for analytics requests: gathers ticket data + KB gap analysis, then
    routes to escalation for SLA + workload data
  - returns intent: resolved | route_to_escalation | waiting_on_customer
     |
     v (if route_to_escalation)
EscalationAgent  [MCP subprocess 3]
  tools: search_tickets, get_ticket_details, update_ticket, assign_ticket,
         add_comment, list_agents, get_agent_workload, get_sla_report
  - evaluates SLA deadline (bumps priority to critical if within 1 hour)
  - assigns to least-loaded available agent regardless of capacity
  - leaves mandatory escalation comment
  - for operational requests (agent leave, bulk redistribution): searches
    tickets by agent, reassigns by SLA deadline priority
  - for analytics/dashboard requests: compiles SLA breach data + agent
    utilization + KB gap summary
  - returns intent: escalated (always terminal)
     |
     v
OrchestratorResult -> stdout summary + examples/<trace_id>.json
```

---

## Agent Responsibilities

| Agent      | Core job                                                           | Terminal intents                    |
|------------|--------------------------------------------------------------------|-------------------------------------|
| Triage     | Classify, create/find ticket, first routing call; routes operational/analytics requests to escalation or resolution | resolved |
| Resolution | KB resolution or human agent assignment; KB gap analysis for analytics requests | resolved, waiting_on_customer |
| Escalation | Force-assign, SLA escalation, bulk redistribution, dashboard reporting | escalated, resolved            |

Each agent is scoped to a fixed set of MCP tools — it cannot call tools outside its
`allowed_tools` list. The LLM never sees out-of-scope tools.

---

## Routing Table

```
(triage,     route_to_resolution)  -> resolution
(triage,     route_to_escalation)  -> escalation
(triage,     resolved)             -> terminal
(triage,     waiting_on_customer)  -> terminal
(resolution, resolved)             -> terminal
(resolution, route_to_escalation)  -> escalation
(resolution, waiting_on_customer)  -> terminal
(escalation, escalated)            -> terminal
(escalation, resolved)             -> terminal
(escalation, waiting_on_customer)  -> terminal
```

---

## Routing Guardrails

**Circular routing guard** — the Orchestrator maintains a `visited` list in SharedContext.
Before routing to the next agent, it checks: if `next_agent in visited`, override to
escalation. If escalation is also visited, terminate. This prevents any agent from being
called twice in a single request.

**MAX_HOPS = 5** — absolute cap regardless of routing logic. If exceeded, the request
terminates with `intent=error`. This guards against future routing additions creating
undetected loops.

---

## AgentMessage Envelope

Every agent returns an `AgentMessage`. This is the only communication interface between
agents — agents never call each other directly.

```python
@dataclass
class AgentMessage:
    sender: Literal["triage", "resolution", "escalation", "orchestrator"]
    intent: Literal[
        "route_to_resolution", "route_to_escalation",
        "resolved", "escalated", "waiting_on_customer", "error"
    ]
    confidence: float       # 0.0–1.0 self-reported
    timestamp: datetime
    trace_id: str
    payload: dict           # keys: summary, actions_taken, ticket_id
```

The LLM inside each agent is required to end its response with a structured JSON block:

```
```json
{
  "intent": "...",
  "confidence": 0.9,
  "summary": "one sentence",
  "actions_taken": ["..."],
  "ticket_id": 42
}
```
```

If parsing fails, the agent returns `intent=error`, which the Orchestrator treats as a
terminal state (does not propagate a broken message downstream).

---

## Shared Context

`SharedContext` is an in-memory `asyncio.Lock`-protected dict keyed by `trace_id`.

```
trace_id -> {
  "original_request": str,
  "visited":          list[str],       # agents already called
  "ticket_id":        int | None,
  "triage_summary":   str,
  "resolution_summary": str,
}
```

Each subsequent agent receives an enriched task string built from this context so it does
not need to rediscover the ticket or repeat prior work.

`asyncio.Lock` is used (not `threading.Lock`) because the entire process runs in a single
asyncio event loop. A threading lock would block the event loop while held.

---

## Trace Logger

`TraceLogger` serves two purposes simultaneously:
1. Real-time stderr logging (same format as Exercise 2, prefixed with `[trace_id]`)
2. In-memory structured accumulation, written to `examples/<trace_id>.json` at completion

Event types recorded: `hop_start`, `reasoning`, `tool_call`, `tool_result`, `hop_end`,
`routing_decision`, `error`, `final`.

---

## Three Independent MCP Subprocesses

As required by the plan, each specialist agent owns a separate `stdio_client` connection,
meaning three independent `uv run python -m src.server` subprocesses are started per
request. This avoids any shared in-process state inside the MCP server and mirrors a
realistic distributed agent deployment.

Initialization is parallelized with `asyncio.gather` before the orchestrator starts routing.

---

## File Structure

```
exercise-3/
  pyproject.toml
  README.md
  ARCHITECTURE.md           <- this file
  src/
    __init__.py
    main.py                 <- entry point, CLI + REPL, MCP wiring
    orchestrator.py         <- Orchestrator, ROUTING_TABLE, guardrails
    messages.py             <- AgentMessage, OrchestratorResult dataclasses
    context.py              <- SharedContext with asyncio.Lock
    trace.py                <- TraceLogger (stderr + JSON accumulation)
    agents/
      __init__.py
      base.py               <- BaseSpecialistAgent (ReAct loop, tool filter, message parsing)
      triage.py             <- TriageAgent
      resolution.py         <- ResolutionAgent
      escalation.py         <- EscalationAgent
  examples/
    <trace_id>.json         <- one file per run
```

---

## How to Run

```bash
# From exercise-3/
cp ../.env.example .env      # fill in ANTHROPIC_API_KEY
uv run helpdesk-multi-agent "My VPN keeps disconnecting after the Windows update"

# Or REPL mode:
uv run helpdesk-multi-agent

# Traces are written to examples/<trace_id>.json automatically.
```

Prerequisites: PostgreSQL running (`docker compose up -d` from repo root),
exercise-1 database seeded (`cd ../exercise-1 && uv run python seed_data.py`).
