# Exercise 3 — Multi-Agent System

Sequential multi-agent pipeline: Triage → Resolution → Escalation, orchestrated with
LLM-driven routing decisions and rule-based guardrails.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design documentation.

## Setup

```bash
# From repo root — start PostgreSQL and seed data (if not already done)
docker compose up -d
cd exercise-1 && uv run python seed_data.py && cd ..

# Install exercise-3 deps
cd exercise-3
uv sync

# Configure environment
cp ../.env.example .env
# Edit .env — set ANTHROPIC_API_KEY
```

## Run

```bash
# Single request
uv run helpdesk-multi-agent "My VPN keeps disconnecting after the Windows update"

# Interactive REPL
uv run helpdesk-multi-agent
```

Structured traces are written to `examples/<trace_id>.json` after each request.

## Example Scenarios

| File | Request | Agents invoked |
|------|---------|----------------|
| `examples/scenario_1_*.json` | *"Triage all open critical tickets: for each one, search the knowledge base for a potential solution. If a solution exists, add it as a comment and mark as in_progress. If no solution exists, escalate to the best available specialist."* | Triage → Resolution → Escalation (conditional) |
| `examples/scenario_2_*.json` | *"Agent Carol Singh is going on leave. Redistribute all her open tickets to other available agents with matching specialties, prioritizing by SLA deadline."* | Triage → Escalation (workload analysis) |
| `examples/scenario_3_*.json` | *"Generate a support operations dashboard: tickets by status, SLA breach rate by category, agent utilization rates, and top 5 most common unresolved issue types with KB gap analysis."* | Triage → Resolution (KB gap analysis) → Escalation (SLA + workload data) |
| `examples/scenario_4_*.json` | *"A major network outage has been reported. Create a critical ticket, find and note all existing open network tickets as potentially related, assign a network specialist, and search the KB for the network outage runbook."* | Triage → Resolution (KB + ticket linking) → Escalation (assignment) |
| `examples/error_handling_circular_guard.json` | Ambiguous ticket request that triggers the circular routing guard (demonstrates error handling requirement) | Triage → Resolution → Escalation (guard override) |
