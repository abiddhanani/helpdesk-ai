# Exercise 4 — Evaluation Framework

Batch evaluation harness for the multi-agent IT help desk system. Runs scenarios from
`test_scenarios/scenarios.yaml` (sc-001 enabled by default; others are commented out),
computes tool accuracy / task completion / efficiency / error recovery metrics, and
scores each response with an LLM judge (4 dimensions, 1-5 scale).

## Setup

```bash
# Prerequisites: PostgreSQL running and mcp-server DB seeded
docker compose up -d                                     # from repo root
cd mcp-server && uv run python seed_data.py && cd ..

# Install eval-harness deps
cd eval-harness
uv sync

# Configure environment
cp ../.env.example .env
# Edit .env — set ANTHROPIC_API_KEY
# Optional: set JUDGE_MODEL (default: claude-haiku-4-5-20251001)
```

## Run

```bash
# Run all 15 scenarios
uv run python src/harness.py

# Or with explicit scenarios path
uv run python src/harness.py --scenarios test_scenarios/scenarios.yaml
```

Reports are written to `results/report_<timestamp>.json` and `results/report_<timestamp>.md`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model for the multi-agent system |
| `ROUTER_MODEL` | `claude-haiku-4-5-20251001` | Cheap model for routing, synthesis, memory compaction |
| `JUDGE_MODEL` | `claude-haiku-4-5-20251001` | Model for LLM-as-judge scoring |
| `DATABASE_URL` | `postgresql://helpdesk:helpdesk@localhost:5432/helpdesk` | Postgres connection |
| `EXERCISE_1_DIR` | auto-detected | Override path to the `mcp-server` directory |

## Metrics

| Metric | Description |
|---|---|
| Tool recall | Fraction of expected tools that were actually called |
| Tool precision | 1 - (forbidden tools called / total tools called) |
| Tool F1 | Harmonic mean of recall and precision |
| Task completion | 0.0 / 0.5 / 1.0 — based on intent match + judge mean >= 3.0 |
| Hop efficiency | Normalized score: min_hops / actual_hops relative to max_hops |
| Error recovery | 1.0 if agent recovered from tool errors, 0.0 if not, null if no errors |

## Judge Dimensions (1-5)

- **Relevance** — Does the response address the actual request?
- **Correctness** — Are facts, ticket IDs, and agent names accurate?
- **Completeness** — Were all required actions taken?
- **Safety** — Does the response avoid hallucinating IDs or assignments?

## Scenario Coverage

| ID | Scenario | Tags |
|---|---|---|
| sc-001 | VPN KB resolution | happy-path, triage-only |
| sc-002 | Password reset KB resolution | happy-path, triage-only |
| sc-003 | macOS dev tools KB resolution | happy-path, triage-only |
| sc-004 | Printer offline KB resolution | happy-path, triage-only |
| sc-005 | New hire workstation setup | multi-hop, assignment |
| sc-006 | SSL cert expired — assignment | multi-hop, assignment |
| sc-007 | Check existing ticket status | read-only |
| sc-008 | Agent workload check | read-only |
| sc-009 | Critical production outage | full-chain, escalation |
| sc-010 | Ransomware alert | escalation, security |
| sc-011 | Invalid ticket ID | error-recovery |
| sc-012 | Already-resolved ticket inquiry | error-recovery |
| sc-013 | Assign to unavailable agent | edge-case, error-recovery |
| sc-014 | All agents at capacity — check workloads | edge-case, workload |
| sc-015 | SLA already breached — ticket 4 critical | edge-case, sla |

