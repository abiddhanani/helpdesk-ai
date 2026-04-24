"""Run all 4 required exercise-3 scenarios and save traces to examples/.

Usage:
    uv run python run_scenarios.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from src.main import run_request

SCENARIOS = [
    (
        "scenario_1",
        (
            "Triage all open critical tickets: for each one, search the knowledge base for a "
            "potential solution. If a solution exists, add it as a comment and mark as "
            "in_progress. If no solution exists, escalate to the best available specialist."
        ),
    ),
    (
        "scenario_2",
        (
            "Agent Carol Singh is going on leave. Redistribute all her open tickets to other "
            "available agents with matching specialties, prioritizing by SLA deadline."
        ),
    ),
    (
        "scenario_3",
        (
            "Generate a support operations dashboard: tickets by status, SLA breach rate by "
            "category, agent utilization rates, and top 5 most common unresolved issue types "
            "with KB gap analysis."
        ),
    ),
    (
        "scenario_4",
        (
            "A major network outage has been reported. Create a critical ticket for this outage, "
            "find and note all existing open network tickets as potentially related (add a comment "
            "linking them), assign a network specialist, and search the KB for the network outage "
            "runbook or troubleshooting steps."
        ),
    ),
]


async def main() -> None:
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://helpdesk:helpdesk@localhost:5432/helpdesk"
    )
    client = AsyncAnthropic(api_key=api_key)

    print(f"Running {len(SCENARIOS)} scenarios with model: {model}\n")

    for label, request in SCENARIOS:
        print(f"=== {label.upper()} ===")
        print(f"Request: {request[:80]}...")
        try:
            summary = await run_request(request, client, model, db_url, label=label)
            print(f"Result:  {summary[:200]}\n")
        except Exception as exc:
            print(f"ERROR: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
