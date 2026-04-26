"""Multi-agent IT Help Desk entry point — Flint Workflow DAG edition.

Uses OpenAI SDK pointed at Anthropic's OpenAI-compatible endpoint:
    OPENAI_API_KEY  = your Anthropic API key
    OPENAI_BASE_URL = https://api.anthropic.com/v1/

Usage:
    uv run helpdesk-multi-agent "My VPN is not working"
    uv run python -m src.main "My VPN is not working"
    uv run python -m src.main            # interactive REPL

Each request spins up a Flint embedded server, runs the triage → resolution +
escalation → synthesis DAG, and prints results. Dashboard: http://localhost:5160/ui/
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from mcp.client.stdio import StdioServerParameters, get_default_environment
from openai import AsyncOpenAI

from flint_ai import configure_engine, shutdown_engine

from src.agents.escalation import EscalationAgent
from src.agents.resolution import ResolutionAgent
from src.agents.triage import TriageAgent
from src.flint_adapter import SpecialistAgentAdapter, SynthesisAdapter
from src.flint_orchestrator import FlintOrchestrator
from src.llm.openai_client import OpenAILLMClient
from src.trace import TraceLogger

EXERCISE_1_DIR = Path(__file__).parent.parent.parent / "mcp-server"
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


async def run_request(
    request: str,
    orchestrator: FlintOrchestrator,
    label: str | None = None,
) -> str:
    trace_id = uuid.uuid4().hex[:8]
    trace_logger = TraceLogger(trace_id)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, orchestrator.run, request, trace_id, trace_logger
    )

    EXAMPLES_DIR.mkdir(exist_ok=True)
    filename = f"{label}_{trace_id}.json" if label else f"{trace_id}.json"
    trace_path = EXAMPLES_DIR / filename
    trace_path.write_text(
        json.dumps(
            {
                "request": request,
                "result": result.to_dict(),
                "trace": trace_logger.dump(),
            },
            indent=2,
        )
    )

    return result.final_summary


async def run() -> None:
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY (or ANTHROPIC_API_KEY) not set.", file=sys.stderr)
        sys.exit(1)

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.anthropic.com/v1/")
    agent_model = os.environ.get("AGENT_MODEL", os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    router_model = os.environ.get("ROUTER_MODEL", "claude-haiku-4-5-20251001")
    db_url = os.environ.get("DATABASE_URL", "postgresql://helpdesk:helpdesk@localhost:5432/helpdesk")

    openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    llm_client = OpenAILLMClient(openai_client)

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "src.server"],
        cwd=str(EXERCISE_1_DIR),
        env={**get_default_environment(), "DATABASE_URL": db_url},
    )

    triage_adapter = SpecialistAgentAdapter(
        "triage", TriageAgent, server_params, llm_client, agent_model, router_model
    )
    resolution_adapter = SpecialistAgentAdapter(
        "resolution", ResolutionAgent, server_params, llm_client, agent_model, router_model
    )
    escalation_adapter = SpecialistAgentAdapter(
        "escalation", EscalationAgent, server_params, llm_client, agent_model, router_model
    )
    synthesis_adapter = SynthesisAdapter(llm_client, router_model)

    configure_engine(
        agents=[triage_adapter, resolution_adapter, escalation_adapter, synthesis_adapter],
        queue_backend="memory",
        store_backend="memory",
    )

    orchestrator = FlintOrchestrator(
        triage_adapter,
        resolution_adapter,
        escalation_adapter,
        synthesis_adapter,
    )

    print(f"Multi-Agent IT Help Desk (Flint DAG). Model: {agent_model} | Router: {router_model}")
    print("Dashboard: http://localhost:5160/ui/\n")

    try:
        if len(sys.argv) > 1:
            request = " ".join(sys.argv[1:])
            summary = await run_request(request, orchestrator)
            print(f"\n{summary}\n")
            return

        print("Type 'exit' to quit.\n")

        loop = asyncio.get_event_loop()
        while True:
            try:
                print("> ", end="", flush=True)
                user_input = await loop.run_in_executor(None, sys.stdin.readline)
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            try:
                summary = await run_request(user_input, orchestrator)
                print(f"\n{summary}\n")
            except Exception as exc:
                print(f"\nError: {exc}\n", file=sys.stderr)
    finally:
        shutdown_engine()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
