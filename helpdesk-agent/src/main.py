"""IT Help Desk Agent — REPL entry point.

Connects to the Exercise 1 MCP server via stdio and runs an interactive loop.
User-facing output goes to stdout; all agent activity logs go to stderr.
"""
import asyncio
import os
import sys
from pathlib import Path

import src.logger as log
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, get_default_environment, stdio_client

from .agent import AgentController

EXERCISE_1_DIR = Path(__file__).parent.parent.parent / "mcp-server"


async def run() -> None:
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://helpdesk:helpdesk@localhost:5432/helpdesk"
    )

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "src.server"],
        cwd=str(EXERCISE_1_DIR),
        env={**get_default_environment(), "DATABASE_URL": db_url},
    )

    anthropic_client = AsyncAnthropic(api_key=api_key)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            controller = AgentController(session, anthropic_client, model)
            await controller.discover_tools()

            print(f"IT Help Desk Agent ready. Model: {model}")
            print(
                f"Loaded {len(controller.tools)} tools. "
                "Type 'exit' to quit, 'reset' to clear memory.\n"
            )

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
                if user_input.lower() == "reset":
                    controller.reset_memory()
                    print("Conversation memory cleared.\n")
                    continue

                try:
                    answer = await controller.run_turn(user_input)
                    print(f"\n{answer}\n")
                except Exception as exc:
                    log.error("Unhandled error during turn", exc)
                    print(f"\nError: {exc}\n")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
