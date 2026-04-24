"""IT Help Desk MCP Server.

Entry point for the FastMCP server. Communicates via stdio transport.
All logging goes to stderr — never stdout (stdout is reserved for JSON-RPC).
"""
import logging
import sys

from mcp.server.fastmcp import FastMCP

# Ensure all logging from this process goes to stderr, not stdout.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

from .database import init_db
from .tools.agents_tools import register_tools as register_agent_tools
from .tools.knowledge import register_tools as register_knowledge_tools
from .tools.memory_tools import register_tools as register_memory_tools
from .tools.tickets import register_tools as register_ticket_tools

# Initialise database tables and indexes on server startup.
init_db()

mcp = FastMCP("IT Help Desk")

register_ticket_tools(mcp)
register_knowledge_tools(mcp)
register_agent_tools(mcp)
register_memory_tools(mcp)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
