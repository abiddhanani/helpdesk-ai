"""Activity logger for the IT Help Desk agent.

All output goes to stderr — stdout is reserved for user-facing answers.
"""
import json
import logging
import sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

_logger = logging.getLogger("agent")


def reasoning(text: str) -> None:
    """Log the LLM's reasoning before a tool call."""
    _logger.info("REASON | %s", text.strip())


def tool_call(name: str, params: dict) -> None:
    """Log a tool invocation with its parameters."""
    _logger.info("ACT    | %s(%s)", name, json.dumps(params, default=str))


def tool_result(name: str, summary: str, is_error: bool = False) -> None:
    """Log the result of a tool call."""
    if is_error:
        _logger.warning("OBSERVE| %s -> %s", name, summary)
    else:
        _logger.info("OBSERVE| %s -> %s", name, summary)


def final_answer(text: str) -> None:
    """Log the agent's final answer."""
    _logger.info("ANSWER | %s", text.strip())


def retry(tool_name: str, attempt: int, reason: str) -> None:
    """Log a retry attempt."""
    _logger.warning("RETRY  | %s attempt %d: %s", tool_name, attempt, reason)


def error(message: str, exc: Exception | None = None) -> None:
    """Log an unrecoverable error."""
    if exc:
        _logger.exception("ERROR  | %s", message)
    else:
        _logger.error("ERROR  | %s", message)
