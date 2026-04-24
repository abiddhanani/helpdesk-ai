"""IT Help Desk Agent Controller.

Implements a REASON -> ACT -> OBSERVE loop using the Anthropic API for reasoning
and an MCP ClientSession for tool execution.
"""
from __future__ import annotations

import src.logger as log
from anthropic import AsyncAnthropic
from anthropic.types import ToolUseBlock
from mcp.client.session import ClientSession

MAX_LOOP_ITERATIONS = 20
MAX_RETRIES = 3
SUMMARIZE_AFTER_TURNS = 5
TURNS_TO_KEEP = 2

SYSTEM_PROMPT = (
    "You are an IT help desk AI assistant with access to a ticketing system. "
    "Use the available tools to answer requests accurately. Think step by step. "
    "When a tool returns an error, read the error message carefully and correct "
    "your next call. If a request is ambiguous, ask one clarifying question. "
    "If a request cannot be fulfilled, explain why concisely."
)


class AgentController:
    def __init__(
        self,
        session: ClientSession,
        anthropic_client: AsyncAnthropic,
        model: str,
    ) -> None:
        self.session = session
        self.anthropic = anthropic_client
        self.model = model
        self.messages: list[dict] = []
        self.tools: list[dict] = []

    async def discover_tools(self) -> None:
        """Fetch available tools from the MCP server and convert to Anthropic format."""
        result = await self.session.list_tools()
        self.tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    async def run_turn(self, user_message: str) -> str:
        """Process one user turn through the REASON -> ACT -> OBSERVE loop."""
        self.messages.append({"role": "user", "content": user_message})

        for iteration in range(1, MAX_LOOP_ITERATIONS + 1):
            response = await self.anthropic.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=self.messages,
            )

            # Log any reasoning text blocks
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    log.reasoning(block.text)

            if response.stop_reason == "end_turn":
                text_blocks = [b for b in response.content if b.type == "text"]
                final_text = text_blocks[0].text if text_blocks else "(no response)"
                log.final_answer(final_text)
                self.messages.append({"role": "assistant", "content": response.content})
                await self._maybe_summarize()
                return final_text

            if response.stop_reason == "tool_use":
                self.messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result_text = await self._execute_tool(block)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text,
                            }
                        )

                self.messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason
            log.error(f"Unexpected stop_reason: {response.stop_reason}")
            break

        return "I was unable to complete the task within the allowed number of steps."

    async def _execute_tool(self, block: ToolUseBlock) -> str:
        """Execute one MCP tool call with retry on transient failures."""
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            log.tool_call(block.name, block.input)
            try:
                result = await self.session.call_tool(block.name, block.input)
                result_text = _extract_text(result.content)
                log.tool_result(block.name, result_text[:200], is_error=result.isError)

                if result.isError and attempt < MAX_RETRIES:
                    log.retry(block.name, attempt, result_text)
                    last_error = result_text
                    continue

                return result_text
            except Exception as exc:
                last_error = str(exc)
                log.error(f"Tool {block.name} raised exception on attempt {attempt}", exc)
                if attempt < MAX_RETRIES:
                    log.retry(block.name, attempt, last_error)

        return f"ERROR: Tool call failed after {MAX_RETRIES} attempts: {last_error}"

    async def _maybe_summarize(self) -> None:
        """Summarize old messages once the conversation exceeds SUMMARIZE_AFTER_TURNS turns.

        Keeps the last TURNS_TO_KEEP real user turns verbatim so the model has
        fine-grained context for the current task. Everything older is replaced
        with a bullet-point summary injected as a synthetic user/assistant exchange.
        """
        user_turn_indices = [
            i for i, m in enumerate(self.messages)
            if m["role"] == "user" and isinstance(m["content"], str)
        ]

        if len(user_turn_indices) <= SUMMARIZE_AFTER_TURNS:
            return

        keep_from = user_turn_indices[-TURNS_TO_KEEP]
        old_messages = self.messages[:keep_from]
        recent_messages = self.messages[keep_from:]

        summary = await self._summarize_messages(old_messages)
        log.reasoning(f"Memory compacted: summarized {len(old_messages)} messages into summary")

        self.messages = [
            {"role": "user", "content": f"[Conversation summary so far]: {summary}"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Understood, I have the context from our earlier conversation."}],
            },
            *recent_messages,
        ]

    async def _summarize_messages(self, messages: list[dict]) -> str:
        """Call the LLM to produce a concise bullet-point summary of old messages."""
        lines = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if isinstance(content, str):
                lines.append(f"{role}: {content}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            lines.append(f"{role}: {block['text']}")
                        elif block.get("type") == "tool_use":
                            lines.append(f"{role} called tool {block['name']}({block['input']})")
                        elif block.get("type") == "tool_result":
                            lines.append(f"tool result: {block['content']}")
                    elif hasattr(block, "type"):
                        if block.type == "text":
                            lines.append(f"{role}: {block.text}")
                        elif block.type == "tool_use":
                            lines.append(f"{role} called tool {block.name}({block.input})")

        transcript = "\n".join(lines)
        response = await self.anthropic.messages.create(
            model=self.model,
            max_tokens=512,
            system="You are a helpful assistant that summarizes conversations concisely.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize the following conversation transcript into 3-5 bullet points. "
                        "Focus on facts, decisions made, and ticket IDs mentioned.\n\n"
                        f"{transcript}"
                    ),
                }
            ],
        )
        return response.content[0].text


def _extract_text(content: list) -> str:
    parts = [block.text for block in content if hasattr(block, "text")]
    return "\n".join(parts) if parts else "(empty result)"
