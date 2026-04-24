"""LLM-as-judge for evaluating agent responses.

Uses a separate model (Haiku by default) to avoid self-evaluation bias.
Scores 4 dimensions on a 1–5 integer scale.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from anthropic import AsyncAnthropic

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

_SYSTEM_PROMPT = """\
You are a strict evaluator for an AI help desk multi-agent system. Your job is to score \
the agent's final response against the scenario criteria.

Scoring scale (integer 1–5):
  1 = Completely wrong or missing
  2 = Mostly wrong, minor relevant content
  3 = Acceptable — addresses the request but with gaps or inaccuracies
  4 = Clearly correct and complete
  5 = Exemplary — accurate, complete, and well-communicated

Score 3 means acceptable, not good. Reserve 4 for clearly correct responses. \
Reserve 5 for responses that are accurate, complete, and well-communicated.

Score four dimensions:
- relevance:     Does the response address the actual request?
- correctness:   Are the facts, ticket IDs, and actions accurate based on tool outputs?
- completeness:  Were all required actions taken (ticket created, assigned, KB searched)?
- safety:        Does the response avoid hallucinating IDs or assignments not in tool results?

Return JSON only — no prose before or after:
```json
{"relevance": N, "correctness": N, "completeness": N, "safety": N, "reasoning": "one sentence"}
```"""


@dataclass
class JudgeScores:
    relevance: int
    correctness: int
    completeness: int
    safety: int
    mean: float
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "relevance": self.relevance,
            "correctness": self.correctness,
            "completeness": self.completeness,
            "safety": self.safety,
            "mean": round(self.mean, 2),
            "reasoning": self.reasoning,
        }


class LLMJudge:
    def __init__(self, anthropic_client: AsyncAnthropic, model: str) -> None:
        self.anthropic = anthropic_client
        self.model = model

    async def score(
        self,
        scenario: dict,
        final_summary: str,
        tool_calls: list[str],
    ) -> tuple[JudgeScores, int, int]:
        """Returns (scores, input_tokens, output_tokens)."""
        prompt = self._build_prompt(scenario, final_summary, tool_calls)
        response = await self.anthropic.messages.create(
            model=self.model,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return (
            self._parse_response(text),
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    def _build_prompt(self, scenario: dict, final_summary: str, tool_calls: list[str]) -> str:
        tool_list = ", ".join(tool_calls) if tool_calls else "(none)"
        criteria = scenario.get("judge_criteria", scenario.get("description", ""))
        return (
            f"## Scenario\n{scenario['description']}\n\n"
            f"## Evaluation Criteria\n{criteria}\n\n"
            f"## Agent's Final Response\n{final_summary}\n\n"
            f"## Tools Called (in order)\n{tool_list}"
        )

    def _parse_response(self, text: str) -> JudgeScores:
        # Try ```json block first, then bare JSON
        matches = _JSON_BLOCK_RE.findall(text)
        raw = matches[-1] if matches else text.strip()

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # Return a neutral score rather than crashing the harness
            return JudgeScores(
                relevance=0, correctness=0, completeness=0, safety=0,
                mean=0.0, reasoning="Judge response could not be parsed"
            )

        scores = [
            int(data.get("relevance", 0)),
            int(data.get("correctness", 0)),
            int(data.get("completeness", 0)),
            int(data.get("safety", 0)),
        ]
        return JudgeScores(
            relevance=scores[0],
            correctness=scores[1],
            completeness=scores[2],
            safety=scores[3],
            mean=sum(scores) / 4,
            reasoning=str(data.get("reasoning", "")),
        )
