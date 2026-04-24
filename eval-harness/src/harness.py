"""Evaluation harness for the multi-agent system.

Loads scenarios from YAML, runs each through exercise-3's multi-agent pipeline,
computes metrics, scores with an LLM judge, and writes JSON + Markdown reports.

Run as a script (not a module) to avoid src namespace conflicts:
    uv run python src/harness.py
    uv run python src/harness.py --scenarios test_scenarios/scenarios.yaml
"""
from __future__ import annotations

# Path setup must happen before any cross-exercise imports.
# This file is executed as a script — exercise-4/src/ is sys.path[0].
# We insert exercise-3/ at position 0 so that:
#   - `from src.X import Y`  resolves to exercise-3's src
#   - `from judge import Z`  resolves to exercise-4/src/judge.py (sys.path[1])
#   - `from metrics import W` resolves to exercise-4/src/metrics.py
import sys
from pathlib import Path

_EX3_DIR = Path(__file__).parent.parent.parent / "multi-agent"
_EX4_SRC = Path(__file__).parent
sys.path.insert(0, str(_EX4_SRC))
sys.path.insert(0, str(_EX3_DIR))

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import uuid  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import yaml  # noqa: E402
from anthropic import AsyncAnthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from mcp.client.session import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, get_default_environment, stdio_client  # noqa: E402

# exercise-3 components (resolved via _EX3_DIR on sys.path)
from src.agents.escalation import EscalationAgent  # noqa: E402
from src.agents.resolution import ResolutionAgent  # noqa: E402
from src.agents.triage import TriageAgent  # noqa: E402
from src.context import SharedContext  # noqa: E402
from src.orchestrator import Orchestrator  # noqa: E402
from src.trace import TraceLogger  # noqa: E402

# exercise-4 local modules (resolved via _EX4_SRC on sys.path)
from judge import JudgeScores, LLMJudge  # noqa: E402
from metrics import ScenarioMetrics, compute_metrics  # noqa: E402

EXERCISE_1_DIR = Path(
    os.environ.get("EXERCISE_1_DIR", str(Path(__file__).parent.parent.parent / "mcp-server"))
)
RESULTS_DIR = Path(__file__).parent.parent / "results"
SCENARIOS_PATH = Path(__file__).parent.parent / "test_scenarios" / "scenarios.yaml"


# Pricing in USD per token (input, output) keyed by model prefix.
# Keys use startswith matching, so "claude-sonnet-4" covers claude-sonnet-4-5, 4-6, etc.
# Prices as of April 2026: https://www.anthropic.com/pricing
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Claude 4 family
    "claude-opus-4":            (15.00 / 1_000_000, 75.00 / 1_000_000),
    "claude-sonnet-4":          ( 3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-haiku-4":           ( 0.80 / 1_000_000,  4.00 / 1_000_000),
    # Claude 3.x family
    "claude-opus-3":            (15.00 / 1_000_000, 75.00 / 1_000_000),
    "claude-sonnet-3-7":        ( 3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-sonnet-3-5":        ( 3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-haiku-3-5":         ( 0.80 / 1_000_000,  4.00 / 1_000_000),
    "claude-haiku-3":           ( 0.25 / 1_000_000,  1.25 / 1_000_000),
}


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    for prefix, (in_price, out_price) in _MODEL_PRICING.items():
        if model.startswith(prefix):
            return input_tokens * in_price + output_tokens * out_price
    # Unknown model — return 0 rather than crashing
    return 0.0


@dataclass
class TokenUsage:
    agent_input_tokens: int
    agent_output_tokens: int
    judge_input_tokens: int
    judge_output_tokens: int
    agent_cost_usd: float
    judge_cost_usd: float

    @property
    def total_input_tokens(self) -> int:
        return self.agent_input_tokens + self.judge_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self.agent_output_tokens + self.judge_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return self.agent_cost_usd + self.judge_cost_usd

    def to_dict(self) -> dict:
        return {
            "agent_input_tokens": self.agent_input_tokens,
            "agent_output_tokens": self.agent_output_tokens,
            "judge_input_tokens": self.judge_input_tokens,
            "judge_output_tokens": self.judge_output_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "agent_cost_usd": round(self.agent_cost_usd, 6),
            "judge_cost_usd": round(self.judge_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


@dataclass
class ScenarioResult:
    scenario_id: str
    description: str
    request: str
    final_intent: str
    final_summary: str
    total_hops: int
    trace: list[dict]
    metrics: ScenarioMetrics
    judge_scores: JudgeScores
    token_usage: TokenUsage
    error: str | None

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "request": self.request,
            "final_intent": self.final_intent,
            "final_summary": self.final_summary,
            "total_hops": self.total_hops,
            "metrics": self.metrics.to_dict(),
            "judge_scores": self.judge_scores.to_dict(),
            "token_usage": self.token_usage.to_dict(),
            "error": self.error,
        }


class EvalHarness:
    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        model: str,
        judge_model: str,
        db_url: str,
    ) -> None:
        self.anthropic = anthropic_client
        self.model = model
        self.judge = LLMJudge(anthropic_client, judge_model)
        self.db_url = db_url

    async def run(self, scenarios_path: Path, results_dir: Path) -> None:
        scenarios = self._load_scenarios(scenarios_path)
        print(f"Running {len(scenarios)} scenarios against the multi-agent system...\n")

        results: list[ScenarioResult] = []
        for i, scenario in enumerate(scenarios, 1):
            sid = scenario.get("id", f"sc-{i:03d}")
            print(f"[{i}/{len(scenarios)}] {sid} — {scenario['description']}")
            result = await self._run_scenario(scenario)
            results.append(result)

            status = "PASS" if result.metrics.intent_match else "FAIL"
            print(
                f"         {status} | intent={result.final_intent} "
                f"hops={result.total_hops} "
                f"f1={result.metrics.tool_f1:.2f} "
                f"judge={result.judge_scores.mean:.1f}/5 "
                f"cost=${result.token_usage.total_cost_usd:.4f}"
            )
            if result.error:
                print(f"         ERROR: {result.error}")

        agg = self._aggregate(results)
        results_dir.mkdir(exist_ok=True)
        self._write_report(results, agg, results_dir)

    async def _run_scenario(self, scenario: dict) -> ScenarioResult:
        trace_id = uuid.uuid4().hex[:8]
        trace_logger = TraceLogger(trace_id)
        context = SharedContext()
        error: str | None = None
        final_intent = "error"
        final_summary = ""
        total_hops = 0
        trace: list[dict] = []

        server_params = StdioServerParameters(
            command="uv",
            args=["run", "python", "-m", "src.server"],
            cwd=str(EXERCISE_1_DIR),
            env={**get_default_environment(), "DATABASE_URL": self.db_url},
        )

        try:
            async with stdio_client(server_params) as (r1, w1):
                async with stdio_client(server_params) as (r2, w2):
                    async with stdio_client(server_params) as (r3, w3):
                        async with ClientSession(r1, w1) as s1:
                            async with ClientSession(r2, w2) as s2:
                                async with ClientSession(r3, w3) as s3:
                                    await asyncio.gather(
                                        s1.initialize(), s2.initialize(), s3.initialize()
                                    )

                                    triage = TriageAgent(s1, self.anthropic, self.model, trace_logger, context)
                                    resolution = ResolutionAgent(s2, self.anthropic, self.model, trace_logger, context)
                                    escalation = EscalationAgent(s3, self.anthropic, self.model, trace_logger, context)

                                    await asyncio.gather(
                                        triage.discover_tools(),
                                        resolution.discover_tools(),
                                        escalation.discover_tools(),
                                    )

                                    orchestrator = Orchestrator(
                                        triage, resolution, escalation, context, trace_logger
                                    )
                                    result = await orchestrator.run(scenario["request"], trace_id)

                                    final_intent = result.final_intent
                                    final_summary = result.final_summary
                                    total_hops = result.total_hops
                                    trace = trace_logger.dump()

        except Exception as exc:
            if isinstance(exc, BaseExceptionGroup):
                error = f"{type(exc.exceptions[0]).__name__}: {exc.exceptions[0]}"
            else:
                error = f"{type(exc).__name__}: {exc}"
            trace = trace_logger.dump()

        metrics = compute_metrics(scenario, final_intent, total_hops, trace)

        # Accumulate agent token usage from trace events
        token_events = [e for e in trace if e.get("event_type") == "llm_tokens"]
        agent_input = sum(e["input_tokens"] for e in token_events)
        agent_output = sum(e["output_tokens"] for e in token_events)
        agent_cost = sum(
            _cost_usd(e["model"], e["input_tokens"], e["output_tokens"])
            for e in token_events
        )

        tool_calls = [e["tool"] for e in trace if e.get("event_type") == "tool_call"]
        judge_input = 0
        judge_output = 0
        try:
            judge_scores, judge_input, judge_output = await self.judge.score(
                scenario, final_summary, tool_calls
            )
        except Exception as exc:
            judge_scores = JudgeScores(
                relevance=0, correctness=0, completeness=0, safety=0,
                mean=0.0, reasoning=f"Judge call failed: {exc}"
            )

        judge_cost = _cost_usd(self.judge.model, judge_input, judge_output)
        token_usage = TokenUsage(
            agent_input_tokens=agent_input,
            agent_output_tokens=agent_output,
            judge_input_tokens=judge_input,
            judge_output_tokens=judge_output,
            agent_cost_usd=agent_cost,
            judge_cost_usd=judge_cost,
        )

        # Finalize task_completion now that judge scores are available
        if metrics.intent_match and judge_scores.mean >= 3.0:
            metrics.task_completion = 1.0
        elif metrics.intent_match:
            metrics.task_completion = 0.5
        else:
            metrics.task_completion = 0.0

        return ScenarioResult(
            scenario_id=scenario.get("id", ""),
            description=scenario.get("description", ""),
            request=scenario["request"],
            final_intent=final_intent,
            final_summary=final_summary,
            total_hops=total_hops,
            trace=trace,
            metrics=metrics,
            judge_scores=judge_scores,
            token_usage=token_usage,
            error=error,
        )

    def _aggregate(self, results: list[ScenarioResult]) -> dict:
        def mean(values: list[float]) -> float:
            return round(sum(values) / len(values), 3) if values else 0.0

        valid = [r for r in results if r.error is None]
        recovery_values = [
            r.metrics.error_recovery
            for r in valid
            if r.metrics.error_recovery is not None
        ]

        total_agent_input = sum(r.token_usage.agent_input_tokens for r in results)
        total_agent_output = sum(r.token_usage.agent_output_tokens for r in results)
        total_judge_input = sum(r.token_usage.judge_input_tokens for r in results)
        total_judge_output = sum(r.token_usage.judge_output_tokens for r in results)
        total_agent_cost = sum(r.token_usage.agent_cost_usd for r in results)
        total_judge_cost = sum(r.token_usage.judge_cost_usd for r in results)

        return {
            "total_scenarios": len(results),
            "scenarios_with_errors": sum(1 for r in results if r.error),
            "intent_match_rate": mean([float(r.metrics.intent_match) for r in valid]),
            "task_completion_mean": mean([r.metrics.task_completion for r in valid]),
            "tool_f1_mean": mean([r.metrics.tool_f1 for r in valid]),
            "hop_efficiency_mean": mean([r.metrics.hop_efficiency for r in valid]),
            "error_recovery_rate": mean(recovery_values) if recovery_values else None,
            "judge_relevance_mean": mean([r.judge_scores.relevance for r in valid]),
            "judge_correctness_mean": mean([r.judge_scores.correctness for r in valid]),
            "judge_completeness_mean": mean([r.judge_scores.completeness for r in valid]),
            "judge_safety_mean": mean([r.judge_scores.safety for r in valid]),
            "judge_mean_overall": mean([r.judge_scores.mean for r in valid]),
            "tokens": {
                "agent_input": total_agent_input,
                "agent_output": total_agent_output,
                "judge_input": total_judge_input,
                "judge_output": total_judge_output,
                "total_input": total_agent_input + total_judge_input,
                "total_output": total_agent_output + total_judge_output,
            },
            "cost_usd": {
                "agent": round(total_agent_cost, 4),
                "judge": round(total_judge_cost, 4),
                "total": round(total_agent_cost + total_judge_cost, 4),
            },
        }

    def _write_report(
        self, results: list[ScenarioResult], agg: dict, results_dir: Path
    ) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        json_path = results_dir / f"report_{ts}.json"
        json_path.write_text(
            json.dumps(
                {
                    "generated_at": ts,
                    "aggregate": agg,
                    "scenarios": [r.to_dict() for r in results],
                },
                indent=2,
            )
        )

        md_path = results_dir / f"report_{ts}.md"
        md_path.write_text(self._render_markdown(results, agg, ts))

        print(f"\nReport written to:\n  {json_path}\n  {md_path}")

    def _render_markdown(self, results: list[ScenarioResult], agg: dict, ts: str) -> str:
        tok = agg["tokens"]
        cost = agg["cost_usd"]
        lines = [
            f"# Evaluation Report — {ts}",
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Score |",
            "|---|---|",
            f"| Total scenarios | {agg['total_scenarios']} |",
            f"| Scenarios with errors | {agg['scenarios_with_errors']} |",
            f"| Intent match rate | {agg['intent_match_rate']:.0%} |",
            f"| Task completion (mean) | {agg['task_completion_mean']:.2f} |",
            f"| Tool F1 (mean) | {agg['tool_f1_mean']:.2f} |",
            f"| Hop efficiency (mean) | {agg['hop_efficiency_mean']:.2f} |",
        ]
        if agg["error_recovery_rate"] is not None:
            lines.append(f"| Error recovery rate | {agg['error_recovery_rate']:.0%} |")
        lines += [
            f"| Judge: Relevance | {agg['judge_relevance_mean']:.1f}/5 |",
            f"| Judge: Correctness | {agg['judge_correctness_mean']:.1f}/5 |",
            f"| Judge: Completeness | {agg['judge_completeness_mean']:.1f}/5 |",
            f"| Judge: Safety | {agg['judge_safety_mean']:.1f}/5 |",
            f"| Judge: Overall mean | {agg['judge_mean_overall']:.1f}/5 |",
            "",
            "## Token Usage & Cost",
            "",
            "| | Input tokens | Output tokens | Cost (USD) |",
            "|---|---|---|---|",
            f"| Agents | {tok['agent_input']:,} | {tok['agent_output']:,} | ${cost['agent']:.4f} |",
            f"| Judge | {tok['judge_input']:,} | {tok['judge_output']:,} | ${cost['judge']:.4f} |",
            f"| **Total** | **{tok['total_input']:,}** | **{tok['total_output']:,}** | **${cost['total']:.4f}** |",
            "",
            "---",
            "",
            "## Per-Scenario Results",
            "",
        ]

        for r in results:
            intent_note = "match" if r.metrics.intent_match else "mismatch"
            status = "PASS" if r.metrics.intent_match else "FAIL"
            u = r.token_usage
            lines += [
                f"### {r.scenario_id} — {r.description} [{status}]",
                "",
                f"- **Request:** {r.request.strip()}",
                f"- **Final intent:** `{r.final_intent}` ({intent_note})",
                f"- **Hops:** {r.metrics.actual_hops} | **Tool calls:** {r.metrics.actual_tool_calls}",
                f"- **Tool F1:** {r.metrics.tool_f1:.2f} | **Hop efficiency:** {r.metrics.hop_efficiency:.2f} | **Task completion:** {r.metrics.task_completion}",
                f"- **Judge:** rel={r.judge_scores.relevance} cor={r.judge_scores.correctness} com={r.judge_scores.completeness} safe={r.judge_scores.safety} mean={r.judge_scores.mean:.1f} | _{r.judge_scores.reasoning}_",
                f"- **Tokens:** agents {u.agent_input_tokens:,}in / {u.agent_output_tokens:,}out (${u.agent_cost_usd:.4f}) | judge {u.judge_input_tokens:,}in / {u.judge_output_tokens:,}out (${u.judge_cost_usd:.4f}) | total ${u.total_cost_usd:.4f}",
            ]
            if r.error:
                lines.append(f"- **Error:** `{r.error}`")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _load_scenarios(path: Path) -> list[dict]:
        with path.open() as f:
            return yaml.safe_load(f)


async def run() -> None:
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    judge_model = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5-20251001")
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://helpdesk:helpdesk@localhost:5432/helpdesk"
    )

    scenarios_path = SCENARIOS_PATH
    if "--scenarios" in sys.argv:
        idx = sys.argv.index("--scenarios")
        scenarios_path = Path(sys.argv[idx + 1])

    anthropic_client = AsyncAnthropic(api_key=api_key)
    harness = EvalHarness(anthropic_client, model, judge_model, db_url)
    await harness.run(scenarios_path, RESULTS_DIR)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
