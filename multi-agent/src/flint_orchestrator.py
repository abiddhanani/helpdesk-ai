"""FlintOrchestrator — replaces hand-rolled Orchestrator with a Flint Workflow DAG.

DAG shape:
    triage ──┬── resolution ──┐
             └── escalation ──┴── synthesis

Flint handles: parallel fan-out, retries, dead-letter queue, and dashboard.
Results are parsed back into OrchestratorResult for compatibility with the
existing trace/eval infrastructure.

Workflow.run() is synchronous (spins up an embedded asyncio engine in a
thread pool internally). Call it from run_in_executor() in async contexts.
"""
from __future__ import annotations

import json
import logging

from flint_ai import Node, Workflow

from src.flint_adapter import SpecialistAgentAdapter, SynthesisAdapter
from src.messages import AgentMessage, Intent, OrchestratorResult, SynthesizedResult
from src.trace import TraceLogger

logger = logging.getLogger("helpdesk.flint_orchestrator")


class FlintOrchestrator:
    def __init__(
        self,
        triage_adapter: SpecialistAgentAdapter,
        resolution_adapter: SpecialistAgentAdapter,
        escalation_adapter: SpecialistAgentAdapter,
        synthesis_adapter: SynthesisAdapter,
        trace_logger: TraceLogger,
    ) -> None:
        self._triage = triage_adapter
        self._resolution = resolution_adapter
        self._escalation = escalation_adapter
        self._synthesis = synthesis_adapter
        self.trace = trace_logger

    def run(self, request: str, trace_id: str) -> OrchestratorResult:
        results = (
            Workflow(f"helpdesk-{trace_id}")
            .add(Node("triage", agent=self._triage, prompt=request))
            .add(
                Node(
                    "resolution",
                    agent=self._resolution,
                    prompt="Resolve this ticket.\n\nTriage context:\n{triage}",
                ).depends_on("triage")
            )
            .add(
                Node(
                    "escalation",
                    agent=self._escalation,
                    prompt="Evaluate for escalation.\n\nTriage context:\n{triage}",
                ).depends_on("triage")
            )
            .add(
                Node(
                    "synthesis",
                    agent=self._synthesis,
                    prompt=(
                        "Synthesize the following parallel branch outputs:\n\n"
                        "Resolution branch:\n{resolution}\n\n"
                        "Escalation branch:\n{escalation}"
                    ),
                ).depends_on("resolution", "escalation")
            )
            .run(verbose=True)
        )
        return self._parse_results(results, trace_id)

    def _parse_results(self, results: dict[str, str], trace_id: str) -> OrchestratorResult:
        hops: list[AgentMessage] = []

        for node_id in ("triage", "resolution", "escalation"):
            raw = results.get(node_id, "")
            if not raw:
                continue
            try:
                data = json.loads(raw)
                hops.append(AgentMessage.model_validate(data))
            except Exception:
                logger.warning("Could not parse %s output: %s", node_id, raw[:200])

        # Parse synthesis result
        synthesis_raw = results.get("synthesis", "")
        final_intent = Intent.ERROR
        final_summary = "Synthesis node produced no output"

        if synthesis_raw:
            try:
                synth = SynthesizedResult.model_validate_json(synthesis_raw)
                final_intent = synth.final_intent
                final_summary = synth.final_summary
            except Exception:
                # Fall back: use triage hop's intent if synthesis failed
                if hops:
                    triage_hop = next((h for h in hops if h.sender == "triage"), None)
                    if triage_hop:
                        final_intent = triage_hop.intent
                        final_summary = triage_hop.output.summary
                logger.warning("Could not parse synthesis output: %s", synthesis_raw[:200])

        self.trace.final(final_intent.value, len(hops))
        return OrchestratorResult(
            trace_id=trace_id,
            final_intent=final_intent,
            final_summary=final_summary,
            hops=hops,
        )
