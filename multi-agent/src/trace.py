"""Per-request structured trace logger with OpenTelemetry integration.

Dual output:
  1. stderr structured logging — real-time visibility (unchanged from exercise-3)
  2. In-memory event list — written to examples/<trace_id>.json at request end
  3. OpenTelemetry spans — emitted to Console exporter by default, OTLP when
     OTEL_EXPORTER_OTLP_ENDPOINT is set

OTel span hierarchy:
    helpdesk.request (root, trace_id as trace)
      helpdesk.hop.<agent> (one per agent activation)
        helpdesk.tool.<tool_name> (one per MCP tool call, child of hop span)

Cache savings and token counts are recorded as span attributes.
Set OTEL_DISABLED=true to suppress all OTel output (e.g. in tests).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import NonRecordingSpan, Span, StatusCode

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("multi-agent")

# ---------------------------------------------------------------------------
# OTel provider setup (module-level singleton)
# ---------------------------------------------------------------------------

def _build_provider() -> TracerProvider | None:
    if os.environ.get("OTEL_DISABLED", "").lower() in ("1", "true", "yes"):
        return None

    resource = Resource.create({"service.name": "helpdesk-multi-agent"})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        except Exception:
            exporter = ConsoleSpanExporter(out=sys.stderr)
    else:
        exporter = ConsoleSpanExporter(out=sys.stderr)

    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


_PROVIDER = _build_provider()
_TRACER = _PROVIDER.get_tracer("helpdesk") if _PROVIDER else None


# ---------------------------------------------------------------------------
# TraceLogger
# ---------------------------------------------------------------------------

class TraceLogger:
    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self._entries: list[dict[str, Any]] = []

        # OTel: one root span per request
        self._root_span: Span = NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
        self._hop_spans: dict[str, Span] = {}
        self._tool_spans: dict[tuple[str, str], Span] = {}  # (agent, tool_call_id)

        if _TRACER:
            self._root_span = _TRACER.start_span(
                "helpdesk.request",
                attributes={"trace_id": trace_id},
            )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _record(self, event_type: str, **fields: Any) -> None:
        entry = {
            "trace_id": self.trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **fields,
        }
        self._entries.append(entry)

    def _hop_span(self, agent: str) -> Span:
        return self._hop_spans.get(agent, self._root_span)

    # -----------------------------------------------------------------------
    # Existing event methods (unchanged API)
    # -----------------------------------------------------------------------

    def hop_start(self, agent: str, task_preview: str) -> None:
        _logger.info("[%s] HOP    | %s <- %s", self.trace_id, agent, task_preview[:120])
        self._record("hop_start", agent=agent, task_preview=task_preview[:120])
        if _TRACER:
            span = _TRACER.start_span(
                f"helpdesk.hop.{agent}",
                context=trace.set_span_in_context(self._root_span),
                attributes={"agent": agent, "task_preview": task_preview[:200]},
            )
            self._hop_spans[agent] = span

    def reasoning(self, agent: str, text: str) -> None:
        _logger.info("[%s] REASON | %s: %s", self.trace_id, agent, text.strip()[:200])
        self._record("reasoning", agent=agent, text=text.strip())
        span = self._hop_spans.get(agent)
        if span:
            span.add_event("reasoning", {"text": text.strip()[:500]})

    def tool_call(self, agent: str, tool: str, params: dict) -> None:
        _logger.info(
            "[%s] ACT    | %s -> %s(%s)",
            self.trace_id, agent, tool, json.dumps(params, default=str)[:200],
        )
        self._record("tool_call", agent=agent, tool=tool, params=params)
        if _TRACER:
            parent = self._hop_spans.get(agent, self._root_span)
            span = _TRACER.start_span(
                f"helpdesk.tool.{tool}",
                context=trace.set_span_in_context(parent),
                attributes={"agent": agent, "tool": tool},
            )
            self._tool_spans[(agent, tool)] = span

    def tool_result(self, agent: str, tool: str, summary: str, is_error: bool) -> None:
        level = logging.WARNING if is_error else logging.INFO
        _logger.log(
            level, "[%s] OBSERVE| %s <- %s: %s",
            self.trace_id, agent, tool, summary[:200],
        )
        self._record(
            "tool_result", agent=agent, tool=tool, summary=summary[:500], is_error=is_error,
        )
        span = self._tool_spans.pop((agent, tool), None)
        if span:
            if is_error:
                span.set_status(StatusCode.ERROR, summary[:200])
            span.add_event("result", {"summary": summary[:500], "is_error": is_error})
            span.end()

    def hop_end(self, agent: str, intent: str, confidence: float, summary: str) -> None:
        _logger.info(
            "[%s] RESULT | %s -> intent=%s confidence=%.2f | %s",
            self.trace_id, agent, intent, confidence, summary[:200],
        )
        self._record("hop_end", agent=agent, intent=intent, confidence=confidence, summary=summary)
        span = self._hop_spans.pop(agent, None)
        if span:
            span.set_attributes({"intent": intent, "confidence": confidence, "summary": summary[:500]})
            span.end()

    def routing_decision(self, from_agent: str, to_agent: str | None, reason: str) -> None:
        destination = to_agent or "(terminal)"
        _logger.info("[%s] ROUTE  | %s -> %s | %s", self.trace_id, from_agent, destination, reason)
        self._record("routing_decision", from_agent=from_agent, to_agent=to_agent, reason=reason)
        self._root_span.add_event(
            "routing_decision",
            {"from": from_agent, "to": destination, "reason": reason[:200]},
        )

    def error(self, message: str) -> None:
        _logger.error("[%s] ERROR  | %s", self.trace_id, message)
        self._record("error", message=message)
        self._root_span.add_event("error", {"message": message[:500]})
        self._root_span.set_status(StatusCode.ERROR, message[:200])

    def llm_tokens(
        self, agent: str, model: str, input_tokens: int, output_tokens: int,
        cache_read_tokens: int = 0, cache_write_tokens: int = 0,
    ) -> None:
        if cache_read_tokens or cache_write_tokens:
            _logger.info(
                "[%s] CACHE  | %s: read=%d write=%d (saved ~$%.4f)",
                self.trace_id, agent, cache_read_tokens, cache_write_tokens,
                cache_read_tokens * 0.0000003,
            )
        self._record(
            "llm_tokens", agent=agent, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens,
        )
        span = self._hop_spans.get(agent, self._root_span)
        span.set_attributes({
            f"{agent}.input_tokens": input_tokens,
            f"{agent}.output_tokens": output_tokens,
            f"{agent}.cache_read_tokens": cache_read_tokens,
            f"{agent}.cache_write_tokens": cache_write_tokens,
            f"{agent}.model": model,
        })

    def final(self, outcome: str, total_hops: int) -> None:
        _logger.info("[%s] FINAL  | outcome=%s hops=%d", self.trace_id, outcome, total_hops)
        self._record("final", outcome=outcome, total_hops=total_hops)
        self._root_span.set_attributes({"outcome": outcome, "total_hops": total_hops})
        self._root_span.end()

    # -----------------------------------------------------------------------
    # New event methods (added in production upgrade)
    # -----------------------------------------------------------------------

    def routing_llm(
        self, from_agent: str, next_agent: str | None, confidence: float,
        reasoning: str, elapsed_ms: float,
    ) -> None:
        destination = next_agent or "(terminal)"
        _logger.info(
            "[%s] ROUTER | %s -> %s (conf=%.2f, %.0fms) | %s",
            self.trace_id, from_agent, destination, confidence, elapsed_ms, reasoning[:120],
        )
        self._record(
            "routing_llm", from_agent=from_agent, next_agent=next_agent,
            confidence=confidence, reasoning=reasoning, elapsed_ms=elapsed_ms,
        )
        self._root_span.add_event(
            "routing_llm",
            {"from": from_agent, "to": destination, "confidence": confidence,
             "reasoning": reasoning[:200], "elapsed_ms": elapsed_ms},
        )

    def retrieval(
        self, agent: str, query: str, result_ids: list[int],
        rrf_scores: list[float], elapsed_ms: float,
    ) -> None:
        _logger.info(
            "[%s] RETRIEV| %s: '%s' -> top=%s", self.trace_id, agent, query[:60], result_ids[:5],
        )
        self._record(
            "retrieval", agent=agent, query=query, result_ids=result_ids,
            rrf_scores=rrf_scores, elapsed_ms=elapsed_ms,
        )
        span = self._hop_spans.get(agent, self._root_span)
        span.add_event(
            "retrieval",
            {"query": query[:200], "result_ids": str(result_ids[:5]),
             "top_rrf_score": rrf_scores[0] if rrf_scores else 0.0, "elapsed_ms": elapsed_ms},
        )

    def parallel_branch(
        self, agent: str, elapsed_ms: float, intent: str, error: str | None = None,
    ) -> None:
        status = f"error={error}" if error else f"intent={intent}"
        _logger.info(
            "[%s] BRANCH | %s completed in %.0fms | %s", self.trace_id, agent, elapsed_ms, status,
        )
        self._record(
            "parallel_branch", agent=agent, elapsed_ms=elapsed_ms, intent=intent, error=error,
        )
        self._root_span.add_event(
            "parallel_branch",
            {"agent": agent, "elapsed_ms": elapsed_ms, "intent": intent,
             "error": error or ""},
        )

    def memory_compact(self, agent: str, evicted_tokens: int, new_fact_tokens: int) -> None:
        _logger.info(
            "[%s] MEMORY | %s compacted %d tokens -> %d fact tokens",
            self.trace_id, agent, evicted_tokens, new_fact_tokens,
        )
        self._record(
            "memory_compact", agent=agent,
            evicted_tokens=evicted_tokens, new_fact_tokens=new_fact_tokens,
        )
        span = self._hop_spans.get(agent, self._root_span)
        span.add_event(
            "memory_compact",
            {"evicted_tokens": evicted_tokens, "new_fact_tokens": new_fact_tokens},
        )

    # -----------------------------------------------------------------------
    # Dump for JSON trace file
    # -----------------------------------------------------------------------

    def dump(self) -> list[dict[str, Any]]:
        return list(self._entries)
