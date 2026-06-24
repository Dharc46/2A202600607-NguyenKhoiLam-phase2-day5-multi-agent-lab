"""Tracing hooks with JSONL export and optional OpenTelemetry spans."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Create a local span and mirror it to OpenTelemetry when installed."""

    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "started_at": datetime.now(UTC).isoformat(),
        "duration_seconds": None,
        "status": "ok",
    }
    otel_context: Any = None
    try:
        trace_module = import_module("opentelemetry.trace")
        tracer = trace_module.get_tracer("multi_agent_research_lab")
        otel_context = tracer.start_as_current_span(name)
        otel_context.__enter__()
    except ImportError:
        otel_context = None
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = f"{type(exc).__name__}: {exc}"
        if otel_context is not None:
            otel_context.__exit__(type(exc), exc, exc.__traceback__)
            otel_context = None
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        if otel_context is not None:
            otel_context.__exit__(None, None, None)


def export_trace(trace: list[dict[str, Any]], path: str | Path) -> Path:
    """Append one complete workflow trace as a JSONL record."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"exported_at": datetime.now(UTC).isoformat(), "events": trace}
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return output
