"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality proxy, citation coverage, and errors."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    answer = state.final_answer or ""
    citations = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    valid_citations = {value for value in citations if 1 <= value <= len(state.sources)}
    citation_coverage = (
        min(1.0, len(valid_citations) / len(state.sources)) if state.sources else 0.0
    )
    quality = _quality_score(answer, citation_coverage, bool(state.errors))
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=state.estimated_cost_usd,
        quality_score=quality,
        citation_coverage=citation_coverage,
        error_rate=1.0 if state.errors else 0.0,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        notes="; ".join(state.errors),
    )
    return state, metrics


def _quality_score(answer: str, citation_coverage: float, has_errors: bool) -> float:
    """Transparent deterministic proxy; peer or model judging can replace it."""

    if not answer:
        return 0.0
    word_count = len(answer.split())
    completeness = min(4.0, word_count / 75)
    structure = 2.0 if any(marker in answer for marker in ("##", "\n-", "\n1.")) else 1.0
    grounding = citation_coverage * 3.0
    penalty = 1.0 if has_errors else 0.0
    return round(max(0.0, min(10.0, completeness + structure + grounding - penalty)), 2)
