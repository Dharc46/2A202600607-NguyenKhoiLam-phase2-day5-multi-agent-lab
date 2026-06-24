"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render comparable metrics and a concise interpretation."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality /10 | Citation coverage | "
        "Error rate | Tokens (in/out) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        coverage = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | "
            f"{coverage} | {item.error_rate:.0%} | {item.input_tokens}/{item.output_tokens} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if metrics:
        best_quality = max(metrics, key=lambda item: item.quality_score or 0)
        fastest = min(metrics, key=lambda item: item.latency_seconds)
        lines.append(
            f"- Highest quality proxy: **{best_quality.run_name}** "
            f"({best_quality.quality_score or 0:.1f}/10)."
        )
        lines.append(
            f"- Lowest latency: **{fastest.run_name}** ({fastest.latency_seconds:.2f}s)."
        )
    lines.extend(
        [
            "- Quality is a deterministic proxy based on completeness, structure, citations, "
            "and errors; use the peer-review rubric for final human scoring.",
            "- Trace records are stored in `reports/traces.jsonl`.",
            "",
            "## Failure modes and mitigations",
            "",
            "- Search/provider failure: retry twice, then continue in explicit degraded mode.",
            "- Infinite routing: bounded by `MAX_ITERATIONS` and workflow recursion limits.",
            "- Slow provider: request and workflow timeouts stop execution.",
            "- Hallucinated citations: the critic checks that inline markers resolve to "
            "known sources.",
        ]
    )
    return "\n".join(lines) + "\n"
