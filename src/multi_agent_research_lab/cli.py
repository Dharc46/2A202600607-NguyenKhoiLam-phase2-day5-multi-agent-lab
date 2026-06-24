"""Command-line entrypoint."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def run_baseline(query: str) -> ResearchState:
    """Run one LLM call over directly retrieved evidence."""

    state = ResearchState(request=ResearchQuery(query=query))
    state.sources = SearchClient().search(query, state.request.max_sources)
    evidence = "\n".join(
        f"[{index}] {source.title}: {source.snippet}"
        for index, source in enumerate(state.sources, start=1)
    )
    response = LLMClient().complete(
        "Answer the research question directly using only the supplied evidence. "
        "Use [n] citations and clearly state uncertainty.",
        f"Question: {query}\n\nEvidence:\n{evidence}",
    )
    sources = "\n".join(
        f"[{index}] {source.title} — {source.url or 'local source'}"
        for index, source in enumerate(state.sources, start=1)
    )
    state.final_answer = f"{response.content}\n\n## Sources\n{sources}"
    state.add_usage(response.input_tokens, response.output_tokens, response.cost_usd)
    state.add_trace_event("baseline.completed", {"source_count": len(state.sources)})
    return state


def run_multi_agent(query: str) -> ResearchState:
    return MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=query)))


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""

    _init()
    state = run_baseline(query)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    result = run_multi_agent(query)
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Markdown report path"),
    ] = Path("benchmark_report.md"),
) -> None:
    """Compare baseline and multi-agent executions and write a report."""

    _init()
    _, baseline_metrics = run_benchmark("baseline", query, run_baseline)
    _, multi_metrics = run_benchmark("multi-agent", query, run_multi_agent)
    report = render_markdown_report([baseline_metrics, multi_metrics])
    path = LocalArtifactStore().write_text(str(output), report)
    console.print(Panel.fit(report, title=f"Benchmark: {path}"))


if __name__ == "__main__":
    app()
