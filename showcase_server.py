"""Local showcase server for Prompt 3: single-call vs multi-agent evaluation."""

from __future__ import annotations

import argparse
import json
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any

from multi_agent_research_lab.services.llm_client import OllamaLLMClient
from multi_agent_research_lab.services.search_client import SearchClient

ROOT = Path(__file__).resolve().parent
SHOWCASE_DIR = ROOT / "showcase"
DEFAULT_MODEL = "qwen3:1.7b"
TOTAL_OUTPUT_BUDGET = 900

PROMPT_3 = """Design a complete internal lab proposal comparing a single-call LLM
system with a multi-agent LLM system on complex research tasks. Include the research
question, hypotheses, task design, datasets, a fair comparison setup, metrics, human
evaluation, statistical considerations, expected results, concrete red-team critique,
and a revised design. Explicitly address token-budget fairness, decomposition versus
inference-time gains, and ways the experiment could be gamed."""


def source_packet(task: str) -> str:
    sources = SearchClient().search(task, max_results=5)
    return "\n".join(
        f"[{index}] {item.title}: {item.snippet} ({item.url})"
        for index, item in enumerate(sources, start=1)
    )


def call_stage(
    client: OllamaLLMClient,
    name: str,
    system: str,
    user: str,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    started = perf_counter()
    response = client.complete(system, user, max_tokens=max_tokens)
    elapsed = perf_counter() - started
    return response.content, {
        "agent": name,
        "latency_seconds": round(elapsed, 3),
        "input_tokens": response.input_tokens or 0,
        "output_tokens": response.output_tokens or 0,
        "budget": max_tokens,
    }


def score_output(text: str, required_terms: list[str]) -> float:
    lower = text.casefold()
    coverage = sum(term.casefold() in lower for term in required_terms) / len(required_terms)
    structure = min(1.0, len(re.findall(r"(?m)^#{1,3}\s|^\d+\.\s|^-\s", text)) / 8)
    specificity = min(1.0, len(text.split()) / 700)
    return round((coverage * 5 + structure * 2 + specificity * 3), 1)


def run_experiment(client: OllamaLLMClient, task: str) -> dict[str, Any]:
    evidence = source_packet(task)
    common = (
        f"Evaluation task:\n{task}\n\nShared source packet:\n{evidence}\n\n"
        "Do not claim that these sources prove empirical superiority. Distinguish proposed "
        "methods from established evidence."
    )

    baseline, baseline_trace = call_stage(
        client,
        "Single-call",
        "You are a rigorous ML research lead. Produce the complete proposal in one call. "
        "Use clear Markdown headings and make every design choice operational. Stay below "
        "700 words and reserve space for the red-team critique and revised design.",
        f"{PROMPT_3}\n\n{common}",
        TOTAL_OUTPUT_BUDGET,
    )

    design, design_trace = call_stage(
        client,
        "Experimentalist",
        "You design controlled LLM experiments. Focus on hypotheses, task sampling, matched "
        "budgets, ablations, metrics, and statistical power. Return terse bullet notes only.",
        f"{PROMPT_3}\n\n{common}\n\nReturn design notes for a downstream writer.",
        180,
    )
    red_team, red_trace = call_stage(
        client,
        "Red team",
        "You are an adversarial methodology reviewer. Find confounds, unfair comparisons, "
        "gaming strategies, leakage, evaluator bias, and invalid causal claims. Return no "
        "more than 8 terse risk-and-fix bullets.",
        f"{PROMPT_3}\n\n{common}\n\nExperimentalist notes:\n{design}",
        180,
    )
    synthesis, writer_trace = call_stage(
        client,
        "Proposal writer",
        "You are a senior research writer. Synthesize the notes into a complete internal lab "
        "proposal below 550 words. Use these exact Markdown headings: Research Question, "
        "Hypotheses, Tasks and Data, Fair Comparison, Metrics and Human Evaluation, "
        "Statistical Plan, Expected Results, Red-Team Critique, Revised Experiment, Final "
        "Caveat. Preserve uncertainty, include every heading, and avoid invented facts.",
        f"{PROMPT_3}\n\n{common}\n\nDesign notes:\n{design}\n\n"
        f"Red-team findings:\n{red_team}",
        540,
    )

    terms = [
        "research question",
        "hypoth",
        "token",
        "metric",
        "human",
        "statistic",
        "red-team",
        "revised",
    ]
    multi_traces = [design_trace, red_trace, writer_trace]
    return {
        "model": client.model,
        "prompt": "Prompt 3 — Experimental Design: Single-Call vs. Multi-Agent",
        "task": task,
        "fairness": {
            "same_model": True,
            "same_sources": True,
            "temperature": 0.2,
            "single_output_budget": TOTAL_OUTPUT_BUDGET,
            "multi_output_budget": sum(item["budget"] for item in multi_traces),
        },
        "baseline": {
            "answer": baseline,
            "latency_seconds": baseline_trace["latency_seconds"],
            "input_tokens": baseline_trace["input_tokens"],
            "output_tokens": baseline_trace["output_tokens"],
            "quality_score": score_output(baseline, terms),
            "trace": [baseline_trace],
        },
        "multi_agent": {
            "answer": synthesis,
            "latency_seconds": round(
                sum(float(item["latency_seconds"]) for item in multi_traces), 3
            ),
            "input_tokens": sum(int(item["input_tokens"]) for item in multi_traces),
            "output_tokens": sum(int(item["output_tokens"]) for item in multi_traces),
            "quality_score": score_output(synthesis, terms),
            "trace": multi_traces,
            "artifacts": {"experimentalist": design, "red_team": red_team},
        },
    }


class ShowcaseHandler(SimpleHTTPRequestHandler):
    """Serve the static showcase and its local-only API."""

    server_version = "ResearchShowcase/1.0"

    def __init__(self, *args: Any, model: str, **kwargs: Any) -> None:
        self.model = model
        super().__init__(*args, directory=str(SHOWCASE_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/status":
            try:
                client = OllamaLLMClient(self.model)
                self._json(
                    {
                        "ok": True,
                        "model": self.model,
                        "installed_models": client.installed_models(),
                    }
                )
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            task = str(payload.get("task") or PROMPT_3).strip()
            if len(task) < 20 or len(task) > 4000:
                raise ValueError("Task must contain between 20 and 4,000 characters.")
            result = run_experiment(OllamaLLMClient(self.model), task)
            self._json({"ok": True, "result": result})
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    handler = lambda *items, **kwargs: ShowcaseHandler(  # noqa: E731
        *items, model=args.model, **kwargs
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Showcase: http://{args.host}:{args.port}")
    print(f"Ollama model: {args.model} (existing local model only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
