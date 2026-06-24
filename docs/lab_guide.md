# Lab Guide: Multi-Agent Research System

## Implemented system

The repository compares:

1. A single-agent baseline that retrieves evidence and performs one synthesis call.
2. A multi-agent workflow with Supervisor, Researcher, Analyst, Writer, and Critic stages.

OpenAI and Tavily are used when their API keys are configured. Deterministic local
providers keep development and tests functional without network access.

## Workflow

The supervisor routes to the first missing artifact:

`researcher -> analyst -> writer -> critic -> done`

Shared Pydantic state records sources, notes, output, usage, errors, routes, and trace
events. Maximum iterations, timeouts, retries, fallbacks, and citation validation constrain
failure behavior.

## Running

```bash
python -m multi_agent_research_lab.cli baseline --query "Explain GraphRAG"
python -m multi_agent_research_lab.cli multi-agent --query "Explain GraphRAG"
python -m multi_agent_research_lab.cli benchmark --query "Explain GraphRAG"
```

The benchmark report is written under `reports/`; workflow traces are appended to
`reports/traces.jsonl`.

## Evaluation

Compare latency, estimated cost, token use, deterministic quality proxy, citation coverage,
and error rate. Apply `docs/peer_review_rubric.md` for final human quality scoring.

## Exit ticket

Use multi-agent orchestration when the task benefits from specialized stages, independent
validation, or detailed observability. Prefer a single agent for simple, low-risk tasks
where orchestration latency and cost exceed the likely quality gain.
