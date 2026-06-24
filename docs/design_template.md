# System Design

## Problem

The system answers open-ended research questions by collecting evidence, comparing claims,
and producing a cited answer for technical learners.

## Why multi-agent?

The baseline is useful for simple questions, but a multi-agent workflow makes evidence
collection, analysis, synthesis, retries, and traces independently observable and testable.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Route and stop safely | Shared state | Next route | Loop or premature stop |
| Researcher | Retrieve and normalize evidence | Query | Sources and notes | Empty/weak sources |
| Analyst | Compare claims and flag gaps | Research notes | Analysis notes | Unsupported inference |
| Writer | Produce audience-aware cited answer | Evidence and analysis | Final answer | Invented citations |
| Critic | Validate citations and workflow health | Final state | Findings | Missed semantic errors |

## Shared state

`ResearchState` stores the request, iteration and route history, normalized sources,
research/analysis/writer outputs, critic findings, token and cost usage, trace events, and
errors. These fields preserve every handoff and make failures reproducible.

## Routing policy

`START -> Supervisor -> Researcher -> Supervisor -> Analyst -> Supervisor -> Writer ->
Critic -> END`. The supervisor selects the first missing artifact. Iteration and recursion
limits prevent loops.

## Guardrails

- Max iterations: 6 by default, configurable with `MAX_ITERATIONS`.
- Timeout: 60 seconds by default for provider calls and fallback workflow execution.
- Retry: provider calls use exponential retry; fallback workers retry twice.
- Fallback: local search and deterministic completion allow offline execution.
- Validation: Pydantic validates state and the critic checks citation references.

## Benchmark plan

Run the three queries in `configs/lab_default.yaml`. Compare latency, estimated cost,
quality proxy, citation coverage, failure rate, and token use. Multi-agent execution is
expected to cost more and run slower, while improving traceability and usually grounding.
