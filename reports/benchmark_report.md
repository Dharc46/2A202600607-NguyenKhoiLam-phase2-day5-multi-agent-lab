# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality /10 | Citation coverage | Error rate | Tokens (in/out) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.00 | 0.0000 | 6.8 | 100% | 0% | 107/104 |
| multi-agent | 0.00 | 0.0000 | 7.2 | 100% | 0% | 251/235 |

## Interpretation

- Highest quality proxy: **multi-agent** (7.2/10).
- Lowest latency: **baseline** (0.00s).
- Quality is a deterministic proxy based on completeness, structure, citations, and errors; use the peer-review rubric for final human scoring.
- Trace records are stored in `reports/traces.jsonl`.

## Failure modes and mitigations

- Search/provider failure: retry twice, then continue in explicit degraded mode.
- Infinite routing: bounded by `MAX_ITERATIONS` and workflow recursion limits.
- Slow provider: request and workflow timeouts stop execution.
- Hallucinated citations: the critic checks that inline markers resolve to known sources.
