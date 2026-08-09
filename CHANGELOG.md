# Changelog

All notable changes to EvalForge are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/); versioning follows [SemVer](https://semver.org/).

## [v0.2.0] - 2026-08-09

### Added — Trajectory-level agent evaluation

The big upgrade: EvalForge now grades the **journey**, not just the final answer.

- **Trajectory schema** (`models/trajectory.py`) — OTel-style span shape: tool name, args, result, thought, latency, tokens, cost, error, ordering. Backward-compatible: `TestResult.trajectory` is optional; v0.1 result files parse unchanged.
- **Process metrics** (`trajectory/metrics.py`, pure code, no LLM judge) — convergence, efficiency (steps/tool-calls per task), loop detection (identical repeated calls), tool validity, error recovery, per-tool cost/latency, budget adherence, per-run aggregation. All deterministic, hand-computed golden tests.
- **Capture layer** (`trajectory/capture.py`, `adapters.py`) — friction ladder: drop-in client shims (OpenAI/Anthropic), one-line `emit_step` recorder, framework callback stubs (LangChain/LangGraph). Evaluator records, never the user.
- **Import** (`trajectory/importers.py`) — the sealed-box path: load a trajectory JSON/JSONL export (own format, OTel-style spans, or minimal JSONL) and get the same report card. Supports full-run per-test maps.
- **Trajectory regression** (`trajectory/regression.py`) — per-tool + per-test deltas, VERDICT (REGRESSED / ok / IMPROVED), disappeared-tool detection, looping-as-regression rule.
- **CLI** — `import-trajectory` command; `compare --trajectory --fail-on-trajectory-regression` CI gate (exits 1 on process regression).
- **Dashboard** — trajectory report card (process hero metrics + per-tool rollup + step timeline), trajectory upload, sample A-vs-B loader, trajectory regression section with verdict badge.

### Demo data

- `examples/gen_demo_trajectories.py` — tiny real ReAct agent (search + calc tools) run on DeepSeek, clean vs loop-prone prompts, same suite/tools/model.
- Committed real trajectories: Run A (clean) vs Run B (loopy) — both 6/6 answers, but Run B loops 16× with +14 llm calls, +8 search calls, +6 calc calls. Verdict: REGRESSED. The pitch: **same pass rate, worse process**.

## [v0.1.0] - 2026-08-08

### Added

- Core eval engine: run test suites against LLM agents, pass/fail per question
- LLM-as-Judge rubric scoring + RAGAS-style metric presets
- Cost & latency tracking (p50/p95/p99)
- CI gate with regression detection
- CLI commands (run, compare, gate, init)
- Streamlit web dashboard with side-by-side regression matrix
- Built for and tested on DeepSeek v4, extensible to any OpenAI-compatible API
