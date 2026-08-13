# Changelog

All notable changes to VerdictLab are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/); versioning follows [SemVer](https://semver.org/).

## [Unreleased] - 2026-08-10

### Added — realistic built-in samples (same format as real runs)

The dashboard's sample data is rebuilt from the ground up. Previously the two
built-in options were two different file formats (output-only RunResults vs a
raw trajectory JSON that the app wrapped with fabricated all-pass answers).
Now all samples are real `verdictlab run` outputs — answers AND trajectories,
one RunResult file per run, loaded through the exact same `load_run` path as
uploads:

- **Real-run sample generator** (`examples/gen_realistic_samples.py`) — a
  tool-using geography agent (ReAct-style `TOOL <entity>` / `ANSWER <value>`
  protocol against a deterministic lookup tool) run against real DeepSeek
  through the same Executor path as the CLI. `examples/geo_facts.yaml` is the
  dataset behind it.
- **Four committed runs** (`examples/geo_*.json`) — `geo_baseline` (6/6,
  1 tool/test), `geo_more_tools` (6/6, 2 tools/test), `geo_no_tool` (4/6,
  memory only), `geo_memory_loopy` (5/6, 2 tools + loops).
- **Three sample pairs in the dashboard** — "same score, more tools",
  "pass rate regressed", "both regressed" — each varying one behavior via
  the system prompt (isolation-of-variables), each producing a distinct
  trajectory-regression verdict.
- **`run_suite()` accepts a custom `generate_fn`** (`cli/run_real.py`) — a
  tool-using agent can be wired through the same one-code-path runner, so
  sample files and user files are produced identically. Default unchanged.
- Dashboard sample loader errors now name the missing file and the
  regeneration command.

### Added — real execution in `verdictlab run` (one run, both dimensions)

`verdictlab run` previously only supported `--no-llm` dry runs. It now executes
a suite against a configured provider and captures trajectories into the same
report, so a single run produces BOTH the answer-level results AND the process
journey:

- **Executor trajectory capture** (`runner/executor.py`) — `capture_trajectories=True`
  gives every test a `StepRecorder`; the generate function receives `(prompt, recorder)`
  and the recorded journey is attached to each `TestResult`. Default remains `False`
  (v1 behavior unchanged).
- **Real-run wiring** (`cli/run_real.py`) — API-key resolution (env or `~/.hermes/.env`),
  provider/model overrides (`--provider`, `--model`), generate fn that emits `llm` steps
  with tokens + cost, and `run_suite()` which attaches the aggregate `trajectory_summary`.
- **Per-test scoring dispatch** (`scoring/registry.py`) — `RegistryScorer` routes each
  response to the scorer matching its `expected.type`, so mixed exact + rubric suites
  run in one executor.
- **CLI** — `verdictlab run suite.yaml` now works for real (previously errored with
  "Real LLM execution requires a configured provider"). `verdictlab run suite.yaml
  --provider deepseek --model deepseek-chat` compares agents/versions.
- **Workflow** — run once per agent, then `verdictlab compare a.json b.json --trajectory`
  shows answer diff AND trajectory regression from the same files; the dashboard upload
  shows both views.

## [v0.2.0] - 2026-08-09

### Added — Trajectory-level agent evaluation

The big upgrade: VerdictLab now grades the **journey**, not just the final answer.

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
