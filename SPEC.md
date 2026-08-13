# VerdictLab — Eval-Driven Agent Framework

## Overview

VerdictLab is an open-source framework for systematically evaluating LLM-powered systems — RAG pipelines, multi-agent orchestrations, and tool-calling agents. It treats evaluation as a first-class CI pipeline stage: define test suites, run your system against them, score across multiple dimensions (correctness, cost, latency, safety), and gate deployments on regressions.

The name combines **Evaluation** + **Forge** (the place where raw materials are shaped into finished tools under controlled, measurable conditions).

---

## Prerequisites (Human Tasks)

| Task | Detail |
|------|--------|
| Python 3.11+ | Available on Mac |
| DeepSeek API key | `DEEPSEEK_API_KEY` in `~/.zshrc` (already set) |
| `uv` for package management | `pip install uv` |
| Git repo | Created at `~/projects/verdictlab/` |

---

## US-1: Core Eval Engine — Test Runner

**AC-1.1:** Given a test suite of N prompts with expected outputs, when the runner executes the suite against a configured LLM, then it returns a pass/fail result for each test case.

**AC-1.2:** Given a test case that failed, when inspected, then the output includes the actual response, the expected response, and a diff/reason for failure.

**AC-1.3:** Given a test suite with 100+ prompts, when the runner executes, then it completes within 5 minutes using concurrent execution (max 10 parallel).

**AC-1.4:** Given a system prompt change between runs, when compared, then results show a regression report highlighting which tests regressed vs improved.

Edge cases:
- Empty test suite → returns empty result, not an error
- LLM API timeout → retries once, then marks test as `error` with timeout reason
- All tests pass → returns `pass: true` with summary stats

---

## US-2: LLM-as-Judge Scoring

**AC-2.1:** Given a test case with an open-ended question (no exact answer), when scored, then a judge LLM evaluates the response against a rubric defined in the test case.

**AC-2.2:** Given a rubric with multiple dimensions (accuracy, completeness, tone), when scored, then the judge returns per-dimension scores 1-5 plus an overall score.

**AC-2.3:** Given a judge evaluation, when examined, then the result includes the judge's reasoning text alongside the score.

Edge cases:
- Judge LLM returns invalid JSON → retries with stricter prompt, then marks as `judge_error`
- Rubric dimension names don't match judge output → detects mismatch, raises warning
- Empty response from target LLM → judge scores 1 with "no response provided"

---

## US-3: Cost & Latency Tracking

**AC-3.1:** Given a test run, when completed, then each test case records input/output token counts and total cost.

**AC-3.2:** Given a test run, when completed, then the report includes aggregate stats: total cost, avg latency, p50/p95/p99 latency.

**AC-3.3:** Given a regression comparison (run A vs run B), when examined, then the report includes cost and latency deltas between runs.

Edge cases:
- Token counts unavailable (open-source model) → shows `N/A`, doesn't crash
- P99 latency requires minimum 10 samples → shows warning if <10 cases

---

## US-4: CI Gate Integration

**AC-4.1:** Given a `verdictlab.toml` or `verdictlab.yaml` config file, when `verdictlab gate` is run, then it loads the config, identifies the regression baseline, runs the suite, and exits with code 0 (pass) or 1 (fail).

**AC-4.2:** Given a config with `allowed_regression: 5%`, when a run shows 3% regression, then the gate passes (within threshold).

**AC-4.3:** Given a config with `allowed_regression: 5%`, when a run shows 8% regression, then the gate fails with a report of what regressed.

**AC-4.4:** Given no prior baseline, when `verdictlab gate` runs, then it creates the baseline automatically and exits 0 (pass — nothing to regress against).

Edge cases:
- Config file missing → exit 1 with clear error: "No config found. Run `verdictlab init` to create one."
- All metrics improved → gate passes, saves new baseline

---

## US-5: CLI & Report Output

**AC-5.1:** Given `verdictlab run <suite>`, when executed, then it runs the suite and outputs results to stdout and saves a JSON report to `verdictlab-output/report-<timestamp>.json`.

**AC-5.2:** Given `verdictlab compare <baseline> <candidate>`, when executed, then it shows a diff table with columns: test name, status, score change, cost change, latency change.

**AC-5.3:** Given `verdictlab init`, when executed in an empty directory, then it creates `verdictlab.yaml`, a `test-suites/` folder with an example suite, and a `.gitignore` for output directory.

Edge cases:
- `compare` with non-existent baseline file → error with paths searched
- `init` in directory with existing config → asks for confirmation before overwriting

---

## US-T1: Trajectory Capture (the journey, not just the answer)

**AC-T1.1:** Given an agent run under evaluation, when the agent makes tool calls, then verdictlab records each call as an ordered step with tool name, args, result, latency, and optional tokens/cost/thought/error.

**AC-T1.2:** Given an agent built on a framework (LangGraph/CrewAI), when the user registers a callback, then the trajectory is captured with zero changes to the agent's code.

**AC-T1.3:** Given a custom agent loop, when the user adds one `emit(...)` line per tool call site, then the trajectory is captured in order.

**AC-T1.4:** Given a sealed-box agent with no hooks, when the user imports a trajectory JSON file, then verdictlab produces the same report card as a captured run.

**AC-T1.5:** Given a captured run, when the report is produced, then each TestResult carries its trajectory and the process metrics layer summarizes it.

Edge cases:
- Empty trajectory → reported as not converged, zeros for metrics, not an error
- Index out of order / duplicated in import → rejected with a clear validation error
- Non-JSON args/result → rejected up front (determinism rule)

---

## US-T2: Process Metrics (pure code, no LLM judge)

**AC-T2.1:** Given a trajectory, when evaluated, then convergence (terminal reason), efficiency (steps, tool calls, repeated identical calls), per-tool stats (calls, errors, latency, cost), validity (unknown tools when the allowed set is known), recovery (errors survived vs died), and budget adherence are computed.

**AC-T2.2:** Given the same trajectory twice, when evaluated, then the metrics are byte-identical (deterministic).

**AC-T2.3:** Given a trajectory with an unknown tool call and no allowed-tools metadata, then validity is skipped (None), not crashed.

Edge cases:
- Zero-step trajectory → converged=False, terminal_reason="empty"
- Trajectory ending at an error step → died_on_error=True
- Repeated identical tool+args calls → counted as repeated_calls with raw evidence exposed

---

## US-T3: Trajectory Regression (run A vs run B)

**AC-T3.1:** Given two runs with trajectories, when compared, then per-tool deltas (calls, error rate, cost) and per-test step deltas are reported.

**AC-T3.2:** Given a tool present in the baseline but absent in the candidate, when compared, then that tool is marked REGRESSED (disappeared-tool rule, mirroring visionforge).

**AC-T3.3:** Given a candidate that loops 3x more than the baseline with the same pass rate, when compared, then the process metrics show the regression even though outcomes match.

**AC-T3.4:** Given `verdictlab compare --trajectory --fail-on-trajectory-regression`, when any tool regressed, then the CLI exits 1 (CI gate).

---

## US-T4: Versioned Showcase (clients see the evolution)

**AC-T4.1:** Given a released version, when viewed on GitHub, then it appears on the Releases page with plain-language notes ("what this version does, what changed, why it matters"), not commit messages.

**AC-T4.2:** Given v0.1.0 and v0.2.0, when a client visits the repo, then each release links a deployed demo of that exact version so the evolution is clickable.

**AC-T4.3:** Given the README, when read, then a "Versions" section links the Releases page.

Edge cases:
- Back-tagging: v1 state tagged `v0.1.0` post-hoc at the pre-v2 commit — release notes written for both versions at once

---

## Data Contracts

### TrajectoryStep
```yaml
index: integer (0-based, strictly increasing)
tool: string
args: object (JSON-safe, max nesting depth 8)
result: string | object | list | null
thought: string | null
latency_ms: float
tokens: { input: int, output: int, total: int } | null
cost_usd: float
error: string | null
```

### Trajectory (attached to TestResult as optional `trajectory`)
```yaml
steps: TrajectoryStep[] (ordered, index == position)
final_answer: string | null
```

### RunResult.trajectory_summary (optional)
```yaml
mean_steps: float
mean_tool_calls: float
total_loops: integer
total_error_steps: integer
per_tool: { tool_name: { calls, errors, total_latency_ms, total_cost_usd, error_rate } }
```

---

### TestSuite
```yaml
name: string
description: string (optional)
tests:
  - id: string
    prompt: string
    expected:
      type: exact | semantic | rubric | function  # scoring method
      value: string | null                        # exact answer or N/A
      rubric:                                     # only for rubric type
        dimensions:
          - name: string
            description: string
            weight: float (0-1, sum to 1)
    metadata:
      tags: string[] (optional)
      cost_limit_usd: float (optional)
```

### RunResult
```yaml
suite_name: string
timestamp: string (ISO 8601)
duration_ms: integer
tests:
  - id: string
    status: pass | fail | error
    response: string | null
    score:
      overall: float (0-1) | null
      dimensions:
        - name: string
          score: integer (1-5)
          reasoning: string
    tokens:
      input: integer
      output: integer
      total: integer
    latency_ms: integer
    cost_usd: float
    error: string | null
summary:
  total: integer
  passed: integer
  failed: integer
  errored: integer
  pass_rate: float
  total_cost_usd: float
  avg_latency_ms: float
  latency_p50: float
  latency_p95: float
  latency_p99: float
```

### GateConfig
```yaml
baseline_dir: string (default: verdictlab-baselines/)
suites:
  - path: string
    allowed_regression_pct: float (default: 5)
judge:
  provider: deepseek | openai | anthropic
  model: string
target:
  provider: deepseek | openai | anthropic
  model: string
concurrency: integer (default: 10)
```

---

## Architecture (Outlined for Architect)

```
verdictlab/
├── verdictlab/              # Package root
│   ├── __init__.py
│   ├── cli/               # CLI commands (typer)
│   │   ├── __init__.py
│   │   ├── main.py        # Click group: run, compare, gate, init
│   │   └── init.py        # Scaffolding logic
│   ├── ui/                # Web dashboard (Streamlit)
│   │   ├── __init__.py
│   │   ├── streamlit_app.py  # Dashboard entry: metrics, matrix, upload
│   │   ├── metrics.py        # RAGAS-style metric presets (rubric templates)
│   │   └── style.css         # Design system (bundled, deploy-safe)
│   ├── runner/            # Test execution engine
│   │   ├── __init__.py
│   │   ├── executor.py    # Concurrent test runner
│   │   └── retry.py       # Retry logic for transient failures
│   ├── scoring/           # Scoring strategies
│   │   ├── __init__.py
│   │   ├── exact.py       # Exact string match
│   │   ├── semantic.py    # Semantic similarity (embedding-based)
│   │   ├── rubric.py      # LLM-as-Judge with rubric
│   │   └── base.py        # Abstract scorer interface
│   ├── judge/             # LLM-as-Judge client
│   │   ├── __init__.py
│   │   ├── client.py      # Multi-provider LLM client
│   │   └── prompts.py     # Judge system prompts
│   ├── models/            # Data models
│   │   ├── __init__.py
│   │   ├── suite.py       # TestSuite model
│   │   └── result.py      # RunResult model
│   ├── tracking/          # Cost & latency tracking
│   │   ├── __init__.py
│   │   ├── counter.py     # Token & cost counter
│   │   └── latency.py     # Latency stats (p50/p95/p99)
│   ├── cli.py             # Typer CLI entry point
│   └── config.py          # Config loader (YAML/TOML)
├── test-suites/           # Example test suites
│   └── example/           # Scaffolded by `verdictlab init`
├── tests/                 # Project's own test suite
│   ├── test_runner.py
│   ├── test_scoring.py
│   ├── test_judge.py
│   └── test_cli.py
├── verdictlab.yaml         # Config file (created by init)
└── pyproject.toml         # Package config + entry point
```

---

## Out of Scope (v0.1)

- ~~GUI / dashboard~~ — **Done:** Streamlit web dashboard added in `verdictlab/ui/` (see README "Web Dashboard")
- Real-time streaming evaluation (batch-only)
- Plugin system for custom scorers (hardcoded strategies, extensible via base class)
- CI provider integrations (exits with code, user pipes to GitHub Actions)
- Persistent result database (file-based JSON reports)
- Production-grade webhook or API server

## Out of Scope (v0.2 — trajectory layer)

- ~~Trajectory capture, metrics, regression, UI~~ — **Done:** v0.2 (US-T1..T3; US-T4 versioning showcase)
- Tracing/observability platform (Langfuse/LangSmith/Phoenix territory) — we grade traces, we don't host/visualize production telemetry
- OTel exporter / telemetry transport — we adopt the span shape as input schema, we don't ship a pipeline
- Agent runtime / framework — we grade trajectories, we don't run arbitrary agents
- New benchmark (τ-bench/SWE-bench territory) and golden-trajectory dataset curation
- LLM-as-judge for trajectory quality (rubric judge deferred to a later minor release)
- Stateful-world simulation (τ-bench-style databases/policies)
- Branch-per-version or commit-history-as-showcase versioning (see US-T4: tags + Releases + per-version demos only)

---

## Extension Point Map

| Scenario | Interface | Implementation | Adding new requires |
|----------|-----------|----------------|-------------------|
| US-1, US-2 | Scorer | ExactScorer, SemanticScorer, RubricScorer | New class, zero modification |
| US-3 | Tracker | CostTracker, LatencyTracker | New class, zero modification |
| New scorer strategy | Scorer | CustomScorer | New class implementing `BaseScorer` |
| New LLM provider | LLMClient | DeepSeekClient, AnthropicClient | New class extending `BaseLLMClient` |
| New output format | Reporter | JSONReporter, MarkdownReporter | New class implementing `BaseReporter` |
| New capture adapter (US-T1) | StepRecorder / adapters | Framework callback, client shim, emit_step | New adapter class, zero core modification |
| New trajectory metric (US-T2) | metrics module | Pure function over Trajectory | New function + golden test |
