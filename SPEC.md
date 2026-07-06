# EvalForge — Eval-Driven Agent Framework

## Overview

EvalForge is an open-source framework for systematically evaluating LLM-powered systems — RAG pipelines, multi-agent orchestrations, and tool-calling agents. It treats evaluation as a first-class CI pipeline stage: define test suites, run your system against them, score across multiple dimensions (correctness, cost, latency, safety), and gate deployments on regressions.

The name combines **Evaluation** + **Forge** (the place where raw materials are shaped into finished tools under controlled, measurable conditions).

---

## Prerequisites (Human Tasks)

| Task | Detail |
|------|--------|
| Python 3.11+ | Available on Mac |
| DeepSeek API key | `DEEPSEEK_API_KEY` in `~/.zshrc` (already set) |
| `uv` for package management | `pip install uv` |
| Git repo | Created at `~/projects/evalforge/` |

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

**AC-4.1:** Given a `evalforge.toml` or `evalforge.yaml` config file, when `evalforge gate` is run, then it loads the config, identifies the regression baseline, runs the suite, and exits with code 0 (pass) or 1 (fail).

**AC-4.2:** Given a config with `allowed_regression: 5%`, when a run shows 3% regression, then the gate passes (within threshold).

**AC-4.3:** Given a config with `allowed_regression: 5%`, when a run shows 8% regression, then the gate fails with a report of what regressed.

**AC-4.4:** Given no prior baseline, when `evalforge gate` runs, then it creates the baseline automatically and exits 0 (pass — nothing to regress against).

Edge cases:
- Config file missing → exit 1 with clear error: "No config found. Run `evalforge init` to create one."
- All metrics improved → gate passes, saves new baseline

---

## US-5: CLI & Report Output

**AC-5.1:** Given `evalforge run <suite>`, when executed, then it runs the suite and outputs results to stdout and saves a JSON report to `evalforge-output/report-<timestamp>.json`.

**AC-5.2:** Given `evalforge compare <baseline> <candidate>`, when executed, then it shows a diff table with columns: test name, status, score change, cost change, latency change.

**AC-5.3:** Given `evalforge init`, when executed in an empty directory, then it creates `evalforge.yaml`, a `test-suites/` folder with an example suite, and a `.gitignore` for output directory.

Edge cases:
- `compare` with non-existent baseline file → error with paths searched
- `init` in directory with existing config → asks for confirmation before overwriting

---

## Data Contracts

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
baseline_dir: string (default: evalforge-baselines/)
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
evalforge/
├── evalforge/              # Package root
│   ├── __init__.py
│   ├── cli/               # CLI commands (typer)
│   │   ├── __init__.py
│   │   ├── main.py        # Click group: run, compare, gate, init
│   │   └── init.py        # Scaffolding logic
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
│   └── example/           # Scaffolded by `evalforge init`
├── tests/                 # Project's own test suite
│   ├── test_runner.py
│   ├── test_scoring.py
│   ├── test_judge.py
│   └── test_cli.py
├── evalforge.yaml         # Config file (created by init)
└── pyproject.toml         # Package config + entry point
```

---

## Out of Scope (v0.1)

- GUI / dashboard (CLI-only for v0.1)
- Real-time streaming evaluation (batch-only)
- Plugin system for custom scorers (hardcoded strategies, extensible via base class)
- CI provider integrations (exits with code, user pipes to GitHub Actions)
- Persistent result database (file-based JSON reports)
- Production-grade webhook or API server

---

## Extension Point Map

| Scenario | Interface | Implementation | Adding new requires |
|----------|-----------|----------------|-------------------|
| US-1, US-2 | Scorer | ExactScorer, SemanticScorer, RubricScorer | New class, zero modification |
| US-3 | Tracker | CostTracker, LatencyTracker | New class, zero modification |
| New scorer strategy | Scorer | CustomScorer | New class implementing `BaseScorer` |
| New LLM provider | LLMClient | DeepSeekClient, AnthropicClient | New class extending `BaseLLMClient` |
| New output format | Reporter | JSONReporter, MarkdownReporter | New class implementing `BaseReporter` |
