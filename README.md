<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-142%20passing-brightgreen" alt="142 tests passing">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/status-v0.1.0--alpha-yellow" alt="v0.1.0-alpha">
</p>

<h1 align="center">⚒️ EvalForge</h1>
<p align="center"><em>Eval-driven agent testing framework — forge reliable AI systems.</em></p>

<p align="center">
  Define test suites. Run LLM systems against them. Score across correctness, cost, latency, and safety. Gate deployments on regressions.
</p>

---

## Why EvalForge?

> *"If there is one skill separating senior AI practitioners in 2026, it is evaluation."* — World AI Expo

Most teams can build an LLM demo. Almost nobody can prove their system is **reliable, safe, and cost-effective** over time. EvalForge bridges that gap — it brings the same disciplined testing culture that software engineering has (CI/CD, regression gates, benchmarks) to the probabilistic world of LLMs.

## Quick Start

```bash
# Install
pip install evalforge

# Scaffold a new evaluation project
evalforge init

# Run the example suite
evalforge run test-suites/example/suite.yaml

# Compare two runs (regression detection)
evalforge compare baselines/run-1.json evalforge-output/report-<ts>.json

# CI gate — exits 0 (pass) or 1 (fail)
evalforge gate
```

## Example Test Suite

```yaml
# test-suites/example/suite.yaml
name: hello-world
tests:
  - id: exact-match
    prompt: "What is the capital of France?"
    expected:
      type: exact
      value: "Paris"

  - id: tone-check
    prompt: "Explain quantum computing to a 5-year-old."
    expected:
      type: rubric
      rubric:
        dimensions:
          - name: accuracy
            description: "Scientifically correct, not misleading"
            weight: 0.4
          - name: simplicity
            description: "Understandable to a child"
            weight: 0.3
          - name: tone
            description: "Encouraging and engaging"
            weight: 0.3
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `evalforge init` | Scaffold a new project with example suite |
| `evalforge run <suite>` | Run a test suite against an LLM |
| `evalforge compare <baseline> <candidate>` | Diff two runs (score, cost, latency) |
| `evalforge gate` | CI gate — checks regression against baseline |
| `evalforge --help` | Show all commands |

## Features

### 🔬 Multi-Strategy Scoring
- **Exact match** — for factual questions with precise answers
- **LLM-as-Judge** — rubric-based evaluation across custom dimensions (accuracy, tone, completeness, etc.)
- **Semantic similarity** — embedding-based comparison for open-ended responses

### 📊 Cost & Latency Tracking  
- Per-test token counts and cost
- Aggregate statistics: avg, p50/p95/p99 latency
- Cost/latency deltas between runs — catch regressions before they hit your wallet

### 🚦 CI Gate Integration
- Define regression thresholds in `evalforge.yaml`
- Auto-creates baselines on first run
- Fails the build when quality/cost/latency degrades beyond limits
- Exit code 0/1 — works with GitHub Actions, GitLab CI, or any CI system

### 🧩 Extensible Architecture
- **Scorers**: `ExactScorer`, `RubricScorer`, `SemanticScorer` — add custom ones via base class
- **LLM Clients**: DeepSeek, OpenAI, Anthropic — add providers via base class
- **Reporters**: JSON, Console, Diff — add formats via base class
- **Trackers**: Cost, Latency — add metrics via base class

## Architecture

```
CLI Layer (typer)
    └─▶ run | compare | gate | init
            │
Core Engine Layer
    ├─ Executor (async + semaphore)
    ├─ Scorer Registry
    ├─ Trackers (cost + latency)
    └─ Reporters (JSON, console, diff)
            │
Infrastructure Layer
    ├─ LLM Client (httpx/async, multi-provider)
    ├─ Config (YAML via pyyaml)
    └─ Models (Pydantic v2)
```

## Configuration

```yaml
# evalforge.yaml
baseline_dir: evalforge-baselines/
suites:
  - path: test-suites/example/suite.yaml
    allowed_regression_pct: 5
judge:
  provider: deepseek
  model: deepseek-chat
target:
  provider: deepseek
  model: deepseek-chat
concurrency: 10
```

## Test Suite Output

```json
{
  "suite_name": "hello-world",
  "timestamp": "2026-07-06T20:00:00Z",
  "duration_ms": 1234,
  "tests": [
    {
      "id": "exact-match",
      "status": "pass",
      "score": { "overall": 1.0 },
      "tokens": { "input": 25, "output": 2, "total": 27 },
      "latency_ms": 340,
      "cost_usd": 0.00005
    }
  ],
  "summary": {
    "total": 1, "passed": 1, "failed": 0,
    "pass_rate": 1.0,
    "total_cost_usd": 0.00005,
    "latency_p95": 340
  }
}
```

## Who Is This For?

- **ML/AI Engineers** who ship LLM features and need to prove they work
- **RAG pipeline builders** — evaluate retrieval + generation quality independently
- **Multi-agent system architects** — benchmark orchestration quality
- **Platform teams** who want a CI gate for LLM changes
- **Freelancers** doing LLM integration — show clients measurable quality

## Project Status

**v0.1.0-alpha** — Core engine, LLM-as-Judge scoring, cost tracking, CI gate, and CLI complete. Built for and tested on DeepSeek v4, extensible to any OpenAI-compatible API.

### Roadmap

- [x] Test runner with concurrent execution
- [x] LLM-as-Judge scoring with rubrics
- [x] Cost & latency tracking
- [x] CI gate with regression detection
- [x] CLI commands (run, compare, gate, init)
- [ ] Web dashboard for visualizing results
- [ ] GitHub Actions integration
- [ ] Real-time streaming evaluation
- [ ] Plugin system for custom scorers

## References

- [SPEC.md](SPEC.md) — Full behavior spec with acceptance criteria
- [ARCHITECTURE.md](ARCHITECTURE.md) — Design, interfaces, extension points
- [gates/SUMMARY.md](gates/SUMMARY.md) — Multi-gate verification results

---

<p align="center">
  Built by <a href="https://github.com/rayyanakmal">@rayyanakmal</a>
</p>
