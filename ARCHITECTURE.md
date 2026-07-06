# EvalForge — Architecture Design

> **Version:** 0.1.0
> **Design Date:** 2026-07-06
> **Status:** Draft

---

## Table of Contents

1. [Component Tree & Data Flow](#1-component-tree--data-flow)
2. [File/Module Structure](#2-filemodule-structure)
3. [Interface Specifications](#3-interface-specifications)
4. [Extension Point Map](#4-extension-point-map)
5. [Tech Stack & Rationale](#5-tech-stack--rationale)
6. [Concurrency Model](#6-concurrency-model)
7. [Scenario-to-Component Mapping](#7-scenario-to-component-mapping)
8. [Spike Verdict Integration](#8-spike-verdict-integration)

---

## 1. Component Tree & Data Flow

### Component Tree (Layered)

```
┌──────────────────────────────────────────────────────────────────┐
│                          CLI Layer (typer)                        │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────┐  │
│  │   run    │  │  compare  │  │   gate    │  │     init      │  │
│  └────┬─────┘  └─────┬─────┘  └─────┬─────┘  └───────┬───────┘  │
│       │              │              │               │           │
│       ▼              ▼              ▼               ▼           │
│  ┌────────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────┐   │
│  │  Executor  │ │  Diff    │ │   Gate      │ │  Init        │   │
│  │  (runner)  │ │ Engine   │ │  Checker    │ │ Scaffolder   │   │
│  └─────┬──────┘ └────┬─────┘ └──────┬─────┘ └──────────────┘   │
└────────┼─────────────┼──────────────┼───────────────────────────┘
         │             │              │
         ▼             ▼              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Core Engine Layer                           │
│  ┌───────────┐  ┌──────────┐  ┌───────────┐  ┌───────────────┐  │
│  │  Executor │  │  Scorer  │  │  Tracker  │  │  Reporter     │  │
│  │  (async)  │  │ Registry │  │  (cost &  │  │  (json/stdout │  │
│  │           │  │          │  │  latency) │  │   /diff)      │  │
│  └─────┬─────┘  └────┬─────┘  └─────┬─────┘  └───────┬───────┘  │
│        │             │              │               │           │
└────────┼─────────────┼──────────────┼───────────────┼───────────┘
         │             │              │               │
         ▼             ▼              ▼               ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Infrastructure Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  LLMClient   │  │   Config     │  │   Models (Pydantic)   │  │
│  │ (httpx/async)│  │  (pyyaml)    │  │  Suite, Result, Gate  │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow — `evalforge run <suite>`

```
                    ┌──────────────────┐
                    │    CLI: run      │
                    │  (cli/main.py)   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Config.load()   │ ── reads evalforge.yaml (optional)
                    │  (config.py)     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  TestSuite.load()│ ── reads suite YAML → Pydantic model
                    │  (models/suite)  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Executor.run()  │
                    │ (runner/executor)│
                    │                  │
                    │ For each test:   │
                    │ ┌──────────────┐ │
                    │ │ LLMClient    │ │ ── httpx async POST → target LLM
                    │ │ .generate()  │ │    returns LLMResponse(content, usage, latency)
                    │ └──────┬───────┘ │
                    │        │         │
                    │ ┌──────▼───────┐ │
                    │ │ Scorer       │ │ ── selects scorer by expected.type:
                    │ │ .score()     │ │    exact → ExactScorer
                    │ └──────┬───────┘ │    rubric → RubricScorer (calls judge LLM)
                    │        │         │    semantic → SemanticScorer
                    │ ┌──────▼───────┐ │
                    │ │ Tracker      │ │ ── records tokens, cost, latency per test
                    │ │ .track()     │ │
                    │ └──────┬───────┘ │
                    │        │         │
                    │  TestResult     │ ── Pydantic model per test
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Tracker         │
                    │  .summarize()    │ ── aggregates: total_cost, p50/p95/p99, pass_rate
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────────┐
                    │  Reporter.generate() │ ── JSONReporter → evalforge-output/report-<ts>.json
                    │  Reporter.write()    │    ConsoleReporter → stdout table
                    └──────────────────────┘
```

### Data Flow — `evalforge gate`

```
  Config.load()
       │
       ▼
  GateChecker.check()
       │
       ├── No baseline? → run suite, save baseline, exit 0
       │
       └── Has baseline? →
             ├── Executor.run(suite) → RunResult (candidate)
             ├── Load baseline RunResult from disk
             ├── DiffEngine.compare(baseline, candidate)
             ├── Check regression against allowed_regression_pct
             │     ├── Within threshold → exit 0, save new baseline
             │     └── Exceeds threshold → exit 1, print regression report
```

---

## 2. File/Module Structure

Every file with a single responsibility.

```
evalforge/
├── evalforge/                    # Package root
│   ├── __init__.py               # Version export: __version__ = "0.1.0"
│   │
│   ├── app.py                    # Typer CLI entry point (defines `app = typer.Typer()`)
│   │
│   ├── config.py                 # Config loader
│   │   #   - GateConfig (Pydantic model)
│   │   #   - load_config(path: Path) → GateConfig
│   │   #   - save_config(config: GateConfig, path: Path) → None
│   │   #   - Supports YAML via pyyaml
│   │   #   Purpose: single source of truth for all runtime configuration
│   │
│   ├── cli/                      # CLI command implementations
│   │   ├── __init__.py           # Empty
│   │   ├── main.py               # Command registrations
│   │   #   #   - run(suite_path, output_dir, concurrency) → None
│   │   #   #   - compare(baseline_path, candidate_path) → None
│   │   #   #   - gate(config_path) → None (calls sys.exit)
│   │   #   #   - init(dir_path, force) → None
│   │   #   #   Purpose: thin command handlers; delegates to engine layer
│   │   │
│   │   └── init.py               # Scaffolding logic
│   │       #   - scaffold_project(target_dir: Path, force: bool) → None
│   │       #   - creates: evalforge.yaml, test-suites/example/suite.yaml, .gitignore
│   │       #   Purpose: bootstraps a new EvalForge project
│   │
│   ├── runner/                   # Test execution engine
│   │   ├── __init__.py           # Empty
│   │   ├── executor.py           # Concurrent test runner
│   │   #   #   - Executor(config: GateConfig, llm_client: LLMClient)
│   │   #   #   - async run(suite: TestSuite, scorer_registry: ScorerRegistry) → RunResult
│   │   #   #   - Uses asyncio.Semaphore(concurrency) for bounded parallelism
│   │   #   #   - Per test: call LLM, score, track → TestResult
│   │   #   #   Purpose: orchestrates test execution; the central engine
│   │   │
│   │   └── retry.py              # Retry logic
│   │       #   - async retry_with_backoff(fn, max_retries=1, base_delay=1.0)
│   │       #   - RetryDecider: decides if error is retryable (timeout=yes, 4xx=no)
│   │       #   Purpose: handles transient LLM API failures (timeouts)
│   │
│   ├── scoring/                  # Scoring strategies
│   │   ├── __init__.py           # Empty
│   │   ├── base.py               # Abstract Scorer interface
│   │   #   #   - class Scorer(ABC)
│   │   #   #   - async score(response: str, expected: Expected) → ScoreResult
│   │   #   #   Purpose: contract all scorers must fulfill
│   │   │
│   │   ├── exact.py              # Exact string match
│   │   #   #   - class ExactScorer(Scorer)
│   │   #   #   - Strips whitespace, case-insensitive comparison
│   │   #   #   - Returns overall=1.0 (match) or 0.0 (no match)
│   │   #   #   Purpose: handles expected.type = "exact"
│   │   │
│   │   ├── semantic.py           # Semantic similarity (embedding-based)
│   │   #   #   - class SemanticScorer(Scorer)
│   │   #   #   - Uses embedding API to compute cosine similarity
│   │   #   #   - Returns overall=similarity_score (0.0–1.0)
│   │   #   #   Purpose: handles expected.type = "semantic"
│   │   │
│   │   ├── rubric.py             # LLM-as-Judge rubric scoring
│   │   #   #   - class RubricScorer(Scorer)
│   │   #   #   - Accepts judge LLMClient in constructor
│   │   #   #   - Builds judge prompt from rubric dimensions
│   │   #   #   - Calls judge LLM with max_tokens=700 (spike minimum)
│   │   #   #   - Parses JSON response → per-dimension scores + reasoning
│   │   #   #   - Retry on JSON parse failure (stricter prompt, 1 retry)
│   │   #   #   - Returns overall=average of dimension scores (normalized 0-1)
│   │   #   #   Purpose: handles expected.type = "rubric"
│   │   │
│   │   ├── function.py           # User-defined function scorer
│   │   #   #   - class FunctionScorer(Scorer)
│   │   #   #   - Accepts a callable: (response, expected) → ScoreResult
│   │   #   #   Purpose: handles expected.type = "function" (extensibility)
│   │   │
│   │   └── registry.py           # Scorer factory
│   │       #   - class ScorerRegistry
│   │       #   - register(name: str, scorer: Scorer) → None
│   │       #   - get(name: str) → Scorer
│   │       #   - Pre-registers: exact, semantic, rubric, function
│   │       #   Purpose: maps expected.type strings → scorer instances
│   │
│   ├── judge/                    # Judge LLM client
│   │   ├── __init__.py           # Empty
│   │   ├── client.py             # Multi-provider LLM client
│   │   #   #   - class LLMClient(ABC)
│   │   #   #   - async generate(messages, max_tokens, temperature) → LLMResponse
│   │   #   #   - class DeepSeekClient(LLMClient) — httpx to api.deepseek.com
│   │   #   #   - class OpenAIClient(LLMClient)   — httpx to api.openai.com
│   │   #   #   - class AnthropicClient(LLMClient) — httpx to api.anthropic.com
│   │   #   #   - Factory: create_client(provider, model, api_key) → LLMClient
│   │   #   #   Purpose: single abstraction for all LLM API calls
│   │   │
│   │   └── prompts.py            # Judge system prompt templates
│   │       #   - JUDGE_SYSTEM_PROMPT: str — base template
│   │       #   - build_rubric_prompt(dimensions: list[RubricDimension]) → str
│   │       #   - STRICT_RETRY_PROMPT: str — used on JSON parse failure
│   │       #   Purpose: isolates prompt engineering from client logic
│   │
│   ├── models/                   # Pydantic data models
│   │   ├── __init__.py           # Re-exports all models
│   │   ├── suite.py              # TestSuite & friends
│   │   #   #   - TestSuite(name, description, tests: list[TestCase])
│   │   #   #   - TestCase(id, prompt, expected: Expected, metadata: TestMetadata)
│   │   #   #   - Expected(type: Literal["exact","semantic","rubric","function"],
│   │   #   #              value: str|None, rubric: list[RubricDimension]|None)
│   │   #   #   - RubricDimension(name, description, weight)
│   │   #   #   - TestMetadata(tags: list[str], cost_limit_usd: float|None)
│   │   #   #   - load_suite(path: Path) → TestSuite (YAML → Pydantic)
│   │   #   #   Purpose: source-of-truth for suite structure; validated on load
│   │   │
│   │   ├── result.py             # RunResult & friends
│   │   #   #   - RunResult(suite_name, timestamp, duration_ms, tests, summary)
│   │   #   #   - TestResult(id, status: Literal["pass","fail","error"],
│   │   #   #                response, score: ScoreResult|None, tokens: TokenCount,
│   │   #   #                latency_ms, cost_usd, error: str|None)
│   │   #   #   - ScoreResult(overall, dimensions: list[DimensionScore]|None, method)
│   │   #   #   - DimensionScore(name, score: 1-5, reasoning: str)
│   │   #   #   - TokenCount(input, output, total)
│   │   #   #   - Summary(total, passed, failed, errored, pass_rate,
│   │   #   #              total_cost_usd, avg_latency_ms, p50, p95, p99)
│   │   #   #   - load_result(path: Path) → RunResult (JSON → Pydantic)
│   │   #   #   Purpose: immutable result structure; serialized as JSON
│   │   │
│   │   └── llm.py                # LLM interaction models
│   │       #   - LLMResponse(content, reasoning_content, usage: Usage, latency_ms, cost_usd)
│   │       #   - Usage(prompt_tokens, completion_tokens, total_tokens)
│   │       #   - Message(role: Literal["system","user","assistant"], content: str)
│   │       #   Purpose: standard shapes for LLM I/O, provider-agnostic
│   │
│   ├── tracking/                 # Cost & latency tracking
│   │   ├── __init__.py           # Empty
│   │   ├── base.py               # Abstract Tracker interface
│   │   #   #   - class Tracker(ABC)
│   │   #   #   - track(result: TestResult) → None
│   │   #   #   - summarize() → TrackingSummary
│   │   #   #   Purpose: contract for all trackers
│   │   │
│   │   ├── cost.py               # Cost tracker
│   │   #   #   - class CostTracker(Tracker)
│   │   #   #   - Accumulates input/output tokens and cost per test
│   │   #   #   - Handling for N/A tokens: stores as None, reports as "N/A"
│   │   #   #   - summarize() returns total_cost_usd, total_input_tokens, total_output_tokens
│   │   #   #   Purpose: handles US-3 cost requirements + N/A edge case
│   │   │
│   │   └── latency.py            # Latency tracker
│   │       #   - class LatencyTracker(Tracker)
│   │       #   - Collects latency_ms per test
│   │       #   - summarize() computes avg, p50, p95, p99
│   │       #   - P99 warning if <10 samples (edge case from US-3/AC-3.3)
│   │       #   Purpose: handles US-3 latency requirements
│   │
│   ├── reporting/                # Report generation (output)
│   │   ├── __init__.py           # Empty
│   │   ├── base.py               # Abstract Reporter interface
│   │   #   #   - class Reporter(ABC)
│   │   #   #   - generate(result: RunResult) → str
│   │   #   #   - write(result: RunResult, path: Path) → None
│   │   #   #   Purpose: contract for all output formats
│   │   │
│   │   ├── json_reporter.py      # JSON file reporter
│   │   #   #   - class JSONReporter(Reporter)
│   │   #   #   - generate() → JSON string (via pydantic .model_dump_json)
│   │   #   #   - write() → writes to evalforge-output/report-{timestamp}.json
│   │   #   #   Purpose: primary persistence format; consumed by compare/gate
│   │   │
│   │   ├── console_reporter.py   # Terminal output reporter
│   │   #   #   - class ConsoleReporter(Reporter)
│   │   #   #   - generate() → formatted text table (pass/fail/error per test)
│   │   #   #   - Summary block: pass_rate, total_cost, latency stats
│   │   #   #   Purpose: human-readable stdout output for `evalforge run`
│   │   │
│   │   └── diff_reporter.py      # Comparison/diff reporter
│   │       #   - class DiffReporter(Reporter)
│   │       #   - generate_diff(baseline: RunResult, candidate: RunResult) → str
│   │       #   - Table columns: test name, status, score Δ, cost Δ, latency Δ
│   │       #   - Highlights regressions (↓) vs improvements (↑)
│   │       #   Purpose: handles `evalforge compare` and gate regression output
│   │
│   └── gate/                     # CI gate logic
│       ├── __init__.py           # Empty
│       └── checker.py            # Regression gate checker
│           #   - class GateChecker
│           #   - check(config: GateConfig) → GateResult (pass/fail + exit code)
│           #   - Handles baseline management (load, create, save)
│           #   - Compares RunResults, checks per-suite regression thresholds
│           #   - No-baseline case: runs suite, saves baseline, returns pass
│           #   Purpose: implements US-4 logic
│
├── test-suites/                  # Example test suites (scaffolded)
│   └── example/
│       └── suite.yaml            # Sample suite with 3 test cases (exact + rubric)
│
├── tests/                        # Project tests (pytest)
│   ├── conftest.py               # Shared fixtures: mock LLMClient, sample suites
│   ├── test_runner.py            # Executor unit tests (AC-1.1, 1.2, 1.3, 1.4)
│   ├── test_scoring.py           # Scorer unit tests (AC-2.1, 2.2, 2.3)
│   ├── test_judge.py             # Judge client unit tests (JSON parse, retry)
│   ├── test_tracking.py          # Tracker unit tests (AC-3.1, 3.2, 3.3)
│   ├── test_reporting.py         # Reporter unit tests
│   ├── test_cli.py               # CLI integration tests (AC-5.1, 5.2, 5.3)
│   └── test_gate.py              # Gate checker tests (AC-4.1, 4.2, 4.3, 4.4)
│
├── evalforge.yaml                # Config file (created by `evalforge init`)
├── pyproject.toml                # Package config + entry point
└── README.md                     # Project documentation
```

---

## 3. Interface Specifications

### 3.1 `Scorer` Interface

```python
# evalforge/scoring/base.py

from abc import ABC, abstractmethod
from evalforge.models.result import ScoreResult, DimensionScore
from evalforge.models.suite import Expected


class Scorer(ABC):
    """Abstract base for all scoring strategies.

    Each scorer evaluates a target LLM response against a test case's
    expected output. The `expected.type` field determines which scorer
    is selected by the ScorerRegistry.

    Implementations: ExactScorer, SemanticScorer, RubricScorer, FunctionScorer
    """

    @abstractmethod
    async def score(self, response: str, expected: Expected) -> ScoreResult:
        """Evaluate a response against expected criteria.

        Args:
            response: The raw text output from the target LLM.
            expected: The Expected model from the test case, containing
                      type, value (for exact/semantic), and optional rubric
                      dimensions.

        Returns:
            ScoreResult with:
              - overall: float 0.0–1.0 (normalized score)
              - dimensions: optional list of DimensionScore (for rubric)
              - method: string identifying the scorer used

        Raises:
            ScoringError: If scoring fails irrecoverably (e.g., judge API down).
                          The executor catches this and marks the test as "error".
        """
        ...


class ScoringError(Exception):
    """Non-retryable scoring failure."""
    pass
```

**Contract for implementations:**

| Method | ExactScorer | RubricScorer | SemanticScorer | FunctionScorer |
|--------|-------------|-------------|----------------|----------------|
| Inputs | `response`, `expected.value` | `response`, `expected.rubric.dimensions` | `response`, `expected.value` | `response`, `expected` |
| Returns `overall` | 1.0 or 0.0 | Average of dimension scores normalized to 0–1 | Cosine similarity 0–1 | User-defined |
| Returns `dimensions` | `None` | List of DimensionScore | `None` | Optional |
| Async? | Yes (trivially) | Yes (calls judge LLM) | Yes (calls embedding API) | Per user impl |
| Side effects | None | Calls judge LLM (counts toward cost) | Calls embedding API | Per user impl |

### 3.2 `LLMClient` Interface

```python
# evalforge/judge/client.py

from abc import ABC, abstractmethod
from evalforge.models.llm import LLMResponse, Message


class LLMClient(ABC):
    """Abstract base for all LLM provider clients.

    Handles API communication, token counting, cost calculation,
    and provider-specific quirks (e.g., DeepSeek reasoning_content).

    Implementations: DeepSeekClient, OpenAIClient, AnthropicClient
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        max_tokens: int = 700,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Send a chat completion request to the LLM provider.

        Args:
            messages: Ordered list of system/user/assistant messages.
            max_tokens: Maximum output tokens. Default 700 (spike-validated
                        minimum for judge reliability).
            temperature: Sampling temperature. Default 0.1 for deterministic scoring.

        Returns:
            LLMResponse with:
              - content: str — the model's text response
              - reasoning_content: str | None — reasoning trace (DeepSeek-specific)
              - usage: Usage(prompt_tokens, completion_tokens, total_tokens)
              - latency_ms: float — wall-clock latency of the API call
              - cost_usd: float — calculated from provider pricing

        Raises:
            LLMTimeoutError: After retries exhausted (transient failure).
            LLMAuthError: 401/403 — not retried.
            LLMError: Other non-retryable failures.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name: 'deepseek', 'openai', 'anthropic'."""
        ...


def create_client(provider: str, model: str, api_key: str) -> LLMClient:
    """Factory: returns the correct LLMClient subclass for the provider.

    Args:
        provider: One of 'deepseek', 'openai', 'anthropic'.
        model: Provider-specific model name (e.g., 'deepseek-v4-flash').
        api_key: API key for the provider.

    Returns:
        Configured LLMClient instance.

    Raises:
        ValueError: Unknown provider string.
    """
```

**Provider-specific notes:**

| Aspect | DeepSeekClient | OpenAIClient | AnthropicClient |
|--------|---------------|--------------|-----------------|
| Base URL | `api.deepseek.com/v1` | `api.openai.com/v1` | `api.anthropic.com/v1` |
| `reasoning_content` | Present (chat.completions) | Not used | Not used |
| Pricing source | Config or hardcoded rates | Config or hardcoded rates | Config or hardcoded rates |
| `max_tokens` minimum | 700 (judge reliability) | Provider default | Provider default |

### 3.3 `Reporter` Interface

```python
# evalforge/reporting/base.py

from abc import ABC, abstractmethod
from pathlib import Path
from evalforge.models.result import RunResult


class Reporter(ABC):
    """Abstract base for all output formats.

    Each reporter transforms a RunResult into a specific output format
    (JSON file, terminal table, diff table, etc.).

    Implementations: JSONReporter, ConsoleReporter, DiffReporter
    """

    @abstractmethod
    def generate(self, result: RunResult) -> str:
        """Convert RunResult to string representation.

        Args:
            result: The complete run result with all test results and summary.

        Returns:
            Formatted string (JSON text, table, etc.).
        """
        ...

    @abstractmethod
    def write(self, result: RunResult, path: Path) -> None:
        """Write the report to a file.

        Args:
            result: The complete run result.
            path: Destination file path. Parent directories are created if needed.

        Raises:
            OSError: If file cannot be written.
        """
        ...
```

**Implementation contracts:**

| Method | JSONReporter | ConsoleReporter | DiffReporter |
|--------|-------------|-----------------|--------------|
| `generate(result)` | `result.model_dump_json(indent=2)` | Rich table: test→status→score→cost→latency | Two-column diff: baseline↔candidate |
| `write(result, path)` | Writes JSON to path | Writes text to stdout (ignores path) | Writes diff table to stdout |
| Extra methods | None | None | `generate_diff(baseline, candidate) → str` |
| Edge case: empty suite | `{"tests": [], "summary": {...}}` | "No tests in suite." | "Both runs empty." |

### 3.4 `Tracker` Interface

```python
# evalforge/tracking/base.py

from abc import ABC, abstractmethod
from evalforge.models.result import TestResult, TrackingSummary


class Tracker(ABC):
    """Abstract base for all metrics trackers.

    Trackers accumulate per-test metrics during a run and produce
    aggregate summaries at the end.

    Implementations: CostTracker, LatencyTracker
    """

    @abstractmethod
    def track(self, result: TestResult) -> None:
        """Record metrics from a single test result.

        Called once per test by the Executor after scoring completes.

        Args:
            result: The completed TestResult (includes tokens, cost, latency).
        """
        ...

    @abstractmethod
    def summarize(self) -> TrackingSummary:
        """Compute aggregate statistics from all tracked results.

        Returns:
            TrackingSummary with:
              - total_cost_usd: float
              - avg_latency_ms: float
              - latency_p50: float
              - latency_p95: float
              - latency_p99: float
              - total_input_tokens: int
              - total_output_tokens: int

        Edge cases:
            - No results tracked → returns zeros, not errors.
            - <10 samples for p99 → sets a warning flag (not an error).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all tracked data. Called before each new run."""
        ...
```

**Implementation specifics:**

| Tracker | What it tracks | Edge case handling |
|---------|---------------|-------------------|
| `CostTracker` | `cost_usd`, `tokens.input`, `tokens.output` per test | Token counts `None` → stored as `None`, reported as `"N/A"` in summary |
| `LatencyTracker` | `latency_ms` per test | <10 samples → sets `warning_p99_unreliable = True`, still computes p99 |

---

## 4. Extension Point Map

Every new feature = new file, never editing existing implementation files.

| Scenario | Interface | Implementation | Adding new scenario requires |
|----------|-----------|----------------|------------------------------|
| **US-1:** Exact match scoring | `Scorer` | `ExactScorer` | Already exists |
| **US-1:** Semantic similarity | `Scorer` | `SemanticScorer` | Already exists |
| **US-2:** Rubric-based LLM judging | `Scorer` | `RubricScorer` | Already exists |
| **US-2:** Custom function scoring | `Scorer` | `FunctionScorer(callable)` | Pass callable at registration time |
| **US-3:** Cost tracking | `Tracker` | `CostTracker` | Already exists |
| **US-3:** Latency stats (p50/p95/p99) | `Tracker` | `LatencyTracker` | Already exists |
| **US-5:** JSON report output | `Reporter` | `JSONReporter` | Already exists |
| **US-5:** Console table output | `Reporter` | `ConsoleReporter` | Already exists |
| **US-5:** Diff/comparison output | `Reporter` | `DiffReporter` | Already exists |
| **Future:** New scoring strategy (e.g., BLEU, ROUGE) | `Scorer` | `BleuScorer` | New file `scoring/bleu.py`, register in registry |
| **Future:** New LLM provider (e.g., Groq, Gemini) | `LLMClient` | `GroqClient` | New class in `judge/client.py` or new file, add to `create_client` factory |
| **Future:** HTML report output | `Reporter` | `HTMLReporter` | New file `reporting/html_reporter.py` |
| **Future:** Database-backed result storage | `Reporter` | `PostgresReporter` | New file `reporting/db_reporter.py` |
| **Future:** Custom tracker (e.g., safety score, toxicity) | `Tracker` | `SafetyTracker` | New file `tracking/safety.py` |
| **Future:** New config format (TOML) | N/A (config.py) | Add `load_config_toml()` | Add function to `config.py`, no interface change |

### Open/Closed Analysis

```
                        ┌──────────────┐
                        │   Executor   │  ← CLOSED for modification
                        │  (runner)    │     Depends on abstractions only
                        └──┬───┬───┬──┘
                           │   │   │
              ┌────────────┼───┼───┼────────────┐
              │            │   │   │            │
              ▼            ▼   │   ▼            ▼
        ┌──────────┐  ┌────────┐ ┌──────────┐ ┌──────────┐
        │  Scorer  │  │Tracker │ │Reporter  │ │LLMClient │
        │  (ABC)   │  │(ABC)   │ │(ABC)     │ │(ABC)     │
        └────┬─────┘  └───┬────┘ └────┬─────┘ └────┬─────┘
             │             │           │            │
       ┌─────┼─────┐       │     ┌─────┼─────┐      ├────────┐
       ▼     ▼     ▼       ▼     ▼     ▼     ▼      ▼        ▼
    Exact Rubric Semantic Cost  JSON Console Diff DeepSeek OpenAI
    Scorer Scorer Scorer Track  Rep  Rep     Rep Client   Client

    All leaves are OPEN for extension — new classes added without
    touching Executor, interfaces, or existing implementations.
```

---

## 5. Tech Stack & Rationale

| Technology | Role | Rationale |
|-----------|------|-----------|
| **Python 3.11+** | Runtime | SPEC requirement; `asyncio` improvements in 3.11 (task groups) |
| **typer** | CLI framework | User requirement; native async support, type-hint-driven, produces clean help text automatically |
| **pydantic v2** | Data modeling + validation | User requirement; validates YAML→model on load, serializes Result→JSON, strict mode for config |
| **httpx** | Async HTTP client | User requirement; async-native, HTTP/2 support, connection pooling for concurrent LLM calls |
| **pyyaml** | YAML parsing | User requirement; loads TestSuite .yaml files and evalforge.yaml config |
| **asyncio** | Concurrency | User requirement; `asyncio.Semaphore(10)` bounds parallelism, `asyncio.gather` for concurrent test execution |
| **pytest** + **pytest-asyncio** | Testing | Standard Python testing; async test support for executor and client tests |
| **rich** (optional) | Terminal formatting | Not specified but recommended for ConsoleReporter tables; can fall back to plain text |

**What we're NOT using (and why):**

| Rejected | Reason |
|----------|--------|
| `click` | typer is built on click and adds type-hint support; typer chosen per user spec |
| `requests` | Synchronous; would block the event loop during concurrent LLM calls |
| `toml` | YAML is more readable for test suites and already chosen per spec; TOML support can be added later |
| `langchain` | Over-engineered for this use case; we need a thin LLM client, not an agent framework |
| `celery` | Too heavy for v0.1; asyncio + semaphore is sufficient for <100 concurrent tests |
| SQL database | Out of scope (v0.1); file-based JSON reports per SPEC |

---

## 6. Concurrency Model

### Design

```
┌──────────────────────────────────────────────────┐
│                  Executor.run()                   │
│                                                   │
│  semaphore = asyncio.Semaphore(10)                │
│                                                   │
│  async def run_one_test(test):                    │
│      async with semaphore:                        │
│          response = await llm_client.generate()   │
│          score = await scorer.score(response)     │
│          trackers.track(TestResult(...))          │
│          return TestResult                        │
│                                                   │
│  results = await asyncio.gather(                  │
│      *(run_one_test(t) for t in suite.tests)      │
│  )                                                │
│                                                   │
│  return RunResult(tests=results, ...)             │
└──────────────────────────────────────────────────┘
```

### Key Decisions

| Decision | Rationale |
|----------|-----------|
| `asyncio.Semaphore(10)` | SPEC AC-1.3: 100+ tests in <5 min. 10 parallel calls is the sweet spot — enough throughput, not too much API contention. Configurable via `evalforge.yaml` `concurrency` field. |
| One coroutine per test | Each test is independent; `asyncio.gather` runs them concurrently up to the semaphore limit. Order is non-deterministic (acceptable — each test is self-contained). |
| Scorer call inside the semaphore | RubricScorer calls the judge LLM, which is also rate-limited. Sharing the semaphore prevents scoring from saturating the API. |
| Retry outside the semaphore? | No — retry is inside the coroutine. If a call fails with timeout, it retries once, then releases the semaphore. The retry itself re-acquires the semaphore implicitly (it's still inside `async with semaphore`). |
| Timeout per call | httpx timeout set to 30s per API call. If the target or judge LLM doesn't respond in 30s → `LLMTimeoutError` → retry once → if still fails → mark test as `error`. |
| Cancellation | If one test fails with a non-retryable error, other tests continue. The executor collects all results, including errors. |

### Concurrency Diagram (sequence)

```
Time ──────────────────────────────────────────────────────────────────────►

  Semaphore slots:  [■■■■■■■■■■] (10 max)

  Test 1 ───[LLM call────][score──][track]───► complete
  Test 2 ───[LLM call──────────────][score][track]───► complete
  Test 3 ───[LLM──timeout──retry──LLM──][score][track]───► complete
  ...
  Test 11 ───waiting for slot...──[LLM call──][score][track]───► complete
  ...
  Test 100 ───waiting for slot...................................──► complete
```

---

## 7. Scenario-to-Component Mapping

Every acceptance criterion has a definitive home.

### US-1: Core Eval Engine — Test Runner

| AC | Component | Method/Path |
|----|-----------|-------------|
| **AC-1.1** (run N prompts, pass/fail per test) | `runner/executor.py` | `Executor.run()` → returns `RunResult` with per-test `status: pass/fail/error` |
| **AC-1.2** (failed test shows actual, expected, diff) | `scoring/exact.py`, `scoring/semantic.py` | `ScoreResult` includes `overall` and `dimensions`; diff for exact is computed in scorer |
| **AC-1.3** (100+ prompts, <5 min, max 10 parallel) | `runner/executor.py` | `asyncio.Semaphore(10)` + `asyncio.gather` |
| **AC-1.4** (system prompt change → regression report) | `gate/checker.py`, `reporting/diff_reporter.py` | `GateChecker.check()` + `DiffReporter.generate_diff()` |
| **Edge: Empty suite** | `runner/executor.py` | `Executor.run()` handles `len(suite.tests) == 0` → returns empty `RunResult` |
| **Edge: LLM timeout** | `runner/retry.py` | `retry_with_backoff()` retries once, then marks `status="error"` with timeout reason |
| **Edge: All pass** | `runner/executor.py` | `Summary.pass_rate == 1.0`, `pass: true` in output |

### US-2: LLM-as-Judge Scoring

| AC | Component | Method/Path |
|----|-----------|-------------|
| **AC-2.1** (open-ended question → judge with rubric) | `scoring/rubric.py` | `RubricScorer.score(response, expected)` builds prompt from `expected.rubric.dimensions` |
| **AC-2.2** (multi-dimension scores 1-5 + overall) | `scoring/rubric.py`, `models/result.py` | `ScoreResult.dimensions` = list of `DimensionScore(score: 1-5)`; `overall` = weighted average normalized |
| **AC-2.3** (judge reasoning alongside score) | `scoring/rubric.py`, `models/result.py` | `DimensionScore.reasoning` populated from judge JSON response |
| **Edge: Invalid JSON** | `scoring/rubric.py` | Catch `json.JSONDecodeError` → retry once with stricter prompt (`STRICT_RETRY_PROMPT`) |
| **Edge: Dimension mismatch** | `scoring/rubric.py` | Validate parsed dimensions against expected rubric; warn on mismatch, still record |
| **Edge: Empty target response** | `scoring/rubric.py` | Detect empty response → return `overall=0.0`, dimensions all = 1, reasoning="no response provided" |

### US-3: Cost & Latency Tracking

| AC | Component | Method/Path |
|----|-----------|-------------|
| **AC-3.1** (per-test token counts + cost) | `tracking/cost.py` | `CostTracker.track(result)` records `result.tokens` and `result.cost_usd` |
| **AC-3.2** (aggregate: total cost, avg/p50/p95/p99 latency) | `tracking/latency.py` | `LatencyTracker.summarize()` computes percentiles from collected latencies |
| **AC-3.3** (regression comparison includes cost/latency deltas) | `reporting/diff_reporter.py` | `DiffReporter.generate_diff()` includes Δ cost and Δ latency columns |
| **Edge: Token counts unavailable** | `tracking/cost.py` | `TokenCount` fields are `int | None`; `CostTracker` stores `None`, reports `"N/A"` |
| **Edge: P99 with <10 samples** | `tracking/latency.py` | `LatencyTracker.summarize()` sets `warning_p99_unreliable=True`, still computes p99 |

### US-4: CI Gate Integration

| AC | Component | Method/Path |
|----|-----------|-------------|
| **AC-4.1** (load config, run suite, exit 0/1) | `gate/checker.py` | `GateChecker.check(config)` → `sys.exit(0 or 1)` |
| **AC-4.2** (3% regression within 5% → pass) | `gate/checker.py` | Compare regression % to `allowed_regression_pct` per suite |
| **AC-4.3** (8% regression >5% → fail with report) | `gate/checker.py`, `reporting/diff_reporter.py` | Gate fails → `DiffReporter` prints regressed tests |
| **AC-4.4** (no baseline → create baseline, pass) | `gate/checker.py` | `GateChecker` detects missing baseline → runs suite → saves as baseline → returns pass |
| **Edge: Config missing** | `config.py` | `load_config()` raises `FileNotFoundError` with message: "No config found. Run `evalforge init`." |
| **Edge: All metrics improved** | `gate/checker.py` | Regression is negative or zero → gate passes, saves new baseline |

### US-5: CLI & Report Output

| AC | Component | Method/Path |
|----|-----------|-------------|
| **AC-5.1** (`evalforge run <suite>` → stdout + JSON report) | `cli/main.py`, `reporting/` | `run()` calls `Executor.run()` → `ConsoleReporter.write(stdout)` + `JSONReporter.write(evalforge-output/report-{ts}.json)` |
| **AC-5.2** (`evalforge compare` → diff table) | `cli/main.py`, `reporting/diff_reporter.py` | `compare()` loads two JSON reports → `DiffReporter.generate_diff()` |
| **AC-5.3** (`evalforge init` → scaffolds project) | `cli/init.py` | `scaffold_project()` creates config, suite folder, .gitignore |
| **Edge: Compare non-existent baseline** | `cli/main.py` | `compare()` checks file existence before loading; raises clear error with paths searched |
| **Edge: Init in existing directory** | `cli/init.py` | `scaffold_project()` checks for existing `evalforge.yaml`; if found, prompts for confirmation (unless `--force`) |

---

## 8. Spike Verdict Integration

The `judge-json-validity` spike yielded concrete, binding constraints for the real build:

| Spike Finding | Architectural Decision | Where Enforced |
|---------------|----------------------|----------------|
| `max_tokens=700` required for 100% JSON parse success | Judge LLM calls default to `max_tokens=700`. Configurable but warns if set lower. | `judge/prompts.py` (default in `build_rubric_prompt`), `judge/client.py` (`generate()` default) |
| `deepseek-v4-flash` emits `reasoning_content` separate from `content` | `LLMResponse` has both fields. Parsers read `content` only. | `models/llm.py` (`LLMResponse.reasoning_content`), `judge/client.py` (DeepSeekClient parses both fields from API response) |
| 100% schema compliance at 700 tokens | RubricScorer validates parsed JSON against expected dimension names | `scoring/rubric.py` (schema validation after parse) |
| No markdown wrapping in DeepSeek responses | JSON parser expects raw `{...}`; no `` ```json `` strip needed. But `try_parse_json` fallback included for other providers. | `scoring/rubric.py` (JSON extraction logic) |
| ~$0.0001/eval, ~0.17s latency | Cost tracking is accurate per-eval; no need for batching or caching in v0.1 | `tracking/cost.py` (per-test cost accumulation) |
| Retry with stricter prompt as fallback | Implemented: on JSON parse failure → prepend `"IMPORTANT: Return ONLY valid JSON"` + retry once | `scoring/rubric.py` (retry path), `judge/prompts.py` (`STRICT_RETRY_PROMPT`) |
| 37% input token cache hits | Not actionable in v0.1; caching is provider-side. Future: could add local prompt cache for repeat judge calls. | Noted for v0.2 |

---

## Design Decisions Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Scorer is async even for sync implementations | Uniform interface; sync scorers `return` immediately but signature is consistent | Sync-only base class → rejected because RubricScorer needs async for LLM call |
| One `LLMClient` for both target and judge | Both target LLM and judge LLM use the same API shapes; differentiate by passing different `LLMClient` instances with different models | Separate `TargetClient` and `JudgeClient` → rejected as over-abstraction; same interface works for both |
| `FunctionScorer` takes a callable, not a file path | Maximum flexibility; users can pass lambdas, imported functions, or partials | Loading from import path → rejected as fragile; callable is simpler |
| Tracker is pull-model (executor pushes TestResults) | Executor owns the loop; trackers are passive accumulators | Push-model where tracker polls → rejected; adds unnecessary complexity |
| Gate saves baseline automatically, no separate command | Matches SPEC AC-4.4: "creates the baseline automatically." Reduces CLI surface area. | `evalforge baseline save` command → rejected; SPEC says auto-create |
| Config is YAML-only for v0.1 | SPEC data contracts are in YAML; pyyaml is a requirement | TOML support → deferred to future extension point |
| `rich` is optional dependency | ConsoleReporter can fall back to plain `print()` tables; rich adds color but isn't critical | Hard dependency → rejected to keep install footprint minimal |
