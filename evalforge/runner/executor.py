"""Concurrent test executor — the central engine of EvalForge.

Executes a TestSuite against a target LLM, scores results, and produces
a RunResult with per-test pass/fail/error status and aggregate summary.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Awaitable, Optional

from evalforge.models.suite import TestSuite, TestCase
from evalforge.models.result import (
    RunResult, TestResult, ScoreResult, Summary, TokenCount,
)
from evalforge.models.llm import LLMResponse
from evalforge.scoring.base import Scorer
from evalforge.runner.retry import retry_with_backoff
from evalforge.tracking.cost import CostTracker
from evalforge.tracking.latency import LatencyTracker

logger = logging.getLogger(__name__)


# Type alias for the LLM generate function
GenerateFn = Callable[[str], Awaitable[LLMResponse]]


# ---------------------------------------------------------------------------
# Comparison report for AC-1.4
# ---------------------------------------------------------------------------

@dataclass
class ChangeItem:
    """A single test's status change between two runs."""
    test_id: str
    baseline_status: str
    candidate_status: str
    cost_delta: float = 0.0
    latency_delta: float = 0.0


@dataclass
class ComparisonReport:
    """Result of comparing two RunResults."""
    regressions: list[ChangeItem] = field(default_factory=list)
    improvements: list[ChangeItem] = field(default_factory=list)
    unchanged: list[ChangeItem] = field(default_factory=list)

    @property
    def regression_count(self) -> int:
        return len(self.regressions)

    @property
    def improvement_count(self) -> int:
        return len(self.improvements)


def compare_results(baseline: RunResult, candidate: RunResult) -> ComparisonReport:
    """Compare two RunResults and produce a regression report.

    A regression = pass→fail or fail→error or pass→error
    An improvement = fail→pass or error→pass

    Args:
        baseline: The previous (baseline) run result.
        candidate: The new (candidate) run result.

    Returns:
        ComparisonReport with regressions, improvements, and unchanged lists.
    """
    baseline_map = {t.id: t for t in baseline.tests}
    candidate_map = {t.id: t for t in candidate.tests}

    report = ComparisonReport()

    all_ids = set(baseline_map.keys()) | set(candidate_map.keys())

    for tid in sorted(all_ids):
        b = baseline_map.get(tid)
        c = candidate_map.get(tid)

        b_status = b.status if b else "missing"
        c_status = c.status if c else "missing"

        b_cost = b.cost_usd if b else 0.0
        c_cost = c.cost_usd if c else 0.0
        b_lat = b.latency_ms if b else 0.0
        c_lat = c.latency_ms if c else 0.0

        item = ChangeItem(
            test_id=tid,
            baseline_status=b_status,
            candidate_status=c_status,
            cost_delta=c_cost - b_cost,
            latency_delta=c_lat - b_lat,
        )

        # Determine regression vs improvement
        # pass → fail, pass → error, fail → error = regression
        if _is_regression(b_status, c_status):
            report.regressions.append(item)
        elif _is_improvement(b_status, c_status):
            report.improvements.append(item)
        else:
            report.unchanged.append(item)

    return report


def _is_regression(before: str, after: str) -> bool:
    """True if after is worse than before."""
    order = {"pass": 0, "fail": 1, "error": 2, "missing": 3}
    return order.get(after, 0) > order.get(before, 0)


def _is_improvement(before: str, after: str) -> bool:
    """True if after is better than before."""
    order = {"pass": 0, "fail": 1, "error": 2, "missing": 3}
    return order.get(after, 0) < order.get(before, 0)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """Concurrent test executor with bounded parallelism.

    Runs a TestSuite against a configured LLM, scores each response,
    and returns a RunResult with per-test status and aggregate summary.

    Usage:
        executor = Executor(generate_fn=my_llm_call, scorer=ExactScorer())
        result = await executor.run(suite)
    """

    def __init__(
        self,
        generate_fn: GenerateFn,
        scorer: Scorer,
        concurrency: int = 10,
    ):
        self.generate_fn = generate_fn
        self.scorer = scorer
        self.concurrency = concurrency

    async def run(self, suite: TestSuite) -> RunResult:
        """Execute all test cases in the suite concurrently.

        Args:
            suite: The TestSuite to execute.

        Returns:
            RunResult with per-test results and aggregate summary.
        """
        start_time = time.monotonic()

        # Initialize trackers for this run
        cost_tracker = CostTracker()
        latency_tracker = LatencyTracker()

        # Edge case: empty suite
        if not suite.tests:
            duration_ms = (time.monotonic() - start_time) * 1000
            return RunResult(
                suite_name=suite.name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration_ms,
                tests=[],
                summary=Summary(total=0, passed=0, failed=0, errored=0, pass_rate=0.0),
            )

        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one_test(test: TestCase) -> TestResult:
            async with semaphore:
                return await self._execute_test_with_tracking(
                    test, cost_tracker, latency_tracker
                )

        results = await asyncio.gather(
            *(run_one_test(t) for t in suite.tests)
        )

        duration_ms = (time.monotonic() - start_time) * 1000

        # Build summary with tracker data
        summary = self._build_summary(results, cost_tracker, latency_tracker)

        return RunResult(
            suite_name=suite.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
            tests=list(results),
            summary=summary,
        )

    async def _execute_test_with_tracking(
        self,
        test: TestCase,
        cost_tracker: CostTracker,
        latency_tracker: LatencyTracker,
    ) -> TestResult:
        """Execute a single test and feed results to trackers."""
        result = await self._execute_test(test)
        cost_tracker.track(result)
        latency_tracker.track(result)
        return result

    async def _execute_test(self, test: TestCase) -> TestResult:
        """Execute a single test case: call LLM, score, return result."""
        t_start = time.monotonic()
        response: Optional[str] = None
        error_msg: Optional[str] = None
        score: Optional[ScoreResult] = None
        tokens: Optional[TokenCount] = None
        cost_usd: float = 0.0

        try:
            # Call LLM with retry on transient failures
            llm_response = await retry_with_backoff(
                lambda: self.generate_fn(test.prompt),
                max_retries=1,
                base_delay=1.0,
            )
            response = llm_response.content
            if llm_response.usage:
                tokens = TokenCount(
                    input=llm_response.usage.prompt_tokens,
                    output=llm_response.usage.completion_tokens,
                    total=llm_response.usage.total_tokens,
                )
            cost_usd = llm_response.cost_usd

            # Score the response
            score = await self.scorer.score(response, test.expected)

        except (TimeoutError, asyncio.TimeoutError) as e:
            error_msg = f"LLM timeout: {e}"
        except Exception as e:
            error_msg = str(e)

        latency_ms = (time.monotonic() - t_start) * 1000

        # Determine status
        if error_msg:
            status = "error"
        elif score and score.overall >= 0.5:
            status = "pass"
        else:
            status = "fail"

        # Build error/reason for failed tests (AC-1.2)
        failure_reason = None
        if status == "fail" and score is not None and score.overall < 0.5:
            failure_reason = (
                f"Expected '{test.expected.value or ''}', got '{response or ''}' "
                f"— no match (score={score.overall})"
            )
        elif status == "error":
            failure_reason = error_msg

        return TestResult(
            id=test.id,
            status=status,
            response=response,
            expected_value=test.expected.value,
            score=score,
            tokens=tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            error=failure_reason,
        )

    @staticmethod
    def _build_summary(
        results: list[TestResult],
        cost_tracker: Optional[CostTracker] = None,
        latency_tracker: Optional[LatencyTracker] = None,
    ) -> Summary:
        """Compute aggregate statistics from test results and trackers."""
        total = len(results)
        passed = sum(1 for r in results if r.status == "pass")
        failed = sum(1 for r in results if r.status == "fail")
        errored = sum(1 for r in results if r.status == "error")

        pass_rate = passed / total if total > 0 else 0.0

        # Use tracker summaries if available, fall back to direct computation
        if cost_tracker is not None:
            cost_summary = cost_tracker.summarize()
            total_cost = cost_summary.total_cost_usd
        else:
            total_cost = sum(r.cost_usd for r in results)

        if latency_tracker is not None:
            lat_summary = latency_tracker.summarize()
            avg_latency = lat_summary.avg_latency_ms
            p50 = lat_summary.latency_p50 if lat_summary.latency_p50 > 0 else None
            p95 = lat_summary.latency_p95 if lat_summary.latency_p95 > 0 else None
            p99 = lat_summary.latency_p99 if lat_summary.latency_p99 > 0 else None
        else:
            latencies = [r.latency_ms for r in results if r.latency_ms > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            p50 = _percentile(latencies, 50) if latencies else None
            p95 = _percentile(latencies, 95) if latencies else None
            p99 = _percentile(latencies, 99) if latencies else None

        return Summary(
            total=total,
            passed=passed,
            failed=failed,
            errored=errored,
            pass_rate=pass_rate,
            total_cost_usd=total_cost,
            avg_latency_ms=avg_latency,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
        )


def _percentile(data: list[float], pct: float) -> float:
    """Compute percentile from a list of values (linear interpolation)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    return sorted_data[f]
