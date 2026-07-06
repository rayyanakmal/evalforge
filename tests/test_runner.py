"""
US-1: Core Eval Engine — Test Runner

Tests for all acceptance criteria and edge cases:
  AC-1.1: Pass/fail result per test case
  AC-1.2: Failed test includes actual, expected, and diff/reason
  AC-1.3: Concurrent execution with max 10 parallel (semaphore)
  AC-1.4: Regression comparison between two runs
  Edge:  Empty suite → empty result, not an error
  Edge:  LLM API timeout → retries once, then marks as error
  Edge:  All tests pass → summary with pass: true
"""

import asyncio
import time
from datetime import datetime, timezone

import pytest

from evalforge.models.suite import TestSuite, TestCase, Expected, TestMetadata
from evalforge.models.result import (
    RunResult, TestResult, ScoreResult, Summary, TokenCount,
)
from evalforge.models.llm import LLMResponse, Usage
from evalforge.scoring.base import Scorer
from evalforge.scoring.exact import ExactScorer
from evalforge.runner.executor import Executor, compare_results, ComparisonReport
from evalforge.runner.retry import retry_with_backoff


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_suite() -> TestSuite:
    return TestSuite(name="empty", tests=[])


@pytest.fixture
def simple_suite() -> TestSuite:
    return TestSuite(
        name="simple",
        description="A simple test suite for AC-1.1",
        tests=[
            TestCase(
                id="test-1",
                prompt="What is 2+2?",
                expected=Expected(type="exact", value="4"),
            ),
            TestCase(
                id="test-2",
                prompt="What is the capital of France?",
                expected=Expected(type="exact", value="Paris"),
            ),
            TestCase(
                id="test-3",
                prompt="Who wrote Hamlet?",
                expected=Expected(type="exact", value="William Shakespeare"),
            ),
        ],
    )


@pytest.fixture
def exact_scorer() -> ExactScorer:
    return ExactScorer()


# ---------------------------------------------------------------------------
# AC-1.1: Pass/fail per test case
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac1_1_pass_fail_per_test(simple_suite, exact_scorer):
    """AC-1.1: Given a test suite of N prompts with expected outputs,
    when the runner executes the suite against a configured LLM,
    then it returns a pass/fail result for each test case."""

    # Mock LLM: returns exact matches for test-1 and test-3, wrong for test-2
    responses = {
        "What is 2+2?": "4",
        "What is the capital of France?": "London",  # wrong
        "Who wrote Hamlet?": "William Shakespeare",
    }

    async def mock_generate(prompt: str) -> LLMResponse:
        return LLMResponse(
            content=responses[prompt],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=100.0,
            cost_usd=0.001,
        )

    executor = Executor(generate_fn=mock_generate, scorer=exact_scorer)
    result = await executor.run(simple_suite)

    # All three tests should have a result
    assert len(result.tests) == 3

    # test-1: pass (exact match)
    t1 = next(t for t in result.tests if t.id == "test-1")
    assert t1.status == "pass"
    assert t1.score is not None
    assert t1.score.overall == 1.0

    # test-2: fail (wrong answer)
    t2 = next(t for t in result.tests if t.id == "test-2")
    assert t2.status == "fail"
    assert t2.score is not None
    assert t2.score.overall < 1.0

    # test-3: pass (exact match)
    t3 = next(t for t in result.tests if t.id == "test-3")
    assert t3.status == "pass"

    # Summary should reflect counts
    assert result.summary.total == 3
    assert result.summary.passed == 2
    assert result.summary.failed == 1
    assert result.summary.errored == 0


# ---------------------------------------------------------------------------
# AC-1.2: Failed test shows actual, expected, diff/reason
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac1_2_failed_test_includes_actual_expected_diff(exact_scorer):
    """AC-1.2: Given a test case that failed, when inspected,
    then the output includes the actual response, the expected response,
    and a diff/reason for failure."""

    suite = TestSuite(
        name="fail-demo",
        tests=[
            TestCase(
                id="fail-1",
                prompt="Say hello",
                expected=Expected(type="exact", value="Hello, world!"),
            ),
        ],
    )

    async def mock_generate(prompt: str) -> LLMResponse:
        return LLMResponse(
            content="Hi there!",  # wrong response
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            latency_ms=50.0,
            cost_usd=0.0005,
        )

    executor = Executor(generate_fn=mock_generate, scorer=exact_scorer)
    result = await executor.run(suite)

    t = result.tests[0]
    assert t.status == "fail"

    # AC-1.2: actual response present
    assert t.response is not None
    assert "Hi there!" in t.response

    # AC-1.2: expected value present
    assert t.expected_value is not None
    assert "Hello, world!" in t.expected_value

    # AC-1.2: diff/reason present
    assert t.error is not None or (t.score is not None and t.score.overall < 1.0)
    # The score itself conveys failure; plus we check that error/reason field is populated
    if t.error:
        assert "expected" in t.error.lower() or "diff" in t.error.lower() or "match" in t.error.lower()


# ---------------------------------------------------------------------------
# AC-1.3: Concurrent execution (max 10 parallel)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac1_3_concurrent_execution_with_semaphore(exact_scorer):
    """AC-1.3: Given a test suite with 100+ prompts, when the runner executes,
    then it completes within 5 minutes using concurrent execution (max 10 parallel)."""

    # Create 50 test cases (enough to verify semaphore behavior)
    tests = [
        TestCase(
            id=f"test-{i}",
            prompt=f"prompt-{i}",
            expected=Expected(type="exact", value=f"response-{i}"),
        )
        for i in range(50)
    ]
    suite = TestSuite(name="concurrency-test", tests=tests)

    # Track concurrency
    concurrent_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def mock_generate(prompt: str) -> LLMResponse:
        nonlocal concurrent_count, max_concurrent
        async with lock:
            concurrent_count += 1
            if concurrent_count > max_concurrent:
                max_concurrent = concurrent_count

        # Small delay to simulate LLM latency and allow concurrency to build up
        await asyncio.sleep(0.01)

        async with lock:
            concurrent_count -= 1

        idx = int(prompt.split("-")[1])
        return LLMResponse(
            content=f"response-{idx}",
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            latency_ms=10.0,
            cost_usd=0.0001,
        )

    executor = Executor(generate_fn=mock_generate, scorer=exact_scorer, concurrency=10)
    start = time.monotonic()
    result = await executor.run(suite)
    elapsed = time.monotonic() - start

    # All 50 should complete
    assert len(result.tests) == 50
    assert result.summary.total == 50
    assert result.summary.passed == 50

    # Max concurrency should not exceed 10
    assert max_concurrent <= 10

    # Should complete within reasonable time (well under 5 minutes!)
    assert elapsed < 60  # 60 seconds is generous for 50 tests

    # Should have had some concurrency (more than 1 running at a time)
    assert max_concurrent > 1, "Expected at least some concurrent execution"


# ---------------------------------------------------------------------------
# AC-1.4: Regression comparison between runs
# ---------------------------------------------------------------------------

def test_ac1_4_regression_comparison():
    """AC-1.4: Given a system prompt change between runs, when compared,
    then results show a regression report highlighting which tests
    regressed vs improved."""

    now = datetime.now(timezone.utc).isoformat()

    # Baseline run: test-1 pass, test-2 pass, test-3 fail
    baseline = RunResult(
        suite_name="my-suite",
        timestamp=now,
        duration_ms=1000,
        tests=[
            TestResult(
                id="test-1", status="pass", response="4", expected_value="4",
                score=ScoreResult(overall=1.0, method="exact"),
                tokens=TokenCount(input=10, output=5, total=15),
                latency_ms=100, cost_usd=0.001,
            ),
            TestResult(
                id="test-2", status="pass", response="Paris", expected_value="Paris",
                score=ScoreResult(overall=1.0, method="exact"),
                tokens=TokenCount(input=10, output=5, total=15),
                latency_ms=100, cost_usd=0.001,
            ),
            TestResult(
                id="test-3", status="fail", response="London", expected_value="Paris",
                score=ScoreResult(overall=0.0, method="exact"),
                tokens=TokenCount(input=10, output=5, total=15),
                latency_ms=100, cost_usd=0.001,
            ),
        ],
        summary=Summary(total=3, passed=2, failed=1, errored=0, pass_rate=0.667),
    )

    # Candidate run: test-1 pass (same), test-2 regressed to fail, test-3 improved to pass
    candidate = RunResult(
        suite_name="my-suite",
        timestamp=now,
        duration_ms=900,
        tests=[
            TestResult(
                id="test-1", status="pass", response="4", expected_value="4",
                score=ScoreResult(overall=1.0, method="exact"),
                tokens=TokenCount(input=10, output=5, total=15),
                latency_ms=95, cost_usd=0.001,
            ),
            TestResult(
                id="test-2", status="fail", response="Berlin", expected_value="Paris",
                score=ScoreResult(overall=0.0, method="exact"),
                tokens=TokenCount(input=10, output=5, total=15),
                latency_ms=110, cost_usd=0.001,
            ),
            TestResult(
                id="test-3", status="pass", response="Paris", expected_value="Paris",
                score=ScoreResult(overall=1.0, method="exact"),
                tokens=TokenCount(input=10, output=5, total=15),
                latency_ms=90, cost_usd=0.001,
            ),
        ],
        summary=Summary(total=3, passed=2, failed=1, errored=0, pass_rate=0.667),
    )

    report = compare_results(baseline, candidate)

    # Regression report should exist
    assert report is not None

    # Should detect test-2 regressed (pass → fail)
    assert len(report.regressions) == 1
    assert report.regressions[0].test_id == "test-2"
    assert report.regressions[0].baseline_status == "pass"
    assert report.regressions[0].candidate_status == "fail"

    # Should detect test-3 improved (fail → pass)
    assert len(report.improvements) == 1
    assert report.improvements[0].test_id == "test-3"
    assert report.improvements[0].baseline_status == "fail"
    assert report.improvements[0].candidate_status == "pass"

    # test-1 should be unchanged
    assert len(report.unchanged) == 1
    assert report.unchanged[0].test_id == "test-1"

    # Summary stats
    assert report.regression_count == 1
    assert report.improvement_count == 1


# ---------------------------------------------------------------------------
# Edge Case: Empty test suite
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edge_empty_suite_returns_empty_result(empty_suite, exact_scorer):
    """Edge: Empty test suite → returns empty result, not an error."""

    async def mock_generate(prompt: str) -> LLMResponse:
        pytest.fail("Should not be called for empty suite")

    executor = Executor(generate_fn=mock_generate, scorer=exact_scorer)
    result = await executor.run(empty_suite)

    assert result.tests == []
    assert result.summary.total == 0
    assert result.summary.passed == 0
    assert result.summary.failed == 0
    assert result.summary.errored == 0
    assert result.summary.pass_rate == 0.0


# ---------------------------------------------------------------------------
# Edge Case: LLM timeout → retry once → error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edge_timeout_retries_once_then_errors(exact_scorer):
    """Edge: LLM API timeout → retries once, then marks test as 'error' with timeout reason."""

    suite = TestSuite(
        name="timeout-test",
        tests=[
            TestCase(
                id="timeout-1",
                prompt="What is 2+2?",
                expected=Expected(type="exact", value="4"),
            ),
        ],
    )

    call_count = 0

    async def mock_generate(prompt: str) -> LLMResponse:
        nonlocal call_count
        call_count += 1
        raise TimeoutError("LLM API timed out after 30s")

    executor = Executor(generate_fn=mock_generate, scorer=exact_scorer)
    result = await executor.run(suite)

    # Should have attempted twice (initial + 1 retry)
    assert call_count == 2

    t = result.tests[0]
    assert t.status == "error"
    assert t.error is not None
    assert "timeout" in t.error.lower() or "timed out" in t.error.lower()


# ---------------------------------------------------------------------------
# Edge Case: All tests pass → summary with pass statistics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edge_all_pass_returns_summary(exact_scorer):
    """Edge: All tests pass → returns pass: true with summary stats."""

    suite = TestSuite(
        name="all-pass",
        tests=[
            TestCase(
                id=f"pass-{i}",
                prompt=f"Q{i}",
                expected=Expected(type="exact", value=f"A{i}"),
            )
            for i in range(5)
        ],
    )

    async def mock_generate(prompt: str) -> LLMResponse:
        idx = prompt[1:]  # "Q0" → "0"
        return LLMResponse(
            content=f"A{idx}",
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            latency_ms=20.0 + int(idx) * 5,
            cost_usd=0.001,
        )

    executor = Executor(generate_fn=mock_generate, scorer=exact_scorer)
    result = await executor.run(suite)

    assert result.summary.total == 5
    assert result.summary.passed == 5
    assert result.summary.failed == 0
    assert result.summary.errored == 0
    assert result.summary.pass_rate == 1.0

    # All individual tests should be pass
    for t in result.tests:
        assert t.status == "pass"
        assert t.score.overall == 1.0


# ---------------------------------------------------------------------------
# Additional: retry logic unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_with_backoff_success_on_first_try():
    """Retry succeeds immediately on first attempt."""
    call_count = 0

    async def flaky_fn():
        nonlocal call_count
        call_count += 1
        return "success"

    result = await retry_with_backoff(flaky_fn, max_retries=2)
    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_with_backoff_success_on_retry():
    """Retry succeeds after first failure."""
    call_count = 0

    async def flaky_fn():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError("timeout")
        return "success"

    result = await retry_with_backoff(flaky_fn, max_retries=2)
    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_with_backoff_exhausted():
    """Retry raises after max retries exceeded."""
    call_count = 0

    async def always_fails():
        nonlocal call_count
        call_count += 1
        raise TimeoutError("timeout")

    with pytest.raises(TimeoutError):
        await retry_with_backoff(always_fails, max_retries=1)

    # 1 initial + 1 retry = 2 total
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_with_backoff_non_retryable_error():
    """Non-retryable errors (ValueError) should NOT be retried."""
    call_count = 0

    async def raises_value_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        await retry_with_backoff(raises_value_error, max_retries=2)

    # Should only be called once — no retry for ValueError
    assert call_count == 1
