"""
US-3: Cost & Latency Tracking

Tests for all acceptance criteria and edge cases:
  AC-3.1: Each test case records input/output token counts and total cost
  AC-3.2: Report includes aggregate stats: total cost, avg latency, p50/p95/p99
  AC-3.3: Regression comparison includes cost and latency deltas
  Edge:  Token counts unavailable → shows N/A, doesn't crash
  Edge:  P99 latency requires minimum 10 samples → shows warning if <10 cases
"""

import asyncio

import pytest

from evalforge.models.result import (
    TestResult, TokenCount, ScoreResult, Summary, RunResult, TrackingSummary,
)
from evalforge.models.suite import TestSuite, TestCase, Expected
from evalforge.models.llm import LLMResponse, Usage
from evalforge.scoring.exact import ExactScorer
from evalforge.tracking.base import Tracker
from evalforge.tracking.cost import CostTracker
from evalforge.tracking.latency import LatencyTracker
from evalforge.runner.executor import Executor, compare_results, ChangeItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_test_result(
    test_id: str = "t1",
    status: str = "pass",
    input_tokens: int | None = 10,
    output_tokens: int | None = 5,
    latency_ms: float = 100.0,
    cost_usd: float = 0.001,
) -> TestResult:
    """Build a TestResult with token/cost/latency fields."""
    tokens = None
    if input_tokens is not None or output_tokens is not None:
        inp = input_tokens or 0
        out = output_tokens or 0
        tokens = TokenCount(input=inp, output=out, total=inp + out)

    return TestResult(
        id=test_id,
        status=status,
        response="mock response",
        expected_value="4",
        score=ScoreResult(overall=1.0, method="exact"),
        tokens=tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


# ===================================================================
# Tracker ABC Tests
# ===================================================================

def test_tracker_is_abstract():
    """Tracker ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Tracker()  # type: ignore[abstract]


# ===================================================================
# CostTracker Tests (AC-3.1)
# ===================================================================

class TestCostTracker:
    """Tests for CostTracker — per-test token & cost tracking."""

    def test_track_records_tokens_and_cost(self):
        """track() records input/output token counts and cost from a TestResult."""
        tracker = CostTracker()
        tr = make_test_result(input_tokens=150, output_tokens=75, cost_usd=0.003)
        tracker.track(tr)

        summary = tracker.summarize()
        assert summary.total_input_tokens == 150
        assert summary.total_output_tokens == 75
        assert summary.total_cost_usd == pytest.approx(0.003)

    def test_summarize_aggregates_multiple_results(self):
        """summarize() correctly aggregates tokens and cost across multiple tests."""
        tracker = CostTracker()
        tracker.track(make_test_result("t1", input_tokens=10, output_tokens=5, cost_usd=0.001))
        tracker.track(make_test_result("t2", input_tokens=20, output_tokens=10, cost_usd=0.002))
        tracker.track(make_test_result("t3", input_tokens=30, output_tokens=15, cost_usd=0.003))

        summary = tracker.summarize()
        assert summary.total_input_tokens == 60
        assert summary.total_output_tokens == 30
        assert summary.total_cost_usd == pytest.approx(0.006)

    def test_edge_na_tokens_no_crash(self):
        """TokenCount fields are None (open-source model) → stored as None, reported as N/A."""
        tracker = CostTracker()
        tr = TestResult(
            id="os-1",
            status="pass",
            response="response",
            score=ScoreResult(overall=1.0, method="exact"),
            tokens=None,  # no token info available
            latency_ms=100.0,
            cost_usd=0.0,
        )
        tracker.track(tr)
        summary = tracker.summarize()
        # Should not crash
        assert summary.total_input_tokens == 0  # None treated as 0 for aggregation
        assert summary.total_output_tokens == 0
        # But the tracker should have recorded the NA flag
        assert tracker.na_count == 1

    def test_reset_clears_all_data(self):
        """reset() clears all accumulated data."""
        tracker = CostTracker()
        tracker.track(make_test_result(input_tokens=10, output_tokens=5, cost_usd=0.001))
        tracker.reset()

        summary = tracker.summarize()
        assert summary.total_input_tokens == 0
        assert summary.total_output_tokens == 0
        assert summary.total_cost_usd == 0.0
        assert tracker.na_count == 0

    def test_summarize_empty_no_crash(self):
        """summarize() on empty tracker returns zeros, not an error."""
        tracker = CostTracker()
        summary = tracker.summarize()
        assert summary.total_input_tokens == 0
        assert summary.total_output_tokens == 0
        assert summary.total_cost_usd == 0.0

    def test_mixed_na_and_valid_tokens(self):
        """Mixing results with and without token info works correctly."""
        tracker = CostTracker()
        tracker.track(make_test_result("t1", input_tokens=10, output_tokens=5, cost_usd=0.001))
        tracker.track(TestResult(
            id="t2", status="pass", response="x",
            score=ScoreResult(overall=1.0, method="exact"),
            tokens=None, latency_ms=50.0, cost_usd=0.0,
        ))
        tracker.track(make_test_result("t3", input_tokens=20, output_tokens=10, cost_usd=0.002))

        summary = tracker.summarize()
        assert summary.total_input_tokens == 30  # 10 + 0 + 20
        assert summary.total_output_tokens == 15  # 5 + 0 + 10
        assert summary.total_cost_usd == pytest.approx(0.003)
        assert tracker.na_count == 1


# ===================================================================
# LatencyTracker Tests (AC-3.2)
# ===================================================================

class TestLatencyTracker:
    """Tests for LatencyTracker — latency stats with percentiles."""

    def test_track_records_latency(self):
        """track() records latency_ms from a TestResult."""
        tracker = LatencyTracker()
        tr = make_test_result(latency_ms=125.5)
        tracker.track(tr)

        summary = tracker.summarize()
        assert summary.avg_latency_ms == pytest.approx(125.5)

    def test_summarize_computes_avg(self):
        """summarize() computes correct average latency."""
        tracker = LatencyTracker()
        for lat in [100.0, 200.0, 300.0]:
            tracker.track(make_test_result(latency_ms=lat))

        summary = tracker.summarize()
        assert summary.avg_latency_ms == pytest.approx(200.0)

    def test_summarize_computes_p50(self):
        """summarize() computes correct p50 (median)."""
        tracker = LatencyTracker()
        for lat in [300.0, 100.0, 200.0, 500.0, 400.0]:
            tracker.track(make_test_result(latency_ms=lat))

        summary = tracker.summarize()
        assert summary.latency_p50 == pytest.approx(300.0)  # median of [100,200,300,400,500]

    def test_summarize_computes_p95(self):
        """summarize() computes correct p95."""
        tracker = LatencyTracker()
        # 20 values — p95 is the 19th when sorted
        for i in range(20):
            tracker.track(make_test_result(latency_ms=float(i + 1)))

        summary = tracker.summarize()
        # p95 of 1..20: index = (20-1)*0.95 = 18.05 → interpolate between idx 18 (19) and 19 (20)
        expected = 19.0 + 0.05 * (20.0 - 19.0)
        assert summary.latency_p95 == pytest.approx(expected)

    def test_summarize_computes_p99(self):
        """summarize() computes correct p99."""
        tracker = LatencyTracker()
        # 100 values
        for i in range(100):
            tracker.track(make_test_result(latency_ms=float(i + 1)))

        summary = tracker.summarize()
        # p99 of 1..100: index = 99*0.99 = 98.01 → interpolate between idx 98 (99) and 99 (100)
        expected = 99.0 + 0.01 * (100.0 - 99.0)
        assert summary.latency_p99 == pytest.approx(expected)

    def test_edge_less_than_10_samples_sets_warning(self):
        """<10 samples → warning_p99_unreliable=True."""
        tracker = LatencyTracker()
        for lat in range(5):
            tracker.track(make_test_result(latency_ms=float(lat * 10)))

        summary = tracker.summarize()
        assert summary.warning_p99_unreliable is True

    def test_edge_10_or_more_samples_no_warning(self):
        """≥10 samples → warning_p99_unreliable=False."""
        tracker = LatencyTracker()
        for lat in range(10):
            tracker.track(make_test_result(latency_ms=float(lat * 10)))

        summary = tracker.summarize()
        assert summary.warning_p99_unreliable is False

    def test_reset_clears_all_data(self):
        """reset() clears all accumulated latencies."""
        tracker = LatencyTracker()
        tracker.track(make_test_result(latency_ms=100.0))
        tracker.reset()

        summary = tracker.summarize()
        assert summary.avg_latency_ms == 0.0
        assert summary.latency_p50 == 0.0
        assert summary.latency_p95 == 0.0
        assert summary.latency_p99 == 0.0

    def test_summarize_empty_no_crash(self):
        """summarize() on empty tracker returns zeros, not an error."""
        tracker = LatencyTracker()
        summary = tracker.summarize()
        assert summary.avg_latency_ms == 0.0
        assert summary.latency_p50 == 0.0
        assert summary.latency_p95 == 0.0
        assert summary.latency_p99 == 0.0


# ===================================================================
# TrackingSummary Model Tests
# ===================================================================

class TestTrackingSummary:
    """Tests for TrackingSummary Pydantic model."""

    def test_defaults_are_zero(self):
        """All numeric fields default to 0.0 or 0."""
        ts = TrackingSummary()
        assert ts.total_cost_usd == 0.0
        assert ts.avg_latency_ms == 0.0
        assert ts.latency_p50 == 0.0
        assert ts.latency_p95 == 0.0
        assert ts.latency_p99 == 0.0
        assert ts.total_input_tokens == 0
        assert ts.total_output_tokens == 0
        assert ts.warning_p99_unreliable is False

    def test_with_warning_flag(self):
        """warning_p99_unreliable field is accessible and settable."""
        ts = TrackingSummary(warning_p99_unreliable=True)
        assert ts.warning_p99_unreliable is True


# ===================================================================
# Integration: AC-3.1 via Executor
# ===================================================================

@pytest.mark.asyncio
async def test_ac3_1_test_case_records_tokens_and_cost():
    """AC-3.1: Given a test run, when completed, then each test case
    records input/output token counts and total cost."""
    suite = TestSuite(
        name="ac3-1",
        tests=[
            TestCase(
                id="t1",
                prompt="What is 2+2?",
                expected=Expected(type="exact", value="4"),
            ),
        ],
    )

    async def mock_generate(prompt: str) -> LLMResponse:
        return LLMResponse(
            content="4",
            usage=Usage(prompt_tokens=150, completion_tokens=75, total_tokens=225),
            latency_ms=120.0,
            cost_usd=0.004,
        )

    executor = Executor(generate_fn=mock_generate, scorer=ExactScorer())
    result = await executor.run(suite)

    t = result.tests[0]
    # Each test should have token info
    assert t.tokens is not None
    assert t.tokens.input == 150
    assert t.tokens.output == 75
    assert t.tokens.total == 225
    # Each test should have cost
    assert t.cost_usd == pytest.approx(0.004)
    # Each test should have latency
    assert t.latency_ms > 0


# ===================================================================
# Integration: AC-3.2 via Executor
# ===================================================================

@pytest.mark.asyncio
async def test_ac3_2_report_includes_aggregate_stats():
    """AC-3.2: Given a test run, when completed, then the report includes
    aggregate stats: total cost, avg latency, p50/p95/p99 latency."""
    suite = TestSuite(
        name="ac3-2",
        tests=[
            TestCase(id=f"t{i}", prompt=f"Q{i}",
                     expected=Expected(type="exact", value=f"A{i}"))
            for i in range(12)
        ],
    )

    async def mock_generate(prompt: str) -> LLMResponse:
        idx = int(prompt[1:])
        # Small sleep so wall-clock latency registers (>0)
        await asyncio.sleep(0.001)
        return LLMResponse(
            content=f"A{idx}",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=50.0 + idx * 10.0,
            cost_usd=0.001,
        )

    executor = Executor(generate_fn=mock_generate, scorer=ExactScorer())
    result = await executor.run(suite)

    s = result.summary
    # Total cost from tracker
    assert s.total_cost_usd == pytest.approx(0.012)  # 12 * 0.001
    # Latency stats exist (exact values depend on wall-clock)
    assert s.avg_latency_ms > 0, "avg latency should be tracked"
    assert s.latency_p50 is not None and s.latency_p50 > 0
    assert s.latency_p95 is not None and s.latency_p95 >= s.latency_p50
    assert s.latency_p99 is not None and s.latency_p99 >= s.latency_p95


# ===================================================================
# Integration: AC-3.3 Comparison includes cost/latency deltas
# ===================================================================

def test_ac3_3_comparison_includes_cost_delta():
    """AC-3.3: Regression comparison includes cost delta between runs."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    baseline = RunResult(
        suite_name="s1", timestamp=now, duration_ms=1000,
        tests=[
            TestResult(id="t1", status="pass", response="x",
                       score=ScoreResult(overall=1.0, method="exact"),
                       tokens=TokenCount(input=10, output=5, total=15),
                       latency_ms=100, cost_usd=0.010),
        ],
        summary=Summary(total=1, passed=1, failed=0, errored=0, pass_rate=1.0,
                        total_cost_usd=0.010),
    )
    candidate = RunResult(
        suite_name="s1", timestamp=now, duration_ms=900,
        tests=[
            TestResult(id="t1", status="pass", response="x",
                       score=ScoreResult(overall=1.0, method="exact"),
                       tokens=TokenCount(input=8, output=4, total=12),
                       latency_ms=80, cost_usd=0.008),
        ],
        summary=Summary(total=1, passed=1, failed=0, errored=0, pass_rate=1.0,
                        total_cost_usd=0.008),
    )

    report = compare_results(baseline, candidate)
    # Check that ChangeItem includes cost delta
    item = report.unchanged[0]
    assert hasattr(item, "cost_delta") or hasattr(item, "baseline_cost_usd")
    # If we extended ChangeItem:
    if hasattr(item, "cost_delta"):
        assert item.cost_delta == pytest.approx(-0.002)  # 0.008 - 0.010


def test_ac3_3_comparison_includes_latency_delta():
    """AC-3.3: Regression comparison includes latency delta between runs."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    baseline = RunResult(
        suite_name="s1", timestamp=now, duration_ms=1000,
        tests=[
            TestResult(id="t1", status="pass", response="x",
                       score=ScoreResult(overall=1.0, method="exact"),
                       tokens=TokenCount(input=10, output=5, total=15),
                       latency_ms=100, cost_usd=0.001),
        ],
        summary=Summary(total=1, passed=1, failed=0, errored=0, pass_rate=1.0),
    )
    candidate = RunResult(
        suite_name="s1", timestamp=now, duration_ms=900,
        tests=[
            TestResult(id="t1", status="pass", response="x",
                       score=ScoreResult(overall=1.0, method="exact"),
                       tokens=TokenCount(input=10, output=5, total=15),
                       latency_ms=150, cost_usd=0.001),  # 50ms slower
        ],
        summary=Summary(total=1, passed=1, failed=0, errored=0, pass_rate=1.0),
    )

    report = compare_results(baseline, candidate)
    item = report.unchanged[0]
    if hasattr(item, "latency_delta"):
        assert item.latency_delta == pytest.approx(50.0)  # 150 - 100


# ===================================================================
# Edge Case: Token counts unavailable (via executor)
# ===================================================================

@pytest.mark.asyncio
async def test_edge_token_counts_unavailable_no_crash():
    """Token counts unavailable (open-source model) → shows N/A, doesn't crash."""
    suite = TestSuite(
        name="no-tokens",
        tests=[
            TestCase(
                id="os-1",
                prompt="Hello",
                expected=Expected(type="exact", value="Hi"),
            ),
        ],
    )

    async def mock_generate(prompt: str) -> LLMResponse:
        return LLMResponse(
            content="Hi",
            usage=None,  # No usage info (open-source model)
            latency_ms=50.0,
            cost_usd=0.0,
        )

    executor = Executor(generate_fn=mock_generate, scorer=ExactScorer())
    result = await executor.run(suite)

    t = result.tests[0]
    # Should not crash; tokens should be None when usage is None
    assert t.tokens is None
    # The test should still pass/fail normally
    assert t.status == "pass"
    # Summary should still work
    assert result.summary.total_cost_usd == 0.0


# ===================================================================
# Edge Case: P99 with <10 samples (via executor)
# ===================================================================

@pytest.mark.asyncio
async def test_edge_p99_less_than_10_samples_handled():
    """P99 latency with <10 cases is still computed but executor handles gracefully."""
    suite = TestSuite(
        name="small-suite",
        tests=[
            TestCase(id=f"t{i}", prompt=f"Q{i}",
                     expected=Expected(type="exact", value=f"A{i}"))
            for i in range(5)  # only 5 tests
        ],
    )

    async def mock_generate(prompt: str) -> LLMResponse:
        idx = int(prompt[1:])
        return LLMResponse(
            content=f"A{idx}",
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            latency_ms=10.0 * idx,
            cost_usd=0.001,
        )

    executor = Executor(generate_fn=mock_generate, scorer=ExactScorer())
    result = await executor.run(suite)

    # Should not crash; p99 may be None or 0 for small datasets
    assert result.summary.total == 5
    # The summary should still have latency_p99 (even if unreliable)
    assert result.summary.latency_p99 is not None or result.summary.latency_p99 == 0.0


# ===================================================================
# Test that CostTracker and LatencyTracker fulfill Tracker ABC
# ===================================================================

def test_cost_tracker_is_tracker_subclass():
    """CostTracker is a valid Tracker subclass."""
    assert issubclass(CostTracker, Tracker)


def test_latency_tracker_is_tracker_subclass():
    """LatencyTracker is a valid Tracker subclass."""
    assert issubclass(LatencyTracker, Tracker)


# ===================================================================
# Test: Tracker.reset is part of the ABC contract
# ===================================================================

def test_tracker_abc_requires_reset():
    """Tracker ABC requires reset() method."""
    assert hasattr(Tracker, "reset")
    from abc import abstractmethod
    # reset should be an abstract method
    assert Tracker.reset.__isabstractmethod__


# ===================================================================
# Test: compare_results includes cost/latency deltas in ChangeItem
# ===================================================================

def test_changeitem_has_cost_and_latency_delta_fields():
    """ChangeItem dataclass includes cost_delta and latency_delta fields."""
    from dataclasses import fields as dc_fields
    field_names = {f.name for f in dc_fields(ChangeItem)}
    assert "cost_delta" in field_names, "ChangeItem must have cost_delta field"
    assert "latency_delta" in field_names, "ChangeItem must have latency_delta field"
