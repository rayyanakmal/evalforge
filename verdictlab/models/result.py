"""RunResult and related Pydantic models."""

from typing import Optional, Literal

from pydantic import BaseModel, Field

from .tokens import TokenCount
from .trajectory import Trajectory

# Backward-compat re-export: existing code imports TokenCount from
# verdictlab.models.result; the class itself lives in models/tokens.py
# to avoid a circular import (result.py -> trajectory.py -> tokens.py).

__all__ = ["TokenCount"]


class DimensionScore(BaseModel):
    """Per-dimension score from rubric evaluation."""
    __test__ = False
    name: str
    score: int = Field(ge=1, le=5)
    reasoning: str = ""


class ScoreResult(BaseModel):
    """Result from scoring a single test response."""
    __test__ = False
    overall: float = Field(ge=0.0, le=1.0)
    dimensions: Optional[list[DimensionScore]] = None
    method: str


class TestResult(BaseModel):
    """Result for a single test case execution."""
    __test__ = False
    id: str
    status: Literal["pass", "fail", "error"]
    response: Optional[str] = None
    expected_value: Optional[str] = None
    score: Optional[ScoreResult] = None
    tokens: Optional[TokenCount] = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: Optional[str] = None
    # v2: the journey. Optional + default None so v1 run files parse unchanged (V4).
    trajectory: Optional[Trajectory] = None


class Summary(BaseModel):
    """Aggregate statistics for a full test run."""
    __test__ = False
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    pass_rate: float = 0.0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None


class TrackingSummary(BaseModel):
    """Aggregate metrics produced by Trackers.

    Produced by CostTracker and LatencyTracker.summarize().
    Separate from Summary to keep tracking concerns isolated.
    """
    __test__ = False
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    warning_p99_unreliable: bool = False


class RunResult(BaseModel):
    """Complete result of a test suite execution."""
    __test__ = False
    suite_name: str
    timestamp: str
    duration_ms: float
    tests: list[TestResult] = Field(default_factory=list)
    summary: Summary = Field(default_factory=Summary)
    # v2: optional aggregate process metrics, populated by the metrics layer.
    trajectory_summary: Optional[dict] = None


def build_summary_from_tests(tests: list[TestResult]) -> Summary:
    """Compute a Summary from raw TestResults (no trackers needed).

    Used by importers and the UI when a RunResult arrives without a summary
    (e.g. imported trajectory reports). Mirrors TestRunner._build_summary's
    fallback path: direct sums, percentiles over positive latencies.
    """
    total = len(tests)
    passed = sum(1 for t in tests if t.status == "pass")
    failed = sum(1 for t in tests if t.status == "fail")
    errored = sum(1 for t in tests if t.status == "error")
    pass_rate = passed / total if total > 0 else 0.0
    total_cost = sum(t.cost_usd or 0.0 for t in tests)
    latencies = [t.latency_ms for t in tests if (t.latency_ms or 0) > 0]
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


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile over a sorted copy (mirrors executor)."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((pct / 100.0) * len(s)) - 1)))
    return s[idx]
