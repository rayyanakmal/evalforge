"""Latency tracker — collects per-test latency and computes percentiles.

Handles the <10 samples edge case: sets warning_p99_unreliable = True
but still computes p99.
"""

from evalforge.models.result import TestResult, TrackingSummary
from evalforge.tracking.base import Tracker


class LatencyTracker(Tracker):
    """Collects per-test latency and computes avg, p50, p95, p99.

    Usage:
        tracker = LatencyTracker()
        for result in test_results:
            tracker.track(result)
        summary = tracker.summarize()
    """

    def __init__(self) -> None:
        self._latencies: list[float] = []

    def track(self, result: TestResult) -> None:
        """Record latency from a TestResult."""
        self._latencies.append(result.latency_ms)

    def summarize(self) -> TrackingSummary:
        """Compute avg, p50, p95, p99 from collected latencies.

        Returns:
            TrackingSummary with latency stats. If <10 samples,
            warning_p99_unreliable is True but p99 is still computed.
        """
        if not self._latencies:
            return TrackingSummary()

        avg = sum(self._latencies) / len(self._latencies)
        sorted_lat = sorted(self._latencies)
        warning = len(self._latencies) < 10

        return TrackingSummary(
            avg_latency_ms=avg,
            latency_p50=_percentile(sorted_lat, 50),
            latency_p95=_percentile(sorted_lat, 95),
            latency_p99=_percentile(sorted_lat, 99),
            warning_p99_unreliable=warning,
        )

    def reset(self) -> None:
        """Clear all accumulated latencies."""
        self._latencies.clear()


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute percentile from a sorted list using linear interpolation."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    return sorted_data[f]
