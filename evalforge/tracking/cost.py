"""Cost tracker — accumulates token counts and cost per test.

Handles the N/A edge case: if token counts are unavailable
(e.g., open-source model), stores None and reports N/A.
"""

from evalforge.models.result import TestResult, TrackingSummary
from evalforge.tracking.base import Tracker


class CostTracker(Tracker):
    """Accumulates input/output token counts and cost across test runs.

    Usage:
        tracker = CostTracker()
        for result in test_results:
            tracker.track(result)
        summary = tracker.summarize()
    """

    def __init__(self) -> None:
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self.na_count: int = 0  # Number of results with unavailable token data

    def track(self, result: TestResult) -> None:
        """Record token counts and cost from a TestResult.

        If result.tokens is None (open-source model), increments na_count
        and records 0 tokens but still tracks cost.
        """
        if result.tokens is None:
            self.na_count += 1
            self._total_cost_usd += result.cost_usd
            return

        inp = result.tokens.input or 0
        out = result.tokens.output or 0
        self._total_input_tokens += inp
        self._total_output_tokens += out
        self._total_cost_usd += result.cost_usd

    def summarize(self) -> TrackingSummary:
        """Return aggregate token and cost statistics."""
        return TrackingSummary(
            total_cost_usd=self._total_cost_usd,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
        )

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0
        self.na_count = 0
