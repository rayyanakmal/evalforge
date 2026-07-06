"""Abstract base for all metrics trackers.

Trackers accumulate per-test metrics during a run and produce
aggregate summaries at the end.

Implementations: CostTracker, LatencyTracker
"""

from abc import ABC, abstractmethod

from evalforge.models.result import TestResult, TrackingSummary


class Tracker(ABC):
    """Abstract base for all metrics trackers.

    Trackers are passive accumulators: the Executor pushes TestResults
    via track(), and at the end of a run asks for summarize().
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
            TrackingSummary with aggregated cost and latency stats.

        Edge cases:
            - No results tracked → returns zeros, not errors.
            - <10 samples for p99 → sets warning_p99_unreliable = True.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all tracked data. Called before each new run."""
        ...
