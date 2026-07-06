"""DiffReporter — comparison diff table for `evalforge compare`.

Produces a table comparing baseline vs. candidate RunResults with columns:
test name, status, score Δ, cost Δ, latency Δ.
Highlights regressions (↓) vs improvements (↑).
"""

import sys
from pathlib import Path

from evalforge.models.result import RunResult, TestResult
from evalforge.reporting.base import Reporter
from evalforge.runner.executor import compare_results


class DiffReporter(Reporter):
    """Generate a diff table comparing two RunResults.

    Usage:
        reporter = DiffReporter()
        diff = reporter.generate_diff(baseline_result, candidate_result)
        print(diff)
    """

    def generate(self, result: RunResult) -> str:
        """Not used directly — use generate_diff() instead.

        Raises:
            NotImplementedError: DiffReporter is designed for pairwise comparison.
        """
        raise NotImplementedError(
            "Use generate_diff(baseline, candidate) for comparison reports."
        )

    def write(self, result: RunResult, path: Path) -> None:
        """Not used directly — use write_diff() instead.

        Raises:
            NotImplementedError: DiffReporter is designed for pairwise comparison.
        """
        raise NotImplementedError(
            "Use write_diff(baseline, candidate) for comparison reports."
        )

    def generate_diff(self, baseline: RunResult, candidate: RunResult) -> str:
        """Generate a diff table comparing two RunResults.

        Columns: test name, status (baseline → candidate), score Δ,
                 cost Δ, latency Δ.

        Args:
            baseline: The previous (baseline) run result.
            candidate: The new (candidate) run result.

        Returns:
            Formatted diff table string.
        """
        report = compare_results(baseline, candidate)
        lines: list[str] = []

        lines.append("")
        lines.append(f"  Baseline:  {baseline.suite_name} ({baseline.timestamp})")
        lines.append(f"  Candidate: {candidate.suite_name} ({candidate.timestamp})")
        lines.append("")

        if not report.regressions and not report.improvements and not report.unchanged:
            lines.append("  Both runs are empty.")
            return "\n".join(lines)

        # Table header
        lines.append(f"  {'Test':<20} {'Status':<18} {'Score Δ':<10} {'Cost Δ':<12} {'Latency Δ':<12}")
        lines.append(f"  {'-'*20} {'-'*18} {'-'*10} {'-'*12} {'-'*12}")

        def status_change(item) -> str:
            """Format status column with indicator."""
            if item.baseline_status == item.candidate_status:
                return f"{item.baseline_status} (no change)"
            arrow = "↓" if _is_worse(item.baseline_status, item.candidate_status) else "↑"
            return f"{item.baseline_status}→{item.candidate_status} {arrow}"

        def score_from_item(item, which: str):
            """Try to get score for a test from baseline or candidate map."""
            bmap = {t.id: t for t in baseline.tests}
            cmap = {t.id: t for t in candidate.tests}
            t = bmap.get(item.test_id) if which == "baseline" else cmap.get(item.test_id)
            if t and t.score:
                return t.score.overall
            return 0.0

        def format_delta(value: float, flip: bool = False) -> str:
            """Format a delta with sign indicator."""
            if value == 0:
                return "0.00"
            actual = -value if flip else value
            sign = "+" if actual > 0 else ""
            return f"{sign}{actual:.4f}"

        # Print regressions first
        if report.regressions:
            lines.append("  [REGRESSIONS]")
            for item in report.regressions:
                b_score = score_from_item(item, "baseline")
                c_score = score_from_item(item, "candidate")
                score_delta = c_score - b_score
                lines.append(
                    f"  {item.test_id:<20} {status_change(item):<18} "
                    f"{format_delta(score_delta, flip=True):<10} "
                    f"{format_delta(item.cost_delta):<12} "
                    f"{format_delta(item.latency_delta):<12}"
                )

        # Print improvements
        if report.improvements:
            lines.append("  [IMPROVEMENTS]")
            for item in report.improvements:
                b_score = score_from_item(item, "baseline")
                c_score = score_from_item(item, "candidate")
                score_delta = c_score - b_score
                lines.append(
                    f"  {item.test_id:<20} {status_change(item):<18} "
                    f"{format_delta(score_delta, flip=True):<10} "
                    f"{format_delta(item.cost_delta):<12} "
                    f"{format_delta(item.latency_delta):<12}"
                )

        # Print unchanged
        if report.unchanged:
            lines.append("  [UNCHANGED]")
            for item in report.unchanged:
                b_score = score_from_item(item, "baseline")
                c_score = score_from_item(item, "candidate")
                score_delta = c_score - b_score
                lines.append(
                    f"  {item.test_id:<20} {status_change(item):<18} "
                    f"{format_delta(score_delta, flip=True):<10} "
                    f"{format_delta(item.cost_delta):<12} "
                    f"{format_delta(item.latency_delta):<12}"
                )

        lines.append("")

        # Summary
        lines.append(f"  Regressions:  {report.regression_count}")
        lines.append(f"  Improvements: {report.improvement_count}")
        lines.append(f"  Unchanged:    {len(report.unchanged)}")
        lines.append("")

        return "\n".join(lines)

    def write_diff(self, baseline: RunResult, candidate: RunResult) -> None:
        """Print the diff table to stdout.

        Args:
            baseline: The previous (baseline) run result.
            candidate: The new (candidate) run result.
        """
        sys.stdout.write(self.generate_diff(baseline, candidate))
        sys.stdout.write("\n")


def _is_worse(before: str, after: str) -> bool:
    """Check if after status is worse than before."""
    order = {"pass": 0, "fail": 1, "error": 2, "missing": 3}
    return order.get(after, 0) > order.get(before, 0)
