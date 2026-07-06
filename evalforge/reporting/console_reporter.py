"""ConsoleReporter — human-readable stdout table.

Outputs a formatted text table with per-test pass/fail/error status,
plus a summary block with aggregate statistics.

Designed for `evalforge run` stdout output.
"""

import sys
from pathlib import Path

from evalforge.models.result import RunResult, TestResult
from evalforge.reporting.base import Reporter


class ConsoleReporter(Reporter):
    """Print a human-readable results table to stdout.

    Usage:
        reporter = ConsoleReporter()
        print(reporter.generate(result))
        # or:
        reporter.write(result, Path("/dev/null"))  # prints to stdout
    """

    def generate(self, result: RunResult) -> str:
        """Format a RunResult as a text table.

        Args:
            result: The complete run result.

        Returns:
            Formatted text with per-test table and summary block.
        """
        lines: list[str] = []

        # Header
        lines.append("")
        lines.append(f"  Suite: {result.suite_name}")
        lines.append(f"  Timestamp: {result.timestamp}")
        lines.append(f"  Duration: {result.duration_ms:.0f}ms")
        lines.append("")

        if not result.tests:
            lines.append("  No tests in suite.")
            return "\n".join(lines)

        # Table header
        lines.append(f"  {'ID':<20} {'Status':<8} {'Score':<8} {'Cost':<10} {'Latency':<10}")
        lines.append(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

        # Table rows
        for t in result.tests:
            status = t.status.upper()
            score_str = f"{t.score.overall:.2f}" if t.score else "N/A"
            cost_str = f"${t.cost_usd:.4f}"
            lat_str = f"{t.latency_ms:.0f}ms"
            lines.append(f"  {t.id:<20} {status:<8} {score_str:<8} {cost_str:<10} {lat_str:<10}")

        lines.append("")

        # Summary
        s = result.summary
        lines.append("  ── Summary ──")
        lines.append(f"  Total:     {s.total}")
        lines.append(f"  Passed:    {s.passed}")
        lines.append(f"  Failed:    {s.failed}")
        lines.append(f"  Errored:   {s.errored}")
        lines.append(f"  Pass Rate: {s.pass_rate:.1%}")
        lines.append(f"  Total Cost: ${s.total_cost_usd:.4f}")
        lines.append(f"  Avg Latency: {s.avg_latency_ms:.0f}ms")
        if s.latency_p50 is not None:
            lines.append(f"  P50 Latency: {s.latency_p50:.0f}ms")
        if s.latency_p95 is not None:
            lines.append(f"  P95 Latency: {s.latency_p95:.0f}ms")
        if s.latency_p99 is not None:
            lines.append(f"  P99 Latency: {s.latency_p99:.0f}ms")
        lines.append("")

        return "\n".join(lines)

    def write(self, result: RunResult, path: Path) -> None:
        """Print the formatted report to stdout.

        Args:
            result: The complete run result.
            path: Ignored — ConsoleReporter always writes to stdout.
        """
        sys.stdout.write(self.generate(result))
        sys.stdout.write("\n")
