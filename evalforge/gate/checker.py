"""CI gate checker — implements US-4 regression gate logic.

GateChecker loads baselines, runs suites, compares results, and decides
whether a deployment should be gated (blocked) due to regressions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable

from evalforge.config import GateConfig, SuiteConfig
from evalforge.models.result import RunResult
from evalforge.models.suite import TestSuite
from evalforge.runner.executor import compare_results

logger = logging.getLogger(__name__)

# Type alias: async callable that runs a suite and returns a RunResult
ExecuteFn = Callable[[SuiteConfig], Awaitable[RunResult]]


@dataclass
class GateResult:
    """Result of a gate check.

    Attributes:
        passed: True if all suites passed their regression thresholds.
        exit_code: 0 for pass, 1 for fail (suitable for sys.exit).
        report: Human-readable summary of what happened.
        baseline_created: True if a new baseline was created (no prior).
    """
    passed: bool
    exit_code: int
    report: str
    baseline_created: bool = False


class GateChecker:
    """Checks test suites for regressions against stored baselines.

    Implements the US-4 gate flow:
      1. For each suite in config, determine if a baseline exists.
      2. If no baseline → run suite, save result as baseline, pass.
      3. If baseline exists → run suite, compare pass rates, check threshold.
      4. Aggregate results across all suites.

    Usage:
        config = load_config(Path("evalforge.yaml"))
        checker = GateChecker(config=config, execute_fn=my_executor)
        result = await checker.check()
        sys.exit(result.exit_code)
    """

    def __init__(
        self,
        config: GateConfig,
        execute_fn: ExecuteFn,
    ):
        self.config = config
        self.execute_fn = execute_fn

    async def check(self) -> GateResult:
        """Run the gate check across all configured suites.

        Returns:
            GateResult with pass/fail status, exit code, and report.
        """
        baseline_dir = Path(self.config.baseline_dir)
        baseline_dir.mkdir(parents=True, exist_ok=True)

        suite_results: list[dict] = []
        any_failed = False
        any_baseline_created = False

        for suite_cfg in self.config.suites:
            suite_name = self._suite_name(suite_cfg)
            baseline_path = baseline_dir / f"{suite_name}.json"
            baseline = self._load_baseline(baseline_path)

            if baseline is None:
                # AC-4.4: No baseline → create one, pass this suite
                candidate = await self.execute_fn(suite_cfg)
                self._save_baseline(candidate, baseline_path)
                any_baseline_created = True
                suite_results.append({
                    "suite": suite_name,
                    "passed": True,
                    "reason": "Baseline created (no prior baseline)",
                })
                continue

            # Baseline exists → run suite and compare
            candidate = await self.execute_fn(suite_cfg)
            regression_pct = self._compute_regression_pct(baseline, candidate)

            threshold = suite_cfg.allowed_regression_pct

            # Use small epsilon for floating-point tolerance
            if regression_pct <= threshold + 1e-9:
                # Within threshold → pass, save new baseline
                self._save_baseline(candidate, baseline_path)
                suite_results.append({
                    "suite": suite_name,
                    "passed": True,
                    "regression_pct": regression_pct,
                    "threshold": threshold,
                    "reason": (
                        f"Regression {regression_pct:.1f}% within "
                        f"allowed {threshold:.1f}%"
                    ),
                })
            else:
                # Exceeds threshold → fail
                any_failed = True
                regressed = self._find_regressed_tests(baseline, candidate)
                suite_results.append({
                    "suite": suite_name,
                    "passed": False,
                    "regression_pct": regression_pct,
                    "threshold": threshold,
                    "regressed_tests": regressed,
                    "reason": (
                        f"Regression {regression_pct:.1f}% exceeds "
                        f"allowed {threshold:.1f}%"
                    ),
                })

        # Build report
        report = self._build_report(suite_results)

        if any_failed:
            return GateResult(
                passed=False,
                exit_code=1,
                report=report,
                baseline_created=any_baseline_created,
            )
        else:
            return GateResult(
                passed=True,
                exit_code=0,
                report=report,
                baseline_created=any_baseline_created,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _suite_name(suite_cfg: SuiteConfig) -> str:
        """Derive a suite name from its YAML file path.

        Example: 'test-suites/example/suite.yaml' → 'suite'
        """
        return Path(suite_cfg.path).stem

    @staticmethod
    def _load_baseline(path: Path) -> RunResult | None:
        """Load a baseline RunResult from a JSON file.

        Returns None if the file does not exist.
        """
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return RunResult.model_validate(data)

    @staticmethod
    def _save_baseline(result: RunResult, path: Path) -> None:
        """Save a RunResult as a baseline JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2))

    @staticmethod
    def _compute_regression_pct(
        baseline: RunResult, candidate: RunResult
    ) -> float:
        """Compute regression as the drop in pass rate (percentage points).

        positive = regression (candidate worse than baseline)
        negative = improvement (candidate better than baseline)
        zero     = no change
        """
        baseline_rate = baseline.summary.pass_rate
        candidate_rate = candidate.summary.pass_rate
        return (baseline_rate - candidate_rate) * 100.0

    @staticmethod
    def _find_regressed_tests(
        baseline: RunResult, candidate: RunResult
    ) -> list[str]:
        """Return IDs of tests that regressed (pass→fail, etc.)."""
        report = compare_results(baseline, candidate)
        return [r.test_id for r in report.regressions]

    @staticmethod
    def _build_report(suite_results: list[dict]) -> str:
        """Build a human-readable report from suite results."""
        lines = ["=== EvalForge Gate Report ===", ""]

        for sr in suite_results:
            status = "PASS" if sr["passed"] else "FAIL"
            lines.append(f"  [{status}] {sr['suite']}: {sr['reason']}")

            if not sr["passed"] and "regressed_tests" in sr:
                regressed = sr["regressed_tests"]
                lines.append(f"          Regressed tests: {', '.join(regressed)}")

        lines.append("")
        total = len(suite_results)
        passed = sum(1 for sr in suite_results if sr["passed"])
        failed = total - passed
        lines.append(f"Summary: {passed}/{total} suites passed, {failed} failed")

        return "\n".join(lines)
