"""
US-4: CI Gate Integration — GateChecker tests.

Tests for all acceptance criteria and edge cases:
  AC-4.1: Gate loads config, identifies baseline, runs suite, exits 0 or 1
  AC-4.2: Regression within threshold → gate passes
  AC-4.3: Regression exceeds threshold → gate fails with report
  AC-4.4: No prior baseline → creates baseline automatically, exits 0
  Edge:  Config file missing → exit 1 with clear error
  Edge:  All metrics improved → gate passes, saves new baseline
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evalforge.config import GateConfig, SuiteConfig
from evalforge.models.suite import TestSuite, TestCase, Expected
from evalforge.models.result import (
    RunResult, TestResult, ScoreResult, Summary, TokenCount,
)
from evalforge.gate.checker import GateChecker, GateResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def sample_suite() -> TestSuite:
    """A simple 4-test suite for gate testing."""
    return TestSuite(
        name="suite",
        description="Suite for gate tests",
        tests=[
            TestCase(
                id="t1",
                prompt="Q1",
                expected=Expected(type="exact", value="A1"),
            ),
            TestCase(
                id="t2",
                prompt="Q2",
                expected=Expected(type="exact", value="A2"),
            ),
            TestCase(
                id="t3",
                prompt="Q3",
                expected=Expected(type="exact", value="A3"),
            ),
            TestCase(
                id="t4",
                prompt="Q4",
                expected=Expected(type="exact", value="A4"),
            ),
        ],
    )


def make_result(
    suite_name: str = "suite",
    test_statuses: list[str] | None = None,
) -> RunResult:
    """Build a RunResult with specified test statuses."""
    if test_statuses is None:
        test_statuses = ["pass", "pass", "pass", "pass"]

    ts = datetime.now(timezone.utc).isoformat()

    tests = []
    for i, status in enumerate(test_statuses):
        overall = 1.0 if status == "pass" else 0.0
        tests.append(TestResult(
            id=f"t{i + 1}",
            status=status,
            response=f"response-{i + 1}",
            expected_value=f"A{i + 1}",
            score=ScoreResult(overall=overall, method="exact"),
            tokens=TokenCount(input=10, output=5, total=15),
            latency_ms=100.0,
            cost_usd=0.001,
        ))

    total = len(tests)
    passed = sum(1 for t in tests if t.status == "pass")
    failed = sum(1 for t in tests if t.status == "fail")

    return RunResult(
        suite_name=suite_name,
        timestamp=ts,
        duration_ms=1000.0,
        tests=tests,
        summary=Summary(
            total=total,
            passed=passed,
            failed=failed,
            errored=0,
            pass_rate=passed / total if total > 0 else 0.0,
            total_cost_usd=total * 0.001,
            avg_latency_ms=100.0,
        ),
    )


# ---------------------------------------------------------------------------
# GateResult Tests
# ---------------------------------------------------------------------------

class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_pass_result(self):
        """GateResult with passed=True and exit_code=0."""
        gr = GateResult(passed=True, exit_code=0, report="All good")
        assert gr.passed is True
        assert gr.exit_code == 0
        assert gr.report == "All good"
        assert gr.baseline_created is False

    def test_fail_result(self):
        """GateResult with passed=False and exit_code=1."""
        gr = GateResult(passed=False, exit_code=1, report="Regression detected")
        assert gr.passed is False
        assert gr.exit_code == 1
        assert "Regression" in gr.report

    def test_baseline_created_flag(self):
        """GateResult tracks whether a baseline was created."""
        gr = GateResult(
            passed=True, exit_code=0, report="Baseline created",
            baseline_created=True,
        )
        assert gr.baseline_created is True


# ---------------------------------------------------------------------------
# AC-4.4: No baseline → create baseline automatically, pass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac4_4_no_baseline_creates_and_passes(sample_suite):
    """AC-4.4: Given no prior baseline, when evalforge gate runs,
    then it creates the baseline automatically and exits 0 (pass)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baselines"
        baseline_dir.mkdir()

        config = GateConfig(
            baseline_dir=str(baseline_dir),
            suites=[SuiteConfig(path="suite.yaml")],
        )

        async def mock_execute(suite_cfg) -> RunResult:
            return make_result("suite", ["pass", "pass", "pass", "pass"])

        checker = GateChecker(config=config, execute_fn=mock_execute)
        result = await checker.check()

        assert result.passed is True
        assert result.exit_code == 0
        assert result.baseline_created is True
        assert "baseline" in result.report.lower()

        # Baseline file should have been created at suite.json
        baseline_files = list(baseline_dir.glob("*.json"))
        assert len(baseline_files) >= 1
        assert (baseline_dir / "suite.json").exists()


@pytest.mark.asyncio
async def test_ac4_4_no_baseline_with_mixed_results(sample_suite):
    """Even with failures, first run creates baseline and passes (AC-4.4)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baselines"
        baseline_dir.mkdir()

        config = GateConfig(
            baseline_dir=str(baseline_dir),
            suites=[SuiteConfig(path="suite.yaml")],
        )

        async def mock_execute(suite_cfg) -> RunResult:
            return make_result("suite", ["pass", "fail", "pass", "fail"])

        checker = GateChecker(config=config, execute_fn=mock_execute)
        result = await checker.check()

        assert result.passed is True
        assert result.exit_code == 0
        assert result.baseline_created is True


# ---------------------------------------------------------------------------
# AC-4.2: Regression within threshold → gate passes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac4_2_regression_within_threshold_passes(sample_suite):
    """AC-4.2: Given allowed_regression=5%, when run shows ≤5% regression,
    then the gate passes (within threshold).

    Baseline: 10/10 pass = 100%
    Candidate: 9/10 pass = 90% → 10pp regression, within 15% threshold.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baselines"
        baseline_dir.mkdir()

        # Pre-create baseline: 10/10 pass = 100% pass rate
        baseline = make_result("suite", ["pass"] * 10)
        (baseline_dir / "suite.json").write_text(
            baseline.model_dump_json(indent=2)
        )

        config = GateConfig(
            baseline_dir=str(baseline_dir),
            suites=[SuiteConfig(path="suite.yaml", allowed_regression_pct=15.0)],
        )

        # Candidate: 9/10 pass = 90% → 10pp regression (within 15% threshold)
        async def mock_execute(suite_cfg) -> RunResult:
            return make_result("suite", ["pass"] * 9 + ["fail"])

        checker = GateChecker(config=config, execute_fn=mock_execute)
        result = await checker.check()

        assert result.passed is True
        assert result.exit_code == 0
        assert result.baseline_created is False


@pytest.mark.asyncio
async def test_ac4_2_small_regression_within_large_threshold_passes(sample_suite):
    """Regression of 1/10 (10%) is within a 15% threshold → passes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baselines"
        baseline_dir.mkdir()

        baseline = make_result("suite", ["pass"] * 10)
        (baseline_dir / "suite.json").write_text(
            baseline.model_dump_json(indent=2)
        )

        config = GateConfig(
            baseline_dir=str(baseline_dir),
            suites=[SuiteConfig(path="suite.yaml", allowed_regression_pct=15.0)],
        )

        async def mock_execute(suite_cfg) -> RunResult:
            return make_result("suite", ["pass"] * 9 + ["fail"])

        checker = GateChecker(config=config, execute_fn=mock_execute)
        result = await checker.check()

        assert result.passed is True


# ---------------------------------------------------------------------------
# AC-4.3: Regression exceeds threshold → gate fails with report
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac4_3_regression_exceeds_threshold_fails(sample_suite):
    """AC-4.3: Given allowed_regression=5%, when run shows >5% regression,
    then the gate fails with a report of what regressed.

    Baseline: 10/10 pass (100%)
    Candidate: 2/10 pass (20%) → 80pp regression → fails
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baselines"
        baseline_dir.mkdir()

        baseline = make_result("suite", ["pass"] * 10)
        (baseline_dir / "suite.json").write_text(
            baseline.model_dump_json(indent=2)
        )

        config = GateConfig(
            baseline_dir=str(baseline_dir),
            suites=[SuiteConfig(path="suite.yaml", allowed_regression_pct=5.0)],
        )

        async def mock_execute(suite_cfg) -> RunResult:
            return make_result("suite", ["pass", "pass"] + ["fail"] * 8)

        checker = GateChecker(config=config, execute_fn=mock_execute)
        result = await checker.check()

        assert result.passed is False
        assert result.exit_code == 1
        assert "regress" in result.report.lower()
        assert "5" in result.report or "threshold" in result.report.lower()


@pytest.mark.asyncio
async def test_ac4_3_report_lists_regressed_tests(sample_suite):
    """Gate failure report identifies which tests regressed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baselines"
        baseline_dir.mkdir()

        baseline = make_result("suite", ["pass", "pass", "pass", "pass"])
        (baseline_dir / "suite.json").write_text(
            baseline.model_dump_json(indent=2)
        )

        config = GateConfig(
            baseline_dir=str(baseline_dir),
            suites=[SuiteConfig(path="suite.yaml", allowed_regression_pct=5.0)],
        )

        # t1 and t3 regressed (pass → fail), t2 and t4 unchanged
        async def mock_execute(suite_cfg) -> RunResult:
            return make_result("suite", ["fail", "pass", "fail", "pass"])

        checker = GateChecker(config=config, execute_fn=mock_execute)
        result = await checker.check()

        assert result.passed is False
        # Report should mention regressed tests
        assert "t1" in result.report.lower() or "regressed" in result.report.lower()


# ---------------------------------------------------------------------------
# AC-4.1: Full end-to-end gate flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac4_1_gate_loads_config_runs_suite_returns_exit_code(sample_suite):
    """AC-4.1: Full gate flow — baseline exists → run → compare → result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baselines"
        baseline_dir.mkdir()

        baseline = make_result("suite", ["pass", "pass", "pass", "pass"])
        (baseline_dir / "suite.json").write_text(
            baseline.model_dump_json(indent=2)
        )

        config = GateConfig(
            baseline_dir=str(baseline_dir),
            suites=[SuiteConfig(path="suite.yaml", allowed_regression_pct=5.0)],
        )

        # Same results → no regression → gate passes
        async def mock_execute(suite_cfg) -> RunResult:
            return make_result("suite", ["pass", "pass", "pass", "pass"])

        checker = GateChecker(config=config, execute_fn=mock_execute)
        result = await checker.check()

        assert result.passed is True
        assert result.exit_code == 0
        assert isinstance(result.report, str)
        assert len(result.report) > 0


# ---------------------------------------------------------------------------
# Edge: All metrics improved → gate passes, saves new baseline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edge_all_improved_passes_and_saves_baseline(sample_suite):
    """All metrics improved: baseline had failures, candidate passes all.
    Gate passes and saves new baseline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baselines"
        baseline_dir.mkdir()

        # Baseline: 2 pass, 2 fail → 50% pass rate
        baseline = make_result("suite", ["pass", "fail", "pass", "fail"])
        baseline_path = baseline_dir / "suite.json"
        baseline_path.write_text(baseline.model_dump_json(indent=2))

        config = GateConfig(
            baseline_dir=str(baseline_dir),
            suites=[SuiteConfig(path="suite.yaml", allowed_regression_pct=5.0)],
        )

        # Candidate: all pass → 100% pass rate → improvement!
        async def mock_execute(suite_cfg) -> RunResult:
            return make_result("suite", ["pass", "pass", "pass", "pass"])

        checker = GateChecker(config=config, execute_fn=mock_execute)
        result = await checker.check()

        assert result.passed is True
        assert result.exit_code == 0

        # New baseline should be saved with updated pass_rate
        updated_baseline = json.loads(baseline_path.read_text())
        assert updated_baseline["summary"]["pass_rate"] == 1.0


# ---------------------------------------------------------------------------
# Edge: Config file missing → clear error
# ---------------------------------------------------------------------------

def test_edge_config_missing_clear_error():
    """Config file missing → FileNotFoundError with message about evalforge init."""
    missing = Path("/nonexistent/path/evalforge.yaml")
    with pytest.raises(FileNotFoundError) as exc_info:
        from evalforge.config import load_config
        load_config(missing)
    msg = str(exc_info.value)
    assert "No config found" in msg or "evalforge init" in msg


# ---------------------------------------------------------------------------
# Multi-suite gate check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_suites_all_pass(sample_suite):
    """Gate with multiple suites: all pass → overall gate passes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baselines"
        baseline_dir.mkdir()

        for name in ["suite-a", "suite-b"]:
            baseline = make_result(name, ["pass"] * 4)
            (baseline_dir / f"{name}.json").write_text(
                baseline.model_dump_json(indent=2)
            )

        config = GateConfig(
            baseline_dir=str(baseline_dir),
            suites=[
                SuiteConfig(path="suite-a.yaml", allowed_regression_pct=5.0),
                SuiteConfig(path="suite-b.yaml", allowed_regression_pct=5.0),
            ],
        )

        call_count = 0

        async def mock_execute(suite_cfg) -> RunResult:
            nonlocal call_count
            call_count += 1
            # suite name from path stem
            name = Path(suite_cfg.path).stem
            return make_result(name, ["pass"] * 4)

        checker = GateChecker(config=config, execute_fn=mock_execute)
        result = await checker.check()

        assert call_count == 2
        assert result.passed is True
        assert result.exit_code == 0


@pytest.mark.asyncio
async def test_multiple_suites_one_fails_overall_fails(sample_suite):
    """Gate with multiple suites: one exceeds regression → overall gate fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = Path(tmpdir) / "baselines"
        baseline_dir.mkdir()

        (baseline_dir / "suite-a.json").write_text(
            make_result("suite-a", ["pass"] * 4).model_dump_json(indent=2)
        )
        (baseline_dir / "suite-b.json").write_text(
            make_result("suite-b", ["pass"] * 4).model_dump_json(indent=2)
        )

        config = GateConfig(
            baseline_dir=str(baseline_dir),
            suites=[
                SuiteConfig(path="suite-a.yaml", allowed_regression_pct=5.0),
                SuiteConfig(path="suite-b.yaml", allowed_regression_pct=5.0),
            ],
        )

        async def mock_execute(suite_cfg) -> RunResult:
            name = Path(suite_cfg.path).stem
            if name == "suite-a":
                return make_result("suite-a", ["pass"] * 4)
            else:
                return make_result("suite-b", ["pass", "fail", "fail", "fail"])

        checker = GateChecker(config=config, execute_fn=mock_execute)
        result = await checker.check()

        assert result.passed is False
        assert result.exit_code == 1
        assert "suite-b" in result.report.lower()
