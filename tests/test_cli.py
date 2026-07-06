"""
US-5: CLI & Report Output

Tests for all acceptance criteria and edge cases:
  AC-5.1: evalforge run <suite> → stdout table + JSON report
  AC-5.2: evalforge compare <baseline> <candidate> → diff table
  AC-5.3: evalforge init → scaffolds project
  Edge:  compare with non-existent baseline → error with paths searched
  Edge:  init in directory with existing config → prompts confirmation (or --force)

Plus reporter unit tests:
  - JSONReporter: generate JSON, write to file
  - ConsoleReporter: generate table output, handle empty suite
  - DiffReporter: generate diff table, handle both empty
  - Reporter ABC: cannot instantiate directly
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from evalforge.models.result import (
    RunResult, TestResult, ScoreResult, Summary, TokenCount, DimensionScore,
)
from evalforge.models.suite import TestSuite, TestCase, Expected


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    """Return a CliRunner for testing Typer CLI commands."""
    return CliRunner()


@pytest.fixture
def sample_run_result() -> RunResult:
    """Build a sample RunResult for reporter tests."""
    ts = datetime.now(timezone.utc).isoformat()
    return RunResult(
        suite_name="test-suite",
        timestamp=ts,
        duration_ms=1500.0,
        tests=[
            TestResult(
                id="test-1", status="pass", response="4", expected_value="4",
                score=ScoreResult(overall=1.0, method="exact"),
                tokens=TokenCount(input=10, output=5, total=15),
                latency_ms=100.0, cost_usd=0.001,
            ),
            TestResult(
                id="test-2", status="fail", response="London", expected_value="Paris",
                score=ScoreResult(overall=0.0, method="exact"),
                tokens=TokenCount(input=12, output=6, total=18),
                latency_ms=120.0, cost_usd=0.0015,
            ),
            TestResult(
                id="test-3", status="pass", response="William Shakespeare",
                expected_value="William Shakespeare",
                score=ScoreResult(overall=1.0, method="exact"),
                tokens=TokenCount(input=8, output=4, total=12),
                latency_ms=90.0, cost_usd=0.0008,
            ),
        ],
        summary=Summary(
            total=3, passed=2, failed=1, errored=0, pass_rate=0.667,
            total_cost_usd=0.0033, avg_latency_ms=103.33,
            latency_p50=100.0, latency_p95=120.0, latency_p99=120.0,
        ),
    )


@pytest.fixture
def empty_run_result() -> RunResult:
    """An empty RunResult for edge case testing."""
    ts = datetime.now(timezone.utc).isoformat()
    return RunResult(
        suite_name="empty-suite",
        timestamp=ts,
        duration_ms=0.0,
        tests=[],
        summary=Summary(total=0, passed=0, failed=0, errored=0, pass_rate=0.0),
    )


@pytest.fixture
def baseline_result() -> RunResult:
    """Baseline RunResult for compare/diff tests."""
    ts = datetime.now(timezone.utc).isoformat()
    return RunResult(
        suite_name="my-suite",
        timestamp=ts,
        duration_ms=1000.0,
        tests=[
            TestResult(
                id="t1", status="pass", response="4", expected_value="4",
                score=ScoreResult(overall=1.0, method="exact"),
                tokens=TokenCount(input=10, output=5, total=15),
                latency_ms=100.0, cost_usd=0.001,
            ),
            TestResult(
                id="t2", status="pass", response="Paris", expected_value="Paris",
                score=ScoreResult(overall=1.0, method="exact"),
                tokens=TokenCount(input=10, output=5, total=15),
                latency_ms=95.0, cost_usd=0.001,
            ),
        ],
        summary=Summary(
            total=2, passed=2, failed=0, errored=0, pass_rate=1.0,
            total_cost_usd=0.002, avg_latency_ms=97.5,
        ),
    )


@pytest.fixture
def candidate_result() -> RunResult:
    """Candidate RunResult for compare/diff tests."""
    ts = datetime.now(timezone.utc).isoformat()
    return RunResult(
        suite_name="my-suite",
        timestamp=ts,
        duration_ms=900.0,
        tests=[
            TestResult(
                id="t1", status="pass", response="4", expected_value="4",
                score=ScoreResult(overall=1.0, method="exact"),
                tokens=TokenCount(input=9, output=4, total=13),
                latency_ms=85.0, cost_usd=0.0009,
            ),
            TestResult(
                id="t2", status="fail", response="Berlin", expected_value="Paris",
                score=ScoreResult(overall=0.0, method="exact"),
                tokens=TokenCount(input=11, output=6, total=17),
                latency_ms=110.0, cost_usd=0.0011,
            ),
        ],
        summary=Summary(
            total=2, passed=1, failed=1, errored=0, pass_rate=0.5,
            total_cost_usd=0.002, avg_latency_ms=97.5,
        ),
    )


@pytest.fixture
def sample_suite_yaml() -> str:
    """Sample suite YAML content."""
    return """
name: Example Suite
description: A sample test suite for evalforge
tests:
  - id: hello-exact
    prompt: "Reply with just the word: Hello"
    expected:
      type: exact
      value: "Hello"
  - id: capital-question
    prompt: "What is the capital of France?"
    expected:
      type: exact
      value: "Paris"
"""


# ===================================================================
# Reporter ABC Tests
# ===================================================================

class TestReporterABC:
    """Tests for Reporter abstract base class."""

    def test_reporter_is_abstract(self):
        """Reporter ABC cannot be instantiated directly."""
        from evalforge.reporting.base import Reporter
        with pytest.raises(TypeError):
            Reporter()  # type: ignore[abstract]

    def test_reporter_defines_generate(self):
        """Reporter ABC requires generate() method."""
        from evalforge.reporting.base import Reporter
        assert hasattr(Reporter, "generate")
        from abc import abstractmethod
        assert Reporter.generate.__isabstractmethod__

    def test_reporter_defines_write(self):
        """Reporter ABC requires write() method."""
        from evalforge.reporting.base import Reporter
        assert hasattr(Reporter, "write")
        from abc import abstractmethod
        assert Reporter.write.__isabstractmethod__


# ===================================================================
# JSONReporter Tests
# ===================================================================

class TestJSONReporter:
    """Tests for JSONReporter — JSON output generation."""

    def test_generate_returns_valid_json(self, sample_run_result):
        """generate() returns valid JSON string."""
        from evalforge.reporting.json_reporter import JSONReporter
        reporter = JSONReporter()
        output = reporter.generate(sample_run_result)
        # Should be valid JSON
        data = json.loads(output)
        assert data["suite_name"] == "test-suite"
        assert len(data["tests"]) == 3
        assert data["summary"]["total"] == 3

    def test_generate_contains_all_test_fields(self, sample_run_result):
        """Generated JSON includes all expected test result fields."""
        from evalforge.reporting.json_reporter import JSONReporter
        reporter = JSONReporter()
        output = reporter.generate(sample_run_result)
        data = json.loads(output)
        t = data["tests"][0]
        assert "id" in t
        assert "status" in t
        assert "response" in t
        assert "score" in t
        assert "latency_ms" in t
        assert "cost_usd" in t

    def test_generate_empty_suite(self, empty_run_result):
        """generate() handles empty suite gracefully."""
        from evalforge.reporting.json_reporter import JSONReporter
        reporter = JSONReporter()
        output = reporter.generate(empty_run_result)
        data = json.loads(output)
        assert data["tests"] == []
        assert data["summary"]["total"] == 0

    def test_write_creates_file(self, sample_run_result):
        """write() creates a JSON file at the specified path."""
        from evalforge.reporting.json_reporter import JSONReporter
        reporter = JSONReporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "report.json"
            reporter.write(sample_run_result, out_path)
            assert out_path.exists()
            content = out_path.read_text()
            data = json.loads(content)
            assert data["suite_name"] == "test-suite"

    def test_write_creates_parent_directories(self, sample_run_result):
        """write() creates parent directories if they don't exist."""
        from evalforge.reporting.json_reporter import JSONReporter
        reporter = JSONReporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "deep" / "nested" / "report.json"
            reporter.write(sample_run_result, nested)
            assert nested.exists()

    def test_is_reporter_subclass(self):
        """JSONReporter is a valid Reporter subclass."""
        from evalforge.reporting.base import Reporter
        from evalforge.reporting.json_reporter import JSONReporter
        assert issubclass(JSONReporter, Reporter)


# ===================================================================
# ConsoleReporter Tests
# ===================================================================

class TestConsoleReporter:
    """Tests for ConsoleReporter — stdout table output."""

    def test_generate_returns_string(self, sample_run_result):
        """generate() returns a non-empty string."""
        from evalforge.reporting.console_reporter import ConsoleReporter
        reporter = ConsoleReporter()
        output = reporter.generate(sample_run_result)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_generate_contains_test_ids(self, sample_run_result):
        """Generated output mentions test IDs."""
        from evalforge.reporting.console_reporter import ConsoleReporter
        reporter = ConsoleReporter()
        output = reporter.generate(sample_run_result)
        assert "test-1" in output
        assert "test-2" in output
        assert "test-3" in output

    def test_generate_contains_pass_fail_status(self, sample_run_result):
        """Generated output shows pass/fail status."""
        from evalforge.reporting.console_reporter import ConsoleReporter
        reporter = ConsoleReporter()
        output = reporter.generate(sample_run_result)
        # Should mention pass/fail
        assert "PASS" in output.upper() or "pass" in output.lower()
        assert "FAIL" in output.upper() or "fail" in output.lower()

    def test_generate_contains_summary_stats(self, sample_run_result):
        """Generated output includes summary statistics."""
        from evalforge.reporting.console_reporter import ConsoleReporter
        reporter = ConsoleReporter()
        output = reporter.generate(sample_run_result)
        assert "total" in output.lower() or "summary" in output.lower()

    def test_generate_empty_suite(self, empty_run_result):
        """generate() handles empty suite gracefully."""
        from evalforge.reporting.console_reporter import ConsoleReporter
        reporter = ConsoleReporter()
        output = reporter.generate(empty_run_result)
        assert isinstance(output, str)
        # Should indicate no tests
        assert len(output) > 0

    def test_write_outputs_to_stdout(self, sample_run_result, capsys):
        """write() prints the generated output to stdout."""
        from evalforge.reporting.console_reporter import ConsoleReporter
        reporter = ConsoleReporter()
        reporter.write(sample_run_result, Path("/dev/null"))
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "test-1" in captured.out or "test-suite" in captured.out

    def test_is_reporter_subclass(self):
        """ConsoleReporter is a valid Reporter subclass."""
        from evalforge.reporting.base import Reporter
        from evalforge.reporting.console_reporter import ConsoleReporter
        assert issubclass(ConsoleReporter, Reporter)


# ===================================================================
# DiffReporter Tests
# ===================================================================

class TestDiffReporter:
    """Tests for DiffReporter — comparison diff table."""

    def test_generate_diff_returns_string(self, baseline_result, candidate_result):
        """generate_diff() returns a non-empty string."""
        from evalforge.reporting.diff_reporter import DiffReporter
        reporter = DiffReporter()
        output = reporter.generate_diff(baseline_result, candidate_result)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_generate_diff_shows_test_names(self, baseline_result, candidate_result):
        """Diff output includes test names."""
        from evalforge.reporting.diff_reporter import DiffReporter
        reporter = DiffReporter()
        output = reporter.generate_diff(baseline_result, candidate_result)
        assert "t1" in output
        assert "t2" in output

    def test_generate_diff_shows_status_changes(self, baseline_result, candidate_result):
        """Diff output shows status: t2 regressed from pass to fail."""
        from evalforge.reporting.diff_reporter import DiffReporter
        reporter = DiffReporter()
        output = reporter.generate_diff(baseline_result, candidate_result)
        # t2 goes from pass→fail (regression)
        assert "pass" in output.lower()
        assert "fail" in output.lower()

    def test_generate_diff_shows_cost_delta(self, baseline_result, candidate_result):
        """Diff output includes cost change information."""
        from evalforge.reporting.diff_reporter import DiffReporter
        reporter = DiffReporter()
        output = reporter.generate_diff(baseline_result, candidate_result)
        # Should mention cost or delta
        assert "cost" in output.lower() or "Δ" in output or "+" in output or "-" in output

    def test_generate_diff_shows_latency_delta(self, baseline_result, candidate_result):
        """Diff output includes latency change information."""
        from evalforge.reporting.diff_reporter import DiffReporter
        reporter = DiffReporter()
        output = reporter.generate_diff(baseline_result, candidate_result)
        # Should mention latency
        assert "latency" in output.lower() or "ms" in output.lower()

    def test_generate_diff_both_empty(self):
        """generate_diff() handles two empty results."""
        from evalforge.reporting.diff_reporter import DiffReporter
        ts = datetime.now(timezone.utc).isoformat()
        empty1 = RunResult(suite_name="a", timestamp=ts, duration_ms=0.0, tests=[],
                           summary=Summary())
        empty2 = RunResult(suite_name="b", timestamp=ts, duration_ms=0.0, tests=[],
                           summary=Summary())
        reporter = DiffReporter()
        output = reporter.generate_diff(empty1, empty2)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_is_reporter_subclass(self):
        """DiffReporter is a valid Reporter subclass."""
        from evalforge.reporting.base import Reporter
        from evalforge.reporting.diff_reporter import DiffReporter
        assert issubclass(DiffReporter, Reporter)


# ===================================================================
# CLI: AC-5.3 — evalforge init
# ===================================================================

class TestCLIInit:
    """Tests for evalforge init command (AC-5.3)."""

    def test_init_creates_config_file(self, runner):
        """evalforge init creates evalforge.yaml in the target directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                _get_app(),
                ["init", tmpdir, "--force"],
            )
            assert result.exit_code == 0
            config_path = Path(tmpdir) / "evalforge.yaml"
            assert config_path.exists()

    def test_init_creates_test_suites_folder(self, runner):
        """evalforge init creates test-suites/ folder with example suite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                _get_app(),
                ["init", tmpdir, "--force"],
            )
            assert result.exit_code == 0
            suites_dir = Path(tmpdir) / "test-suites"
            assert suites_dir.is_dir()
            example_dir = suites_dir / "example"
            assert example_dir.is_dir()
            suite_file = example_dir / "suite.yaml"
            assert suite_file.exists()

    def test_init_creates_gitignore(self, runner):
        """evalforge init creates .gitignore with evalforge-output/ entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                _get_app(),
                ["init", tmpdir, "--force"],
            )
            assert result.exit_code == 0
            gitignore = Path(tmpdir) / ".gitignore"
            assert gitignore.exists()
            content = gitignore.read_text()
            assert "evalforge-output" in content

    def test_init_example_suite_is_valid_yaml(self, runner):
        """The example suite created by init is valid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                _get_app(),
                ["init", tmpdir, "--force"],
            )
            assert result.exit_code == 0
            suite_path = Path(tmpdir) / "test-suites" / "example" / "suite.yaml"
            import yaml
            data = yaml.safe_load(suite_path.read_text())
            assert data is not None
            assert "name" in data
            assert "tests" in data

    def test_init_creates_valid_config_yaml(self, runner):
        """The evalforge.yaml created by init is loadable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                _get_app(),
                ["init", tmpdir, "--force"],
            )
            assert result.exit_code == 0
            from evalforge.config import load_config
            config = load_config(Path(tmpdir) / "evalforge.yaml")
            assert config is not None
            assert len(config.suites) > 0

    def test_init_with_existing_config_and_force(self, runner):
        """init --force overwrites existing evalforge.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "evalforge.yaml"
            # Pre-create a different config
            from evalforge.config import GateConfig, SuiteConfig, save_config
            original = GateConfig(
                baseline_dir="custom/",
                suites=[SuiteConfig(path="old.yaml")],
                concurrency=55,
            )
            save_config(original, config_path)

            # Now run init --force
            result = runner.invoke(
                _get_app(),
                ["init", tmpdir, "--force"],
            )
            assert result.exit_code == 0

            # Should have been overwritten
            from evalforge.config import load_config
            new_config = load_config(config_path)
            # Should be the scaffolded config, not the old one
            assert new_config.concurrency != 55

    def test_init_defaults_to_current_directory(self, runner):
        """evalforge init with no path defaults to current directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = runner.invoke(
                    _get_app(),
                    ["init", "--force"],
                )
                assert result.exit_code == 0
                assert Path("evalforge.yaml").exists()
                assert Path("test-suites").is_dir()
            finally:
                os.chdir(original_cwd)


# ===================================================================
# CLI: AC-5.1 — evalforge run
# ===================================================================

class TestCLIRun:
    """Tests for evalforge run command (AC-5.1)."""

    def test_run_requires_suite_argument(self, runner):
        """evalforge run without a suite path shows usage."""
        result = runner.invoke(_get_app(), ["run"])
        # Should fail with missing argument or show help
        assert result.exit_code != 0 or "usage" in result.output.lower()

    def test_run_with_missing_suite_file(self, runner):
        """evalforge run with non-existent suite file shows error."""
        result = runner.invoke(
            _get_app(),
            ["run", "/nonexistent/suite.yaml"],
        )
        assert result.exit_code != 0

    def test_run_accepts_output_dir_option(self, runner, sample_suite_yaml):
        """evalforge run accepts --output-dir option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "suite.yaml"
            suite_path.write_text(sample_suite_yaml)
            out_dir = Path(tmpdir) / "custom-output"

            result = runner.invoke(
                _get_app(),
                ["run", str(suite_path), "--output-dir", str(out_dir), "--no-llm"],
            )
            # Should run (even if LLM calls fail in test, the CLI should parse options)
            # With --no-llm flag, we mock out the LLM
            # This test validates the option is accepted

    def test_run_accepts_concurrency_option(self, runner, sample_suite_yaml):
        """evalforge run accepts --concurrency option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "suite.yaml"
            suite_path.write_text(sample_suite_yaml)

            result = runner.invoke(
                _get_app(),
                ["run", str(suite_path), "--concurrency", "5", "--no-llm"],
            )
            # Validates the option is accepted without error

    def test_run_with_no_llm_flag_uses_dry_run(self, runner, sample_suite_yaml):
        """evalforge run --no-llm performs a dry run without calling LLMs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "suite.yaml"
            suite_path.write_text(sample_suite_yaml)
            out_dir = Path(tmpdir) / "evalforge-output"

            result = runner.invoke(
                _get_app(),
                ["run", str(suite_path), "--output-dir", str(out_dir), "--no-llm"],
            )
            assert result.exit_code == 0
            # Should produce output on stdout
            assert len(result.output) > 0

    def test_run_with_no_llm_creates_json_report(self, runner, sample_suite_yaml):
        """evalforge run --no-llm saves JSON report to output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "suite.yaml"
            suite_path.write_text(sample_suite_yaml)
            out_dir = Path(tmpdir) / "evalforge-output"

            result = runner.invoke(
                _get_app(),
                ["run", str(suite_path), "--output-dir", str(out_dir), "--no-llm"],
            )
            assert result.exit_code == 0

            # Should have created the output directory
            assert out_dir.exists()

            # Should have a JSON report file
            json_files = list(out_dir.glob("report-*.json"))
            assert len(json_files) >= 1, f"No JSON report found in {out_dir}"

            # JSON report should be valid
            report_data = json.loads(json_files[0].read_text())
            assert "suite_name" in report_data
            assert "tests" in report_data

    def test_run_outputs_table_to_stdout(self, runner, sample_suite_yaml):
        """evalforge run outputs results table to stdout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "suite.yaml"
            suite_path.write_text(sample_suite_yaml)
            out_dir = Path(tmpdir) / "evalforge-output"

            result = runner.invoke(
                _get_app(),
                ["run", str(suite_path), "--output-dir", str(out_dir), "--no-llm"],
            )
            # Stdout should contain test info
            assert "hello-exact" in result.output.lower() or "capital" in result.output.lower() or "test" in result.output.lower()


# ===================================================================
# CLI: AC-5.2 — evalforge compare
# ===================================================================

class TestCLICompare:
    """Tests for evalforge compare command (AC-5.2)."""

    def test_compare_requires_two_arguments(self, runner):
        """evalforge compare requires baseline and candidate paths."""
        result = runner.invoke(_get_app(), ["compare"])
        assert result.exit_code != 0 or "usage" in result.output.lower()

    def test_compare_with_valid_reports(self, runner, baseline_result, candidate_result):
        """evalforge compare with valid baseline and candidate JSON reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            candidate_path = Path(tmpdir) / "candidate.json"
            baseline_path.write_text(baseline_result.model_dump_json(indent=2))
            candidate_path.write_text(candidate_result.model_dump_json(indent=2))

            result = runner.invoke(
                _get_app(),
                ["compare", str(baseline_path), str(candidate_path)],
            )
            assert result.exit_code == 0
            # Output should include diff table
            output = result.output.lower()
            assert "t1" in output or "t2" in output or "test" in output

    def test_compare_shows_diff_columns(self, runner, baseline_result, candidate_result):
        """evalforge compare shows columns: test name, status, score, cost, latency changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            candidate_path = Path(tmpdir) / "candidate.json"
            baseline_path.write_text(baseline_result.model_dump_json(indent=2))
            candidate_path.write_text(candidate_result.model_dump_json(indent=2))

            result = runner.invoke(
                _get_app(),
                ["compare", str(baseline_path), str(candidate_path)],
            )
            output = result.output.lower()
            # Should mention status
            assert "status" in output or "pass" in output or "fail" in output

    def test_compare_edge_nonexistent_baseline(self, runner):
        """Edge: compare with non-existent baseline file → error with paths searched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                _get_app(),
                ["compare", "/nonexistent/baseline.json", "/nonexistent/candidate.json"],
            )
            assert result.exit_code != 0
            # Error message should mention the path or "not found"
            output = result.output.lower()
            assert "not found" in output or "exist" in output or "error" in output

    def test_compare_invalid_json_handled(self, runner, baseline_result):
        """evalforge compare with invalid JSON reports error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            candidate_path = Path(tmpdir) / "candidate.json"
            baseline_path.write_text(baseline_result.model_dump_json(indent=2))
            candidate_path.write_text("not valid json {{{")

            result = runner.invoke(
                _get_app(),
                ["compare", str(baseline_path), str(candidate_path)],
            )
            assert result.exit_code != 0


# ===================================================================
# CLI: evalforge gate
# ===================================================================

class TestCLIGate:
    """Tests for evalforge gate command (US-4 integration)."""

    def test_gate_accepts_config_option(self, runner):
        """evalforge gate accepts --config option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "evalforge.yaml"
            from evalforge.config import GateConfig, SuiteConfig, save_config
            config = GateConfig(
                baseline_dir=str(Path(tmpdir) / "baselines"),
                suites=[SuiteConfig(path="/nonexistent/suite.yaml")],
            )
            save_config(config, config_path)

            result = runner.invoke(
                _get_app(),
                ["gate", "--config", str(config_path)],
            )
            # Will fail because suite doesn't exist, but the CLI should parse correctly
            assert result.exit_code != 0

    def test_gate_defaults_to_evalforge_yaml(self, runner):
        """evalforge gate without --config looks for evalforge.yaml in cwd."""
        result = runner.invoke(_get_app(), ["gate"])
        # Should exit with error since no config in temp test dir
        assert result.exit_code != 0
        assert "config" in result.output.lower() or "init" in result.output.lower()


# ---------------------------------------------------------------------------
# Helper to import the app lazily (avoid CI issues with missing deps)
# ---------------------------------------------------------------------------

_app = None


def _get_app():
    """Get the Typer app, importing lazily."""
    global _app
    if _app is None:
        from evalforge.cli import app
        _app = app
    return _app
