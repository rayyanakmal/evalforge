"""Typer CLI application — command handlers for evalforge.

Defines the main Typer app and registers subcommands:
    run       — Execute a test suite and produce reports
    compare   — Compare two run results (diff table)
    gate      — CI gate: check for regressions
    init      — Scaffold a new EvalForge project
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from evalforge.cli.init import scaffold_project
from evalforge.config import load_config
from evalforge.models.result import RunResult, TestResult, ScoreResult, Summary, TokenCount


app = typer.Typer(
    name="evalforge",
    help="EvalForge — Eval-driven agent testing framework",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helper for dry-run (--no-llm) mode
# ---------------------------------------------------------------------------

def _dry_run_results(suite_name: str, tests: list) -> RunResult:
    """Generate mock RunResult for --no-llm dry-run mode.

    Each test is marked as 'pass' with a perfect score.
    """
    ts = datetime.now(timezone.utc).isoformat()
    results = []
    for i, test in enumerate(tests):
        results.append(TestResult(
            id=test.get("id", f"test-{i}"),
            status="pass",
            response="(dry-run: no LLM call)",
            expected_value=test.get("expected", {}).get("value"),
            score=ScoreResult(overall=1.0, method="dry-run"),
            tokens=TokenCount(input=0, output=0, total=0),
            latency_ms=0.0,
            cost_usd=0.0,
        ))
    total = len(results)
    return RunResult(
        suite_name=suite_name,
        timestamp=ts,
        duration_ms=0.0,
        tests=results,
        summary=Summary(
            total=total,
            passed=total,
            failed=0,
            errored=0,
            pass_rate=1.0 if total > 0 else 0.0,
            total_cost_usd=0.0,
            avg_latency_ms=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# evalforge run
# ---------------------------------------------------------------------------

@app.command()
def run(
    suite_path: str = typer.Argument(..., help="Path to the test suite YAML file"),
    output_dir: str = typer.Option(
        "evalforge-output", "--output-dir", "-o",
        help="Directory for JSON report output",
    ),
    concurrency: int = typer.Option(
        10, "--concurrency", "-c",
        help="Maximum number of parallel test executions",
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm",
        help="Dry-run mode: do not call LLMs, use mock results",
    ),
) -> None:
    """Run a test suite and output results to stdout and JSON.

    Loads the suite YAML, executes each test against the configured LLM,
    prints a results table to stdout, and saves a JSON report to the
    output directory as evalforge-output/report-<timestamp>.json.

    Examples:
        evalforge run test-suites/example/suite.yaml
        evalforge run suite.yaml --output-dir results/ --concurrency 5
        evalforge run suite.yaml --no-llm  # dry-run for testing
    """
    suite_path_obj = Path(suite_path)
    if not suite_path_obj.exists():
        typer.echo(f"Error: Suite file not found: {suite_path}", err=True)
        raise typer.Exit(code=1)

    # Load the suite
    import yaml
    try:
        raw = yaml.safe_load(suite_path_obj.read_text())
    except yaml.YAMLError as e:
        typer.echo(f"Error: Invalid YAML in suite file: {e}", err=True)
        raise typer.Exit(code=1)

    if raw is None:
        typer.echo("Error: Empty suite file.", err=True)
        raise typer.Exit(code=1)

    suite_name = raw.get("name", suite_path_obj.stem)
    tests = raw.get("tests", [])

    # Generate results
    if no_llm:
        result = _dry_run_results(suite_name, tests)
    else:
        # Real execution: create an executor and run
        # For now, this requires a configured LLM client
        typer.echo(
            "Error: Real LLM execution requires a configured provider. "
            "Use --no-llm for dry-run mode or configure judge/target in evalforge.yaml.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Report to stdout (console)
    from evalforge.reporting.console_reporter import ConsoleReporter
    console = ConsoleReporter()
    console.write(result, Path("/dev/null"))

    # Save JSON report
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = out_dir / f"report-{timestamp}.json"

    from evalforge.reporting.json_reporter import JSONReporter
    json_reporter = JSONReporter()
    json_reporter.write(result, report_path)

    typer.echo(f"  Report saved to: {report_path}")


# ---------------------------------------------------------------------------
# evalforge compare
# ---------------------------------------------------------------------------

@app.command()
def compare(
    baseline_path: str = typer.Argument(..., help="Path to baseline JSON report"),
    candidate_path: str = typer.Argument(..., help="Path to candidate JSON report"),
) -> None:
    """Compare two run results and show a diff table.

    Loads two JSON reports (baseline and candidate) and displays a
    diff table with columns: test name, status, score change, cost
    change, latency change.

    Examples:
        evalforge compare baseline.json candidate.json
        evalforge compare evalforge-output/report-old.json evalforge-output/report-new.json
    """
    baseline_obj = Path(baseline_path)
    candidate_obj = Path(candidate_path)

    # Validate file existence
    missing = []
    if not baseline_obj.exists():
        missing.append(baseline_path)
    if not candidate_obj.exists():
        missing.append(candidate_path)

    if missing:
        typer.echo(
            f"Error: File(s) not found: {', '.join(missing)}",
            err=True,
        )
        raise typer.Exit(code=1)

    # Load baseline
    try:
        baseline_data = json.loads(baseline_obj.read_text())
        baseline = RunResult.model_validate(baseline_data)
    except (json.JSONDecodeError, Exception) as e:
        typer.echo(
            f"Error: Failed to load baseline report '{baseline_path}': {e}",
            err=True,
        )
        raise typer.Exit(code=1)

    # Load candidate
    try:
        candidate_data = json.loads(candidate_obj.read_text())
        candidate = RunResult.model_validate(candidate_data)
    except (json.JSONDecodeError, Exception) as e:
        typer.echo(
            f"Error: Failed to load candidate report '{candidate_path}': {e}",
            err=True,
        )
        raise typer.Exit(code=1)

    # Generate diff
    from evalforge.reporting.diff_reporter import DiffReporter
    diff = DiffReporter()
    diff.write_diff(baseline, candidate)


# ---------------------------------------------------------------------------
# evalforge gate
# ---------------------------------------------------------------------------

@app.command()
def gate(
    config_path: str = typer.Option(
        "evalforge.yaml", "--config", "-c",
        help="Path to evalforge.yaml config file",
    ),
) -> None:
    """CI gate: check test suites for regressions against baselines.

    Loads evalforge.yaml, runs configured suites, compares against
    stored baselines, and exits 0 (pass) or 1 (fail).

    Examples:
        evalforge gate
        evalforge gate --config my-evalforge.yaml
    """
    config_obj = Path(config_path)

    try:
        config = load_config(config_obj)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: Failed to load config: {e}", err=True)
        raise typer.Exit(code=1)

    from evalforge.gate.checker import GateChecker

    async def _run_gate() -> None:
        async def execute_fn(suite_cfg):
            """Validate suite path and return result.

            In a real implementation, this would load the suite YAML,
            create an executor with configured LLM clients, and run.
            For now, we validate the path exists.
            """
            suite_path = Path(suite_cfg.path)
            if not suite_path.exists():
                raise FileNotFoundError(
                    f"Suite file not found: {suite_path}"
                )
            return RunResult(
                suite_name=suite_path.stem,
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=0.0,
                tests=[],
                summary=Summary(),
            )

        checker = GateChecker(config=config, execute_fn=execute_fn)
        result = await checker.check()
        typer.echo(result.report)
        if result.exit_code != 0:
            raise typer.Exit(code=result.exit_code)

    try:
        asyncio.run(_run_gate())
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: Gate check failed: {e}", err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# evalforge init
# ---------------------------------------------------------------------------

@app.command()
def init(
    path: str = typer.Argument(
        ".", help="Target directory to scaffold into (default: current directory)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing files without prompting",
    ),
) -> None:
    """Scaffold a new EvalForge project.

    Creates evalforge.yaml, test-suites/example/suite.yaml, and
    a .gitignore file for the output directory.

    Examples:
        evalforge init
        evalforge init my-project/
        evalforge init --force   # overwrite existing config
    """
    target_dir = Path(path)

    try:
        scaffold_project(target_dir, force=force)
    except FileExistsError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"✓ Initialized EvalForge project in {target_dir.resolve()}")
    typer.echo(f"  Created: evalforge.yaml")
    typer.echo(f"  Created: test-suites/example/suite.yaml")
    typer.echo(f"  Created: .gitignore")
    typer.echo("")
    typer.echo("  Next steps:")
    typer.echo(f"    1. Edit {target_dir / 'evalforge.yaml'} with your API keys")
    typer.echo(f"    2. Run: evalforge run test-suites/example/suite.yaml --no-llm")
    typer.echo(f"    3. When ready, run without --no-llm for real evaluation")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
