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
    trajectory: bool = typer.Option(
        False, "--trajectory",
        help="Also compare trajectory (process) quality and print the trajectory regression block",
    ),
    fail_on_trajectory_regression: bool = typer.Option(
        False, "--fail-on-trajectory-regression",
        help="Exit code 1 when the trajectory verdict is REGRESSED (CI gate)",
    ),
) -> None:
    """Compare two run results and show a diff table.

    Loads two JSON reports (baseline and candidate) and displays a
    diff table with columns: test name, status, score change, cost
    change, latency change. With --trajectory, also compares process
    quality (per-tool deltas + verdict).

    Examples:
        evalforge compare baseline.json candidate.json
        evalforge compare evalforge-output/report-old.json evalforge-output/report-new.json
        evalforge compare old.json new.json --trajectory --fail-on-trajectory-regression
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

    # Trajectory (process) comparison — opt-in
    if trajectory:
        from evalforge.trajectory.regression import (
            compare_trajectories,
            format_trajectory_regression,
        )
        traj_rep = compare_trajectories(baseline, candidate)
        typer.echo(format_trajectory_regression(traj_rep))
        if (
            fail_on_trajectory_regression
            and traj_rep.get("verdict") == "REGRESSED"
        ):
            typer.echo(
                "Trajectory regression detected — exiting 1 (CI gate).",
                err=True,
            )
            raise typer.Exit(code=1)


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
# evalforge import-trajectory
# ---------------------------------------------------------------------------

@app.command("import-trajectory")
def import_trajectory(
    path: str = typer.Argument(..., help="Path to trajectory JSON/JSONL file"),
    out: str = typer.Option(
        "report.json", "--out",
        help="Output report path (.json or .md)",
    ),
    suite_name: str = typer.Option(
        "imported", "--suite-name",
        help="Suite name to use in the report",
    ),
    allowed_tools: str = typer.Option(
        None, "--allowed-tools",
        help="Comma-separated allowed tool names (enables validity metric)",
    ),
) -> None:
    """Import a trajectory file and produce a process report card.

    The A fallback path (D3): sealed-box agents with no hooks export
    their trace; evalforge grades it. Accepts our own JSON format,
    OTel-style spans, minimal JSONL, or a per-test-id map of
    trajectories (a full run export).

    Examples:
        evalforge import-trajectory trajectory.json --out report.json
        evalforge import-trajectory run_a.json --out run_a_report.json
        evalforge import-trajectory trajectory.json --out report.md --allowed-tools search,calculator
    """
    from evalforge.models import TestResult, Trajectory
    from evalforge.trajectory.importers import (
        load_trajectories_file,
        load_trajectory_json,
    )
    from evalforge.trajectory.metrics import (
        compute_budget,
        compute_convergence,
        compute_efficiency,
        compute_recovery,
        compute_tool_stats,
        compute_validity,
        summarize_trajectories,
    )

    allowed = {t.strip() for t in allowed_tools.split(",")} if allowed_tools else None

    try:
        # Detect shape: a per-test map of trajectories, or a single trajectory.
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        is_map = (
            isinstance(raw, dict)
            and bool(raw)
            and all(isinstance(v, dict) and "steps" in v for v in raw.values())
        )
    except (ValueError, json.JSONDecodeError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError:
        typer.echo(f"Error: trajectory file not found: {path}", err=True)
        raise typer.Exit(code=1)

    try:
        if is_map:
            trajs = load_trajectories_file(path)
            tests = [
                TestResult(
                    id=tid,
                    status="pass" if compute_convergence(t)["converged"] else "fail",
                    response=t.final_answer,
                    trajectory=t,
                )
                for tid, t in trajs.items()
            ]
            summary = summarize_trajectories(tests)
        else:
            traj: Trajectory = load_trajectory_json(path)
            test = TestResult(
                id="imported",
                status="pass" if compute_convergence(traj)["converged"] else "fail",
                response=traj.final_answer,
                trajectory=traj,
            )
            tests = [test]
            summary = summarize_trajectories(tests)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    report = RunResult(
        suite_name=suite_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        duration_ms=0.0,
        tests=tests,
    )

    # Process metrics section — always the summarize_trajectories shape so
    # UI/regression consume one schema. Run-level validity folded in when
    # --allowed-tools is provided.
    summary["validity"] = _run_validity(tests, allowed)
    report.trajectory_summary = summary

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".md":
        out_path.write_text(_trajectory_markdown(report), encoding="utf-8")
    else:
        out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    typer.echo(f"Imported {len(tests)} trajectory test(s) from {path}")
    typer.echo(f"  Mean steps: {summary['mean_steps']:.2f}, "
              f"total loops: {summary['total_loops']}")
    typer.echo(f"  Report written to {out_path}")


def _run_validity(tests, allowed_tools) -> Optional[dict]:
    """Aggregate validity across a run's tests; None when no allowed set."""
    if allowed_tools is None:
        return None
    invalid: set[str] = set()
    count = 0
    for t in tests:
        if t.trajectory is None:
            continue
        for s in t.trajectory.steps:
            if s.tool not in allowed_tools:
                count += 1
                invalid.add(s.tool)
    return {"invalid_calls": count, "invalid_tool_names": sorted(invalid)}


def _trajectory_markdown(report: RunResult) -> str:
    """Render a trajectory report as markdown for human reading."""
    m = report.trajectory_summary or {}
    lines = [
        "# Trajectory Report",
        "",
        f"- Suite: {report.suite_name}",
        f"- Timestamp: {report.timestamp}",
        f"- Tests: {len(report.tests)}",
        "",
        "## Process metrics",
        "",
    ]
    lines.append(f"- Mean steps: {m.get('mean_steps', 0.0):.2f}")
    lines.append(f"- Mean tool calls: {m.get('mean_tool_calls', 0.0):.2f}")
    lines.append(f"- Total loops (repeated calls): {m.get('total_loops', 0)}")
    lines.append(f"- Total error steps: {m.get('total_error_steps', 0)}")
    lines.append("")
    lines.append("## Per-tool")
    lines.append("")
    lines.append("| Tool | Calls | Errors | Latency (ms) | Cost (USD) | Error rate |")
    lines.append("|------|-------|--------|---------------|------------|------------|")
    for tool, e in (m.get("per_tool") or {}).items():
        lines.append(
            f"| {tool} | {e['calls']} | {e['errors']} | "
            f"{e['total_latency_ms']:.1f} | {e['total_cost_usd']:.4f} | "
            f"{e['error_rate']:.2f} |"
        )
    val = m.get("validity")
    if val:
        lines.append("")
        lines.append(f"## Validity: {val['invalid_calls']} invalid calls "
                     f"(unknown tools: {val['invalid_tool_names'] or 'none'})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
