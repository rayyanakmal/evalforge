"""Pure display functions for the trajectory layer — unit-testable, no Streamlit.

These convert trajectory metrics/regression dicts into DataFrame/html shapes
the dashboard renders. Streamlit widgets stay thin in streamlit_app.py.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from verdictlab.models import Trajectory


def trajectory_metrics_df(summary: Optional[dict]) -> pd.DataFrame:
    """Process hero metrics: one row per metric, machine + display columns."""
    if not summary:
        return pd.DataFrame(columns=["metric", "value", "detail"])
    rows = [
        ("Mean steps", f"{summary.get('mean_steps', 0.0):.2f}", "avg steps per test"),
        ("Mean tool calls", f"{summary.get('mean_tool_calls', 0.0):.2f}", "avg tool invocations per test"),
        ("Loops (repeated calls)", f"{summary.get('total_loops', 0)}", "identical tool+args repeats"),
        ("Error steps", f"{summary.get('total_error_steps', 0)}", "steps that errored"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "detail"])


def per_tool_df(summary: Optional[dict]) -> pd.DataFrame:
    """Per-tool rollup table from a trajectory_summary."""
    if not summary:
        return pd.DataFrame(columns=["tool", "calls", "errors", "latency_ms", "cost_usd", "error_rate"])
    rows = []
    for tool, e in (summary.get("per_tool") or {}).items():
        rows.append({
            "tool": tool,
            "calls": e.get("calls", 0),
            "errors": e.get("errors", 0),
            "latency_ms": round(e.get("total_latency_ms", 0.0), 1),
            "cost_usd": round(e.get("total_cost_usd", 0.0), 4),
            "error_rate": round(e.get("error_rate", 0.0), 2),
        })
    return pd.DataFrame(rows)


def steps_df(trajectory: Optional[Trajectory]) -> pd.DataFrame:
    """Step timeline for one test: numbered tool calls with args/result."""
    if trajectory is None:
        return pd.DataFrame(columns=["#", "tool", "args", "result", "latency_ms", "error"])
    rows = []
    for s in trajectory.steps:
        args = s.args if isinstance(s.args, dict) else {}
        rows.append({
            "#": s.index + 1,
            "tool": s.tool,
            "args": _short(args, 60),
            "result": _short(s.result, 80),
            "latency_ms": round(s.latency_ms, 1),
            "error": s.error or "",
        })
    return pd.DataFrame(rows)


def regression_df(rep: Optional[dict]) -> pd.DataFrame:
    """Per-tool delta table from a compare_trajectories result."""
    if not rep:
        return pd.DataFrame(columns=["tool", "calls_delta", "error_rate_delta", "cost_delta", "verdict"])
    rows = []
    for tool, e in (rep.get("per_tool") or {}).items():
        verdict = "REGRESSED" if e.get("regressed") else ("IMPROVED" if e.get("improved") else "ok")
        rows.append({
            "tool": tool,
            "calls_delta": _signed(e.get("calls_delta", 0)),
            "error_rate_delta": _signed(round(e.get("error_rate_delta", 0.0), 2)),
            "cost_delta": _signed(round(e.get("cost_delta", 0.0), 4)),
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def verdict_badge_html(verdict: Optional[str]) -> str:
    """Badge HTML for the regression verdict banner."""
    if verdict is None:
        return ""
    tone = "danger" if verdict == "REGRESSED" else "success"
    return (
        f'<span class="badge {tone}">{verdict}</span>'
    )


def _short(value, max_len: int) -> str:
    """Trim a value for table display."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _signed(value) -> str:
    if value == 0:
        return "0"
    return f"+{value}" if value > 0 else str(value)
