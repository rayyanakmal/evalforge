"""Trajectory regression — run A vs run B on process quality (D7).

Mirrors visionforge's compare philosophy: per-tool + per-test deltas with a
PASS/REGRESSED verdict, strict on anything that got worse. Pure code, no LLM.
"""

from __future__ import annotations

from typing import Optional

from evalforge.models import RunResult
from evalforge.trajectory.metrics import (
    compute_efficiency,
    compute_recovery,
    compute_tool_stats,
)


def compare_trajectories(
    base: RunResult,
    candidate: RunResult,
    threshold: float = 0.05,
    cost_threshold: float = 0.01,
) -> dict:
    """Compare trajectory quality between two runs.

    Returns a dict with:
    - verdict: "PASS" | "REGRESSED" | None (None when neither run has
      trajectory data)
    - per_tool: {tool: {calls_delta, error_rate_delta, cost_delta,
                        regressed, disappeared, improved}}
    - per_test: {test_id: {steps_delta, repeated_calls_delta, cost_delta}}
    """
    base_traj = {t.id: t.trajectory for t in base.tests if t.trajectory is not None}
    cand_traj = {t.id: t.trajectory for t in candidate.tests if t.trajectory is not None}
    if not base_traj and not cand_traj:
        return {"verdict": None, "per_tool": {}, "per_test": {}}

    # Per-tool rollup for each run
    base_stats = _rollup(base_traj)
    cand_stats = _rollup(cand_traj)

    per_tool: dict[str, dict] = {}
    for tool in sorted(set(base_stats) | set(cand_stats)):
        b = base_stats.get(tool, _zero())
        c = cand_stats.get(tool, _zero())
        entry = {
            "calls_delta": c["calls"] - b["calls"],
            "error_rate_delta": c["error_rate"] - b["error_rate"],
            "cost_delta": c["cost"] - b["cost"],
            "disappeared": tool in base_stats and tool not in cand_stats,
            "appeared": tool not in base_stats and tool in cand_stats,
        }
        # Regression rules (mirror visionforge strictness):
        # - disappeared tool = REGRESSED
        # - more calls (looping) beyond threshold = REGRESSED
        # - error rate up beyond threshold = REGRESSED
        # - cost up beyond cost_threshold = REGRESSED
        entry["regressed"] = bool(
            entry["disappeared"]
            or (
                not entry["appeared"]
                and entry["calls_delta"] > threshold
            )
            or entry["error_rate_delta"] > threshold
            or entry["cost_delta"] > cost_threshold
        )
        entry["improved"] = bool(
            (not entry["regressed"])
            and (
                entry["error_rate_delta"] < -threshold
                or entry["cost_delta"] < -cost_threshold
                or entry["calls_delta"] < -threshold
            )
        )
        per_tool[tool] = entry

    # Per-test step deltas
    per_test: dict[str, dict] = {}
    for tid in sorted(set(base_traj) | set(cand_traj)):
        if tid in base_traj and tid in cand_traj:
            b_eff = compute_efficiency(base_traj[tid])
            c_eff = compute_efficiency(cand_traj[tid])
            b_cost = sum(s.cost_usd for s in base_traj[tid].steps)
            c_cost = sum(s.cost_usd for s in cand_traj[tid].steps)
            per_test[tid] = {
                "steps_delta": c_eff["steps"] - b_eff["steps"],
                "repeated_calls_delta": c_eff["repeated_calls"] - b_eff["repeated_calls"],
                "cost_delta": c_cost - b_cost,
            }
        elif tid in cand_traj:  # appeared in candidate
            per_test[tid] = {"steps_delta": None, "repeated_calls_delta": None, "cost_delta": None, "appeared": True}
        else:  # disappeared from candidate
            per_test[tid] = {"steps_delta": None, "repeated_calls_delta": None, "cost_delta": None, "disappeared": True}

    any_regressed = any(v["regressed"] for v in per_tool.values())
    return {
        "verdict": "REGRESSED" if any_regressed else "PASS",
        "per_tool": per_tool,
        "per_test": per_test,
    }


def format_trajectory_regression(rep: dict) -> str:
    """Render the regression dict as a human-readable report block."""
    lines: list[str] = []
    lines.append("  [TRAJECTORY REGRESSION]")
    verdict = rep.get("verdict")
    if verdict is None:
        lines.append("  No trajectory data in either run — nothing to compare.")
        return "\n".join(lines)

    per_tool = rep.get("per_tool", {})
    if not per_tool:
        lines.append("  No tool-level trajectory data to compare.")
    else:
        lines.append(f"  {'Tool':<12} {'Calls Δ':<8} {'ErrRate Δ':<10} {'Cost Δ':<10} Verdict")
        lines.append(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
        for tool, e in per_tool.items():
            verdict_txt = "REGRESSED" if e["regressed"] else ("IMPROVED" if e["improved"] else "ok")
            lines.append(
                f"  {tool:<12} {e['calls_delta']:>+6} {e['error_rate_delta']:>+9.2f} "
                f"{e['cost_delta']:>+9.4f} {verdict_txt}"
            )
    lines.append(f"  Verdict: {verdict}")
    return "\n".join(lines)


def _rollup(traj_map: dict) -> dict[str, dict]:
    """Aggregate per-tool stats across a run's trajectories."""
    out: dict[str, dict] = {}
    for traj in traj_map.values():
        stats = compute_tool_stats(traj)["per_tool"]
        for tool, entry in stats.items():
            agg = out.setdefault(tool, _zero())
            agg["calls"] += entry["calls"]
            agg["errors"] += entry["errors"]
            agg["cost"] += entry["total_cost_usd"]
    for entry in out.values():
        entry["error_rate"] = entry["errors"] / entry["calls"] if entry["calls"] else 0.0
    return out


def _zero() -> dict:
    return {"calls": 0, "errors": 0, "cost": 0.0, "error_rate": 0.0}
