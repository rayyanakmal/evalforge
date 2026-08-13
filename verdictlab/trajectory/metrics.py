"""Trajectory process metrics — pure functions over Trajectory.

All functions are deterministic (V3): same trajectory in, byte-identical dict
out. No randomness, no network, no LLM. Every function has a hand-computed
golden test in tests/test_trajectory_metrics.py (V2).

Conventions:
- ``llm``-tooled steps count as *steps* but not as *tool calls* (an LLM span
  is a thought, not a tool invocation).
- A "repeated call" = a step whose (tool, args) was already seen earlier in
  the trajectory. Exposed as raw counts; loop verdicts happen in the
  regression layer with thresholds, never as magic here.
- Trajectories with no steps produce zeros, not errors (tracker pattern).
"""

from __future__ import annotations

from typing import Optional

from verdictlab.models import TestResult, Trajectory, TrajectoryStep

# Tools that record model completions rather than tool invocations.
_LLM_TOOL = "llm"


def _steps(trajectory: Trajectory) -> list[TrajectoryStep]:
    return trajectory.steps


def compute_convergence(trajectory: Trajectory) -> dict:
    """Did the agent reach a final answer, and how did the run terminate?"""
    steps = _steps(trajectory)
    if not steps:
        return {"converged": False, "terminal_reason": "empty"}
    if trajectory.final_answer:
        return {"converged": True, "terminal_reason": "final_answer"}
    if steps[-1].error:
        return {"converged": False, "terminal_reason": "error"}
    return {"converged": False, "terminal_reason": "max_steps"}


def compute_efficiency(trajectory: Trajectory) -> dict:
    """How many steps/tool calls did the run take, and how much looping?"""
    steps = _steps(trajectory)
    tool_calls = [s for s in steps if s.tool != _LLM_TOOL]

    seen: set[tuple[str, str]] = set()
    repeated_calls = 0
    max_repeat_run = 0
    current_run = 0
    prev_key: Optional[tuple[str, str]] = None

    # Loop detection covers TOOL calls only — llm steps are the agent's
    # thoughts between calls, and repeating an llm step is expected in a
    # ReAct loop (the model is consulted after every tool result).
    for s in steps:
        if s.tool == _LLM_TOOL:
            continue
        key = (s.tool, _stable_args(s.args))
        if prev_key is not None and key == prev_key:
            current_run += 1
        else:
            current_run = 1
        max_repeat_run = max(max_repeat_run, current_run)
        prev_key = key

        if key in seen:
            repeated_calls += 1
        else:
            seen.add(key)

    return {
        "steps": len(steps),
        "tool_calls": len(tool_calls),
        "steps_to_answer": len(steps),
        "tool_calls_to_answer": len(tool_calls),
        "repeated_calls": repeated_calls,
        "max_repeat_run": max_repeat_run,
    }


def compute_tool_stats(trajectory: Trajectory) -> dict:
    """Per-tool rollup: calls, errors, latency, cost, error rate; diversity."""
    per_tool: dict[str, dict] = {}
    for s in _steps(trajectory):
        entry = per_tool.setdefault(
            s.tool,
            {"calls": 0, "errors": 0, "total_latency_ms": 0.0,
             "total_cost_usd": 0.0, "error_rate": 0.0},
        )
        entry["calls"] += 1
        if s.error:
            entry["errors"] += 1
        entry["total_latency_ms"] += s.latency_ms
        entry["total_cost_usd"] += s.cost_usd
    for entry in per_tool.values():
        entry["error_rate"] = entry["errors"] / entry["calls"] if entry["calls"] else 0.0
    return {
        "per_tool": dict(sorted(per_tool.items())),
        "tool_diversity": len(per_tool),
    }


def compute_validity(trajectory: Trajectory, allowed_tools: Optional[set[str]]) -> Optional[dict]:
    """Unknown-tool detection. Returns None when no allowed set is known.

    The suite may declare ``metadata.allowed_tools``; without it we cannot
    judge validity, so we skip rather than guess (AC-T2.3).
    """
    if allowed_tools is None:
        return None
    invalid = sorted({
        s.tool for s in _steps(trajectory)
        if s.tool not in allowed_tools
    })
    return {
        "invalid_calls": sum(1 for s in _steps(trajectory) if s.tool not in allowed_tools),
        "invalid_tool_names": invalid,
    }


def compute_recovery(trajectory: Trajectory) -> dict:
    """Error handling: did the run survive errors or die on one?"""
    steps = _steps(trajectory)
    error_steps = [s for s in steps if s.error]
    return {
        "error_steps": len(error_steps),
        "recovered_after_error": sum(1 for s in error_steps if s is not steps[-1]),
        "died_on_error": bool(steps and steps[-1].error),
    }


def compute_budget(
    trajectory: Trajectory,
    cost_limit_usd: Optional[float] = None,
    max_steps: Optional[int] = None,
) -> dict:
    """Budget adherence. Report-only (Q2 default): no enforcement here."""
    steps = _steps(trajectory)
    total_cost = sum(s.cost_usd for s in steps)
    over_budget = cost_limit_usd is not None and total_cost > cost_limit_usd
    over_max_steps = max_steps is not None and len(steps) > max_steps
    return {
        "over_budget": bool(over_budget),
        "over_max_steps": bool(over_max_steps),
        "total_cost_usd": total_cost,
    }


def summarize_trajectories(results: list[TestResult]) -> dict:
    """Aggregate process metrics across a run (RunResult.trajectory_summary).

    Skips results with no trajectory (v1-shaped results stay harmless).
    pass_rate_by_tool_usage: for each tool, the pass rate of tests that used it.
    """
    trajectories: list[Trajectory] = [
        r.trajectory for r in results if r.trajectory is not None
    ]
    with_traj = [r for r in results if r.trajectory is not None]
    if not trajectories:
        return {
            "mean_steps": 0.0, "mean_tool_calls": 0.0, "total_loops": 0,
            "total_error_steps": 0, "per_tool": {},
            "pass_rate_by_tool_usage": {},
        }

    mean_steps = sum(len(t.steps) for t in trajectories) / len(trajectories)
    mean_tool_calls = sum(
        1 for t in trajectories for s in t.steps if s.tool != _LLM_TOOL
    ) / len(trajectories)

    total_loops = sum(
        compute_efficiency(t)["repeated_calls"] for t in trajectories
    )
    total_error_steps = sum(
        compute_recovery(t)["error_steps"] for t in trajectories
    )

    per_tool: dict[str, dict] = {}
    usage: dict[str, list[bool]] = {}
    for r in with_traj:
        stats = compute_tool_stats(r.trajectory)["per_tool"]
        for tool, entry in stats.items():
            agg = per_tool.setdefault(
                tool,
                {"calls": 0, "errors": 0, "total_latency_ms": 0.0,
                 "total_cost_usd": 0.0, "error_rate": 0.0},
            )
            agg["calls"] += entry["calls"]
            agg["errors"] += entry["errors"]
            agg["total_latency_ms"] += entry["total_latency_ms"]
            agg["total_cost_usd"] += entry["total_cost_usd"]
            usage.setdefault(tool, []).append(r.status == "pass")
    for entry in per_tool.values():
        entry["error_rate"] = entry["errors"] / entry["calls"] if entry["calls"] else 0.0

    pass_rate_by_tool_usage = {
        tool: (sum(passes) / len(passes)) if passes else 0.0
        for tool, passes in usage.items()
    }

    return {
        "mean_steps": mean_steps,
        "mean_tool_calls": mean_tool_calls,
        "total_loops": total_loops,
        "total_error_steps": total_error_steps,
        "per_tool": dict(sorted(per_tool.items())),
        "pass_rate_by_tool_usage": dict(sorted(pass_rate_by_tool_usage.items())),
    }


def _stable_args(args: dict) -> str:
    """Deterministic string key for (tool, args) repetition detection."""
    return repr(sorted(args.items()))
