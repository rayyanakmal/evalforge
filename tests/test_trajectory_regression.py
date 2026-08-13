"""P4 tests: trajectory regression — per-tool + per-test deltas + verdict."""

import pytest

from verdictlab.models import RunResult, TestResult, Trajectory, TrajectoryStep


def step(index, tool="search", **overrides):
    data = {"index": index, "tool": tool, "args": {}, "result": "ok"}
    data.update(overrides)
    return TrajectoryStep(**data)


def run_result(suite_name, test_map):
    """Build a RunResult from {test_id: Trajectory}."""
    tests = [
        TestResult(id=tid, status="pass", trajectory=traj)
        for tid, traj in test_map.items()
    ]
    return RunResult(
        suite_name=suite_name, timestamp="ts", duration_ms=1.0, tests=tests,
    )


class TestCompareTrajectories:
    def test_clean_vs_loop_regression(self):
        """The demo case: same pass rate, candidate loops more.

        base:  t1 = [search, calc]              (2 steps, 0 repeats)
        cand:  t1 = [search, search, calc]      (3 steps, 1 repeat)
        search calls 1 -> 2, cost 0.001 -> 0.002: regressed.
        """
        from verdictlab.trajectory.regression import compare_trajectories

        base = run_result("base", {
            "t1": Trajectory(
                steps=[step(0, cost_usd=0.001), step(1, tool="calc", cost_usd=0.0005)],
                final_answer="2"),
        })
        cand = run_result("cand", {
            "t1": Trajectory(
                steps=[step(0, cost_usd=0.001), step(1, cost_usd=0.001), step(2, tool="calc", cost_usd=0.0005)],
                final_answer="2"),
        })
        rep = compare_trajectories(base, cand)
        assert rep["verdict"] == "REGRESSED"
        assert "search" in rep["per_tool"]
        assert rep["per_tool"]["search"]["regressed"] is True
        assert rep["per_tool"]["search"]["calls_delta"] == 1
        assert rep["per_tool"]["calc"]["regressed"] is False
        # per-test delta
        assert rep["per_test"]["t1"]["steps_delta"] == 1
        assert rep["per_test"]["t1"]["repeated_calls_delta"] == 1

    def test_disappeared_tool_is_regressed(self):
        """Tool in base, absent in candidate -> REGRESSED (mirror visionforge)."""
        from verdictlab.trajectory.regression import compare_trajectories

        base = run_result("base", {
            "t1": Trajectory(steps=[step(0), step(1, tool="calc")], final_answer="x"),
        })
        cand = run_result("cand", {
            "t1": Trajectory(steps=[step(0)], final_answer="x"),
        })
        rep = compare_trajectories(base, cand)
        assert rep["verdict"] == "REGRESSED"
        assert rep["per_tool"]["calc"]["regressed"] is True
        assert rep["per_tool"]["calc"]["disappeared"] is True

    def test_improvement(self):
        """Fewer errors -> IMPROVED, not regressed."""
        from verdictlab.trajectory.regression import compare_trajectories

        base = run_result("base", {
            "t1": Trajectory(steps=[step(0, error="boom")], final_answer=None),
        })
        cand = run_result("cand", {
            "t1": Trajectory(steps=[step(0)], final_answer="ok"),
        })
        rep = compare_trajectories(base, cand)
        assert rep["verdict"] == "PASS"
        assert rep["per_tool"]["search"]["regressed"] is False
        assert rep["per_tool"]["search"]["error_rate_delta"] == pytest.approx(-1.0)

    def test_equal_runs_pass(self):
        from verdictlab.trajectory.regression import compare_trajectories

        traj = Trajectory(steps=[step(0)], final_answer="x")
        base = run_result("base", {"t1": traj})
        cand = run_result("cand", {"t1": traj})
        rep = compare_trajectories(base, cand)
        assert rep["verdict"] == "PASS"
        assert rep["per_tool"]["search"]["regressed"] is False

    def test_no_trajectories_returns_no_verdict(self):
        from verdictlab.trajectory.regression import compare_trajectories

        base = RunResult(suite_name="b", timestamp="ts", duration_ms=1.0,
                         tests=[TestResult(id="t1", status="pass")])
        cand = RunResult(suite_name="c", timestamp="ts", duration_ms=1.0,
                         tests=[TestResult(id="t1", status="pass")])
        rep = compare_trajectories(base, cand)
        assert rep["verdict"] is None
        assert rep["per_tool"] == {}

    def test_threshold_respects_small_deltas(self):
        """Cost delta below threshold is not a regression."""
        from verdictlab.trajectory.regression import compare_trajectories

        base = run_result("base", {
            "t1": Trajectory(steps=[step(0, cost_usd=0.001)]),
        })
        cand = run_result("cand", {
            "t1": Trajectory(steps=[step(0, cost_usd=0.001001), step(1, tool="calc")]),
        })
        rep = compare_trajectories(base, cand, cost_threshold=0.01)
        assert rep["verdict"] == "PASS"


class TestRegressionOutput:
    def test_format_regression_report(self):
        """The CLI-facing report: tool rows + verdict line."""
        from verdictlab.trajectory.regression import (
            compare_trajectories, format_trajectory_regression,
        )

        base = run_result("base", {
            "t1": Trajectory(steps=[step(0, cost_usd=0.001)], final_answer="x"),
        })
        cand = run_result("cand", {
            "t1": Trajectory(steps=[step(0, cost_usd=0.001), step(1, cost_usd=0.001)], final_answer="x"),
        })
        rep = compare_trajectories(base, cand)
        text = format_trajectory_regression(rep)
        assert "REGRESSED" in text
        assert "search" in text
