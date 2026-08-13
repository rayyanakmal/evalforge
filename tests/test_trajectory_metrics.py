"""P2 golden tests: trajectory process metrics, hand-computed expected values.

Every metric is a pure function over Trajectory. Expected values below were
computed by hand from the fixture shapes — see comments per test.
"""

import pytest

from verdictlab.models import TestResult, Trajectory, TrajectoryStep


def step(index, tool="search", **overrides):
    data = {"index": index, "tool": tool, "args": {}, "result": "ok"}
    data.update(overrides)
    return TrajectoryStep(**data)


def traj(steps, final_answer=None):
    return Trajectory(steps=steps, final_answer=final_answer)


class TestConvergence:
    def test_empty_trajectory(self):
        from verdictlab.trajectory.metrics import compute_convergence
        m = compute_convergence(traj([]))
        assert m["converged"] is False
        assert m["terminal_reason"] == "empty"

    def test_final_answer_converged(self):
        from verdictlab.trajectory.metrics import compute_convergence
        m = compute_convergence(traj([step(0)], final_answer="ok"))
        assert m["converged"] is True
        assert m["terminal_reason"] == "final_answer"

    def test_ends_in_error(self):
        from verdictlab.trajectory.metrics import compute_convergence
        m = compute_convergence(traj([
            step(0),
            step(1, tool="calc", error="boom"),
        ]))
        assert m["converged"] is False
        assert m["terminal_reason"] == "error"

    def test_no_answer_no_error_is_max_steps(self):
        from verdictlab.trajectory.metrics import compute_convergence
        m = compute_convergence(traj([step(0), step(1, tool="calc")]))
        assert m["converged"] is False
        assert m["terminal_reason"] == "max_steps"


class TestEfficiency:
    def test_steps_and_tool_calls_exclude_llm(self):
        """llm steps count as steps but NOT as tool calls."""
        from verdictlab.trajectory.metrics import compute_efficiency
        t = traj([
            step(0, tool="llm"),
            step(1, tool="search", args={"q": "x"}),
            step(2, tool="llm"),
        ], final_answer="x")
        m = compute_efficiency(t)
        assert m["steps"] == 3
        assert m["tool_calls"] == 1
        assert m["steps_to_answer"] == 3
        assert m["tool_calls_to_answer"] == 1

    def test_repeated_calls_and_max_repeat_run(self):
        """step1 repeats step0's call (search q=a) -> repeated; step3 repeats
        step0 again but non-consecutive. The consecutive run is 2 (step0+step1)."""
        from verdictlab.trajectory.metrics import compute_efficiency
        t = traj([
            step(0, args={"q": "a"}),
            step(1, args={"q": "a"}),           # repeat of (0)
            step(2, tool="calc", args={"expr": "1+1"}),
            step(3, args={"q": "a"}),           # repeat of (0), non-consecutive
        ], final_answer="a")
        m = compute_efficiency(t)
        assert m["steps"] == 4
        assert m["tool_calls"] == 4
        assert m["repeated_calls"] == 2
        assert m["max_repeat_run"] == 2

    def test_zero_repeats_clean_trajectory(self):
        from verdictlab.trajectory.metrics import compute_efficiency
        t = traj([
            step(0, args={"q": "a"}),
            step(1, tool="calc", args={"expr": "1+1"}),
        ], final_answer="2")
        m = compute_efficiency(t)
        assert m["repeated_calls"] == 0
        assert m["max_repeat_run"] == 1

    def test_empty_trajectory_zeros(self):
        from verdictlab.trajectory.metrics import compute_efficiency
        m = compute_efficiency(traj([]))
        assert m["steps"] == 0
        assert m["tool_calls"] == 0
        assert m["repeated_calls"] == 0
        assert m["max_repeat_run"] == 0


class TestToolStats:
    def test_per_tool_rollup(self):
        """search called twice (1 error), calc once. Latency and cost summed."""
        from verdictlab.trajectory.metrics import compute_tool_stats
        t = traj([
            step(0, latency_ms=100.0, cost_usd=0.001),
            step(1, tool="calc", args={"expr": "1+1"}, latency_ms=20.0, cost_usd=0.0001),
            step(2, latency_ms=50.0, cost_usd=0.002, error="timeout"),
        ], final_answer="2")
        m = compute_tool_stats(t)
        assert set(m["per_tool"].keys()) == {"search", "calc"}
        assert m["per_tool"]["search"] == {
            "calls": 2, "errors": 1, "total_latency_ms": 150.0,
            "total_cost_usd": 0.003, "error_rate": 0.5,
        }
        assert m["per_tool"]["calc"] == {
            "calls": 1, "errors": 0, "total_latency_ms": 20.0,
            "total_cost_usd": 0.0001, "error_rate": 0.0,
        }
        assert m["tool_diversity"] == 2

    def test_empty_trajectory(self):
        from verdictlab.trajectory.metrics import compute_tool_stats
        m = compute_tool_stats(traj([]))
        assert m["per_tool"] == {}
        assert m["tool_diversity"] == 0


class TestValidity:
    def test_unknown_tool_detected(self):
        from verdictlab.trajectory.metrics import compute_validity
        t = traj([
            step(0),
            step(1, tool="calc"),
        ], final_answer="x")
        m = compute_validity(t, allowed_tools={"search"})
        assert m["invalid_calls"] == 1
        assert m["invalid_tool_names"] == ["calc"]

    def test_all_valid(self):
        from verdictlab.trajectory.metrics import compute_validity
        t = traj([step(0), step(1, tool="calc")], final_answer="x")
        m = compute_validity(t, allowed_tools={"search", "calc"})
        assert m["invalid_calls"] == 0
        assert m["invalid_tool_names"] == []

    def test_no_allowed_tools_skips(self):
        from verdictlab.trajectory.metrics import compute_validity
        m = compute_validity(traj([step(0)]), allowed_tools=None)
        assert m is None


class TestRecovery:
    def test_recovered_after_error(self):
        from verdictlab.trajectory.metrics import compute_recovery
        t = traj([
            step(0),
            step(1, tool="calc", error="boom"),
            step(2, tool="search", args={"q": "retry"}),
        ], final_answer="ok")
        m = compute_recovery(t)
        assert m["error_steps"] == 1
        assert m["recovered_after_error"] == 1
        assert m["died_on_error"] is False

    def test_died_on_error(self):
        from verdictlab.trajectory.metrics import compute_recovery
        t = traj([step(0), step(1, error="boom")])
        m = compute_recovery(t)
        assert m["error_steps"] == 1
        assert m["recovered_after_error"] == 0
        assert m["died_on_error"] is True

    def test_no_errors(self):
        from verdictlab.trajectory.metrics import compute_recovery
        m = compute_recovery(traj([step(0), step(1)], final_answer="x"))
        assert m["error_steps"] == 0
        assert m["recovered_after_error"] == 0
        assert m["died_on_error"] is False


class TestBudget:
    def test_under_budget(self):
        from verdictlab.trajectory.metrics import compute_budget
        t = traj([step(0, cost_usd=0.001), step(1, cost_usd=0.002)])
        m = compute_budget(t, cost_limit_usd=0.01, max_steps=10)
        assert m["over_budget"] is False
        assert m["over_max_steps"] is False
        assert m["total_cost_usd"] == 0.003

    def test_over_budget(self):
        from verdictlab.trajectory.metrics import compute_budget
        t = traj([step(0, cost_usd=0.02)])
        m = compute_budget(t, cost_limit_usd=0.01, max_steps=10)
        assert m["over_budget"] is True

    def test_over_max_steps(self):
        from verdictlab.trajectory.metrics import compute_budget
        t = traj([step(i) for i in range(5)])
        m = compute_budget(t, cost_limit_usd=None, max_steps=3)
        assert m["over_max_steps"] is True

    def test_no_limits_no_flags(self):
        from verdictlab.trajectory.metrics import compute_budget
        m = compute_budget(traj([step(0)]), cost_limit_usd=None, max_steps=None)
        assert m["over_budget"] is False
        assert m["over_max_steps"] is False


class TestSummarize:
    def _result(self, tid, steps_list, final_answer, status="pass"):
        return TestResult(
            id=tid, status=status,
            trajectory=Trajectory(steps=steps_list, final_answer=final_answer),
        )

    def test_aggregates_across_tests(self):
        """3 tests:
        t1: 2 steps, 1 repeated, 1 error (search x2)   — FAIL (died on error)
        t2: 4 steps, 2 repeated, 0 errors (search, calc) — PASS
        t3: 1 step,  0 repeated, 0 errors (search)        — FAIL
        mean_steps = (2+4+1)/3 = 7/3; mean_tool_calls same (no llm steps).
        total_loops = 1+2+0 = 3; total_error_steps = 1.
        per_tool: search calls = 2+3+1 = 6, calc calls = 1.
        pass_rate_by_tool_usage: search used by all 3 -> 1/3 pass (t2 only);
                                 calc used by t2 only -> 1/1 pass.
        """
        from verdictlab.trajectory.metrics import summarize_trajectories
        results = [
            self._result("t1", [
                step(0, args={"q": "a"}),
                step(1, args={"q": "a"}, error="boom"),
            ], None, status="fail"),
            self._result("t2", [
                step(0, args={"q": "a"}),
                step(1, args={"q": "a"}),
                step(2, tool="calc", args={"expr": "1+1"}),
                step(3, args={"q": "a"}),
            ], "a", status="pass"),
            self._result("t3", [step(0, args={"q": "b"})], "b", status="fail"),
        ]
        s = summarize_trajectories(results)
        assert s["mean_steps"] == pytest.approx(7 / 3)
        assert s["mean_tool_calls"] == pytest.approx(7 / 3)
        assert s["total_loops"] == 3
        assert s["total_error_steps"] == 1
        assert s["per_tool"]["search"]["calls"] == 6
        assert s["per_tool"]["calc"]["calls"] == 1
        assert s["per_tool"]["search"]["errors"] == 1
        assert s["pass_rate_by_tool_usage"]["search"] == pytest.approx(1 / 3)
        assert s["pass_rate_by_tool_usage"]["calc"] == pytest.approx(1.0)

    def test_results_without_trajectory_skipped(self):
        from verdictlab.trajectory.metrics import summarize_trajectories
        results = [
            TestResult(id="no-traj", status="pass"),   # v1-shaped, no trajectory
            TestResult(id="with-traj", status="pass",
                       trajectory=Trajectory(steps=[step(0)], final_answer="x")),
        ]
        s = summarize_trajectories(results)
        assert s["mean_steps"] == pytest.approx(1.0)
        assert s["total_loops"] == 0

    def test_empty_input(self):
        from verdictlab.trajectory.metrics import summarize_trajectories
        s = summarize_trajectories([])
        assert s["mean_steps"] == 0.0
        assert s["mean_tool_calls"] == 0.0
        assert s["per_tool"] == {}
        assert s["pass_rate_by_tool_usage"] == {}
