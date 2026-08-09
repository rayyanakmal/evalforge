"""P1 exit-gate tests: trajectory schema contracts + backward compatibility.

Covers Task 1.1 (Trajectory/TrajectoryStep validators) and Task 1.2
(TestResult.trajectory optional, RunResult.trajectory_summary optional).
"""

import json

import pytest
from pydantic import ValidationError

from evalforge.models import (
    RunResult, TestResult, TokenCount, Trajectory, TrajectoryStep,
)


def make_step(index=0, tool="search", **overrides):
    data = {
        "index": index,
        "tool": tool,
        "args": {"q": "hk islands"},
        "result": {"count": 3},
    }
    data.update(overrides)
    return TrajectoryStep(**data)


class TestTrajectoryStep:
    def test_minimal_step(self):
        step = TrajectoryStep(index=0, tool="search")
        assert step.args == {}
        assert step.result is None
        assert step.thought is None
        assert step.latency_ms == 0.0
        assert step.tokens is None
        assert step.cost_usd == 0.0
        assert step.error is None

    def test_full_step(self):
        step = make_step(
            thought="need population data",
            latency_ms=120.5,
            tokens=TokenCount(input=10, output=5, total=15),
            cost_usd=0.0042,
        )
        assert step.latency_ms == 120.5
        assert step.tokens.total == 15

    def test_empty_tool_rejected(self):
        with pytest.raises(ValidationError):
            TrajectoryStep(index=0, tool="")

    def test_negative_index_rejected(self):
        with pytest.raises(ValidationError):
            TrajectoryStep(index=-1, tool="search")

    def test_negative_latency_rejected(self):
        with pytest.raises(ValidationError):
            make_step(latency_ms=-1.0)

    def test_args_must_be_json_safe(self):
        with pytest.raises(ValidationError):
            make_step(args={"bad": object()})

    def test_args_dict_keys_must_be_str(self):
        with pytest.raises(ValidationError):
            make_step(args={1: "one"})

    def test_result_json_safe(self):
        step = make_step(result={"nested": [1, 2, {"three": True}]})
        assert step.result["nested"][2]["three"] is True

    def test_result_non_json_rejected(self):
        with pytest.raises(ValidationError):
            make_step(result={"obj": object()})

    def test_deep_nesting_rejected(self):
        # 10 levels deep exceeds the 8-level cap
        payload = "x"
        for _ in range(10):
            payload = [payload]
        with pytest.raises(ValidationError):
            make_step(args={"deep": payload})

    def test_serializes_to_json(self):
        step = make_step()
        roundtrip = json.loads(step.model_dump_json())
        assert roundtrip["tool"] == "search"
        assert roundtrip["args"] == {"q": "hk islands"}


class TestTrajectory:
    def test_empty_trajectory(self):
        traj = Trajectory()
        assert traj.steps == []
        assert traj.final_answer is None

    def test_ordered_steps_accepted(self):
        traj = Trajectory(
            steps=[make_step(0), make_step(1, tool="calculator")],
            final_answer="42",
        )
        assert len(traj.steps) == 2
        assert traj.final_answer == "42"

    def test_index_must_match_position(self):
        with pytest.raises(ValidationError):
            Trajectory(steps=[make_step(0), make_step(2, tool="calculator")])

    def test_duplicate_index_rejected(self):
        with pytest.raises(ValidationError):
            Trajectory(steps=[make_step(0), make_step(0)])

    def test_starting_index_must_be_zero(self):
        with pytest.raises(ValidationError):
            Trajectory(steps=[make_step(1)])

    def test_json_roundtrip(self):
        traj = Trajectory(
            steps=[make_step(0), make_step(1, tool="calculator", result="3")],
            final_answer="3",
        )
        roundtrip = Trajectory.model_validate_json(traj.model_dump_json())
        assert roundtrip.steps[1].tool == "calculator"
        assert roundtrip.final_answer == "3"


class TestBackwardCompat:
    """V4: v1 run files parse unchanged; trajectory is optional."""

    def test_v1_test_result_without_trajectory(self):
        tr = TestResult(
            id="t1", status="pass", response="hello",
            expected_value="hello", latency_ms=10.0, cost_usd=0.001,
        )
        assert tr.trajectory is None

    def test_v1_run_result_parses(self):
        rr = RunResult(
            suite_name="demo", timestamp="2026-01-01T00:00:00", duration_ms=5.0,
            tests=[
                TestResult(id="t1", status="pass"),
                TestResult(id="t2", status="fail", error="mismatch"),
            ],
        )
        assert rr.trajectory_summary is None
        assert len(rr.tests) == 2

    def test_test_result_with_trajectory(self):
        tr = TestResult(
            id="t1", status="pass",
            trajectory=Trajectory(steps=[make_step(0)]),
        )
        assert tr.trajectory is not None
        assert tr.trajectory.steps[0].tool == "search"

    def test_run_result_with_trajectory_summary(self):
        rr = RunResult(
            suite_name="demo", timestamp="ts", duration_ms=1.0,
            trajectory_summary={"mean_steps": 3.0},
        )
        assert rr.trajectory_summary["mean_steps"] == 3.0

    def test_v1_json_report_still_validates(self):
        """A v1 report file (no trajectory keys) must load into RunResult."""
        v1_json = {
            "suite_name": "legacy",
            "timestamp": "2026-01-01T00:00:00",
            "duration_ms": 12.0,
            "tests": [
                {"id": "a", "status": "pass", "response": "ok", "latency_ms": 5.0},
            ],
            "summary": {"total": 1, "passed": 1, "failed": 0, "errored": 0,
                        "pass_rate": 1.0, "total_cost_usd": 0.0,
                        "avg_latency_ms": 5.0, "latency_p50": None,
                        "latency_p95": None, "latency_p99": None},
        }
        rr = RunResult.model_validate(v1_json)
        assert rr.tests[0].status == "pass"
        assert rr.tests[0].trajectory is None
