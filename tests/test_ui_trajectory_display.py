"""P6 tests: pure trajectory display functions (no Streamlit import)."""

import pandas as pd

from evalforge.models import Trajectory, TrajectoryStep
from evalforge.ui.trajectory_display import (
    per_tool_df,
    regression_df,
    steps_df,
    trajectory_metrics_df,
    verdict_badge_html,
)


def _summary():
    return {
        "mean_steps": 7.0,
        "mean_tool_calls": 3.0,
        "total_loops": 16,
        "total_error_steps": 2,
        "per_tool": {
            "search": {"calls": 20, "errors": 1, "total_latency_ms": 800.0,
                       "total_cost_usd": 0.002, "error_rate": 0.05},
            "calculator": {"calls": 12, "errors": 0, "total_latency_ms": 300.0,
                           "total_cost_usd": 0.0, "error_rate": 0.0},
        },
    }


class TestTrajectoryMetricsDf:
    def test_rows(self):
        df = trajectory_metrics_df(_summary())
        assert len(df) == 4
        assert "Loops (repeated calls)" in df["metric"].values
        row = df[df["metric"] == "Loops (repeated calls)"].iloc[0]
        assert row["value"] == "16"

    def test_none_summary(self):
        df = trajectory_metrics_df(None)
        assert df.empty


class TestPerToolDf:
    def test_columns_and_values(self):
        df = per_tool_df(_summary())
        assert list(df.columns) == ["tool", "calls", "errors", "latency_ms", "cost_usd", "error_rate"]
        search = df[df["tool"] == "search"].iloc[0]
        assert search["calls"] == 20
        assert search["error_rate"] == 0.05

    def test_empty(self):
        df = per_tool_df({})
        assert df.empty


class TestStepsDf:
    def _traj(self):
        return Trajectory(
            steps=[
                TrajectoryStep(index=0, tool="search", args={"q": "hk"},
                               result='{"name": "lantau"}', latency_ms=800.0),
                TrajectoryStep(index=1, tool="calculator", args={"expr": "1+1"},
                               result="2", latency_ms=150.0),
            ],
            final_answer="2",
        )

    def test_rows(self):
        df = steps_df(self._traj())
        assert len(df) == 2
        assert df.iloc[0]["#"] == 1
        assert df.iloc[0]["tool"] == "search"
        assert df.iloc[1]["result"] == "2"

    def test_long_result_trimmed(self):
        traj = Trajectory(steps=[
            TrajectoryStep(index=0, tool="search", result="x" * 500),
        ])
        df = steps_df(traj)
        assert len(df.iloc[0]["result"]) <= 80

    def test_none(self):
        assert steps_df(None).empty


class TestRegressionDf:
    def _rep(self):
        return {
            "verdict": "REGRESSED",
            "per_tool": {
                "search": {"calls_delta": 8, "error_rate_delta": 0.0,
                           "cost_delta": 0.0, "regressed": True, "improved": False},
                "calculator": {"calls_delta": -2, "error_rate_delta": -0.1,
                               "cost_delta": -0.001, "regressed": False, "improved": True},
                "llm": {"calls_delta": 0, "error_rate_delta": 0.0,
                        "cost_delta": 0.0, "regressed": False, "improved": False},
            },
        }

    def test_verdict_column(self):
        df = regression_df(self._rep())
        assert len(df) == 3
        assert df[df["tool"] == "search"].iloc[0]["verdict"] == "REGRESSED"
        assert df[df["tool"] == "calculator"].iloc[0]["verdict"] == "IMPROVED"
        assert df[df["tool"] == "llm"].iloc[0]["verdict"] == "ok"

    def test_signed_deltas(self):
        df = regression_df(self._rep())
        assert df[df["tool"] == "search"].iloc[0]["calls_delta"] == "+8"
        assert df[df["tool"] == "calculator"].iloc[0]["calls_delta"] == "-2"

    def test_empty(self):
        assert regression_df({}).empty


class TestVerdictBadge:
    def test_regressed_tone(self):
        assert "danger" in verdict_badge_html("REGRESSED")
        assert "REGRESSED" in verdict_badge_html("REGRESSED")

    def test_pass_tone(self):
        assert "success" in verdict_badge_html("PASS")

    def test_none(self):
        assert verdict_badge_html(None) == ""
