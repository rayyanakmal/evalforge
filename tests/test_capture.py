"""P3 tests: capture layer — StepRecorder, emit_step, importers, adapters."""

import json
import time

import pytest

from evalforge.models import Trajectory, TrajectoryStep
from evalforge.trajectory.capture import StepRecorder, record
from evalforge.trajectory.importers import load_trajectory_json, load_trajectories_file


class TestStepRecorder:
    def test_emit_collects_steps_in_order(self):
        rec = StepRecorder()
        rec.emit(tool="search", args={"q": "hk"})
        rec.emit(tool="calculator", args={"expr": "1+1"}, result="2")
        t = rec.trajectory()
        assert len(t.steps) == 2
        assert t.steps[0].index == 0
        assert t.steps[1].index == 1
        assert t.steps[0].tool == "search"
        assert t.steps[1].result == "2"

    def test_final_answer_sets_trajectory_answer(self):
        rec = StepRecorder()
        rec.emit(tool="search", args={"q": "x"})
        rec.finish("the answer")
        t = rec.trajectory()
        assert t.final_answer == "the answer"

    def test_call_context_times_and_captures_result(self):
        rec = StepRecorder()
        with rec.call(tool="calculator", args={"expr": "1+1"}) as step:
            time.sleep(0.01)
            step.result = "2"
        t = rec.trajectory()
        assert len(t.steps) == 1
        assert t.steps[0].result == "2"
        assert t.steps[0].latency_ms >= 5.0

    def test_call_context_captures_exception_as_error(self):
        rec = StepRecorder()
        with pytest.raises(RuntimeError):
            with rec.call(tool="search", args={"q": "boom"}) as step:
                step.result = "partial"
                raise RuntimeError("tool crashed")
        t = rec.trajectory()
        assert len(t.steps) == 1
        assert t.steps[0].error is not None
        assert "tool crashed" in t.steps[0].error

    def test_empty_recorder(self):
        rec = StepRecorder()
        t = rec.trajectory()
        assert t.steps == []
        assert t.final_answer is None

    def test_emit_with_tokens_and_cost(self):
        rec = StepRecorder()
        rec.emit(tool="llm", args={"prompt": "hi"}, tokens_in=10, tokens_out=5, cost_usd=0.001)
        step = rec.trajectory().steps[0]
        assert step.tokens.input == 10
        assert step.tokens.output == 5
        assert step.tokens.total == 15
        assert step.cost_usd == 0.001


class TestRecordContext:
    def test_record_returns_emit_function(self):
        with record() as emit:
            assert callable(emit)
            emit(tool="search", args={"q": "a"})
        # context manager does not auto-build a trajectory; recorder via capture

    def test_record_emits_order(self):
        rec = StepRecorder()
        with record(recorder=rec) as emit:
            emit(tool="search", args={"q": "a"})
            emit(tool="calc", args={"expr": "1"}, result="1")
        assert len(rec.trajectory().steps) == 2


class TestImporters:
    def test_load_trajectory_json_own_format(self, tmp_path):
        payload = {
            "steps": [
                {"index": 0, "tool": "search", "args": {"q": "hk"}, "result": "islands"},
            ],
            "final_answer": "3 islands",
        }
        p = tmp_path / "traj.json"
        p.write_text(json.dumps(payload))
        t = load_trajectory_json(str(p))
        assert t.steps[0].tool == "search"
        assert t.final_answer == "3 islands"

    def test_load_trajectory_json_otel_style(self, tmp_path):
        """OTel-style span JSON: name/attributes instead of tool/args."""
        payload = {
            "spans": [
                {"name": "tool.search", "attributes": {"args": {"q": "hk"}},
                 "result": "islands", "status": "ok"},
            ],
            "final_answer": "3 islands",
        }
        p = tmp_path / "otel.json"
        p.write_text(json.dumps(payload))
        t = load_trajectory_json(str(p))
        assert t.steps[0].tool == "search"
        assert t.steps[0].args == {"q": "hk"}
        assert t.final_answer == "3 islands"

    def test_load_trajectory_json_minimal_jsonl(self, tmp_path):
        lines = [
            {"tool": "search", "args": {"q": "a"}, "result": "1"},
            {"tool": "calc", "args": {"expr": "1+1"}, "result": "2"},
        ]
        p = tmp_path / "traj.jsonl"
        p.write_text("\n".join(json.dumps(l) for l in lines))
        t = load_trajectory_json(str(p))
        assert len(t.steps) == 2
        assert t.steps[0].tool == "search"
        assert t.steps[1].tool == "calc"

    def test_bad_order_raises_friendly_error(self, tmp_path):
        payload = {
            "steps": [
                {"index": 1, "tool": "search"},
                {"index": 0, "tool": "calc"},
            ],
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="index"):
            load_trajectory_json(str(p))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_trajectory_json(str(tmp_path / "nope.json"))

    def test_load_trajectories_file_map(self, tmp_path):
        payload = {
            "t1": {"steps": [{"index": 0, "tool": "search"}], "final_answer": "a"},
            "t2": {"steps": [{"index": 0, "tool": "calc", "result": "2"}], "final_answer": "2"},
        }
        p = tmp_path / "runs.json"
        p.write_text(json.dumps(payload))
        m = load_trajectories_file(str(p))
        assert set(m.keys()) == {"t1", "t2"}
        assert m["t2"].steps[0].tool == "calc"

    def test_load_trajectories_file_missing_keys(self, tmp_path):
        p = tmp_path / "runs.json"
        p.write_text(json.dumps({"t1": {"steps": []}}))
        with pytest.raises(ValueError, match="t1|t2"):
            load_trajectories_file(str(p), required_ids=["t1", "t2"])


class TestAdapters:
    def test_wrap_openai_shim_emits_llm_step(self):
        """The shim wraps a fake client; every completion call emits an llm step."""
        from evalforge.trajectory.adapters import wrap_openai

        class FakeResponse:
            def __init__(self, text):
                self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]
                self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

        class FakeClient:
            def __init__(self):
                self.calls = 0
            def chat_completions_create(self, **kwargs):
                self.calls += 1
                return FakeResponse("hello")

        rec = StepRecorder()
        inner = FakeClient()
        wrapped = wrap_openai(inner, recorder=rec)
        out = wrapped.chat_completions_create(model="deepseek", messages=[{"role": "user", "content": "hi"}])
        assert out.choices[0].message.content == "hello"
        assert inner.calls == 1
        steps = rec.trajectory().steps
        assert len(steps) == 1
        assert steps[0].tool == "llm"
        assert steps[0].result == "hello"
        assert steps[0].tokens.total == 15

    def test_wrap_openai_without_recorder_raises(self):
        from evalforge.trajectory.adapters import wrap_openai
        with pytest.raises(ValueError, match="recorder"):
            wrap_openai(object())

    def test_framework_callback_registration_requires_recorder(self):
        from evalforge.trajectory.adapters import (
            register_langchain_callback, register_langgraph_callback,
        )
        with pytest.raises(ValueError, match="recorder"):
            register_langchain_callback(None)
        with pytest.raises(ValueError, match="recorder"):
            register_langgraph_callback(None)
