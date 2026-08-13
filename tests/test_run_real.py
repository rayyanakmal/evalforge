"""Tests for the real-execution runner (verdictlab.cli.run_real).

Covers the v0.2.x promise: `verdictlab run` executes against a configured
provider AND captures trajectories into the same report, so one file
carries both the answer-level data and the process journey.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from verdictlab.config import GateConfig, ProviderConfig
from verdictlab.models.llm import LLMResponse, Message, Usage
from verdictlab.models.suite import TestSuite, TestCase, Expected
from verdictlab.cli.run_real import (
    build_clients,
    build_generate_fn,
    build_scorer,
    resolve_api_key,
    run_suite,
)


class _FakeClient:
    """Minimal stand-in for an LLMClient."""

    provider_name = "deepseek"
    model = "deepseek-chat"

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    async def generate(self, messages, max_tokens=700, temperature=0.1):
        self.calls.append(messages)
        prompt = messages[-1].content if messages else ""
        content = self.responses.get(prompt, "Paris")
        return LLMResponse(
            content=content,
            usage=Usage(prompt_tokens=11, completion_tokens=4, total_tokens=15),
            latency_ms=120.0,
            cost_usd=0.0009,
        )


def _suite() -> TestSuite:
    return TestSuite(
        name="demo",
        tests=[
            TestCase(
                id="t1",
                prompt="What is the capital of France?",
                expected=Expected(type="exact", value="Paris"),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# build_generate_fn — trajectory capture in capture mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_fn_emits_llm_step_in_capture_mode():
    client = _FakeClient(responses={"What is the capital of France?": "Paris"})
    generate = build_generate_fn(client)
    recorder = MagicMock()

    resp = await generate("What is the capital of France?", recorder)

    assert resp.content == "Paris"
    recorder.emit.assert_called_once()
    _, kwargs = recorder.emit.call_args
    assert kwargs["tool"] == "llm"
    assert kwargs["result"] == "Paris"
    assert kwargs["cost_usd"] == 0.0009


@pytest.mark.asyncio
async def test_generate_fn_works_without_recorder():
    """v1 contract: plain prompt → response (no recorder passed)."""
    client = _FakeClient(responses={"Hello": "hi"})
    generate = build_generate_fn(client)

    resp = await generate("Hello")

    assert resp.content == "hi"
    assert len(client.calls) == 1
    assert isinstance(client.calls[0][0], Message)


# ---------------------------------------------------------------------------
# build_clients / resolve_api_key
# ---------------------------------------------------------------------------

def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
    assert resolve_api_key("deepseek") == "sk-test-123"


def test_resolve_api_key_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    # Point at a temp home with no .env
    with patch("verdictlab.cli.run_real.Path.home") as home:
        home.return_value = Path("/nonexistent-home")
        assert resolve_api_key("deepseek") is None


def test_build_clients_defaults_to_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    with patch("verdictlab.cli.run_real.create_client") as create:
        create.return_value = _FakeClient()
        target, judge = build_clients(None, None, None)
        assert target is judge  # same client reused when no judge configured


def test_build_clients_uses_config():
    config = GateConfig(
        target=ProviderConfig(provider="deepseek", model="deepseek-chat"),
        judge=ProviderConfig(provider="deepseek", model="deepseek-chat"),
    )
    with patch("verdictlab.cli.run_real.create_client") as create:
        create.return_value = _FakeClient()
        with patch("verdictlab.cli.run_real.resolve_api_key", return_value="sk-x"):
            target, judge = build_clients(config, None, None)
            create.assert_called_once()
            assert target is judge


# ---------------------------------------------------------------------------
# build_scorer — dispatcher covers exact + rubric
# ---------------------------------------------------------------------------

def test_build_scorer_registers_exact_and_rubric():
    client = _FakeClient()
    scorer = build_scorer(client)
    # RegistryScorer exposes .registry with exact + rubric registered
    assert "exact" in scorer.registry.list_scorers()
    assert "rubric" in scorer.registry.list_scorers()


# ---------------------------------------------------------------------------
# run_suite — single run produces both dimensions
# ---------------------------------------------------------------------------

def test_run_suite_returns_trajectory_summary(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    with patch("verdictlab.cli.run_real.build_clients") as build:
        client = _FakeClient(responses={"What is the capital of France?": "Paris"})
        build.return_value = (client, client)

        result = run_suite(_suite(), config=None, provider=None, model=None)

    assert len(result.tests) == 1
    t = result.tests[0]
    assert t.status == "pass"
    assert t.response == "Paris"
    # Both dimensions present in the single result:
    assert t.trajectory is not None
    assert len(t.trajectory.steps) == 1
    assert t.trajectory.steps[0].tool == "llm"
    assert t.trajectory.final_answer == "Paris"
    # Aggregate process report attached (dashboard trajectory view + compare --trajectory)
    assert result.trajectory_summary is not None
    assert result.trajectory_summary.get("total_tests") == 1 or "mean_steps" in result.trajectory_summary


def test_run_suite_json_roundtrip_preserves_trajectory(monkeypatch, tmp_path):
    """The saved JSON report must carry trajectories so compare --trajectory works."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    with patch("verdictlab.cli.run_real.build_clients") as build:
        client = _FakeClient(responses={"What is the capital of France?": "Paris"})
        build.return_value = (client, client)

        result = run_suite(_suite(), config=None, provider=None, model=None)

    import json
    payload = json.loads(result.model_dump_json())
    assert payload["tests"][0]["trajectory"] is not None
    assert payload["tests"][0]["trajectory"]["steps"][0]["tool"] == "llm"
    assert "trajectory_summary" in payload


def test_run_suite_accepts_custom_generate_fn(monkeypatch):
    """A custom generate_fn (e.g. a tool-using agent) replaces the default
    single-call builder; multi-step journeys are captured and summarized."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    async def generate(prompt, recorder=None):
        if recorder is not None:
            recorder.emit(tool="llm", args={}, result="Paris", thought="think")
            recorder.emit(tool="lookup", args={"query": "france"}, result={"fact": "Paris"})
        return LLMResponse(
            content="Paris",
            usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            latency_ms=10.0,
            cost_usd=0.0001,
        )

    with patch("verdictlab.cli.run_real.build_clients") as build:
        client = _FakeClient()
        build.return_value = (client, client)

        result = run_suite(
            _suite(), config=None, provider=None, model=None, generate_fn=generate
        )

    t = result.tests[0]
    assert t.status == "pass"
    assert t.response == "Paris"
    assert t.trajectory is not None
    assert [s.tool for s in t.trajectory.steps] == ["llm", "lookup"]
    assert result.trajectory_summary is not None
