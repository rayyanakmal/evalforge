"""Real-execution runner for `verdictlab run`.

Wires the Executor to a configured LLM client and captures trajectories
during the run, so a single `verdictlab run` produces a RunResult with
BOTH answer-level data AND the process journey (v0.2.x: one file, both
dimensions). Previously the CLI only supported --no-llm dry runs.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Callable, Optional

from verdictlab.config import GateConfig, ProviderConfig
from verdictlab.judge.client import LLMClient, create_client
from verdictlab.models.llm import Message
from verdictlab.models.result import RunResult
from verdictlab.models.suite import TestSuite
from verdictlab.scoring.registry import (
    RegistryScorer,
    ScorerRegistry,
    create_default_registry,
)

logger = logging.getLogger(__name__)


def resolve_api_key(provider: str) -> Optional[str]:
    """Find an API key for a provider from the environment.

    Checks the conventional env var (DEEPSEEK_API_KEY, OPENAI_API_KEY,
    ANTHROPIC_API_KEY), then a ~/.hermes/.env file (Hermes convention).
    """
    env_var = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider.lower())

    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val.strip()

    # Hermes-style .env fallback
    env_file = Path.home() / ".hermes" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            if env_var and key.strip() == env_var:
                return value.strip()

    return None


def _provider_from_config(config: GateConfig, role: str) -> ProviderConfig:
    """Pick a provider config by role ('target' or 'judge')."""
    if role == "judge" and config.judge:
        return config.judge
    if config.target:
        return config.target
    if config.judge:
        return config.judge
    raise ValueError(
        f"No {role} provider configured. Add a 'target' (and optionally "
        "'judge') section to verdictlab.yaml, or use --provider/--model."
    )


def build_clients(
    config: Optional[GateConfig],
    provider_override: Optional[str],
    model_override: Optional[str],
) -> tuple[LLMClient, LLMClient]:
    """Build (target_client, judge_client) from config + CLI overrides.

    The target is the agent under test; the judge scores rubric tests.
    If no judge is configured, the target client is reused (cheap and
    sensible for the common single-provider case).
    """
    target_cfg = _provider_from_config(config, "target") if config else None
    judge_cfg = (config.judge or config.target) if config else None

    provider = provider_override or (target_cfg.provider if target_cfg else "deepseek")
    model = model_override or (target_cfg.model if target_cfg else "deepseek-chat")

    api_key = resolve_api_key(provider)
    if not api_key:
        raise ValueError(
            f"No API key found for provider '{provider}'. "
            f"Set {provider.upper()}_API_KEY in the environment or ~/.hermes/.env."
        )

    target_client = create_client(provider, model, api_key)
    judge_client = None
    if judge_cfg:
        judge_provider = judge_cfg.provider or provider
        judge_model = judge_cfg.model or model
        judge_key = resolve_api_key(judge_provider) or api_key
        if judge_provider == provider and judge_model == model:
            judge_client = target_client
        else:
            judge_client = create_client(judge_provider, judge_model, judge_key)
    else:
        judge_client = target_client

    return target_client, judge_client


def build_scorer(judge_client: LLMClient) -> RegistryScorer:
    """Build a per-test-type dispatcher scorer (exact + rubric)."""
    registry: ScorerRegistry = create_default_registry(
        rubric_judge_client=judge_client
    )
    return RegistryScorer(registry)


def build_generate_fn(client: LLMClient):
    """Build the Executor generate_fn bound to an LLM client.

    In capture mode the function receives (prompt, recorder) and emits an
    ``llm`` step per API call with tokens/cost. Single-call eval: one llm
    step per test — the journey is the LLM call itself. Custom agent loops
    can emit more steps (tools) before returning the final response.
    """

    async def generate(prompt: str, recorder=None):
        messages = [Message(role="user", content=prompt)]
        resp = await client.generate(messages, max_tokens=700, temperature=0.1)
        if recorder is not None:
            usage = resp.usage
            recorder.emit(
                tool="llm",
                args={"model": client.model, "messages": [m.model_dump() for m in messages]},
                result=resp.content,
                latency_ms=resp.latency_ms,
                tokens_in=usage.prompt_tokens if usage else None,
                tokens_out=usage.completion_tokens if usage else None,
                cost_usd=resp.cost_usd,
            )
        return resp

    return generate


def run_suite(
    suite: TestSuite,
    config: Optional[GateConfig],
    provider: Optional[str],
    model: Optional[str],
    concurrency: int = 10,
    generate_fn: Optional[Callable] = None,
) -> RunResult:
    """Execute a suite against a configured provider, capturing trajectories.

    Returns a RunResult whose tests carry trajectories and whose
    trajectory_summary is populated — so the same JSON report feeds both
    `verdictlab compare --trajectory` and the dashboard trajectory view.

    generate_fn: optional custom executor generate function. When provided
    (e.g. a tool-using agent), it replaces the default single-call builder
    and receives ``(prompt, recorder)`` in capture mode — same contract,
    richer journeys.
    """
    from verdictlab.runner.executor import Executor
    from verdictlab.trajectory.metrics import summarize_trajectories

    target_client, judge_client = build_clients(config, provider, model)
    scorer = build_scorer(judge_client)
    if generate_fn is None:
        generate_fn = build_generate_fn(target_client)

    executor = Executor(
        generate_fn=generate_fn,
        scorer=scorer,
        concurrency=concurrency,
        capture_trajectories=True,
    )

    result = asyncio.run(executor.run(suite))
    # Attach the aggregate process report so the JSON carries both dimensions.
    result.trajectory_summary = summarize_trajectories(result.tests)
    return result
