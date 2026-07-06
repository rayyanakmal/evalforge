"""
US-2: LLM-as-Judge Scoring — Judge Client Tests

Tests for:
  - LLMClient ABC (abstract, cannot instantiate)
  - create_client factory
  - DeepSeekClient construction and generate()
  - Judge prompts: build_rubric_prompt, STRICT_RETRY_PROMPT
  - max_tokens default >= 700 (spike constraint)
"""

import json
import pytest

from evalforge.models.llm import Message, LLMResponse
from evalforge.models.suite import RubricDimension
from evalforge.judge.client import (
    LLMClient,
    DeepSeekClient,
    create_client,
    LLMTimeoutError,
    LLMAuthError,
    LLMError,
)
from evalforge.judge.prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_rubric_prompt,
    STRICT_RETRY_PROMPT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dimensions() -> list[RubricDimension]:
    return [
        RubricDimension(name="accuracy", description="Factual correctness", weight=0.5),
        RubricDimension(name="completeness", description="How complete the answer is", weight=0.3),
        RubricDimension(name="tone", description="Professional and helpful tone", weight=0.2),
    ]


@pytest.fixture
def sample_messages() -> list[Message]:
    return [
        Message(role="system", content="You are a judge."),
        Message(role="user", content="Score this response."),
    ]


# ---------------------------------------------------------------------------
# LLMClient ABC tests
# ---------------------------------------------------------------------------

def test_llm_client_is_abstract():
    """LLMClient cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]


def test_deepseek_client_is_concrete():
    """DeepSeekClient can be instantiated."""
    client = DeepSeekClient(api_key="test-key", model="deepseek-v4-flash")
    assert client.provider_name == "deepseek"
    assert client.model == "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# create_client factory
# ---------------------------------------------------------------------------

def test_create_client_deepseek():
    """create_client with 'deepseek' returns DeepSeekClient."""
    client = create_client("deepseek", "deepseek-v4-flash", "test-key")
    assert isinstance(client, DeepSeekClient)
    assert client.provider_name == "deepseek"


def test_create_client_unknown_provider():
    """create_client with unknown provider raises ValueError."""
    with pytest.raises(ValueError, match="Unknown provider"):
        create_client("unknown-provider", "model-x", "key")


# ---------------------------------------------------------------------------
# DeepSeekClient.generate() — mock tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deepseek_generate_returns_llm_response(sample_messages):
    """DeepSeekClient.generate() returns LLMResponse with content and usage."""
    import httpx
    from unittest.mock import AsyncMock, patch

    mock_response = httpx.Response(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "content": '{"accuracy": 5, "completeness": 4, "tone": 3, "overall": 4, "reasoning": "Good answer"}',
                        "reasoning_content": "Let me evaluate...",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        },
    )

    async def mock_post(*args, **kwargs):
        return mock_response

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        client = DeepSeekClient(api_key="test-key", model="deepseek-v4-flash")
        result = await client.generate(sample_messages)

    assert isinstance(result, LLMResponse)
    assert "accuracy" in result.content
    assert result.reasoning_content == "Let me evaluate..."
    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 150
    assert result.latency_ms > 0
    assert result.cost_usd > 0


@pytest.mark.asyncio
async def test_deepseek_generate_default_max_tokens_is_700(sample_messages):
    """DeepSeekClient.generate() defaults max_tokens to 700 (spike constraint)."""
    import httpx
    from unittest.mock import AsyncMock, patch

    captured_payload = {}

    async def mock_post(url, **kwargs):
        captured_payload["json"] = kwargs.get("json", {})
        return httpx.Response(
            status_code=200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        client = DeepSeekClient(api_key="test-key", model="deepseek-v4-flash")
        # Don't pass max_tokens — should default to 700
        await client.generate(sample_messages)

    assert captured_payload["json"]["max_tokens"] == 700


@pytest.mark.asyncio
async def test_deepseek_generate_custom_max_tokens(sample_messages):
    """DeepSeekClient.generate() accepts custom max_tokens."""
    import httpx
    from unittest.mock import AsyncMock, patch

    captured_payload = {}

    async def mock_post(url, **kwargs):
        captured_payload["json"] = kwargs.get("json", {})
        return httpx.Response(
            status_code=200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        client = DeepSeekClient(api_key="test-key", model="deepseek-v4-flash")
        await client.generate(sample_messages, max_tokens=300)

    assert captured_payload["json"]["max_tokens"] == 300


@pytest.mark.asyncio
async def test_deepseek_generate_http_error_raises_llm_error(sample_messages):
    """DeepSeekClient raises LLMError on HTTP error (non-401/403)."""
    import httpx
    from unittest.mock import patch

    async def mock_post(*args, **kwargs):
        return httpx.Response(status_code=500, json={"error": "Internal server error"})

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        client = DeepSeekClient(api_key="test-key", model="deepseek-v4-flash")
        with pytest.raises(LLMError, match="500"):
            await client.generate(sample_messages)


@pytest.mark.asyncio
async def test_deepseek_generate_401_raises_auth_error(sample_messages):
    """DeepSeekClient raises LLMAuthError on 401."""
    import httpx
    from unittest.mock import patch

    async def mock_post(*args, **kwargs):
        return httpx.Response(status_code=401, json={"error": "Unauthorized"})

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        client = DeepSeekClient(api_key="test-key", model="deepseek-v4-flash")
        with pytest.raises(LLMAuthError):
            await client.generate(sample_messages)


@pytest.mark.asyncio
async def test_deepseek_generate_timeout_raises_timeout_error(sample_messages):
    """DeepSeekClient raises LLMTimeoutError on timeout."""
    import httpx
    from unittest.mock import patch

    async def mock_post(*args, **kwargs):
        raise httpx.TimeoutException("Request timed out")

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        client = DeepSeekClient(api_key="test-key", model="deepseek-v4-flash")
        with pytest.raises(LLMTimeoutError):
            await client.generate(sample_messages)


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

def test_judge_system_prompt_includes_rubric_instructions():
    """JUDGE_SYSTEM_PROMPT contains rubric evaluation instructions."""
    assert "expert evaluation judge" in JUDGE_SYSTEM_PROMPT.lower()
    assert "json" in JUDGE_SYSTEM_PROMPT.lower()
    assert "reasoning" in JUDGE_SYSTEM_PROMPT.lower()


def test_build_rubric_prompt_includes_dimensions(sample_dimensions):
    """build_rubric_prompt includes all dimension names and descriptions."""
    prompt = build_rubric_prompt(sample_dimensions)

    assert "accuracy" in prompt
    assert "Factual correctness" in prompt
    assert "completeness" in prompt
    assert "How complete the answer is" in prompt
    assert "tone" in prompt
    assert "Professional and helpful tone" in prompt


def test_build_rubric_prompt_includes_score_range():
    """build_rubric_prompt instructs scores 1-5."""
    prompt = build_rubric_prompt([
        RubricDimension(name="accuracy", description="Correctness", weight=1.0),
    ])
    assert "1-5" in prompt or "1 to 5" in prompt


def test_build_rubric_prompt_demands_json_output():
    """build_rubric_prompt instructs JSON-only output."""
    prompt = build_rubric_prompt([
        RubricDimension(name="accuracy", description="Correctness", weight=1.0),
    ])
    assert "json" in prompt.lower()
    assert "{" in prompt  # Should contain expected JSON format


def test_strict_retry_prompt_is_stricter():
    """STRICT_RETRY_PROMPT has stricter JSON instructions."""
    assert "IMPORTANT" in STRICT_RETRY_PROMPT.upper() or "important" in STRICT_RETRY_PROMPT.lower()
    assert "json" in STRICT_RETRY_PROMPT.lower()
    assert "only" in STRICT_RETRY_PROMPT.lower()  # "ONLY JSON" constraint


def test_build_rubric_prompt_empty_dimensions():
    """build_rubric_prompt with empty dimensions should still return valid prompt."""
    prompt = build_rubric_prompt([])
    assert len(prompt) > 0
    assert "json" in prompt.lower()
