"""
US-2: LLM-as-Judge Scoring Tests

Tests for all acceptance criteria and edge cases:
  AC-2.1: Open-ended question scored against rubric by judge LLM
  AC-2.2: Multi-dimension scores 1-5 + overall score
  AC-2.3: Judge reasoning text alongside scores
  Edge:  Invalid JSON → retry with stricter prompt, then judge_error
  Edge:  Dimension mismatch → detects mismatch, raises warning
  Edge:  Empty response → score 1 with "no response provided"
"""

import json
import warnings
import pytest

from evalforge.models.suite import Expected, RubricDimension
from evalforge.models.result import ScoreResult, DimensionScore
from evalforge.models.llm import LLMResponse, Usage
from evalforge.scoring.base import ScoringError
from evalforge.scoring.rubric import RubricScorer
from evalforge.judge.client import LLMClient, LLMError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rubric_expected() -> Expected:
    return Expected(
        type="rubric",
        rubric=[
            RubricDimension(name="accuracy", description="Factual correctness of the answer", weight=0.5),
            RubricDimension(name="completeness", description="How completely the question is answered", weight=0.3),
            RubricDimension(name="tone", description="Professional and helpful tone", weight=0.2),
        ],
    )


@pytest.fixture
def good_judge_response() -> dict:
    """A well-formed judge JSON response matching the rubric dimensions."""
    return {
        "accuracy": 5,
        "completeness": 4,
        "tone": 3,
        "overall": 4,
        "reasoning": "The answer is factually correct and mostly complete. "
                      "Tone could be slightly more professional but is acceptable.",
    }


# ---------------------------------------------------------------------------
# Mock LLM Client for judge tests
# ---------------------------------------------------------------------------

class MockJudgeClient(LLMClient):
    """Mock LLM client that returns a pre-programmed response."""

    def __init__(self, responses: list[dict], provider_name: str = "mock"):
        self._responses = responses
        self._call_count = 0
        self._provider_name = provider_name
        self.last_messages: list = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def call_count(self) -> int:
        return self._call_count

    async def generate(self, messages, max_tokens=700, temperature=0.1) -> LLMResponse:
        self._call_count += 1
        self.last_messages = messages
        idx = min(self._call_count - 1, len(self._responses) - 1)
        resp = self._responses[idx]
        return LLMResponse(
            content=json.dumps(resp),
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            latency_ms=200.0,
            cost_usd=0.0001,
        )


# ---------------------------------------------------------------------------
# AC-2.1: Open-ended question scored against rubric
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac2_1_rubric_scores_open_ended_question(rubric_expected, good_judge_response):
    """AC-2.1: Given a test case with an open-ended question (no exact answer),
    when scored, then a judge LLM evaluates the response against a rubric
    defined in the test case."""
    mock_client = MockJudgeClient(responses=[good_judge_response])
    scorer = RubricScorer(judge_client=mock_client)

    response = "Paris is the capital of France. It is a major European city known for the Eiffel Tower."
    result = await scorer.score(response, rubric_expected)

    # Should have called the judge LLM
    assert mock_client.call_count >= 1

    # Should return a valid ScoreResult
    assert isinstance(result, ScoreResult)
    assert result.method == "rubric"
    assert 0.0 <= result.overall <= 1.0
    assert result.dimensions is not None
    assert len(result.dimensions) > 0


# ---------------------------------------------------------------------------
# AC-2.2: Multi-dimension scores 1-5 + overall score
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac2_2_multi_dimension_scores(rubric_expected, good_judge_response):
    """AC-2.2: Given a rubric with multiple dimensions (accuracy, completeness, tone),
    when scored, then the judge returns per-dimension scores 1-5 plus an overall score."""
    mock_client = MockJudgeClient(responses=[good_judge_response])
    scorer = RubricScorer(judge_client=mock_client)

    response = "Paris is the capital of France."
    result = await scorer.score(response, rubric_expected)

    # All three dimensions should have scores
    assert result.dimensions is not None
    dim_names = {d.name for d in result.dimensions}
    assert "accuracy" in dim_names
    assert "completeness" in dim_names
    assert "tone" in dim_names

    # Each dimension score should be 1-5
    for dim in result.dimensions:
        assert 1 <= dim.score <= 5, f"Dimension {dim.name} score {dim.score} not in 1-5"

    # Overall should be between 0 and 1 (normalized)
    assert 0.0 <= result.overall <= 1.0

    # Overall should be the weighted average of dimension scores normalized to 0-1
    # accuracy=5 (weight 0.5), completeness=4 (weight 0.3), tone=3 (weight 0.2)
    # weighted avg = 5*0.5 + 4*0.3 + 3*0.2 = 2.5 + 1.2 + 0.6 = 4.3
    # normalized = 4.3 / 5 = 0.86
    expected_overall = (5 * 0.5 + 4 * 0.3 + 3 * 0.2) / 5.0
    assert abs(result.overall - expected_overall) < 0.01, \
        f"Expected overall ~{expected_overall}, got {result.overall}"


# ---------------------------------------------------------------------------
# AC-2.3: Judge reasoning text alongside scores
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac2_3_judge_reasoning_included(rubric_expected, good_judge_response):
    """AC-2.3: Given a judge evaluation, when examined,
    then the result includes the judge's reasoning text alongside the score."""
    mock_client = MockJudgeClient(responses=[good_judge_response])
    scorer = RubricScorer(judge_client=mock_client)

    response = "Paris is the capital of France."
    result = await scorer.score(response, rubric_expected)

    # Every dimension should have reasoning
    assert result.dimensions is not None
    for dim in result.dimensions:
        assert dim.reasoning, f"Dimension {dim.name} has no reasoning"
        assert len(dim.reasoning) > 0

    # The overall reasoning should be present (in at least one dimension or aggregated)
    all_reasoning = " ".join(d.reasoning for d in result.dimensions)
    assert len(all_reasoning) > 10, "Reasoning text is too short"


# ---------------------------------------------------------------------------
# Edge Case: Invalid JSON → retry with stricter prompt → judge_error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edge_invalid_json_retries_then_errors(rubric_expected):
    """Edge: Judge LLM returns invalid JSON → retries with stricter prompt,
    then marks as judge_error if still invalid."""
    # First response: invalid JSON (not JSON at all)
    # Second response: also invalid (still not parseable)
    bad_responses = [
        "This is not JSON at all, just plain text.",
        "Still not valid {broken json",
    ]

    class InvalidJSONClient(MockJudgeClient):
        async def generate(self, messages, max_tokens=700, temperature=0.1) -> LLMResponse:
            self._call_count += 1
            self.last_messages = messages
            idx = min(self._call_count - 1, len(self._responses) - 1)
            return LLMResponse(
                content=self._responses[idx],  # raw string, not JSON-encoded
                usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
                latency_ms=100.0,
                cost_usd=0.0001,
            )

    mock_client = InvalidJSONClient(responses=bad_responses)
    scorer = RubricScorer(judge_client=mock_client)

    response = "Paris is the capital of France."

    with pytest.raises(ScoringError, match=r"(?i)judge"):
        await scorer.score(response, rubric_expected)

    # Should have tried twice (original + retry)
    assert mock_client.call_count == 2

    # Second call should have stricter prompt
    second_messages = mock_client.last_messages
    assert second_messages is not None


@pytest.mark.asyncio
async def test_edge_invalid_json_succeeds_on_retry(rubric_expected):
    """Edge: Judge LLM returns invalid JSON first, then valid on retry."""
    bad_then_good = [
        "This is not JSON",  # first try fails
        {  # second try succeeds (will be JSON-encoded by MockJudgeClient)
            "accuracy": 5,
            "completeness": 4,
            "tone": 3,
            "overall": 4,
            "reasoning": "Good on retry",
        },
    ]

    class RetryClient(MockJudgeClient):
        async def generate(self, messages, max_tokens=700, temperature=0.1) -> LLMResponse:
            self._call_count += 1
            self.last_messages = messages
            idx = min(self._call_count - 1, len(self._responses) - 1)
            raw = self._responses[idx]
            content = raw if isinstance(raw, str) else json.dumps(raw)
            return LLMResponse(
                content=content,
                usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                latency_ms=200.0,
                cost_usd=0.0001,
            )

    mock_client = RetryClient(responses=bad_then_good)
    scorer = RubricScorer(judge_client=mock_client)

    response = "Paris is the capital of France."
    result = await scorer.score(response, rubric_expected)

    # Should have called twice
    assert mock_client.call_count == 2

    # Result should be valid
    assert result.dimensions is not None
    assert len(result.dimensions) == 3
    assert result.overall > 0


# ---------------------------------------------------------------------------
# Edge Case: Dimension mismatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edge_dimension_mismatch_warns(rubric_expected):
    """Edge: Rubric dimension names don't match judge output → detects mismatch, raises warning."""
    mismatched_response = {
        "accuracy": 5,
        "completeness": 4,
        "professionalism": 3,  # "tone" expected, got "professionalism"
        "overall": 4,
        "reasoning": "Good answer.",
    }

    mock_client = MockJudgeClient(responses=[mismatched_response])
    scorer = RubricScorer(judge_client=mock_client)

    response = "Paris is the capital of France."

    with pytest.warns(UserWarning, match="mismatch|missing|unexpected"):
        result = await scorer.score(response, rubric_expected)

    # Should still return a result, even with mismatch
    assert result.dimensions is not None
    # The "tone" dimension should be present (possibly with default score)
    dim_names = {d.name for d in result.dimensions}
    assert "tone" in dim_names or "professionalism" in dim_names


# ---------------------------------------------------------------------------
# Edge Case: Empty response from target LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edge_empty_response_scores_1(rubric_expected, good_judge_response):
    """Edge: Empty response from target LLM → judge scores 1 with 'no response provided'."""
    # The scorer should handle this before calling the judge
    scorer = RubricScorer(judge_client=MockJudgeClient(responses=[good_judge_response]))

    # Empty response
    result = await scorer.score("", rubric_expected)

    assert result.overall == 0.0
    assert result.dimensions is not None
    for dim in result.dimensions:
        assert dim.score == 1
        assert "no response" in dim.reasoning.lower() or "empty" in dim.reasoning.lower()


@pytest.mark.asyncio
async def test_edge_whitespace_only_response_scores_1(rubric_expected, good_judge_response):
    """Edge: Whitespace-only response → treated as empty, scores 1."""
    scorer = RubricScorer(judge_client=MockJudgeClient(responses=[good_judge_response]))

    result = await scorer.score("   \n\t  ", rubric_expected)

    assert result.overall == 0.0
    assert result.dimensions is not None
    for dim in result.dimensions:
        assert dim.score == 1


# ---------------------------------------------------------------------------
# Additional: Scorer validates max_tokens constraint
# ---------------------------------------------------------------------------

def test_rubric_scorer_enforces_max_tokens_minimum():
    """RubricScorer defaults to max_tokens=700 (spike-validated minimum)."""
    scorer = RubricScorer(judge_client=MockJudgeClient(responses=[]))
    assert scorer.max_tokens >= 700, "max_tokens must be >= 700 per spike findings"


def test_rubric_scorer_rejects_low_max_tokens():
    """RubricScorer warns if max_tokens is set below 700."""
    with pytest.warns(UserWarning, match="700"):
        RubricScorer(
            judge_client=MockJudgeClient(responses=[]),
            max_tokens=200,
        )



