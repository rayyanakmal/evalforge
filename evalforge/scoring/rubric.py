"""Rubric-based LLM-as-Judge scoring.

Evaluates target LLM responses by asking a judge LLM to score against
a rubric with multiple dimensions (accuracy, completeness, tone, etc.).
"""

import json
import logging
import warnings
from typing import Optional

from evalforge.models.result import ScoreResult, DimensionScore
from evalforge.models.suite import Expected, RubricDimension
from evalforge.models.llm import Message
from evalforge.scoring.base import Scorer, ScoringError
from evalforge.judge.client import LLMClient
from evalforge.judge.prompts import build_rubric_prompt, STRICT_RETRY_PROMPT

logger = logging.getLogger(__name__)


class RubricScorer(Scorer):
    """Scores responses using an LLM judge against a defined rubric.

    Calls a judge LLM (e.g., DeepSeek) with a system prompt describing
    the rubric dimensions. The judge returns JSON with per-dimension
    scores (1-5), an overall score, and reasoning.

    Edge cases handled:
      - Empty/whitespace response → auto-scores 1 for all dimensions
      - Invalid JSON from judge → retry once with stricter prompt
      - Dimension name mismatch → warns but still returns result
    """

    MIN_MAX_TOKENS = 700  # Spike-validated minimum for reliable JSON output

    def __init__(
        self,
        judge_client: LLMClient,
        max_tokens: int = 700,
    ):
        """Initialize the rubric scorer.

        Args:
            judge_client: LLM client for the judge model.
            max_tokens: Max tokens for judge output. Must be >= 700 per spike
                        findings for reliable JSON parsing.
        """
        self.judge_client = judge_client
        self.max_tokens = max_tokens

        if max_tokens < self.MIN_MAX_TOKENS:
            warnings.warn(
                f"max_tokens={max_tokens} is below the spike-validated minimum "
                f"of {self.MIN_MAX_TOKENS}. Judge JSON parsing may be unreliable.",
                UserWarning,
            )

    async def score(self, response: str, expected: Expected) -> ScoreResult:
        """Score a response against the rubric in expected.

        Args:
            response: The target LLM's response to evaluate.
            expected: Expected model containing rubric dimensions.

        Returns:
            ScoreResult with overall score 0.0–1.0 and per-dimension scores.

        Raises:
            ScoringError: If judge fails irrecoverably (invalid JSON after retry).
        """
        # Edge case: empty response
        if not response or not response.strip():
            return self._empty_response_result(expected)

        dimensions = expected.rubric or []

        if not dimensions:
            # No rubric defined — can't score
            return ScoreResult(
                overall=0.0,
                dimensions=[],
                method="rubric",
            )

        # Build judge prompt
        system_prompt = build_rubric_prompt(dimensions)
        user_prompt = f"Response to evaluate:\n{response}"

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        # First attempt: call judge
        parsed = None
        for attempt in range(2):
            try:
                judge_response = await self.judge_client.generate(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.1,
                )
                parsed = self._parse_judge_json(judge_response.content)

                if parsed is not None:
                    break

                # JSON parse failed — retry with stricter prompt
                logger.warning(
                    "Judge returned invalid JSON on attempt %d. Retrying with stricter prompt.",
                    attempt + 1,
                )
                messages[0] = Message(
                    role="system",
                    content=STRICT_RETRY_PROMPT + "\n\n" + system_prompt,
                )

            except Exception as e:
                if attempt == 0:
                    logger.warning("Judge call failed: %s. Retrying.", e)
                    messages[0] = Message(
                        role="system",
                        content=STRICT_RETRY_PROMPT + "\n\n" + system_prompt,
                    )
                else:
                    raise ScoringError(
                        f"Judge failed after retry: {e}"
                    ) from e

        if parsed is None:
            raise ScoringError(
                "Judge returned invalid JSON on both attempts. "
                "Cannot score this response."
            )

        # Validate dimensions and build result
        return self._build_result(parsed, dimensions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_judge_json(content: str) -> Optional[dict]:
        """Try to extract and parse JSON from judge response.

        Handles: raw JSON, ```json blocks, { ... } extraction.
        """
        if not content or not content.strip():
            return None

        text = content.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try ```json ... ``` block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # Try ``` ... ``` block
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # Try { ... } extraction
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _build_result(
        self, parsed: dict, dimensions: list[RubricDimension]
    ) -> ScoreResult:
        """Build a ScoreResult from parsed judge JSON.

        Validates dimension names match, warns on mismatch.
        Computes overall as weighted average normalized to 0-1.
        """
        # Build dimension name → (weight, description) map
        dim_map = {d.name: d for d in dimensions}

        dimension_scores = []
        total_weighted = 0.0
        total_weight = 0.0

        seen_names = set()

        # Extract scores for expected dimensions
        for dim in dimensions:
            if dim.name in parsed:
                raw_score = parsed[dim.name]
                score = self._clamp_score(raw_score)
                seen_names.add(dim.name)
            else:
                # Missing dimension — default to 1
                score = 1
                warnings.warn(
                    f"Judge response missing dimension '{dim.name}'. Defaulting to score=1.",
                    UserWarning,
                )

            # Reasoning: use per-dimension reasoning if available, else overall
            reasoning_key = f"{dim.name}_reasoning"
            reasoning = parsed.get(reasoning_key, parsed.get("reasoning", ""))

            dimension_scores.append(
                DimensionScore(
                    name=dim.name,
                    score=score,
                    reasoning=reasoning,
                )
            )

            total_weighted += score * dim.weight
            total_weight += dim.weight

        # Check for unexpected dimensions in judge output
        extra_dims = set(parsed.keys()) - seen_names - {"overall", "reasoning"}
        # Also filter out _reasoning keys
        extra_dims = {k for k in extra_dims if not k.endswith("_reasoning")}

        if extra_dims:
            warnings.warn(
                f"Judge returned unexpected dimensions not in rubric: {extra_dims}. "
                f"Expected: {list(dim_map.keys())}.",
                UserWarning,
            )

        # Compute overall: weighted average normalized to 0-1
        if total_weight > 0:
            overall = (total_weighted / total_weight) / 5.0
        else:
            overall = 0.0

        return ScoreResult(
            overall=overall,
            dimensions=dimension_scores,
            method="rubric",
        )

    @staticmethod
    def _clamp_score(raw: object) -> int:
        """Clamp a raw score value to the 1-5 range."""
        try:
            score = int(float(str(raw)))  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return 1
        return max(1, min(5, score))

    def _empty_response_result(self, expected: Expected) -> ScoreResult:
        """Build a result for an empty/whitespace response.

        Per spec: scores 1 for all dimensions with 'no response provided'.
        """
        dimensions = expected.rubric or []

        if not dimensions:
            return ScoreResult(overall=0.0, dimensions=[], method="rubric")

        dim_scores = [
            DimensionScore(
                name=d.name,
                score=1,
                reasoning="No response provided.",
            )
            for d in dimensions
        ]

        return ScoreResult(
            overall=0.0,
            dimensions=dim_scores,
            method="rubric",
        )
