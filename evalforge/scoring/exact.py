"""Exact string match scorer."""

from evalforge.models.result import ScoreResult
from evalforge.models.suite import Expected
from evalforge.scoring.base import Scorer


class ExactScorer(Scorer):
    """Scores by exact string match (case-insensitive, whitespace-normalized)."""

    async def score(self, response: str, expected: Expected) -> ScoreResult:
        """Compare response to expected.value with normalization."""
        actual = response.strip().lower()
        want = (expected.value or "").strip().lower()

        passed = actual == want

        return ScoreResult(
            overall=1.0 if passed else 0.0,
            method="exact",
        )
