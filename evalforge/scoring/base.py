"""Abstract base for all scoring strategies."""

from abc import ABC, abstractmethod

from evalforge.models.result import ScoreResult
from evalforge.models.suite import Expected


class Scorer(ABC):
    """Abstract base for all scoring strategies.

    Each scorer evaluates a target LLM response against a test case's
    expected output.
    """

    @abstractmethod
    async def score(self, response: str, expected: Expected) -> ScoreResult:
        """Evaluate a response against expected criteria.

        Args:
            response: Raw text output from target LLM.
            expected: Expected model from the test case.

        Returns:
            ScoreResult with overall score 0.0–1.0.
        """
        ...


class ScoringError(Exception):
    """Non-retryable scoring failure."""
    pass
