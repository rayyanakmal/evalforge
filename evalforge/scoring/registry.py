"""Scorer registry — maps expected.type strings to Scorer instances.

Provides a factory pattern for selecting the right scorer based on
a test case's expected.type field.

Pre-registered scorers: exact, rubric
Future: semantic, function
"""

from evalforge.scoring.base import Scorer
from evalforge.scoring.exact import ExactScorer
from evalforge.scoring.rubric import RubricScorer


class ScorerRegistry:
    """Registry mapping scorer names to Scorer instances.

    Usage:
        registry = ScorerRegistry()
        registry.register("rubric", RubricScorer(judge_client))
        scorer = registry.get("rubric")
        result = await scorer.score(response, expected)
    """

    def __init__(self):
        self._scorers: dict[str, Scorer] = {}

    def register(self, name: str, scorer: Scorer) -> None:
        """Register a scorer instance under a name.

        Args:
            name: The scorer name (matches expected.type values).
            scorer: A Scorer instance.
        """
        self._scorers[name] = scorer

    def get(self, name: str) -> Scorer:
        """Retrieve a scorer by name.

        Args:
            name: The scorer name to look up.

        Returns:
            The registered Scorer instance.

        Raises:
            KeyError: If no scorer is registered under the given name.
        """
        if name not in self._scorers:
            raise KeyError(
                f"No scorer registered for '{name}'. "
                f"Available: {list(self._scorers.keys())}"
            )
        return self._scorers[name]

    def list_scorers(self) -> list[str]:
        """Return list of registered scorer names."""
        return list(self._scorers.keys())


def create_default_registry(
    rubric_judge_client=None,
) -> ScorerRegistry:
    """Create a ScorerRegistry pre-loaded with the default scorers.

    Args:
        rubric_judge_client: LLMClient for the RubricScorer.
                             If None, RubricScorer is not registered.

    Returns:
        ScorerRegistry with exact and optionally rubric scorer registered.
    """
    registry = ScorerRegistry()
    registry.register("exact", ExactScorer())

    if rubric_judge_client is not None:
        registry.register("rubric", RubricScorer(judge_client=rubric_judge_client))

    return registry
