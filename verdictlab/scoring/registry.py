"""Scorer registry — maps expected.type strings to Scorer instances.

Provides a factory pattern for selecting the right scorer based on
a test case's expected.type field.

Pre-registered scorers: exact, rubric
Future: semantic, function
"""

from verdictlab.scoring.base import Scorer, ScoringError
from verdictlab.scoring.exact import ExactScorer
from verdictlab.scoring.rubric import RubricScorer
from verdictlab.models.result import ScoreResult
from verdictlab.models.suite import Expected


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


class RegistryScorer(Scorer):
    """Delegates per-test scoring to the scorer matching expected.type.

    Lets a single Executor run a mixed suite (exact + rubric + …) by
    dispatching each response to the registered scorer for its expected
    type — the runner gets one scorer, suites keep per-test types.
    """

    def __init__(self, registry: ScorerRegistry):
        self.registry = registry

    async def score(self, response: str, expected: Expected) -> ScoreResult:
        try:
            scorer = self.registry.get(expected.type)
        except KeyError as e:
            raise ScoringError(
                f"No scorer registered for expected.type={expected.type!r}. "
                f"Available: {sorted(self.registry.list_scorers())}"
            ) from e
        return await scorer.score(response, expected)
