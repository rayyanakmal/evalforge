"""RAGAS-style canonical metric presets for the verdictlab dashboard.

These are the industry-standard metric names from the RAGAS framework
(faithfulness, answer relevancy, context precision, context recall,
answer correctness). They map onto verdictlab's existing RubricDimension
model — each metric is just a rubric with one scored dimension plus
supporting context, evaluated by the LLM judge.

No RAGAS dependency: these are prompt templates, not imports.
"""

from verdictlab.models.suite import RubricDimension


# Metric name -> (description, rubric dimensions)
# Each metric is expressed as a single-dimension rubric so it plugs
# into verdictlab's existing RubricScorer unchanged.
RAGAS_METRICS: dict[str, dict] = {
    "Faithfulness": {
        "description": (
            "Is the answer supported by the retrieved context? "
            "Scores how many claims in the answer can be verified "
            "against the provided source documents."
        ),
        "rubric": [
            RubricDimension(
                name="faithfulness",
                description=(
                    "Score 1-5: 1 = answer contains claims entirely "
                    "unsupported by the context (hallucination); "
                    "5 = every claim is directly supported by the "
                    "retrieved context. Penalize invented sources, "
                    "numbers, or facts."
                ),
                weight=1.0,
            )
        ],
    },
    "Answer Relevancy": {
        "description": (
            "Does the answer actually address the question? "
            "Irrelevant or off-topic answers score low even if fluent."
        ),
        "rubric": [
            RubricDimension(
                name="answer_relevancy",
                description=(
                    "Score 1-5: 1 = answer does not address the "
                    "question at all; 5 = answer directly and "
                    "completely addresses what was asked. "
                    "Penalize tangents and boilerplate."
                ),
                weight=1.0,
            )
        ],
    },
    "Context Precision": {
        "description": (
            "Is the retrieved context relevant? Measures whether the "
            "retrieved chunks contain the information needed, i.e. "
            "retrieval quality at the top of the result list."
        ),
        "rubric": [
            RubricDimension(
                name="context_precision",
                description=(
                    "Score 1-5: 1 = retrieved chunks are mostly "
                    "irrelevant noise; 5 = the most relevant chunks "
                    "rank first and are on-topic. Penalize noisy "
                    "retrieval that buries the answer."
                ),
                weight=1.0,
            )
        ],
    },
    "Context Recall": {
        "description": (
            "Was all necessary information retrieved? Measures whether "
            "the retrieval captured everything needed to answer, i.e. "
            "missing-chunk detection."
        ),
        "rubric": [
            RubricDimension(
                name="context_recall",
                description=(
                    "Score 1-5: 1 = retrieval missed most information "
                    "needed to answer; 5 = retrieval captured all "
                    "information required. Penalize answers that "
                    "needed a chunk that was never retrieved."
                ),
                weight=1.0,
            )
        ],
    },
    "Answer Correctness": {
        "description": (
            "Does the answer match the ground truth? Compares the "
            "answer against the expected value for factual agreement "
            "beyond keyword overlap."
        ),
        "rubric": [
            RubricDimension(
                name="answer_correctness",
                description=(
                    "Score 1-5: 1 = answer contradicts the expected "
                    "answer; 5 = answer is factually consistent with "
                    "the expected answer in all key points. Compare "
                    "meaning, not wording."
                ),
                weight=1.0,
            )
        ],
    },
}


def metric_names() -> list[str]:
    """Return the RAGAS-style metric names for UI dropdowns."""
    return list(RAGAS_METRICS.keys())


def metric_rubric(name: str) -> list[RubricDimension]:
    """Return the rubric dimensions for a named metric."""
    return RAGAS_METRICS[name]["rubric"]


def metric_description(name: str) -> str:
    """Return the human description for a named metric."""
    return RAGAS_METRICS[name]["description"]
