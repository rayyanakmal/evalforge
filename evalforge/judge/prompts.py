"""Judge system prompt templates for LLM-as-Judge scoring.

Isolates prompt engineering from client logic.
"""

from evalforge.models.suite import RubricDimension


JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluation judge. Your task is to evaluate a response "
    "to a question against a set of rubric dimensions.\n\n"
    "Return ONLY valid JSON (no markdown, no extra text) with this exact structure:\n"
    '{{"<dimension_name>": <score 1-5>, ..., '
    '"overall": <score 1-5>, "reasoning": "<brief overall justification>"}}\n\n'
    "Scores:\n"
    "  1 = Poor — completely misses the mark\n"
    "  2 = Below average — significant issues\n"
    "  3 = Acceptable — meets minimum requirements\n"
    "  4 = Good — solid response with minor issues\n"
    "  5 = Excellent — thorough, accurate, and well-presented\n\n"
    "IMPORTANT: Include EVERY rubric dimension in your JSON response. "
    "Do not skip any dimension. Always include the 'overall' and 'reasoning' fields."
)


STRICT_RETRY_PROMPT = (
    "IMPORTANT: Your previous response was not valid JSON. "
    "You MUST return ONLY valid JSON and nothing else. "
    "No markdown formatting, no code fences, no additional text. "
    "Start your response with '{{' and end with '}}'. "
    "Include ALL required dimensions. "
    "Failure to return valid JSON will be treated as a critical error.\n\n"
)


def build_rubric_prompt(dimensions: list[RubricDimension]) -> str:
    """Build the judge system prompt including rubric dimensions.

    Args:
        dimensions: List of rubric dimensions to evaluate.

    Returns:
        Complete system prompt string with dimension descriptions and JSON format.
    """
    if not dimensions:
        # Fallback: generic prompt
        return (
            JUDGE_SYSTEM_PROMPT
            + "\n\nNo specific dimensions provided. Evaluate the response overall."
        )

    dim_lines = []
    json_template_parts = []
    for dim in dimensions:
        dim_lines.append(f"  - {dim.name}: {dim.description} (weight: {dim.weight})")
        json_template_parts.append(f'"{dim.name}": <1-5>')

    dim_section = "\n".join(dim_lines)
    json_template = "{" + ", ".join(json_template_parts) + ', "overall": <1-5>, "reasoning": "<justification>"}'

    prompt = (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"Rubric Dimensions to Evaluate:\n"
        f"{dim_section}\n\n"
        f"Expected JSON format:\n"
        f"{json_template}\n\n"
        f"Evaluate the response against EACH dimension above. "
        f"Provide a score of 1-5 for every dimension, an overall score of 1-5, "
        f"and a brief reasoning justifying your scores."
    )

    return prompt
