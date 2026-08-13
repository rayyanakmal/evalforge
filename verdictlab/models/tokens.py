"""Token usage models, extracted to avoid circular imports.

TrajectoryStep (trajectory.py) and TestResult (result.py) both reference
TokenCount, and result.py will reference Trajectory — so TokenCount lives
here in its own module. result.py re-exports it for backward compatibility
(`from verdictlab.models.result import TokenCount` keeps working).
"""

from typing import Optional

from pydantic import BaseModel


class TokenCount(BaseModel):
    """Token usage for a single LLM call.

    Fields are Optional[int] to support open-source models
    that don't report token counts (N/A edge case per US-3).
    """
    __test__ = False
    input: Optional[int] = 0
    output: Optional[int] = 0
    total: Optional[int] = 0
