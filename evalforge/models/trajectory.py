"""Trajectory models — the journey an agent takes, not just its final answer.

Shape follows the OTel GenAI semantic-convention span (tool, args, result,
thought, latency, tokens, error, ordering) so traces are not vendor-locked
(D5). These models are pure data: no scoring, no metrics — that lives in
``evalforge/trajectory/metrics.py`` (Phase 2).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .tokens import TokenCount

# Max nesting depth for args/result — protects against pathological payloads
# while keeping the common shapes (flat dicts, small nested lists) supported.
_MAX_JSON_DEPTH = 8


class TrajectoryStep(BaseModel):
    """One recorded step in an agent's run: a single tool call (or LLM call)."""

    __test__ = False

    index: int = Field(ge=0, description="Zero-based position in the trajectory; strictly increasing")
    tool: str = Field(min_length=1, description="Tool or action name, e.g. 'search', 'calculator', 'llm'")
    args: dict[str, Any] = Field(default_factory=dict, description="JSON-safe tool arguments")
    result: Optional[Any] = Field(default=None, description="Tool result: str, dict, list, or None")
    thought: Optional[str] = Field(default=None, description="Agent reasoning before the call, if exposed")
    latency_ms: float = Field(default=0.0, ge=0.0)
    tokens: Optional[TokenCount] = Field(default=None, description="Token usage, when the provider reports it")
    cost_usd: float = Field(default=0.0, ge=0.0)
    error: Optional[str] = Field(default=None, description="Error message if the call failed; None on success")

    @field_validator("args", "result", mode="before")
    @classmethod
    def _check_json_safe(cls, value: Any) -> Any:
        """Reject non-JSON-serializable payloads up front (V3 determinism)."""
        if value is None:
            return value
        _assert_json_safe(value, depth=0)
        return value


def _assert_json_safe(value: Any, depth: int) -> None:
    """Recursively verify a value survives json.dumps; depth-capped."""
    if depth > _MAX_JSON_DEPTH:
        raise ValueError(f"payload exceeds max nesting depth of {_MAX_JSON_DEPTH}")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"dict keys must be strings, got {type(k).__name__}")
            _assert_json_safe(v, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_json_safe(item, depth + 1)
        return
    raise ValueError(f"value of type {type(value).__name__} is not JSON-serializable")


class Trajectory(BaseModel):
    """A full agent run: ordered steps plus the final answer."""

    __test__ = False

    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_answer: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _indexes_strictly_increasing(self) -> "Trajectory":
        """Indexes must be 0,1,2,... with no gaps or duplicates (V1/V3)."""
        for i, step in enumerate(self.steps):
            if step.index != i:
                raise ValueError(
                    f"step index must equal position (got index={step.index} at position {i}); "
                    "trajectories are ordered records"
                )
        return self
