"""RunResult and related Pydantic models."""

from typing import Optional, Literal

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    """Per-dimension score from rubric evaluation."""
    __test__ = False
    name: str
    score: int = Field(ge=1, le=5)
    reasoning: str = ""


class ScoreResult(BaseModel):
    """Result from scoring a single test response."""
    __test__ = False
    overall: float = Field(ge=0.0, le=1.0)
    dimensions: Optional[list[DimensionScore]] = None
    method: str


class TokenCount(BaseModel):
    """Token usage for a single LLM call."""
    __test__ = False
    input: int = 0
    output: int = 0
    total: int = 0


class TestResult(BaseModel):
    """Result for a single test case execution."""
    __test__ = False
    id: str
    status: Literal["pass", "fail", "error"]
    response: Optional[str] = None
    expected_value: Optional[str] = None
    score: Optional[ScoreResult] = None
    tokens: Optional[TokenCount] = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: Optional[str] = None


class Summary(BaseModel):
    """Aggregate statistics for a full test run."""
    __test__ = False
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    pass_rate: float = 0.0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None


class RunResult(BaseModel):
    """Complete result of a test suite execution."""
    __test__ = False
    suite_name: str
    timestamp: str
    duration_ms: float
    tests: list[TestResult] = Field(default_factory=list)
    summary: Summary = Field(default_factory=Summary)
