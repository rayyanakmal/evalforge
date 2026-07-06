"""TestSuite and related Pydantic models."""

from typing import Optional, Literal

from pydantic import BaseModel, Field


class RubricDimension(BaseModel):
    """A single scoring dimension for rubric-based evaluation."""
    __test__ = False
    name: str
    description: str
    weight: float = Field(ge=0.0, le=1.0)


class Expected(BaseModel):
    """Expected output specification for a test case."""
    __test__ = False
    type: Literal["exact", "semantic", "rubric", "function"]
    value: Optional[str] = None
    rubric: Optional[list[RubricDimension]] = None


class TestMetadata(BaseModel):
    """Optional metadata for a test case."""
    __test__ = False
    tags: Optional[list[str]] = None
    cost_limit_usd: Optional[float] = None


class TestCase(BaseModel):
    """A single test case within a suite."""
    __test__ = False
    id: str
    prompt: str
    expected: Expected
    metadata: Optional[TestMetadata] = None


class TestSuite(BaseModel):
    """A collection of test cases to evaluate."""
    __test__ = False
    name: str
    description: Optional[str] = None
    tests: list[TestCase] = Field(default_factory=list)
