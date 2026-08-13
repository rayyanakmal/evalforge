"""LLM interaction models — provider-agnostic."""

from typing import Optional, Literal

from pydantic import BaseModel


class Usage(BaseModel):
    """Token usage from an LLM API call."""
    __test__ = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""
    __test__ = False
    content: str
    reasoning_content: Optional[str] = None
    usage: Optional[Usage] = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class Message(BaseModel):
    """A single message in a chat conversation."""
    __test__ = False
    role: Literal["system", "user", "assistant"]
    content: str
