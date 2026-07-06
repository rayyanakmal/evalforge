"""Multi-provider LLM client for judge and target LLM calls.

Provides a single abstraction for all LLM API calls: DeepSeek, OpenAI, Anthropic.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from evalforge.models.llm import LLMResponse, Message, Usage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM API call times out."""
    pass


class LLMAuthError(LLMError):
    """Raised on authentication failures (401/403)."""
    pass


# ---------------------------------------------------------------------------
# Abstract LLM Client
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract base for all LLM provider clients.

    Handles API communication, token counting, cost calculation,
    and provider-specific quirks.

    Implementations: DeepSeekClient, OpenAIClient, AnthropicClient
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        max_tokens: int = 700,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Send a chat completion request to the LLM provider.

        Args:
            messages: Ordered list of system/user/assistant messages.
            max_tokens: Maximum output tokens. Default 700 (spike-validated).
            temperature: Sampling temperature. Default 0.1 for deterministic scoring.

        Returns:
            LLMResponse with content, usage, latency_ms, and cost_usd.

        Raises:
            LLMTimeoutError: After timeout.
            LLMAuthError: 401/403.
            LLMError: Other failures.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name: 'deepseek', 'openai', 'anthropic'."""
        ...


# ---------------------------------------------------------------------------
# DeepSeek Client
# ---------------------------------------------------------------------------

class DeepSeekClient(LLMClient):
    """LLM client for DeepSeek API (api.deepseek.com).

    Handles reasoning_content, token counting, and cost calculation.
    """

    BASE_URL = "https://api.deepseek.com/v1/chat/completions"

    # DeepSeek pricing per million tokens (approximate)
    PRICING_INPUT_PER_MTOK = 0.07
    PRICING_OUTPUT_PER_MTOK = 0.28

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url or self.BASE_URL
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "deepseek"

    async def generate(
        self,
        messages: list[Message],
        max_tokens: int = 700,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Send a chat completion request to DeepSeek API."""
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        start = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._base_url,
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"DeepSeek API timed out: {e}") from e

        latency_ms = (time.monotonic() - start) * 1000

        if response.status_code == 401 or response.status_code == 403:
            raise LLMAuthError(
                f"DeepSeek API authentication failed ({response.status_code}): "
                f"{response.text[:200]}"
            )

        if response.status_code != 200:
            raise LLMError(
                f"DeepSeek API error ({response.status_code}): {response.text[:500]}"
            )

        data = response.json()
        choice = data["choices"][0]
        message_data = choice["message"]
        usage_data = data.get("usage", {})

        # Calculate cost
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)

        cost_usd = (
            prompt_tokens / 1_000_000 * self.PRICING_INPUT_PER_MTOK
            + completion_tokens / 1_000_000 * self.PRICING_OUTPUT_PER_MTOK
        )

        return LLMResponse(
            content=message_data.get("content", ""),
            reasoning_content=message_data.get("reasoning_content"),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_client(provider: str, model: str, api_key: str) -> LLMClient:
    """Factory: returns the correct LLMClient subclass for the provider.

    Args:
        provider: One of 'deepseek', 'openai', 'anthropic'.
        model: Provider-specific model name.
        api_key: API key for the provider.

    Returns:
        Configured LLMClient instance.

    Raises:
        ValueError: Unknown provider string.
    """
    provider_lower = provider.lower()

    if provider_lower == "deepseek":
        return DeepSeekClient(api_key=api_key, model=model)
    elif provider_lower == "openai":
        raise NotImplementedError("OpenAI client not yet implemented")
    elif provider_lower == "anthropic":
        raise NotImplementedError("Anthropic client not yet implemented")
    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Supported providers: deepseek, openai, anthropic"
        )
