"""Retry logic for transient LLM API failures."""

import asyncio
import logging
from typing import TypeVar, Callable, Awaitable

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_EXCEPTIONS = (TimeoutError, asyncio.TimeoutError)


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    max_retries: int = 1,
    base_delay: float = 1.0,
) -> T:
    """Execute an async function with retry on transient failures.

    Only retries on RETRYABLE_EXCEPTIONS (TimeoutError, asyncio.TimeoutError).
    Non-retryable exceptions (ValueError, etc.) propagate immediately.

    Args:
        fn: Async callable to execute.
        max_retries: Maximum retry attempts (total calls = 1 + max_retries).
        base_delay: Base delay in seconds between retries.

    Returns:
        The result of the successful call.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Retryable error on attempt %d/%d: %s. Retrying in %.1fs...",
                    attempt + 1, max_retries + 1, e, delay,
                )
                await asyncio.sleep(delay)
        # Non-retryable exceptions propagate immediately

    # Exhausted all retries
    raise last_exception  # type: ignore[misc]
