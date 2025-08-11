"""
Error mapping utilities for Anthropic Batches
"""
from __future__ import annotations

import json
from typing import Optional

import httpx
import litellm


def raise_mapped_error(resp: httpx.Response, operation: str) -> None:
    """Map Anthropic HTTP errors to litellm exceptions with consistent messages."""
    status = resp.status_code
    try:
        payload = resp.json()
    except Exception:
        payload = {"error": resp.text}
    error_message = f"{operation} failed. Error code: {status} - {json.dumps(payload, ensure_ascii=False)}"

    if status == 400:
        raise litellm.BadRequestError(message=error_message, model="n/a", llm_provider="anthropic", response=resp)  # type: ignore
    if status == 401:
        raise litellm.AuthenticationError(message=error_message, model="n/a", llm_provider="anthropic", response=resp)  # type: ignore
    if status == 403:
        raise litellm.PermissionDeniedError(message=error_message, model="n/a", llm_provider="anthropic", response=resp)  # type: ignore
    if status == 404:
        raise litellm.NotFoundError(message=error_message, model="n/a", llm_provider="anthropic", response=resp)  # type: ignore
    if status == 408:
        raise litellm.APITimeoutError(message=error_message, model="n/a", llm_provider="anthropic", response=resp)  # type: ignore
    if status == 409:
        # Align with OpenAI-style conflict as BadRequest
        raise litellm.BadRequestError(message=error_message, model="n/a", llm_provider="anthropic", response=resp)  # type: ignore
    if status == 422:
        raise litellm.UnprocessableEntityError(message=error_message, model="n/a", llm_provider="anthropic", response=resp)  # type: ignore
    if status == 429:
        raise litellm.RateLimitError(message=error_message, model="n/a", llm_provider="anthropic", response=resp)  # type: ignore
    if status == 503:
        raise litellm.ServiceUnavailableError(message=error_message, model="n/a", llm_provider="anthropic", response=resp)  # type: ignore
    if status >= 500:
        raise litellm.APIError(message=error_message, model="n/a", llm_provider="anthropic", response=resp)  # type: ignore
    # default
    raise litellm.APIConnectionError(message=error_message, model="n/a", llm_provider="anthropic", response=resp)  # type: ignore 