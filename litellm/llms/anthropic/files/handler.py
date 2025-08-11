"""
Anthropic Files (Results) Handler
- Supports fetching Message Batch results via results_url or msgbatch id
"""
from __future__ import annotations

import os
from typing import Optional

import httpx

import litellm
from litellm.types.llms.openai import HttpxBinaryResponseContent as _Bin


class AnthropicFilesHandler:
    def __init__(self) -> None:
        self.default_api_base = "https://api.anthropic.com"
        self.api_version = "2023-06-01"

    def _build_url(self, file_id: str, api_base: Optional[str]) -> str:
        if file_id.startswith("http://") or file_id.startswith("https://"):
            return file_id
        base = (api_base or self.default_api_base).rstrip("/")
        if file_id.startswith("msgbatch_"):
            return f"{base}/v1/messages/batches/{file_id}/results"
        # fallback - treat as direct url
        return file_id

    def _build_headers(self, api_key: Optional[str], extra_headers: Optional[dict]) -> dict:
        key = (
            api_key
            or litellm.api_key
            or getattr(litellm, "anthropic_key", None)
            or os.getenv("ANTHROPIC_API_KEY")
        )
        if key is None:
            raise litellm.AuthenticationError(
                message="Missing ANTHROPIC_API_KEY for retrieving file content",
                model="n/a",
                llm_provider="anthropic",
            )
        headers = {
            "x-api-key": key,
            "anthropic-version": self.api_version,
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def file_content(
        self,
        file_id: str,
        api_base: Optional[str],
        api_key: Optional[str],
        timeout: float,
        extra_headers: Optional[dict] = None,
    ) -> _Bin:
        url = self._build_url(file_id=file_id, api_base=api_base)
        headers = self._build_headers(api_key=api_key, extra_headers=extra_headers)
        with httpx.Client(timeout=timeout) as client_httpx:
            resp = client_httpx.get(url, headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise litellm.APIError(
                    message=f"Anthropic file content fetch failed: {e}",
                    model="n/a",
                    llm_provider="anthropic",
                    response=resp,
                )
            return _Bin(resp)  # type: ignore 