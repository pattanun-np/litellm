"""
Anthropic Message Batches API Handler

Handles Anthropic message batch operations without requiring file uploads.
Anthropic batches work differently from other providers - they don't require JSONL files.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

import httpx

from .errors import raise_mapped_error
from .transformation import (
    build_requests_from_jsonl,
    to_litellm_batch,
)
from litellm.types.llms.openai import (
    Batch,
    CancelBatchRequest,
    CreateBatchRequest,
    RetrieveBatchRequest,
)
from litellm.types.utils import LiteLLMBatch


class AnthropicBatchesAPI:
    """
    Anthropic Message Batches API Handler
    
    Anthropic batches work differently from other providers:
    - No file upload required
    - Direct message batch creation
    - Uses Anthropic's native batch format
    """

    def __init__(self):
        self.api_base = "https://api.anthropic.com"
        self.api_version = "2023-06-01"

    def _get_base_url(self, api_base: Optional[str]) -> str:
        base = api_base or self.api_base
        if base.endswith("/"):
            base = base[:-1]
        return base

    def _get_batches_url(self, api_base: Optional[str]) -> str:
        base = self._get_base_url(api_base)
        return f"{base}/v1/messages/batches"

    def _build_headers(self, api_key: Optional[str]) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "x-api-key": api_key or "",
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }
        return headers

    def create_batch(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        max_retries: Optional[int] = None,
        create_batch_data: CreateBatchRequest = None,
        _is_async: bool = False,
    ) -> Union[LiteLLMBatch, Any]:
        """
        Create an Anthropic message batch
        
        For Anthropic, input_file_id can contain JSONL content directly
        or be a reference to uploaded content.
        """
        if _is_async:
            return self._acreate_batch(
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                create_batch_data=create_batch_data,
            )

        # Extract fields
        if isinstance(create_batch_data, dict):
            input_file_id = create_batch_data.get("input_file_id", "")
        else:
            input_file_id = create_batch_data.input_file_id  # type: ignore

        # Build requests from JSONL content (if provided)
        requests_payload: List[Dict[str, Any]] = build_requests_from_jsonl(input_file_id)
        if len(requests_payload) == 0 and input_file_id:
            raise ValueError("Anthropic batches require JSONL content in input_file_id with per-line request bodies.")

        url = self._get_batches_url(api_base)
        headers = self._build_headers(api_key)
        payload: Dict[str, Any] = {"requests": requests_payload}

        client = httpx.Client(timeout=timeout or 600.0)
        try:
            resp = client.post(url, headers=headers, json=payload)
        finally:
            client.close()
        if resp.is_error:
            raise_mapped_error(resp, operation="Create batch")

        resp_json = resp.json()
        return to_litellm_batch(resp_json)

    async def _acreate_batch(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        max_retries: Optional[int] = None,
        create_batch_data: CreateBatchRequest = None,
    ) -> LiteLLMBatch:
        """Async version of create_batch"""
        return self.create_batch(
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            create_batch_data=create_batch_data,
            _is_async=False,
        )

    def retrieve_batch(
        self,
        batch_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        max_retries: Optional[int] = None,
        _is_async: bool = False,
    ) -> Union[LiteLLMBatch, Any]:
        """Retrieve an Anthropic message batch"""
        if _is_async:
            return self._aretrieve_batch(
                batch_id=batch_id,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )

        url = f"{self._get_batches_url(api_base)}/{batch_id}"
        headers = self._build_headers(api_key)
        client = httpx.Client(timeout=timeout or 600.0)
        try:
            resp = client.get(url, headers=headers)
        finally:
            client.close()
        if resp.is_error:
            raise_mapped_error(resp, operation="Retrieve batch")

        return to_litellm_batch(resp.json())

    async def _aretrieve_batch(
        self,
        batch_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        max_retries: Optional[int] = None,
    ) -> LiteLLMBatch:
        """Async version of retrieve_batch"""
        return self.retrieve_batch(
            batch_id=batch_id,
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            _is_async=False,
        )

    def list_batches(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        max_retries: Optional[int] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        _is_async: bool = False,
    ) -> Union[List[LiteLLMBatch], Any]:
        """List Anthropic message batches"""
        if _is_async:
            return self._alist_batches(
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                litellm_params=litellm_params,
            )

        url = self._get_batches_url(api_base)
        headers = self._build_headers(api_key)
        client = httpx.Client(timeout=timeout or 600.0)
        try:
            resp = client.get(url, headers=headers)
        finally:
            client.close()
        if resp.is_error:
            raise_mapped_error(resp, operation="List batches")

        resp_json = resp.json()
        items = resp_json.get("data", [])
        mapped = [to_litellm_batch(item) for item in items]
        return {
            "data": mapped,
            "has_more": resp_json.get("has_more", False),
            "object": "list",
            "first_id": resp_json.get("first_id"),
            "last_id": resp_json.get("last_id"),
        }

    async def _alist_batches(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        max_retries: Optional[int] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> List[LiteLLMBatch]:
        """Async version of list_batches"""
        return self.list_batches(
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            litellm_params=litellm_params,
            _is_async=False,
        )

    def cancel_batch(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        max_retries: Optional[int] = None,
        cancel_batch_data: CancelBatchRequest = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        _is_async: bool = False,
    ) -> Union[Batch, Any]:
        """Cancel an Anthropic message batch"""
        if _is_async:
            return self._acancel_batch(
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                cancel_batch_data=cancel_batch_data,
                litellm_params=litellm_params,
            )

        # Extract batch id
        if isinstance(cancel_batch_data, dict):
            batch_id = cancel_batch_data.get("batch_id", "")
        else:
            batch_id = cancel_batch_data.batch_id  # type: ignore
        if not batch_id:
            raise ValueError("'batch_id' is required to cancel a batch.")

        url = f"{self._get_batches_url(api_base)}/{batch_id}/cancel"
        headers = self._build_headers(api_key)
        client = httpx.Client(timeout=timeout or 600.0)
        try:
            resp = client.post(url, headers=headers)
        finally:
            client.close()
        if resp.is_error:
            raise_mapped_error(resp, operation="Cancel batch")

        batch = to_litellm_batch(resp.json())
        # Return as Batch (OpenAI type) compatible object
        return Batch(
            id=batch.id,
            object=batch.object,
            endpoint=batch.endpoint,
            input_file_id=batch.input_file_id,
            completion_window=batch.completion_window,
            status=batch.status,
            created_at=batch.created_at,
            metadata=batch.metadata,
            request_counts=batch.request_counts,
            usage=batch.usage,
            errors=batch.errors,
            output_file_id=batch.output_file_id,
            error_file_id=batch.error_file_id,
            expired_at=batch.expired_at,
            expires_at=batch.expires_at,
            failed_at=batch.failed_at,
            finalizing_at=batch.finalizing_at,
            in_progress_at=batch.in_progress_at,
            cancelled_at=batch.cancelled_at,
            cancelling_at=batch.cancelling_at,
            completed_at=batch.completed_at,
        )

    async def _acancel_batch(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        max_retries: Optional[int] = None,
        cancel_batch_data: CancelBatchRequest = None,
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> Batch:
        """Async version of cancel_batch"""
        return self.cancel_batch(
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            cancel_batch_data=cancel_batch_data,
            litellm_params=litellm_params,
            _is_async=False,
        ) 