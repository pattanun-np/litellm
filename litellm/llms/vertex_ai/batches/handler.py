import json
from typing import Any, Coroutine, Dict, Optional, Union

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.openai import CreateBatchRequest
from litellm.types.llms.vertex_ai import (
    VERTEX_CREDENTIALS_TYPES,
    VertexAIBatchPredictionJob,
)
from litellm.types.utils import LiteLLMBatch

from .transformation import VertexAIBatchTransformation


class VertexAIBatchPrediction(VertexLLM):
    def __init__(self, gcs_bucket_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcs_bucket_name = gcs_bucket_name

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
        sync_handler = _get_httpx_client()

        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        default_api_base = self.create_vertex_batch_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
        )

        if len(default_api_base.split(":")) > 1:
            endpoint = default_api_base.split(":")[-1]
        else:
            endpoint = ""

        _, api_base = self._check_custom_proxy(
            api_base=api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint,
            stream=None,
            auth_header=None,
            url=default_api_base,
        )

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        vertex_batch_request: VertexAIBatchPredictionJob = VertexAIBatchTransformation.transform_openai_batch_request_to_vertex_ai_batch_request(
            request=create_batch_data
        )

        if _is_async is True:
            return self._async_create_batch(
                vertex_batch_request=vertex_batch_request,
                api_base=api_base,
                headers=headers,
            )

        response = sync_handler.post(
            url=api_base,
            headers=headers,
            data=json.dumps(vertex_batch_request),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    async def _async_create_batch(
        self,
        vertex_batch_request: VertexAIBatchPredictionJob,
        api_base: str,
        headers: Dict[str, str],
    ) -> LiteLLMBatch:
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )
        response = await client.post(
            url=api_base,
            headers=headers,
            data=json.dumps(vertex_batch_request),
        )
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    def create_vertex_batch_url(
        self,
        vertex_location: str,
        vertex_project: str,
    ) -> str:
        """Return the base url for the vertex garden models"""
        #  POST https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION/batchPredictionJobs
        return f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/batchPredictionJobs"

    def retrieve_batch(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
        sync_handler = _get_httpx_client()
# 
        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        default_api_base = self.create_vertex_batch_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
        )

        # Append batch_id to the URL
        default_api_base = f"{default_api_base}/{batch_id}"

        if len(default_api_base.split(":")) > 1:
            endpoint = default_api_base.split(":")[-1]
        else:
            endpoint = ""

        _, api_base = self._check_custom_proxy(
            api_base=api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint,
            stream=None,
            auth_header=None,
            url=default_api_base,
        )

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        if _is_async is True:
            return self._async_retrieve_batch(
                api_base=api_base,
                headers=headers,
            )

        response = sync_handler.get(
            url=api_base,
            headers=headers,
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    async def _async_retrieve_batch(
        self,
        api_base: str,
        headers: Dict[str, str],
    ) -> LiteLLMBatch:
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )
        response = await client.get(
            url=api_base,
            headers=headers,
        )
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    def cancel_batch(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
        sync_handler = _get_httpx_client()

        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        default_base = self.create_vertex_batch_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
        )
        # POST .../batchPredictionJobs/{id}:cancel
        cancel_url = f"{default_base}/{batch_id}:cancel"

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        if _is_async is True:
            return self._async_cancel_batch(cancel_url=cancel_url, headers=headers)

        try:
            response = sync_handler.post(url=cancel_url, headers=headers, json={})
        except Exception as e:
            # httpx.HTTPStatusError or transport errors – try to extract response text
            status_text = getattr(e, "text", None) or getattr(e, "message", None) or str(e)
            resp_obj = getattr(e, "response", None)
            body_dict = None
            clean_message = status_text
            try:
                body_dict = json.loads(status_text)
                clean_message = (
                    body_dict.get("error", {}).get("message", status_text)
                    if isinstance(body_dict, dict)
                    else status_text
                )
            except Exception:
                pass
            raise litellm.BadRequestError(
                message=clean_message,
                model="n/a",
                llm_provider="vertex_ai",
                response=resp_obj,
                body=body_dict,
            )
        if response.status_code not in (200, 204):
            # Map to BadRequestError with provider-specific detail
            try:
                detail = response.text
            except Exception:
                detail = str(response)
            body_dict = None
            clean_message = detail
            try:
                body_dict = json.loads(detail)
                clean_message = (
                    body_dict.get("error", {}).get("message", detail)
                    if isinstance(body_dict, dict)
                    else detail
                )
            except Exception:
                pass
            raise litellm.BadRequestError(
                message=clean_message,
                model="n/a",
                llm_provider="vertex_ai",
                response=response,
                body=body_dict,
            )

        # After cancelling (or if already cancelling), fetch current state
        return self.retrieve_batch(
            _is_async=False,
            batch_id=batch_id,
            api_base=api_base,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def _async_cancel_batch(self, cancel_url: str, headers: Dict[str, str]) -> LiteLLMBatch:
        async_client = get_async_httpx_client(llm_provider=litellm.LlmProviders.VERTEX_AI)
        try:
            resp = await async_client.post(url=cancel_url, headers=headers, json={})
        except Exception as e:
            status_text = getattr(e, "text", None) or getattr(e, "message", None) or str(e)
            resp_obj = getattr(e, "response", None)
            body_dict = None
            clean_message = status_text
            try:
                body_dict = json.loads(status_text)
                clean_message = (
                    body_dict.get("error", {}).get("message", status_text)
                    if isinstance(body_dict, dict)
                    else status_text
                )
            except Exception:
                pass
            raise litellm.BadRequestError(
                message=clean_message,
                model="n/a",
                llm_provider="vertex_ai",
                response=resp_obj,
                body=body_dict,
            )
        if resp.status_code not in (200, 204):
            body = await resp.aread()
            text_body = body.decode("utf-8", errors="ignore")
            body_dict = None
            clean_message = text_body
            try:
                body_dict = json.loads(text_body)
                clean_message = (
                    body_dict.get("error", {}).get("message", text_body)
                    if isinstance(body_dict, dict)
                    else text_body
                )
            except Exception:
                pass
            raise litellm.BadRequestError(
                message=clean_message,
                model="n/a",
                llm_provider="vertex_ai",
                response=resp,
                body=body_dict,
            )
        # Build retrieve URL and fetch actual status
        if cancel_url.endswith(":cancel"):
            retrieve_url = cancel_url[: -len(":cancel")]
        else:
            retrieve_url = cancel_url.split(":cancel")[0]
        try:
            get_resp = await async_client.get(url=retrieve_url, headers=headers)
            get_resp.raise_for_status()
        except Exception as e:
            status_text = getattr(e, "text", None) or getattr(e, "message", None) or str(e)
            resp_obj = getattr(e, "response", None)
            body_dict = None
            clean_message = status_text
            try:
                body_dict = json.loads(status_text)
                clean_message = (
                    body_dict.get("error", {}).get("message", status_text)
                    if isinstance(body_dict, dict)
                    else status_text
                )
            except Exception:
                pass
            raise litellm.BadRequestError(
                message=clean_message,
                model="n/a",
                llm_provider="vertex_ai",
                response=resp_obj,
                body=body_dict,
            )
        _json_response = get_resp.json()
        return VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )

    def list_batches(
        self,
        _is_async: bool,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        after: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Union[Dict[str, Any], Coroutine[Any, Any, Dict[str, Any]]]:
        sync_handler = _get_httpx_client()

        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        base_url = self.create_vertex_batch_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
        )

        # Vertex list endpoint is the same base without trailing id
        if len(base_url.split(":")) > 1:
            endpoint = base_url.split(":")[-1]
        else:
            endpoint = ""

        _, list_url = self._check_custom_proxy(
            api_base=api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint,
            stream=None,
            auth_header=None,
            url=base_url,
        )

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        params: Dict[str, Any] = {}
        if limit is not None:
            params["pageSize"] = limit
        if after is not None:
            params["pageToken"] = after

        if _is_async is True:
            return self._async_list_batches(list_url=list_url, headers=headers, params=params)

        try:
            response = sync_handler.get(url=list_url, headers=headers, params=params)
        except Exception as e:
            status_text = getattr(e, "text", None) or getattr(e, "message", None) or str(e)
            resp_obj = getattr(e, "response", None)
            body_dict = None
            clean_message = status_text
            try:
                body_dict = json.loads(status_text)
                clean_message = (
                    body_dict.get("error", {}).get("message", status_text)
                    if isinstance(body_dict, dict)
                    else status_text
                )
            except Exception:
                pass
            raise litellm.BadRequestError(
                message=clean_message,
                model="n/a",
                llm_provider="vertex_ai",
                response=resp_obj,
                body=body_dict,
            )
        if response.status_code != 200:
            detail = response.text
            body_dict = None
            clean_message = detail
            try:
                body_dict = json.loads(detail)
                clean_message = (
                    body_dict.get("error", {}).get("message", detail)
                    if isinstance(body_dict, dict)
                    else detail
                )
            except Exception:
                pass
            raise litellm.BadRequestError(
                message=clean_message,
                model="n/a",
                llm_provider="vertex_ai",
                response=response,
                body=body_dict,
            )

        data = response.json() or {}
        items = data.get("batchPredictionJobs", []) or []
        mapped = [
            VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(r)
            for r in items
        ]
        return {
            "object": "list",
            "data": mapped,
            "has_more": True if data.get("nextPageToken") else False,
            "first_id": mapped[0].id if mapped else None,
            "last_id": mapped[-1].id if mapped else None,
            "next_page_token": data.get("nextPageToken"),
        }

    async def _async_list_batches(
        self,
        list_url: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        client = get_async_httpx_client(llm_provider=litellm.LlmProviders.VERTEX_AI)
        try:
            resp = await client.get(url=list_url, headers=headers, params=params)
        except Exception as e:
            status_text = getattr(e, "text", None) or getattr(e, "message", None) or str(e)
            resp_obj = getattr(e, "response", None)
            body_dict = None
            clean_message = status_text
            try:
                body_dict = json.loads(status_text)
                clean_message = (
                    body_dict.get("error", {}).get("message", status_text)
                    if isinstance(body_dict, dict)
                    else status_text
                )
            except Exception:
                pass
            raise litellm.BadRequestError(
                message=clean_message,
                model="n/a",
                llm_provider="vertex_ai",
                response=resp_obj,
                body=body_dict,
            )
        if resp.status_code != 200:
            text_body = resp.text
            body_dict = None
            clean_message = text_body
            try:
                body_dict = json.loads(text_body)
                clean_message = (
                    body_dict.get("error", {}).get("message", text_body)
                    if isinstance(body_dict, dict)
                    else text_body
                )
            except Exception:
                pass
            raise litellm.BadRequestError(
                message=clean_message,
                model="n/a",
                llm_provider="vertex_ai",
                response=resp,
                body=body_dict,
            )

        data = resp.json() or {}
        items = data.get("batchPredictionJobs", []) or []
        mapped = [
            VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(r)
            for r in items
        ]
        return {
            "object": "list",
            "data": mapped,
            "has_more": True if data.get("nextPageToken") else False,
            "first_id": mapped[0].id if mapped else None,
            "last_id": mapped[-1].id if mapped else None,
            "next_page_token": data.get("nextPageToken"),
        }
