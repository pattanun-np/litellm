import asyncio
import json
import urllib.parse
from typing import Any, Coroutine, Optional, Union

import httpx

from litellm import LlmProviders
from litellm.integrations.gcs_bucket.gcs_bucket_base import (
    GCSBucketBase,
    GCSLoggingConfig,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.openai import CreateFileRequest, OpenAIFileObject, HttpxBinaryResponseContent as _Bin
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES

from .transformation import VertexAIJsonlFilesTransformation

vertex_ai_files_transformation = VertexAIJsonlFilesTransformation()


class VertexAIFilesHandler(GCSBucketBase):
    """
    Handles Calling VertexAI in OpenAI Files API format v1/files/*

    This implementation uploads files on GCS Buckets
    """

    def __init__(self):
        super().__init__()
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=LlmProviders.VERTEX_AI,
        )

    async def async_create_file(
        self,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> OpenAIFileObject:
        gcs_logging_config: GCSLoggingConfig = await self.get_gcs_logging_config(
            kwargs={}
        )
        headers = await self.construct_request_headers(
            vertex_instance=gcs_logging_config["vertex_instance"],
            service_account_json=gcs_logging_config["path_service_account"],
        )
        bucket_name = gcs_logging_config["bucket_name"]
        (
            logging_payload,
            object_name,
        ) = vertex_ai_files_transformation.transform_openai_file_content_to_vertex_ai_file_content(
            openai_file_content=create_file_data.get("file")
        )
        gcs_upload_response = await self._log_json_data_on_gcs(
            headers=headers,
            bucket_name=bucket_name,
            object_name=object_name,
            logging_payload=logging_payload,
        )

        return vertex_ai_files_transformation.transform_gcs_bucket_response_to_openai_file_object(
            create_file_data=create_file_data,
            gcs_upload_response=gcs_upload_response,
        )

    def create_file(
        self,
        _is_async: bool,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[OpenAIFileObject, Coroutine[Any, Any, OpenAIFileObject]]:
        """
        Creates a file on VertexAI GCS Bucket

        Only supported for Async litellm.acreate_file
        """

        if _is_async:
            return self.async_create_file(
                create_file_data=create_file_data,
                api_base=api_base,
                vertex_credentials=vertex_credentials,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            return asyncio.run(
                self.async_create_file(
                    create_file_data=create_file_data,
                    api_base=api_base,
                    vertex_credentials=vertex_credentials,
                    vertex_project=vertex_project,
                    vertex_location=vertex_location,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            )

    def _parse_gs_uri(self, gs_uri: str) -> Optional[tuple[str, str]]:
        # expects gs://bucket/path/to/folder or file
        if not gs_uri.startswith("gs://"):
            return None
        without_scheme = gs_uri[len("gs://") :]
        parts = without_scheme.split("/", 1)
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1]

    async def _list_gcs_objects(self, bucket: str, prefix: str, headers: dict) -> list[dict]:
        # JSON API: https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={prefix}
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={urllib.parse.quote(prefix, safe='') }"
        resp = await self.async_httpx_client.get(url=url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])

    async def _download_gcs_object_raw(self, bucket: str, object_name: str, headers: dict) -> httpx.Response:
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(object_name, safe='') }?alt=media"
        resp = await self.async_httpx_client.get(url=url, headers=headers)
        resp.raise_for_status()
        return resp

    def get_results_content(
        self,
        file_id: str,
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[dict] = None,
    ) -> _Bin:
        """
        Given gs://bucket/prefix (results folder) or gs://bucket/prefix/predictions.jsonl,
        locate output file and return its contents as HttpxBinaryResponseContent.
        """
        # Prepare auth headers via existing GCS config
        gcs_logging_config: GCSLoggingConfig = asyncio.run(self.get_gcs_logging_config(kwargs={}))
        headers = asyncio.run(
            self.construct_request_headers(
                vertex_instance=gcs_logging_config["vertex_instance"],
                service_account_json=gcs_logging_config["path_service_account"],
            )
        )
        if extra_headers:
            headers.update(extra_headers)

        parsed = self._parse_gs_uri(file_id)
        if parsed is None:
            raise ValueError("Expected gs:// URI for vertex_ai results")
        bucket, object_path = parsed
        # If path points to a file directly, try downloading
        async def _run() -> _Bin:
            try:
                if object_path.endswith(".jsonl"):
                    resp = await self._download_gcs_object_raw(bucket=bucket, object_name=object_path, headers=headers)
                    return _Bin(resp)  # type: ignore
                # Otherwise list objects under prefix
                items = await self._list_gcs_objects(bucket=bucket, prefix=object_path, headers=headers)
                if not items:
                    raise httpx.HTTPStatusError("No output files under prefix", request=None, response=httpx.Response(503))
                # Prefer predictions.jsonl
                selected = None
                for it in items:
                    name = it.get("name", "")
                    if name.endswith("predictions.jsonl"):
                        selected = name
                        break
                if selected is None:
                    # pick first item that's not input.jsonl
                    for it in items:
                        name = it.get("name", "")
                        if name and not name.endswith("input.jsonl"):
                            selected = name
                            break
                if selected is None:
                    # fallback to first
                    selected = items[0].get("name")
                resp = await self._download_gcs_object_raw(bucket=bucket, object_name=selected, headers=headers)
                return _Bin(resp)  # type: ignore
            except httpx.HTTPStatusError as e:
                raise

        return asyncio.run(_run())
