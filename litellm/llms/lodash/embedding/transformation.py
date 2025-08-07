from typing import List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage


class LodashError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Union[dict, httpx.Headers] = {},
    ):
        request = httpx.Request(
            method="POST", url="http://127.0.0.1:8000/api/v1/app2/gateway/embeddings"
        )
        response = httpx.Response(status_code=status_code, request=request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
            request=request,
            response=response,
        )


class LodashEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: Lodash AI Embeddings API
    """

    def __init__(self) -> None:
        pass

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base:
            if not api_base.endswith("/embeddings"):
                api_base = f"{api_base}/embeddings"
            return api_base
        return "http://127.0.0.1:8000/api/v1/app2/gateway/embeddings/local"

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "encoding_format",
            "dimensions",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to Lodash params
        """
        if "encoding_format" in non_default_params:
            optional_params["encoding_format"] = non_default_params["encoding_format"]
        if "dimensions" in non_default_params:
            optional_params["dimensions"] = non_default_params["dimensions"]
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            api_key = (
                get_secret_str("LODASH_API_KEY")
                or get_secret_str("LODASH_AI_API_KEY")
                or get_secret_str("LODASH_AI_TOKEN")
            )
        return {
            "x-token": api_key,
            "Content-Type": "application/json",
        }

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the input to match Lodash API format
        
        Remove 'lodash/' prefix from model name since Lodash API 
        expects just the model name (e.g., 'all-MiniLM-L6-v2')
        """
        # Remove 'lodash/' prefix if present
        clean_model = model
        if model.startswith("lodash/"):
            clean_model = model.replace("lodash/", "")
        
        data = {
            "input": input,
            "model": clean_model,
        }

        # Add optional parameters
        for param in ["encoding_format", "dimensions"]:
            if param in optional_params:
                data[param] = optional_params[param]

        return data

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        """
        Transform Lodash API response to OpenAI format
        
        Lodash API returns:
        {
          "data": [
            [0.1, 0.2, 0.3, ...],
            [0.4, 0.5, 0.6, ...],
          ]
        }
        
        Transform to OpenAI format:
        {
          "object": "list",
          "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, ...]},
            {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6, ...]},
          ],
          "model": "lodash/all-MiniLM-L6-v2",
          "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }
        """
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise LodashError(
                status_code=raw_response.status_code,
                message=f"Failed to decode response: {str(e)}",
            )

        # Handle error responses
        if raw_response.status_code != 200:
            error_message = response_json.get("error", {}).get("message", "Unknown error")
            raise LodashError(
                status_code=raw_response.status_code,
                message=error_message,
            )

        # Extract raw embedding arrays from Lodash response
        raw_embeddings = response_json.get("data", [])
        
        # Transform Lodash format to OpenAI format
        openai_data = []
        for i, embedding_array in enumerate(raw_embeddings):
            openai_data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding_array
            })

        # Set response in OpenAI format
        model_response.object = "list"
        model_response.data = openai_data
        model_response.model = model
        
        # Estimate usage information (since Lodash doesn't provide it)
        num_inputs = len(raw_embeddings)
        estimated_tokens_per_input = 5  # rough estimate
        estimated_total_tokens = num_inputs * estimated_tokens_per_input
        
        model_response.usage = Usage(
            prompt_tokens=estimated_total_tokens,
            total_tokens=estimated_total_tokens,
        )

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return LodashError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        ) 