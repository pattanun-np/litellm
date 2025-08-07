import pytest
import asyncio
from unittest.mock import Mock, patch
import httpx

from litellm.llms.lodash.embedding.transformation import (
    LodashEmbeddingConfig,
    LodashError,
)
from litellm.types.utils import EmbeddingResponse
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


class TestLodashEmbeddingConfig:
    def test_init(self):
        """Test LodashEmbeddingConfig initialization"""
        config = LodashEmbeddingConfig()
        assert config is not None

    def test_get_complete_url_default(self):
        """Test default API URL"""
        config = LodashEmbeddingConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="text-embedding-ada-002",
            optional_params={},
            litellm_params={},
        )
        assert url == "http://127.0.0.1:8000/api/v1/app2/gateway/embeddings/local"

    def test_get_complete_url_custom_base(self):
        """Test custom API base URL"""
        config = LodashEmbeddingConfig()
        custom_base = "https://custom.api.com/v1"
        url = config.get_complete_url(
            api_base=custom_base,
            api_key="test-key",
            model="text-embedding-ada-002",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/embeddings"

    def test_get_complete_url_already_has_embeddings(self):
        """Test API base URL that already ends with /embeddings"""
        config = LodashEmbeddingConfig()
        custom_base = "https://custom.api.com/v1/embeddings"
        url = config.get_complete_url(
            api_base=custom_base,
            api_key="test-key",
            model="text-embedding-ada-002",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/embeddings"

    def test_get_supported_openai_params(self):
        """Test supported OpenAI parameters"""
        config = LodashEmbeddingConfig()
        params = config.get_supported_openai_params("text-embedding-ada-002")
        expected_params = ["encoding_format", "dimensions"]
        assert params == expected_params

    def test_map_openai_params(self):
        """Test mapping of OpenAI parameters"""
        config = LodashEmbeddingConfig()
        non_default_params = {
            "encoding_format": "float",
            "dimensions": 1536,
            "user": "test-user",  # Not supported
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="text-embedding-ada-002",
            drop_params=False,
        )
        
        assert result["encoding_format"] == "float"
        assert result["dimensions"] == 1536
        assert "user" not in result

    @patch("litellm.llms.lodash.embedding.transformation.get_secret_str")
    def test_validate_environment_with_env_var(self, mock_get_secret):
        """Test environment validation with API key from environment"""
        # Configure mock to return the API key for the first call
        mock_get_secret.return_value = "env-api-key"
        
        config = LodashEmbeddingConfig()
        headers = config.validate_environment(
            headers={},
            model="text-embedding-ada-002",
            messages=[],
            optional_params={},
            litellm_params={},
        )
        
        assert headers["x-token"] == "env-api-key"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_with_direct_key(self):
        """Test environment validation with direct API key"""
        config = LodashEmbeddingConfig()
        headers = config.validate_environment(
            headers={},
            model="text-embedding-ada-002",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="direct-api-key",
        )
        
        assert headers["x-token"] == "direct-api-key"
        assert headers["Content-Type"] == "application/json"

    def test_transform_embedding_request(self):
        """Test request transformation"""
        config = LodashEmbeddingConfig()
        input_data = ["Hello world", "How are you?"]
        model = "lodash/all-MiniLM-L6-v2"
        optional_params = {
            "encoding_format": "float",
            "dimensions": 1536,
        }
        
        request_data = config.transform_embedding_request(
            model=model,
            input=input_data,
            optional_params=optional_params,
            headers={},
        )
        
        assert request_data["input"] == input_data
        assert request_data["model"] == "all-MiniLM-L6-v2"  # Should have prefix removed
        assert request_data["encoding_format"] == "float"
        assert request_data["dimensions"] == 1536

    def test_transform_embedding_response_success(self):
        """Test successful response transformation"""
        config = LodashEmbeddingConfig()
        
        # Mock Lodash API response format
        response_data = {
            "data": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        }
        
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        
        model_response = EmbeddingResponse()
        logging_obj = Mock(spec=Logging)
        
        result = config.transform_embedding_response(
            model="text-embedding-ada-002",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key="test-key",
            request_data={},
            optional_params={},
            litellm_params={},
        )
        
        assert result.object == "list"
        assert len(result.data) == 2
        assert result.data[0]["object"] == "embedding"
        assert result.data[0]["index"] == 0
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result.data[1]["object"] == "embedding"
        assert result.data[1]["index"] == 1
        assert result.data[1]["embedding"] == [0.4, 0.5, 0.6]
        assert result.model == "text-embedding-ada-002"
        assert result.usage.prompt_tokens == 10  # 2 inputs * 5 tokens each
        assert result.usage.total_tokens == 10

    def test_transform_embedding_response_error(self):
        """Test error response transformation"""
        config = LodashEmbeddingConfig()
        
        error_response = {
            "error": {
                "message": "Invalid API key",
                "type": "authentication_error",
            }
        }
        
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.json.return_value = error_response
        
        model_response = EmbeddingResponse()
        logging_obj = Mock(spec=Logging)
        
        with pytest.raises(LodashError) as exc_info:
            config.transform_embedding_response(
                model="text-embedding-ada-002",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key="test-key",
                request_data={},
                optional_params={},
                litellm_params={},
            )
        
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.message

    def test_transform_embedding_response_json_decode_error(self):
        """Test JSON decode error handling"""
        config = LodashEmbeddingConfig()
        
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        
        model_response = EmbeddingResponse()
        logging_obj = Mock(spec=Logging)
        
        with pytest.raises(LodashError) as exc_info:
            config.transform_embedding_response(
                model="text-embedding-ada-002",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key="test-key",
                request_data={},
                optional_params={},
                litellm_params={},
            )
        
        assert exc_info.value.status_code == 200
        assert "Failed to decode response" in exc_info.value.message

    def test_get_error_class(self):
        """Test error class creation"""
        config = LodashEmbeddingConfig()
        error = config.get_error_class(
            error_message="Test error",
            status_code=500,
            headers={"Content-Type": "application/json"},
        )
        
        assert isinstance(error, LodashError)
        assert error.status_code == 500
        assert error.message == "Test error"


class TestLodashError:
    def test_lodash_error_init(self):
        """Test LodashError initialization"""
        error = LodashError(
            status_code=400,
            message="Bad request",
            headers={"Content-Type": "application/json"},
        )
        
        assert error.status_code == 400
        assert error.message == "Bad request"
        assert error.request.method == "POST"
        assert "embeddings" in str(error.request.url)
        assert error.response.status_code == 400


class TestLodashProviderDetection:
    def test_get_llm_provider_lodash_models(self):
        """Test that lodash models are detected correctly"""
        lodash_models = [
            "lodash/all-MiniLM-L6-v2"
        ]
        
        for model in lodash_models:
            model_name, provider, _, _ = get_llm_provider(model)
            assert provider == "lodash"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 