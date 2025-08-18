# Lodash AI

LiteLLM supports all Lodash AI embedding models via the `/embeddings` endpoint.

## Quick Start

```python
import litellm

# Set API Key
import os
os.environ["LODASH_API_KEY"] = "your-api-key"

# Basic embedding call
response = litellm.embedding(
    model="lodash/all-MiniLM-L6-v2",
    input=["Hello world", "How are you?"]
)

print(response)
```

## Supported Models

| Model Name | Context Window | Input Cost (per 1M tokens) |
|------------|----------------|----------------------------|
| lodash/all-MiniLM-L6-v2 | 8,191 | $0.13 |

## Authentication

### Environment Variables
Set your Lodash AI API key and optionally the API base URL:

```bash
export LODASH_API_KEY="your-api-key"
export LODASH_API_BASE="https://api.lodash.ai/v1"  # Optional
```

**Supported Environment Variables:**
- `LODASH_API_KEY` / `LODASH_AI_API_KEY` / `LODASH_AI_TOKEN` - API authentication key
- `LODASH_API_BASE` - Primary API base URL (highest priority)
- `LODASH_BASE_URL` - Alternative base URL
- `LODASH_API_URL` - Additional base URL option

### Direct Parameter
You can also pass the API key directly in your code:

```python
import litellm

response = litellm.embedding(
    model="lodash/all-MiniLM-L6-v2",
    input=["Hello world"],
    api_key="your-api-key"
)
```

## Usage Examples

### Basic Embedding

```python
import litellm
import os

os.environ["LODASH_API_KEY"] = "your-api-key"

response = litellm.embedding(
    model="lodash/all-MiniLM-L6-v2",
    input="The quick brown fox jumps over the lazy dog"
)

print(f"Embedding: {response.data[0]['embedding']}")
print(f"Usage: {response.usage}")
```

### Multiple Texts

```python
import litellm
import os

os.environ["LODASH_API_KEY"] = "your-api-key"

texts = [
    "Machine learning is fascinating",
    "Python is a versatile programming language",
    "Data science involves statistical analysis"
]

response = litellm.embedding(
    model="lodash/all-MiniLM-L6-v2",
    input=texts
)

for i, embedding_data in enumerate(response.data):
    print(f"Text {i+1} embedding: {embedding_data['embedding'][:5]}...")
```

### With Custom Dimensions

```python
import litellm
import os

os.environ["LODASH_API_KEY"] = "your-api-key"

response = litellm.embedding(
    model="lodash/all-MiniLM-L6-v2",
    input="Hello world",
    dimensions=512  # Reduce dimensions for faster processing
)

print(f"Embedding dimensions: {len(response.data[0]['embedding'])}")
```

### Async Usage

```python
import litellm
import asyncio
import os

os.environ["LODASH_API_KEY"] = "your-api-key"

async def get_embeddings():
    response = await litellm.aembedding(
        model="lodash/all-MiniLM-L6-v2",
        input=["Async embedding example"]
    )
    return response

# Run async function
response = asyncio.run(get_embeddings())
print(response)
```

## Supported Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input` | string or array | Input text(s) to embed | Required |
| `model` | string | Model identifier | Required |
| `encoding_format` | string | Format for embeddings (float/base64) | "float" |
| `dimensions` | integer | Number of dimensions in embeddings | Model default |
| `api_key` | string | Lodash AI API key | Environment variable |
| `api_base` | string | Base URL for API calls | Environment variable or default |

## LiteLLM Proxy Server

You can use Lodash AI with the LiteLLM Proxy Server for centralized access control and logging.

### Configuration

Create a `config.yaml` file:

```yaml
model_list:
  - model_name: lodash-ada-002
    litellm_params:
      model: lodash/all-MiniLM-L6-v2
      api_key: os.environ/LODASH_API_KEY
  
  - model_name: lodash-3-small
    litellm_params:
      model: lodash/clip-ViT-B-32-multilingual-v1
      api_key: os.environ/LODASH_API_KEY

general_settings:
  master_key: your-master-key
```

### Start the Proxy

```bash
export LODASH_API_KEY="your-api-key"
export LODASH_API_BASE="https://api.lodash.ai/v1"  # Optional: custom API base
litellm --config config.yaml
```

### Use the Proxy

```python
import openai

client = openai.OpenAI(
    api_key="your-master-key",
    base_url="http://localhost:4000"
)

response = client.embeddings.create(
    model="lodash-ada-002",
    input="Hello from proxy!"
)

print(response)
```

## Error Handling

The Lodash provider raises `LodashError` for API-related errors:

```python
import litellm
from litellm.llms.lodash.embedding.transformation import LodashError

try:
    response = litellm.embedding(
        model="lodash/all-MiniLM-L6-v2",
        input="Hello world",
        api_key="invalid-key"
    )
except LodashError as e:
    print(f"Lodash API Error: {e.message}")
    print(f"Status Code: {e.status_code}")
except Exception as e:
    print(f"Other error: {e}")
```

## Custom API Base

You can configure a custom Lodash AI API base URL in several ways:

### Method 1: Environment Variable (Recommended)

```bash
export LODASH_API_BASE="https://your-custom-domain.com/api/v1"
export LODASH_API_KEY="your-api-key"
```

```python
import litellm

# Uses environment variable automatically
response = litellm.embedding(
    model="lodash/all-MiniLM-L6-v2",
    input="Hello world"
)
```

### Method 2: Direct Parameter

```python
import litellm

response = litellm.embedding(
    model="lodash/all-MiniLM-L6-v2",
    input="Hello world",
    api_base="https://your-custom-domain.com/api/v1",
    api_key="your-api-key"
)
```

### Environment Variable Priority

LiteLLM checks environment variables in this order:
1. `LODASH_API_BASE` (highest priority)
2. `LODASH_BASE_URL`
3. `LODASH_API_URL`
4. Default fallback URL

### URL Handling

- URLs are automatically appended with `/embeddings` if not present
- Both trailing slash and non-trailing slash formats are supported
- Custom endpoints like `/embeddings/local` are preserved

## Rate Limits and Best Practices

1. **Batch Processing**: Process multiple texts in a single request when possible
2. **Error Handling**: Always implement proper error handling for production use
3. **API Key Security**: Store API keys securely using environment variables
4. **Environment Configuration**: Use environment variables for API base URLs in different environments
5. **Monitoring**: Use LiteLLM's logging features to monitor usage and costs

## Troubleshooting

### Common Issues

1. **Authentication Error (401)**
   - Verify your API key is correct
   - Ensure the environment variable is set properly

2. **Invalid Model Error**
   - Check that you're using a supported model name
   - Ensure the model name includes the "lodash/" prefix

3. **Connection/URL Errors**
   - Verify `LODASH_API_BASE` environment variable is set correctly
   - Check if the API endpoint is accessible
   - Ensure URL format is correct (with or without trailing slash)

4. **Rate Limiting (429)**
   - Implement exponential backoff
   - Reduce request frequency

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import litellm

litellm.set_verbose = True

response = litellm.embedding(
    model="lodash/all-MiniLM-L6-v2",
    input="Debug example"
)
```

### Environment-Specific Configuration

**Development Environment:**
```bash
export LODASH_API_BASE="http://localhost:8000/api/v1/app2/gateway"
export LODASH_API_KEY="dev-api-key"
```

**Staging Environment:**
```bash
export LODASH_API_BASE="https://staging-api.lodash.ai/v1"
export LODASH_API_KEY="staging-api-key"
```

**Production Environment:**
```bash
export LODASH_API_BASE="https://api.lodash.ai/v1"
export LODASH_API_KEY="prod-api-key"
```

## Support

For issues specific to the Lodash AI integration in LiteLLM:
- [LiteLLM Issues](https://github.com/BerriAI/litellm/issues)

For Lodash AI API issues:
- Contact Lodash AI support directly 