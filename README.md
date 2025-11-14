# LiteLLM Plugin Structure

A comprehensive plugin structure for extending [LiteLLM](https://github.com/BerriAI/litellm) with custom providers and request/response interceptors.

## Features

- **Custom Providers**: Create custom LLM providers that integrate with LiteLLM's unified API
- **Request/Response Interception**: Modify requests before sending and responses before returning
- **Plugin Registry**: Easy registration and management of custom providers
- **Type-Safe**: Full type hints and IDE support
- **Examples Included**: Working examples for both providers and loggers

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/tool-proxy.git
cd tool-proxy

# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/tool-proxy.git
cd tool-proxy

# Install the package
pip install -e .
```

## Quick Start

### 1. Creating a Custom Provider

Custom providers allow you to add support for new LLM APIs or create mock providers for testing.

```python
from litellm_plugin.base import BaseCustomProvider
from litellm.types.utils import ModelResponse, Message, Choices, Usage
import time

class MyCustomProvider(BaseCustomProvider):
    def completion(self, model, messages, **kwargs):
        # Extract user message
        user_message = messages[-1]["content"]

        # Create response
        response = ModelResponse()
        response.model = model
        response.created = int(time.time())

        response.choices = [Choices(
            index=0,
            message=Message(
                role="assistant",
                content=f"Response to: {user_message}"
            ),
            finish_reason="stop"
        )]

        response.usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )

        return response
```

**Register and use:**

```python
import litellm
from litellm_plugin.registry import register_provider, initialize_plugins

# Register your provider
register_provider("my_custom", MyCustomProvider)

# Initialize all registered providers
initialize_plugins()

# Use your provider
response = litellm.completion(
    model="my_custom/my-model",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### 2. Creating a Custom Logger (Request/Response Interceptor)

Custom loggers allow you to intercept and modify requests/responses, implement content filtering, or add observability.

```python
from litellm_plugin.logger import BaseCustomLogger
from litellm.types.utils import ModelResponse

class ResponseModifier(BaseCustomLogger):
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        # Modify the response before it's returned to the user
        if hasattr(response, "choices") and len(response.choices) > 0:
            original = response.choices[0].message.content
            response.choices[0].message.content = f"[Modified] {original}"
        return response

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Log successful completions
        duration = end_time - start_time
        print(f"Request completed in {duration:.2f}s")
```

**Register and use:**

```python
import litellm

# Register the logger
litellm.callbacks = [ResponseModifier()]

# Use LiteLLM normally - responses will be automatically modified
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Response will have "[Modified] " prefix
print(response.choices[0].message.content)
```

## Project Structure

```
tool-proxy/
├── litellm_plugin/
│   ├── __init__.py           # Package initialization
│   ├── base.py               # BaseCustomProvider class
│   ├── logger.py             # BaseCustomLogger class
│   ├── registry.py           # Plugin registry system
│   └── examples/
│       ├── __init__.py
│       ├── echo_provider.py      # Example: Echo provider
│       ├── mock_api_provider.py  # Example: Mock API provider
│       └── response_modifier.py  # Example: Response interceptors
├── pyproject.toml           # Project configuration (uv/pip)
├── uv.lock                  # Lock file for reproducible installs
├── example_usage.py         # Usage examples
└── README.md                # This file
```

## Architecture

### Two Extension Approaches

This plugin structure supports two ways to extend LiteLLM:

#### 1. Custom Providers (`BaseCustomProvider`)

Custom providers add support for new LLM APIs or create mock/test providers.

**Use cases:**
- Integrate proprietary LLM APIs
- Create mock providers for testing
- Add support for local models
- Implement custom routing logic

**Key methods:**
- `completion()` - Synchronous text completion
- `acompletion()` - Async text completion
- `streaming()` - Synchronous streaming
- `astreaming()` - Async streaming
- `embedding()` - Text embeddings
- `image_generation()` - Image generation

#### 2. Custom Loggers (`BaseCustomLogger`)

Custom loggers intercept and modify requests/responses in the LiteLLM pipeline.

**Use cases:**
- Modify requests before sending to LLMs
- Filter or validate response content
- Add observability/monitoring
- Implement content safety filters
- Track usage and costs

**Key hooks:**
- `async_pre_call_hook()` - Modify requests before sending
- `async_post_call_success_hook()` - Modify responses (non-streaming only)
- `async_log_success_event()` - Log successful completions
- `async_log_failure_event()` - Log failed completions

**Important:** Response modification only works for non-streaming responses. For streaming, you can observe but not modify.

## Examples

### Example 1: Echo Provider

A simple provider that echoes back user messages:

```python
from litellm_plugin.examples.echo_provider import EchoProvider
from litellm_plugin.registry import register_provider, initialize_plugins
import litellm

register_provider("echo", EchoProvider)
initialize_plugins()

response = litellm.completion(
    model="echo/test",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)  # Output: "Echo: Hello!"
```

### Example 2: Mock API Provider

A provider that simulates an external API:

```python
from litellm_plugin.examples.mock_api_provider import MockAPIProvider
from litellm_plugin.registry import register_provider, initialize_plugins
import litellm

register_provider("mock_api", MockAPIProvider)
initialize_plugins()

response = litellm.completion(
    model="mock_api/gpt-mock",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="test-key",
    temperature=0.7,
    max_tokens=50
)

print(response.choices[0].message.content)
```

### Example 3: Response Modifier

Add a prefix to all LLM responses:

```python
from litellm_plugin.examples.response_modifier import ResponseModifier
import litellm

litellm.callbacks = [ResponseModifier(prefix="[Verified] ")]

response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

print(response.choices[0].message.content)  # Starts with "[Verified] "
```

### Example 4: Content Filter

Filter blocked words from responses:

```python
from litellm_plugin.examples.response_modifier import ContentFilter
import litellm

litellm.callbacks = [ContentFilter(blocked_words=["spam", "advertisement"])]

response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Tell me about products"}]
)

# Any blocked words will be replaced with [FILTERED]
print(response.choices[0].message.content)
```

### Example 5: Request Logger

Log all requests and responses:

```python
from litellm_plugin.examples.response_modifier import RequestLogger
import litellm

litellm.callbacks = [RequestLogger(verbose=True)]

# All requests will be logged with timing and token usage
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Development

### Setup Development Environment

```bash
# Install all dependencies including dev dependencies
uv sync --all-extras

# Or install with specific groups
uv sync --extra dev
```

### Development Commands

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=litellm_plugin --cov-report=html

# Format code
uv run ruff format litellm_plugin/

# Lint code
uv run ruff check litellm_plugin/

# Fix linting issues automatically
uv run ruff check --fix litellm_plugin/

# Type checking
uv run mypy litellm_plugin/

# Run the example script
uv run python example_usage.py
```

### Creating a New Provider

1. Create a new file in `litellm_plugin/examples/` or your own module
2. Inherit from `BaseCustomProvider`
3. Implement the required methods (at minimum `completion()`)
4. Register your provider with the registry
5. Initialize and use

### Creating a New Logger

1. Create a new file in `litellm_plugin/examples/` or your own module
2. Inherit from `BaseCustomLogger`
3. Implement the hooks you need
4. Add to `litellm.callbacks` list
5. Use LiteLLM normally

## Configuration with LiteLLM Proxy

To use custom loggers with the LiteLLM proxy server, add them to your config file:

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
      api_key: your-api-key

litellm_settings:
  callbacks: ["litellm_plugin.examples.response_modifier.ResponseModifier"]
```

## API Reference

### BaseCustomProvider

Base class for custom LLM providers.

**Methods:**
- `completion(model, messages, **kwargs) -> ModelResponse`
- `streaming(model, messages, **kwargs) -> Iterator[GenericStreamingChunk]`
- `acompletion(model, messages, **kwargs) -> ModelResponse`
- `astreaming(model, messages, **kwargs) -> AsyncIterator[GenericStreamingChunk]`
- `embedding(model, input, **kwargs) -> EmbeddingResponse`
- `aembedding(model, input, **kwargs) -> EmbeddingResponse`
- `image_generation(model, prompt, **kwargs) -> ImageResponse`
- `aimage_generation(model, prompt, **kwargs) -> ImageResponse`

### BaseCustomLogger

Base class for request/response interceptors.

**Hooks:**
- `async_pre_call_hook(user_api_key_dict, cache, data, call_type) -> Dict`
- `async_post_call_success_hook(data, user_api_key_dict, response) -> ModelResponse`
- `async_log_success_event(kwargs, response_obj, start_time, end_time) -> None`
- `async_log_failure_event(kwargs, response_obj, start_time, end_time) -> None`
- `async_post_call_streaming_hook(user_api_key_dict, response) -> Any`

### PluginRegistry

Registry for managing custom providers.

**Methods:**
- `register(provider_name, provider_class)` - Register a provider
- `unregister(provider_name)` - Unregister a provider
- `get_provider(provider_name)` - Get provider instance
- `list_providers()` - List all registered providers
- `initialize()` - Initialize all providers with LiteLLM
- `initialize_provider(provider_name)` - Initialize specific provider

**Convenience functions:**
- `register_provider(name, class)` - Register with global registry
- `get_provider(name)` - Get from global registry
- `initialize_plugins()` - Initialize all with LiteLLM
- `get_registry()` - Get global registry instance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Related Projects

- [LiteLLM](https://github.com/BerriAI/litellm) - The main LiteLLM project
- [LiteLLM Proxy](https://docs.litellm.ai/docs/proxy/quick_start) - LiteLLM proxy server

## Support

For issues and questions:
- Open an issue on GitHub
- Check the [LiteLLM documentation](https://docs.litellm.ai/)
- Review the examples in `litellm_plugin/examples/`