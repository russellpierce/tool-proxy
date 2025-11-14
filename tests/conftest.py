"""Pytest configuration and shared fixtures for LiteLLM plugin tests."""

import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock

import pytest


# Mock LiteLLM types when not available
try:
    from litellm.types.utils import (
        ModelResponse,
        Usage,
        Choices,
        Message,
        StreamingChoices,
        Delta,
    )
    from litellm import CustomLLM
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

    # Create mock types for testing
    class ModelResponse:
        def __init__(self):
            self.id = ""
            self.choices = []
            self.created = 0
            self.model = ""
            self.object = ""
            self.usage = None

    class Usage:
        def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens

    class Message:
        def __init__(self, content="", role="assistant"):
            self.content = content
            self.role = role

    class Choices:
        def __init__(self, finish_reason="stop", index=0, message=None):
            self.finish_reason = finish_reason
            self.index = index
            self.message = message or Message()

    class Delta:
        def __init__(self, content="", role="assistant"):
            self.content = content
            self.role = role

    class StreamingChoices:
        def __init__(self, finish_reason=None, index=0, delta=None):
            self.finish_reason = finish_reason
            self.index = index
            self.delta = delta or Delta()

    CustomLLM = object


@pytest.fixture
def mock_model_response():
    """Create a mock ModelResponse object."""
    response = ModelResponse()
    response.id = "chatcmpl-test123"
    response.created = int(time.time())
    response.model = "test-model"
    response.object = "chat.completion"
    response.usage = Usage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )
    response.choices = [
        Choices(
            finish_reason="stop",
            index=0,
            message=Message(
                content="This is a test response",
                role="assistant"
            )
        )
    ]
    return response


@pytest.fixture
def mock_streaming_response():
    """Create a mock streaming response chunk."""
    response = ModelResponse()
    response.id = "chatcmpl-test123"
    response.created = int(time.time())
    response.model = "test-model"
    response.object = "chat.completion.chunk"
    response.choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(content="test ", role="assistant")
        )
    ]
    return response


@pytest.fixture
def mock_messages():
    """Create sample chat messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]


@pytest.fixture
def mock_completion_kwargs():
    """Create sample completion kwargs."""
    return {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Test message"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False
    }


@pytest.fixture
def mock_user_api_key_dict():
    """Create sample user API key dictionary for hooks."""
    return {
        "api_key": "test-key-123",
        "user_id": "user-123",
        "team_id": "team-456",
        "key_name": "test-key",
        "metadata": {
            "environment": "test"
        },
        "permissions": {
            "models": ["gpt-3.5-turbo", "gpt-4"]
        }
    }


@pytest.fixture
def mock_cache():
    """Create a mock cache object."""
    cache = MagicMock()
    cache.get_cache = AsyncMock(return_value=None)
    cache.set_cache = AsyncMock()
    cache.async_get_cache = AsyncMock(return_value=None)
    cache.async_set_cache = AsyncMock()
    return cache


@pytest.fixture
def mock_litellm_call_data():
    """Create sample data dict for pre_call_hook."""
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Test"}],
        "temperature": 0.7,
        "max_tokens": 100,
        "litellm_call_id": "call-123",
        "metadata": {}
    }


@pytest.fixture
def sample_embedding_response():
    """Create a sample embedding response."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3, 0.4] * 192,  # 768 dimensions
                "index": 0
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }


@pytest.fixture
def sample_image_generation_response():
    """Create a sample image generation response."""
    return {
        "created": int(time.time()),
        "data": [
            {
                "url": "https://example.com/image.png",
                "b64_json": None
            }
        ]
    }


@pytest.fixture
def mock_exception():
    """Create a mock exception for failure testing."""
    return Exception("Test error message")


@pytest.fixture
def mock_request_headers():
    """Create mock request headers."""
    return {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
        "X-LiteLLM-Call-ID": "call-123"
    }


@pytest.fixture
async def async_generator_helper():
    """Helper to create async generators for testing streaming."""
    async def _create_async_gen(items: List[Any]):
        for item in items:
            yield item
    return _create_async_gen


@pytest.fixture
def litellm_available():
    """Check if LiteLLM is available."""
    return LITELLM_AVAILABLE


# Markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "requires_litellm: mark test as requiring LiteLLM installation"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require LiteLLM if it's not installed."""
    if not LITELLM_AVAILABLE:
        skip_litellm = pytest.mark.skip(reason="LiteLLM not installed")
        for item in items:
            if "requires_litellm" in item.keywords:
                item.add_marker(skip_litellm)
