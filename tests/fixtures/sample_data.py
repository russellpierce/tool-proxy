"""Sample data for testing LiteLLM plugins."""

from typing import Dict, List, Any


# Sample chat messages
SAMPLE_MESSAGES: List[Dict[str, str]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
]

SAMPLE_SINGLE_MESSAGE: List[Dict[str, str]] = [
    {"role": "user", "content": "What is the weather today?"}
]

SAMPLE_LONG_CONVERSATION: List[Dict[str, str]] = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I write a function in Python?"},
    {"role": "assistant", "content": "Here's how to write a function in Python:\n\n```python\ndef my_function():\n    pass\n```"},
    {"role": "user", "content": "Can you add parameters?"},
]

# Sample completion parameters
SAMPLE_COMPLETION_PARAMS: Dict[str, Any] = {
    "model": "gpt-3.5-turbo",
    "messages": SAMPLE_MESSAGES,
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": False
}

SAMPLE_STREAMING_PARAMS: Dict[str, Any] = {
    "model": "gpt-3.5-turbo",
    "messages": SAMPLE_MESSAGES,
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": True
}

# Sample user API key dictionaries
SAMPLE_USER_API_KEY_DICT: Dict[str, Any] = {
    "api_key": "sk-test123456",
    "user_id": "user-abc123",
    "team_id": "team-xyz789",
    "key_name": "development-key",
    "max_parallel_requests": 10,
    "metadata": {
        "environment": "development",
        "department": "engineering"
    },
    "permissions": {
        "models": ["gpt-3.5-turbo", "gpt-4"],
        "endpoints": ["/chat/completions", "/embeddings"]
    }
}

SAMPLE_ADMIN_KEY_DICT: Dict[str, Any] = {
    "api_key": "sk-admin-key",
    "user_id": "admin-user",
    "team_id": "admin-team",
    "key_name": "admin-key",
    "max_parallel_requests": 100,
    "metadata": {
        "environment": "production",
        "role": "admin"
    },
    "permissions": {
        "models": ["*"],
        "endpoints": ["*"]
    }
}

# Sample call data for hooks
SAMPLE_CALL_DATA: Dict[str, Any] = {
    "model": "gpt-3.5-turbo",
    "messages": SAMPLE_MESSAGES,
    "temperature": 0.7,
    "max_tokens": 100,
    "litellm_call_id": "call-abc123",
    "metadata": {
        "user_api_key": "sk-test123456",
        "user_id": "user-abc123"
    }
}

# Sample embedding parameters
SAMPLE_EMBEDDING_PARAMS: Dict[str, Any] = {
    "model": "text-embedding-ada-002",
    "input": "Hello world",
}

SAMPLE_EMBEDDING_BATCH_PARAMS: Dict[str, Any] = {
    "model": "text-embedding-ada-002",
    "input": ["Hello world", "How are you?", "Testing embeddings"],
}

# Sample image generation parameters
SAMPLE_IMAGE_GEN_PARAMS: Dict[str, Any] = {
    "model": "dall-e-3",
    "prompt": "A cute cat playing with a ball of yarn",
    "n": 1,
    "size": "1024x1024"
}

# Sample response content
SAMPLE_ASSISTANT_RESPONSES: List[str] = [
    "Hello! I'm doing well, thank you for asking. How can I help you today?",
    "I'm an AI assistant, so I don't have feelings, but I'm functioning properly and ready to assist you!",
    "I'm here and ready to help! What would you like to know?",
]

# Sample streaming chunks
SAMPLE_STREAMING_CHUNKS: List[str] = [
    "Hello",
    "!",
    " I'm",
    " doing",
    " well",
    ",",
    " thank",
    " you",
    " for",
    " asking",
    "."
]

# Sample error scenarios
SAMPLE_ERROR_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "rate_limit": {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded"
        },
        "status_code": 429
    },
    "invalid_api_key": {
        "error": {
            "message": "Invalid API key",
            "type": "invalid_request_error",
            "code": "invalid_api_key"
        },
        "status_code": 401
    },
    "model_not_found": {
        "error": {
            "message": "Model not found",
            "type": "invalid_request_error",
            "code": "model_not_found"
        },
        "status_code": 404
    },
    "context_length_exceeded": {
        "error": {
            "message": "Maximum context length exceeded",
            "type": "invalid_request_error",
            "code": "context_length_exceeded"
        },
        "status_code": 400
    }
}

# Sample cache keys and values
SAMPLE_CACHE_KEYS: List[str] = [
    "litellm:sk-test123:completion",
    "litellm:sk-test123:embedding",
    "litellm:user-abc123:rate_limit",
]

SAMPLE_CACHE_VALUES: Dict[str, Any] = {
    "completion": {
        "response": "Cached response",
        "timestamp": 1234567890,
        "ttl": 60
    },
    "rate_limit": {
        "count": 5,
        "reset_at": 1234567950
    }
}

# Call types
CALL_TYPES: List[str] = [
    "completion",
    "text_completion",
    "embeddings",
    "image_generation",
    "moderation",
    "audio_transcription"
]
