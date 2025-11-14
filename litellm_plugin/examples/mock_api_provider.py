"""
Example mock API provider that simulates an external LLM API.

This demonstrates how to implement a provider that would interact with
an external API endpoint.
"""

import time
from typing import Any, Optional

try:
    from litellm.types.utils import (
        Choices,
        Message,
        ModelResponse,
        Usage,
    )
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    ModelResponse = dict
    Choices = dict
    Message = dict
    Usage = dict

from litellm_plugin.base import BaseCustomProvider


class MockAPIProvider(BaseCustomProvider):
    """
    A mock API provider that simulates calling an external LLM API.

    This provider demonstrates:
    - API key handling
    - Custom API base URLs
    - Error handling
    - Response formatting

    Example:
        import litellm
        from litellm_plugin.examples.mock_api_provider import MockAPIProvider
        from litellm_plugin.registry import register_provider, initialize_plugins

        # Register and initialize
        register_provider("mock_api", MockAPIProvider)
        initialize_plugins()

        # Use the provider
        response = litellm.completion(
            model="mock_api/gpt-mock",
            messages=[{"role": "user", "content": "Hello!"}],
            api_key="your-api-key",  # Optional
            api_base="https://api.example.com/v1",  # Optional
        )
    """

    def __init__(self, default_api_key: Optional[str] = None):
        """
        Initialize the mock API provider.

        Args:
            default_api_key: Optional default API key
        """
        super().__init__()
        self.provider_name = "mock_api"
        self.default_api_key = default_api_key

    def completion(
        self,
        model: str,
        messages: list,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Simulate an API completion call.

        Args:
            model: The model identifier
            messages: List of message dicts
            api_base: Optional API base URL
            api_key: Optional API key
            custom_llm_provider: Optional provider identifier
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            ModelResponse with simulated API response

        Raises:
            ValueError: If API key is not provided
        """
        # Check for API key
        used_api_key = api_key or self.default_api_key
        if not used_api_key:
            raise ValueError(
                "API key is required. Provide it via api_key parameter or "
                "set it in the provider initialization."
            )

        # Extract the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        # Simulate API processing
        # In a real implementation, you would:
        # 1. Make HTTP request to api_base
        # 2. Include api_key in headers
        # 3. Send messages in the required format
        # 4. Parse the response

        # Extract parameters
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 100)

        # Simulate response based on parameters
        response_content = self._generate_mock_response(
            user_message,
            temperature,
            max_tokens,
        )

        if not LITELLM_AVAILABLE:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response_content,
                        }
                    }
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": len(user_message.split()) + len(response_content.split()),
                },
            }

        # Create proper LiteLLM response
        response = ModelResponse()
        response.model = model
        response.created = int(time.time())

        # Create the message
        message = Message(
            role="assistant",
            content=response_content,
        )

        # Create choice
        choice = Choices(
            index=0,
            message=message,
            finish_reason="stop",
        )

        response.choices = [choice]

        # Add usage information
        prompt_tokens = len(user_message.split())
        completion_tokens = len(response_content.split())

        response.usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        return response

    def _generate_mock_response(
        self,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Generate a mock response.

        Args:
            user_message: The user's input
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate

        Returns:
            Mock response string
        """
        # Simple mock response generation
        base_response = (
            f"This is a mock API response to: '{user_message}'. "
            f"Temperature: {temperature}, Max tokens: {max_tokens}."
        )

        # Truncate if needed (simple word-based truncation)
        words = base_response.split()
        if len(words) > max_tokens:
            return " ".join(words[:max_tokens]) + "..."

        return base_response
