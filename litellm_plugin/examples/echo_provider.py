"""
Example echo provider that returns user messages back.

This is a simple example demonstrating how to implement a custom LiteLLM provider.
"""

import time
from typing import Any, Optional, Union

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


class EchoProvider(BaseCustomProvider):
    """
    A simple echo provider that returns the user's message back.

    This provider is useful for testing and demonstrating the plugin structure.
    It implements only the basic completion method.

    Example:
        import litellm
        from litellm_plugin.examples.echo_provider import EchoProvider
        from litellm_plugin.registry import register_provider, initialize_plugins

        # Register and initialize
        register_provider("echo", EchoProvider)
        initialize_plugins()

        # Use the provider
        response = litellm.completion(
            model="echo/test",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)  # Output: "Echo: Hello!"
    """

    def __init__(self):
        """Initialize the echo provider."""
        super().__init__()
        self.provider_name = "echo"

    def completion(
        self,
        model: str,
        messages: list,
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Echo back the last user message.

        Args:
            model: The model identifier (ignored for echo)
            messages: List of message dicts
            api_base: Optional API base URL (ignored)
            custom_llm_provider: Optional provider identifier (ignored)
            **kwargs: Additional parameters (ignored)

        Returns:
            ModelResponse with echoed message
        """
        # Extract the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        # Create response content
        echo_content = f"Echo: {user_message}"

        if not LITELLM_AVAILABLE:
            # Return a simple dict if litellm is not available
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": echo_content,
                        }
                    }
                ],
                "model": model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        # Create proper LiteLLM response
        response = ModelResponse()
        response.model = model
        response.created = int(time.time())

        # Create the message
        message = Message(
            role="assistant",
            content=echo_content,
        )

        # Create choice
        choice = Choices(
            index=0,
            message=message,
            finish_reason="stop",
        )

        response.choices = [choice]

        # Add usage information
        response.usage = Usage(
            prompt_tokens=len(user_message.split()),
            completion_tokens=len(echo_content.split()),
            total_tokens=len(user_message.split()) + len(echo_content.split()),
        )

        return response
