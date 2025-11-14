"""
Custom logger for intercepting and modifying LiteLLM requests/responses.

This module provides a base class for creating custom loggers that can:
- Intercept requests before they're sent to LLMs
- Modify responses before they're sent to users
- Log success/failure events for observability
"""

from typing import Any, Dict, Optional

try:
    from litellm.integrations.custom_logger import CustomLogger as LiteLLMCustomLogger
    from litellm.types.utils import ModelResponse
    LITELLM_AVAILABLE = True
except ImportError:
    # Fallback when litellm is not installed
    LiteLLMCustomLogger = object
    ModelResponse = Any
    LITELLM_AVAILABLE = False


class BaseCustomLogger(LiteLLMCustomLogger if LiteLLMCustomLogger != object else object):
    """
    Base class for custom LiteLLM loggers.

    This class provides hooks for intercepting and modifying requests/responses:
    - async_pre_call_hook: Modify requests before sending to LLM
    - async_post_call_success_hook: Modify responses before sending to user
    - async_log_success_event: Log successful completions
    - async_log_failure_event: Log failed completions

    Example:
        class MyLogger(BaseCustomLogger):
            async def async_post_call_success_hook(self, data, user_api_key_dict, response):
                # Modify response
                response.choices[0].message.content = "Modified: " + response.choices[0].message.content
                return response
    """

    def __init__(self):
        """Initialize the custom logger."""
        if LITELLM_AVAILABLE and LiteLLMCustomLogger != object:
            super().__init__()
        self.logger_name = self.__class__.__name__

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Dict[str, Any],
        cache: Any,
        data: Dict[str, Any],
        call_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Hook called before sending request to LLM.

        Use this to:
        - Modify request parameters
        - Add/remove messages
        - Change model selection
        - Validate inputs

        Args:
            user_api_key_dict: User API key information
            cache: Cache object
            data: Request data (model, messages, etc.)
            call_type: Type of call (e.g., "completion", "embedding")

        Returns:
            Modified data dict or None to use original data

        Example:
            async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
                # Add a system message
                if "messages" in data:
                    data["messages"].insert(0, {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    })
                return data
        """
        return None

    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Dict[str, Any],
        response: ModelResponse,
    ) -> ModelResponse:
        """
        Hook called after successful LLM response (non-streaming only).

        Use this to:
        - Modify response content
        - Filter/validate outputs
        - Add custom headers
        - Reject responses

        Note: This only works for non-streaming responses. For streaming,
        you can observe but not modify the response.

        Args:
            data: Original request data
            user_api_key_dict: User API key information
            response: LLM response object

        Returns:
            Modified ModelResponse

        Example:
            async def async_post_call_success_hook(self, data, user_api_key_dict, response):
                # Add a prefix to the response
                content = response.choices[0].message.content
                response.choices[0].message.content = f"[Verified] {content}"
                return response
        """
        return response

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: ModelResponse,
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Hook called to log successful completions.

        Use this for:
        - Observability/monitoring
        - Usage tracking
        - Analytics
        - Debugging

        Args:
            kwargs: Request parameters
            response_obj: Response object
            start_time: Request start timestamp
            end_time: Request end timestamp

        Example:
            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                duration = end_time - start_time
                print(f"Request to {kwargs.get('model')} took {duration:.2f}s")
                print(f"Tokens used: {response_obj.usage.total_tokens}")
        """
        pass

    async def async_log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Hook called to log failed completions.

        Use this for:
        - Error tracking
        - Alerting
        - Debugging
        - Retry logic

        Args:
            kwargs: Request parameters
            response_obj: Error/exception object
            start_time: Request start timestamp
            end_time: Request end timestamp

        Example:
            async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
                print(f"Request failed: {response_obj}")
                print(f"Model: {kwargs.get('model')}")
                print(f"Error: {str(response_obj)}")
        """
        pass

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: Dict[str, Any],
        response: Any,
    ) -> Any:
        """
        Hook called for streaming responses.

        Note: You can observe streaming responses but cannot modify them
        as they are already being sent to the user.

        Args:
            user_api_key_dict: User API key information
            response: Streaming response object

        Returns:
            The response object (modifications won't affect output)

        Example:
            async def async_post_call_streaming_hook(self, user_api_key_dict, response):
                print("Streaming response started")
                return response
        """
        return response
