"""
Example custom logger that modifies LLM responses.

This demonstrates how to use the CustomLogger approach to intercept
and modify requests/responses.
"""

from typing import Any, Dict

try:
    from litellm.types.utils import ModelResponse
    LITELLM_AVAILABLE = True
except ImportError:
    ModelResponse = Any
    LITELLM_AVAILABLE = False

from litellm_plugin.logger import BaseCustomLogger


class ResponseModifier(BaseCustomLogger):
    """
    Example logger that adds a prefix to all LLM responses.

    This demonstrates the async_post_call_success_hook for modifying
    responses before they're sent to users.

    Example:
        import litellm
        from litellm_plugin.examples.response_modifier import ResponseModifier

        # Register the logger
        litellm.callbacks = [ResponseModifier()]

        # Use LiteLLM normally
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        # Response will have "[Verified] " prefix
    """

    def __init__(self, prefix: str = "[Verified] "):
        """
        Initialize the response modifier.

        Args:
            prefix: Prefix to add to all responses
        """
        super().__init__()
        self.prefix = prefix
        print(f"ResponseModifier initialized with prefix: '{prefix}'")

    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Dict[str, Any],
        response: ModelResponse,
    ) -> ModelResponse:
        """
        Add prefix to response content.

        Args:
            data: Original request data
            user_api_key_dict: User API key information
            response: LLM response object

        Returns:
            Modified response with prefix
        """
        if hasattr(response, "choices") and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                original_content = choice.message.content
                if original_content:
                    choice.message.content = self.prefix + original_content

        return response

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: ModelResponse,
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Log successful completions.

        Args:
            kwargs: Request parameters
            response_obj: Response object
            start_time: Request start timestamp
            end_time: Request end timestamp
        """
        duration = end_time - start_time
        model = kwargs.get("model", "unknown")

        if hasattr(response_obj, "usage") and response_obj.usage:
            tokens = response_obj.usage.total_tokens
        else:
            tokens = "N/A"

        print(
            f"[ResponseModifier] {model} | "
            f"Duration: {duration:.2f}s | "
            f"Tokens: {tokens}"
        )


class ContentFilter(BaseCustomLogger):
    """
    Example logger that filters/validates response content.

    This demonstrates rejecting responses based on content rules.

    Example:
        import litellm
        from litellm_plugin.examples.response_modifier import ContentFilter

        # Register the logger with blocked words
        litellm.callbacks = [ContentFilter(blocked_words=["spam", "advertisement"])]

        # Use LiteLLM normally
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me about products"}]
        )
    """

    def __init__(self, blocked_words: list = None):
        """
        Initialize the content filter.

        Args:
            blocked_words: List of words to filter from responses
        """
        super().__init__()
        self.blocked_words = blocked_words or []
        print(f"ContentFilter initialized with {len(self.blocked_words)} blocked words")

    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Dict[str, Any],
        response: ModelResponse,
    ) -> ModelResponse:
        """
        Filter blocked words from response.

        Args:
            data: Original request data
            user_api_key_dict: User API key information
            response: LLM response object

        Returns:
            Filtered response
        """
        if hasattr(response, "choices") and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content
                if content:
                    # Filter blocked words
                    for word in self.blocked_words:
                        content = content.replace(word, "[FILTERED]")
                    choice.message.content = content

        return response


class RequestLogger(BaseCustomLogger):
    """
    Example logger that logs all requests and responses.

    This demonstrates using both pre and post hooks for complete observability.

    Example:
        import litellm
        from litellm_plugin.examples.response_modifier import RequestLogger

        # Register the logger
        litellm.callbacks = [RequestLogger()]

        # All requests will be logged
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the request logger.

        Args:
            verbose: Whether to log detailed information
        """
        super().__init__()
        self.verbose = verbose
        self.request_count = 0

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Dict[str, Any],
        cache: Any,
        data: Dict[str, Any],
        call_type: str,
    ) -> Dict[str, Any]:
        """
        Log request before sending to LLM.

        Args:
            user_api_key_dict: User API key information
            cache: Cache object
            data: Request data
            call_type: Type of call

        Returns:
            Original data (unmodified)
        """
        self.request_count += 1

        if self.verbose:
            model = data.get("model", "unknown")
            print(f"\n[RequestLogger] Request #{self.request_count}")
            print(f"  Type: {call_type}")
            print(f"  Model: {model}")

            if "messages" in data:
                print(f"  Messages: {len(data['messages'])}")

        return data

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: ModelResponse,
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Log successful completion.

        Args:
            kwargs: Request parameters
            response_obj: Response object
            start_time: Request start timestamp
            end_time: Request end timestamp
        """
        if self.verbose:
            duration = end_time - start_time
            print(f"[RequestLogger] ✓ Success in {duration:.2f}s")

            if hasattr(response_obj, "usage") and response_obj.usage:
                print(f"  Tokens: {response_obj.usage.total_tokens}")

    async def async_log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Log failed completion.

        Args:
            kwargs: Request parameters
            response_obj: Error object
            start_time: Request start timestamp
            end_time: Request end timestamp
        """
        duration = end_time - start_time
        print(f"[RequestLogger] ✗ Failed after {duration:.2f}s")
        print(f"  Error: {str(response_obj)}")
