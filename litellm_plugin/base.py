"""
Base class for custom LiteLLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Coroutine, Iterator, Optional, Union

try:
    from litellm import CustomLLM
    from litellm.types.utils import (
        EmbeddingResponse,
        GenericStreamingChunk,
        ImageResponse,
        ModelResponse,
    )
except ImportError:
    # Fallback for when litellm is not installed
    CustomLLM = object
    ModelResponse = Any
    GenericStreamingChunk = Any
    ImageResponse = Any
    EmbeddingResponse = Any


class CustomStreamWrapper:
    """Wrapper for custom streaming responses."""
    pass


class BaseCustomProvider(CustomLLM if CustomLLM != object else ABC):
    """
    Base class for implementing custom LiteLLM providers.

    This class provides a template for creating custom providers that can be
    registered with LiteLLM. Override the methods you need for your use case.

    Example:
        class MyProvider(BaseCustomProvider):
            def completion(self, model: str, messages: list, **kwargs):
                # Your implementation here
                return ModelResponse(...)
    """

    def __init__(self) -> None:
        """Initialize the custom provider."""
        if CustomLLM != object:
            super().__init__()
        self.provider_name = self.__class__.__name__

    @abstractmethod
    def completion(
        self,
        model: str,
        messages: list,
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Synchronous text completion.

        Args:
            model: The model name/identifier
            messages: List of message dicts with 'role' and 'content'
            api_base: Optional API base URL
            custom_llm_provider: Optional provider identifier
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse or CustomStreamWrapper for streaming
        """
        raise NotImplementedError("completion() must be implemented")

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[GenericStreamingChunk]:
        """
        Synchronous streaming completion.

        Args:
            model: The model name/identifier
            messages: List of message dicts
            api_base: Optional API base URL
            custom_llm_provider: Optional provider identifier
            **kwargs: Additional parameters

        Yields:
            GenericStreamingChunk objects
        """
        raise NotImplementedError("streaming() not implemented for this provider")

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Asynchronous text completion.

        Args:
            model: The model name/identifier
            messages: List of message dicts
            api_base: Optional API base URL
            custom_llm_provider: Optional provider identifier
            **kwargs: Additional parameters

        Returns:
            ModelResponse or CustomStreamWrapper for streaming
        """
        raise NotImplementedError("acompletion() not implemented for this provider")

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenericStreamingChunk]:
        """
        Asynchronous streaming completion.

        Args:
            model: The model name/identifier
            messages: List of message dicts
            api_base: Optional API base URL
            custom_llm_provider: Optional provider identifier
            **kwargs: Additional parameters

        Yields:
            GenericStreamingChunk objects asynchronously
        """
        raise NotImplementedError("astreaming() not implemented for this provider")
        # Make this a proper async generator
        if False:
            yield

    def image_generation(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageResponse:
        """
        Synchronous image generation.

        Args:
            model: The model name/identifier
            prompt: Text prompt for image generation
            **kwargs: Additional parameters (size, quality, etc.)

        Returns:
            ImageResponse with generated image(s)
        """
        raise NotImplementedError("image_generation() not implemented for this provider")

    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> ImageResponse:
        """
        Asynchronous image generation.

        Args:
            model: The model name/identifier
            prompt: Text prompt for image generation
            **kwargs: Additional parameters

        Returns:
            ImageResponse with generated image(s)
        """
        raise NotImplementedError("aimage_generation() not implemented for this provider")

    def embedding(
        self,
        model: str,
        input: Union[str, list],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Synchronous text embedding generation.

        Args:
            model: The model name/identifier
            input: Text or list of texts to embed
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse with embeddings
        """
        raise NotImplementedError("embedding() not implemented for this provider")

    async def aembedding(
        self,
        model: str,
        input: Union[str, list],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Asynchronous text embedding generation.

        Args:
            model: The model name/identifier
            input: Text or list of texts to embed
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse with embeddings
        """
        raise NotImplementedError("aembedding() not implemented for this provider")
