"""Tests for BaseCustomProvider."""

import pytest
from typing import Union, AsyncIterator
from unittest.mock import MagicMock

from litellm_plugin.base import BaseCustomProvider, CustomStreamWrapper

try:
    from litellm.types.utils import (
        ModelResponse,
        GenericStreamingChunk,
        ImageResponse,
        EmbeddingResponse,
    )
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    ModelResponse = object
    GenericStreamingChunk = object
    ImageResponse = object
    EmbeddingResponse = object


# Test implementation of BaseCustomProvider
class TestProvider(BaseCustomProvider):
    """Test provider implementation."""

    def completion(self, model: str, messages: list, **kwargs):
        """Implement required abstract method."""
        # Create a mock response
        if LITELLM_AVAILABLE:
            from tests.fixtures.mock_litellm import create_mock_response
            return create_mock_response(
                content=f"Response from {model}",
                model=model
            )
        else:
            return MagicMock()


class TestBaseCustomProvider:
    """Test suite for BaseCustomProvider."""

    def test_provider_initialization(self):
        """Test that provider initializes correctly."""
        provider = TestProvider()
        assert provider.provider_name == "TestProvider"

    def test_provider_name_matches_class_name(self):
        """Test provider_name is set to class name."""
        class CustomNamedProvider(BaseCustomProvider):
            def completion(self, model: str, messages: list, **kwargs):
                return MagicMock()

        provider = CustomNamedProvider()
        assert provider.provider_name == "CustomNamedProvider"

    def test_completion_abstract_method_must_be_implemented(self):
        """Test that completion must be implemented."""
        # When litellm is available, BaseCustomProvider inherits from CustomLLM
        # which doesn't enforce abstract methods the same way
        # So we check that calling completion raises NotImplementedError instead
        class IncompleteProvider(BaseCustomProvider):
            pass

        # Try to create instance - may or may not raise depending on litellm availability
        try:
            provider = IncompleteProvider()
            # If we can create it, calling completion should raise NotImplementedError
            with pytest.raises(NotImplementedError):
                provider.completion("test", [])
        except TypeError:
            # If we can't create it, that's also acceptable
            pass

    def test_completion_method_signature(self, mock_messages):
        """Test completion method can be called with correct signature."""
        provider = TestProvider()
        response = provider.completion(
            model="test-model",
            messages=mock_messages,
            api_base="https://api.example.com",
            custom_llm_provider="test",
            temperature=0.7
        )
        assert response is not None

    def test_completion_with_minimal_args(self, mock_messages):
        """Test completion with only required arguments."""
        provider = TestProvider()
        response = provider.completion(
            model="test-model",
            messages=mock_messages
        )
        assert response is not None

    def test_streaming_not_implemented_by_default(self, mock_messages):
        """Test that streaming raises NotImplementedError by default."""
        provider = TestProvider()
        with pytest.raises(NotImplementedError, match="streaming\\(\\) not implemented"):
            provider.streaming(
                model="test-model",
                messages=mock_messages
            )

    @pytest.mark.asyncio
    async def test_acompletion_not_implemented_by_default(self, mock_messages):
        """Test that acompletion raises NotImplementedError by default."""
        provider = TestProvider()
        with pytest.raises(NotImplementedError, match="acompletion\\(\\) not implemented"):
            await provider.acompletion(
                model="test-model",
                messages=mock_messages
            )

    @pytest.mark.asyncio
    async def test_astreaming_not_implemented_by_default(self, mock_messages):
        """Test that astreaming raises NotImplementedError by default."""
        provider = TestProvider()
        with pytest.raises(NotImplementedError, match="astreaming\\(\\) not implemented"):
            async for _ in provider.astreaming(
                model="test-model",
                messages=mock_messages
            ):
                pass

    def test_image_generation_not_implemented_by_default(self):
        """Test that image_generation raises NotImplementedError by default."""
        provider = TestProvider()
        with pytest.raises(NotImplementedError, match="image_generation\\(\\) not implemented"):
            provider.image_generation(
                model="dall-e-3",
                prompt="A test image"
            )

    @pytest.mark.asyncio
    async def test_aimage_generation_not_implemented_by_default(self):
        """Test that aimage_generation raises NotImplementedError by default."""
        provider = TestProvider()
        with pytest.raises(NotImplementedError, match="aimage_generation\\(\\) not implemented"):
            await provider.aimage_generation(
                model="dall-e-3",
                prompt="A test image"
            )

    def test_embedding_not_implemented_by_default(self):
        """Test that embedding raises NotImplementedError by default."""
        provider = TestProvider()
        with pytest.raises(NotImplementedError, match="embedding\\(\\) not implemented"):
            provider.embedding(
                model="text-embedding-ada-002",
                input="test text"
            )

    @pytest.mark.asyncio
    async def test_aembedding_not_implemented_by_default(self):
        """Test that aembedding raises NotImplementedError by default."""
        provider = TestProvider()
        with pytest.raises(NotImplementedError, match="aembedding\\(\\) not implemented"):
            await provider.aembedding(
                model="text-embedding-ada-002",
                input="test text"
            )

    def test_provider_accepts_additional_kwargs(self, mock_messages):
        """Test that provider methods accept additional kwargs."""
        provider = TestProvider()
        # Should not raise even with extra parameters
        response = provider.completion(
            model="test-model",
            messages=mock_messages,
            temperature=0.9,
            max_tokens=100,
            custom_param="value"
        )
        assert response is not None


class TestFullyImplementedProvider:
    """Test a fully implemented provider."""

    class FullProvider(BaseCustomProvider):
        """Provider with all methods implemented."""

        def completion(self, model: str, messages: list, **kwargs):
            from tests.fixtures.mock_litellm import create_mock_response
            return create_mock_response(content="Full completion")

        def streaming(self, model: str, messages: list, **kwargs):
            from tests.fixtures.mock_litellm import MockModelResponse, MockStreamingChoices, MockDelta
            chunks = ["Hello", " world"]
            for chunk in chunks:
                response = MockModelResponse(
                    model=model,
                    object="chat.completion.chunk",
                    choices=[MockStreamingChoices(delta=MockDelta(content=chunk))]
                )
                yield response

        async def acompletion(self, model: str, messages: list, **kwargs):
            from tests.fixtures.mock_litellm import create_mock_response
            return create_mock_response(content="Async completion")

        async def astreaming(self, model: str, messages: list, **kwargs):
            from tests.fixtures.mock_litellm import MockModelResponse, MockStreamingChoices, MockDelta
            chunks = ["Async", " streaming"]
            for chunk in chunks:
                response = MockModelResponse(
                    model=model,
                    object="chat.completion.chunk",
                    choices=[MockStreamingChoices(delta=MockDelta(content=chunk))]
                )
                yield response

        def embedding(self, model: str, input, **kwargs):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        async def aembedding(self, model: str, input, **kwargs):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        def image_generation(self, model: str, prompt: str, **kwargs):
            return {"data": [{"url": "https://example.com/image.png"}]}

        async def aimage_generation(self, model: str, prompt: str, **kwargs):
            return {"data": [{"url": "https://example.com/image.png"}]}

    def test_full_provider_completion(self, mock_messages):
        """Test fully implemented provider completion."""
        provider = self.FullProvider()
        response = provider.completion("test-model", mock_messages)
        assert response is not None

    def test_full_provider_streaming(self, mock_messages):
        """Test fully implemented provider streaming."""
        provider = self.FullProvider()
        chunks = list(provider.streaming("test-model", mock_messages))
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_full_provider_acompletion(self, mock_messages):
        """Test fully implemented provider async completion."""
        provider = self.FullProvider()
        response = await provider.acompletion("test-model", mock_messages)
        assert response is not None

    @pytest.mark.asyncio
    async def test_full_provider_astreaming(self, mock_messages):
        """Test fully implemented provider async streaming."""
        provider = self.FullProvider()
        chunks = []
        async for chunk in provider.astreaming("test-model", mock_messages):
            chunks.append(chunk)
        assert len(chunks) == 2

    def test_full_provider_embedding(self):
        """Test fully implemented provider embedding."""
        provider = self.FullProvider()
        response = provider.embedding("text-embedding-ada-002", "test")
        assert "data" in response

    @pytest.mark.asyncio
    async def test_full_provider_aembedding(self):
        """Test fully implemented provider async embedding."""
        provider = self.FullProvider()
        response = await provider.aembedding("text-embedding-ada-002", "test")
        assert "data" in response

    def test_full_provider_image_generation(self):
        """Test fully implemented provider image generation."""
        provider = self.FullProvider()
        response = provider.image_generation("dall-e-3", "A test image")
        assert "data" in response

    @pytest.mark.asyncio
    async def test_full_provider_aimage_generation(self):
        """Test fully implemented provider async image generation."""
        provider = self.FullProvider()
        response = await provider.aimage_generation("dall-e-3", "A test image")
        assert "data" in response


class TestCustomStreamWrapper:
    """Test CustomStreamWrapper class."""

    def test_custom_stream_wrapper_exists(self):
        """Test that CustomStreamWrapper can be instantiated."""
        wrapper = CustomStreamWrapper()
        assert wrapper is not None

    def test_custom_stream_wrapper_is_class(self):
        """Test that CustomStreamWrapper is a class."""
        assert isinstance(CustomStreamWrapper, type)
