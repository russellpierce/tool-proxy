"""Tests for MockAPIProvider example."""

import pytest

from litellm_plugin.examples.mock_api_provider import MockAPIProvider


class TestMockAPIProviderBasics:
    """Test basic MockAPIProvider functionality."""

    def test_provider_initialization_without_api_key(self):
        """Test provider initializes without default API key."""
        provider = MockAPIProvider()
        assert provider.provider_name == "mock_api"
        assert provider.default_api_key is None

    def test_provider_initialization_with_api_key(self):
        """Test provider initializes with default API key."""
        provider = MockAPIProvider(default_api_key="test-key-123")
        assert provider.provider_name == "mock_api"
        assert provider.default_api_key == "test-key-123"

    def test_completion_requires_api_key(self):
        """Test completion raises error when no API key provided."""
        provider = MockAPIProvider()
        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(ValueError, match="API key is required"):
            provider.completion(
                model="mock_api/gpt-mock",
                messages=messages
            )

    def test_completion_with_provided_api_key(self):
        """Test completion works with provided API key."""
        provider = MockAPIProvider()
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages,
            api_key="provided-key"
        )

        assert response is not None

    def test_completion_with_default_api_key(self):
        """Test completion works with default API key."""
        provider = MockAPIProvider(default_api_key="default-key")
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )

        assert response is not None

    def test_provided_api_key_overrides_default(self):
        """Test that provided API key overrides default."""
        provider = MockAPIProvider(default_api_key="default-key")
        messages = [{"role": "user", "content": "Test"}]

        # Should not raise even though provided key is different
        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages,
            api_key="override-key"
        )

        assert response is not None


class TestMockAPIProviderResponse:
    """Test MockAPIProvider response generation."""

    def test_response_includes_user_message(self):
        """Test response references the user's message."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "What is AI?"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]

        assert "What is AI?" in content
        assert "mock API response" in content

    def test_response_includes_temperature(self):
        """Test response includes temperature parameter."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages,
            temperature=0.9
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]

        assert "Temperature: 0.9" in content

    def test_response_includes_max_tokens(self):
        """Test response includes max_tokens parameter."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages,
            max_tokens=50
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]

        assert "Max tokens: 50" in content

    def test_response_uses_default_parameters(self):
        """Test response uses default parameters when not provided."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]

        # Should use defaults: temperature=0.7, max_tokens=100
        assert "Temperature: 0.7" in content
        assert "Max tokens: 100" in content


class TestMockAPIProviderTokenTruncation:
    """Test token truncation in MockAPIProvider."""

    def test_response_truncates_at_max_tokens(self):
        """Test response is truncated if it exceeds max_tokens."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Short"}]

        # Use very small max_tokens to trigger truncation
        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages,
            max_tokens=5
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]

        # Response should be truncated and end with "..."
        word_count = len(content.split())
        assert word_count <= 6  # max_tokens + potential ellipsis
        assert content.endswith("...")

    def test_response_not_truncated_when_under_max_tokens(self):
        """Test response is not truncated when under max_tokens."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]

        # Use large max_tokens
        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages,
            max_tokens=1000
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]

        # Response should not be truncated
        assert not content.endswith("...")


class TestMockAPIProviderUsageTracking:
    """Test usage/token tracking in MockAPIProvider."""

    def test_usage_information_included(self):
        """Test response includes usage information."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Hello world"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )

        if hasattr(response, 'usage'):
            assert response.usage is not None
            assert response.usage.prompt_tokens > 0
            assert response.usage.completion_tokens > 0
            assert response.usage.total_tokens > 0
        else:
            assert "usage" in response
            assert response["usage"]["prompt_tokens"] > 0
            assert response["usage"]["completion_tokens"] > 0
            assert response["usage"]["total_tokens"] > 0

    def test_usage_counts_match(self):
        """Test usage token counts are consistent."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Test message"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )

        if hasattr(response, 'usage'):
            prompt = response.usage.prompt_tokens
            completion = response.usage.completion_tokens
            total = response.usage.total_tokens
        else:
            prompt = response["usage"]["prompt_tokens"]
            completion = response["usage"]["completion_tokens"]
            total = response["usage"]["total_tokens"]

        assert total == prompt + completion

    def test_prompt_tokens_reflect_input(self):
        """Test prompt tokens reflect the input message length."""
        provider = MockAPIProvider(default_api_key="test-key")

        # Short message
        messages_short = [{"role": "user", "content": "Hi"}]
        response_short = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages_short
        )

        # Long message
        messages_long = [{"role": "user", "content": "This is a much longer message with many words"}]
        response_long = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages_long
        )

        if hasattr(response_short, 'usage'):
            prompt_short = response_short.usage.prompt_tokens
            prompt_long = response_long.usage.prompt_tokens
        else:
            prompt_short = response_short["usage"]["prompt_tokens"]
            prompt_long = response_long["usage"]["prompt_tokens"]

        assert prompt_long > prompt_short


class TestMockAPIProviderMultipleMessages:
    """Test MockAPIProvider with multiple messages."""

    def test_extracts_last_user_message(self):
        """Test provider extracts the last user message."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"}
        ]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]

        assert "Second question" in content
        assert "First question" not in content

    def test_handles_no_user_messages(self):
        """Test provider handles messages without user role."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [
            {"role": "system", "content": "You are helpful."}
        ]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )

        # Should handle gracefully
        assert response is not None


class TestMockAPIProviderParameters:
    """Test parameter handling in MockAPIProvider."""

    def test_accepts_api_base_parameter(self):
        """Test provider accepts api_base parameter."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages,
            api_base="https://custom-api.example.com/v1"
        )

        assert response is not None

    def test_accepts_custom_llm_provider_parameter(self):
        """Test provider accepts custom_llm_provider parameter."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages,
            custom_llm_provider="custom"
        )

        assert response is not None

    def test_accepts_additional_kwargs(self):
        """Test provider accepts additional kwargs."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            custom_param="value"
        )

        assert response is not None


class TestMockAPIProviderResponseFormat:
    """Test response formatting in MockAPIProvider."""

    def test_response_has_correct_model(self):
        """Test response includes the correct model."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="mock_api/custom-model-v2",
            messages=messages
        )

        if hasattr(response, 'model'):
            assert response.model == "mock_api/custom-model-v2"
        else:
            assert response["model"] == "mock_api/custom-model-v2"

    def test_response_has_timestamp(self):
        """Test response includes timestamp."""
        import time
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]

        before = int(time.time())
        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )
        after = int(time.time())

        if hasattr(response, 'created'):
            assert before <= response.created <= after

    def test_response_has_finish_reason(self):
        """Test response includes finish_reason."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )

        if hasattr(response, 'choices'):
            assert response.choices[0].finish_reason == "stop"

    def test_response_message_role_is_assistant(self):
        """Test response message has assistant role."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )

        if hasattr(response, 'choices'):
            assert response.choices[0].message.role == "assistant"
        else:
            assert response["choices"][0]["message"]["role"] == "assistant"


class TestMockAPIProviderEdgeCases:
    """Test edge cases for MockAPIProvider."""

    def test_empty_api_key_string_fails(self):
        """Test that empty API key string is treated as no key."""
        provider = MockAPIProvider(default_api_key="")
        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(ValueError, match="API key is required"):
            provider.completion(
                model="mock_api/gpt-mock",
                messages=messages
            )

    def test_special_characters_in_message(self):
        """Test handling special characters in messages."""
        provider = MockAPIProvider(default_api_key="test-key")
        special_content = "Test with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        messages = [{"role": "user", "content": special_content}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]

        assert special_content in content

    def test_very_long_message(self):
        """Test handling very long messages."""
        provider = MockAPIProvider(default_api_key="test-key")
        long_message = "word " * 10000
        messages = [{"role": "user", "content": long_message.strip()}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )

        assert response is not None

    def test_unicode_in_message(self):
        """Test handling Unicode characters."""
        provider = MockAPIProvider(default_api_key="test-key")
        messages = [{"role": "user", "content": "Hello 世界 🌍"}]

        response = provider.completion(
            model="mock_api/gpt-mock",
            messages=messages
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]

        assert "Hello 世界 🌍" in content
