"""Tests for EchoProvider example."""

import pytest
import time

from litellm_plugin.examples.echo_provider import EchoProvider


class TestEchoProviderBasics:
    """Test basic EchoProvider functionality."""

    def test_echo_provider_initialization(self):
        """Test EchoProvider initializes correctly."""
        provider = EchoProvider()
        assert provider.provider_name == "echo"

    def test_completion_echoes_user_message(self):
        """Test completion echoes the last user message."""
        provider = EchoProvider()
        messages = [
            {"role": "user", "content": "Hello, world!"}
        ]

        response = provider.completion(
            model="echo/test",
            messages=messages
        )

        # Check response structure
        assert response is not None
        if hasattr(response, 'choices'):
            # LiteLLM available
            assert len(response.choices) > 0
            content = response.choices[0].message.content
            assert content == "Echo: Hello, world!"
        else:
            # Dict response when LiteLLM not available
            assert "choices" in response
            content = response["choices"][0]["message"]["content"]
            assert content == "Echo: Hello, world!"

    def test_completion_with_multiple_messages(self):
        """Test completion echoes the last user message from conversation."""
        provider = EchoProvider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"}
        ]

        response = provider.completion(
            model="echo/test",
            messages=messages
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
            assert content == "Echo: Second message"
        else:
            content = response["choices"][0]["message"]["content"]
            assert content == "Echo: Second message"

    def test_completion_with_empty_message(self):
        """Test completion handles empty user message."""
        provider = EchoProvider()
        messages = [
            {"role": "user", "content": ""}
        ]

        response = provider.completion(
            model="echo/test",
            messages=messages
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
            assert content == "Echo: "
        else:
            content = response["choices"][0]["message"]["content"]
            assert content == "Echo: "

    def test_completion_with_no_user_messages(self):
        """Test completion when there are no user messages."""
        provider = EchoProvider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        response = provider.completion(
            model="echo/test",
            messages=messages
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
            assert content == "Echo: "
        else:
            content = response["choices"][0]["message"]["content"]
            assert content == "Echo: "


class TestEchoProviderResponse:
    """Test EchoProvider response formatting."""

    def test_response_has_correct_model(self):
        """Test response includes the model name."""
        provider = EchoProvider()
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="echo/custom-model",
            messages=messages
        )

        if hasattr(response, 'model'):
            assert response.model == "echo/custom-model"
        else:
            assert response["model"] == "echo/custom-model"

    def test_response_has_usage_information(self):
        """Test response includes usage information."""
        provider = EchoProvider()
        messages = [{"role": "user", "content": "Hello world"}]

        response = provider.completion(
            model="echo/test",
            messages=messages
        )

        if hasattr(response, 'usage'):
            assert response.usage is not None
            assert response.usage.prompt_tokens >= 0
            assert response.usage.completion_tokens >= 0
            assert response.usage.total_tokens >= 0
        else:
            assert "usage" in response
            assert "prompt_tokens" in response["usage"]
            assert "completion_tokens" in response["usage"]
            assert "total_tokens" in response["usage"]

    def test_response_has_timestamp(self):
        """Test response includes a timestamp."""
        provider = EchoProvider()
        messages = [{"role": "user", "content": "Test"}]

        before = int(time.time())
        response = provider.completion(
            model="echo/test",
            messages=messages
        )
        after = int(time.time())

        if hasattr(response, 'created'):
            assert before <= response.created <= after
        # Dict response doesn't have created timestamp in this implementation

    def test_response_has_finish_reason(self):
        """Test response includes finish_reason."""
        provider = EchoProvider()
        messages = [{"role": "user", "content": "Test"}]

        response = provider.completion(
            model="echo/test",
            messages=messages
        )

        if hasattr(response, 'choices'):
            assert response.choices[0].finish_reason == "stop"
        else:
            # Dict response doesn't include finish_reason in this implementation
            pass


class TestEchoProviderTokenCounting:
    """Test token counting in EchoProvider."""

    def test_token_count_simple_message(self):
        """Test token counting for simple message."""
        provider = EchoProvider()
        messages = [{"role": "user", "content": "one two three"}]

        response = provider.completion(
            model="echo/test",
            messages=messages
        )

        if hasattr(response, 'usage'):
            # Tokens are counted by splitting on whitespace
            assert response.usage.prompt_tokens == 3  # "one two three"
            # Echo content: "Echo: one two three" = 4 words
            assert response.usage.completion_tokens == 4
            assert response.usage.total_tokens == 7

    def test_token_count_empty_message(self):
        """Test token counting for empty message."""
        provider = EchoProvider()
        messages = [{"role": "user", "content": ""}]

        response = provider.completion(
            model="echo/test",
            messages=messages
        )

        if hasattr(response, 'usage'):
            # Empty string split results in []
            assert response.usage.prompt_tokens == 0  # len([])
            # "Echo: " splits to ['Echo:']
            assert response.usage.completion_tokens == 1  # len(['Echo:'])
            assert response.usage.total_tokens == 1


class TestEchoProviderParameters:
    """Test EchoProvider parameter handling."""

    def test_completion_accepts_optional_parameters(self):
        """Test completion accepts but ignores optional parameters."""
        provider = EchoProvider()
        messages = [{"role": "user", "content": "Test"}]

        # Should not raise even with extra parameters
        response = provider.completion(
            model="echo/test",
            messages=messages,
            api_base="https://api.example.com",
            custom_llm_provider="custom",
            temperature=0.9,
            max_tokens=100,
            top_p=0.95
        )

        assert response is not None

    def test_completion_ignores_api_base(self):
        """Test that api_base parameter is ignored."""
        provider = EchoProvider()
        messages = [{"role": "user", "content": "Hello"}]

        response1 = provider.completion(
            model="echo/test",
            messages=messages,
            api_base="https://api1.example.com"
        )

        response2 = provider.completion(
            model="echo/test",
            messages=messages,
            api_base="https://api2.example.com"
        )

        # Responses should be the same regardless of api_base
        if hasattr(response1, 'choices'):
            content1 = response1.choices[0].message.content
            content2 = response2.choices[0].message.content
        else:
            content1 = response1["choices"][0]["message"]["content"]
            content2 = response2["choices"][0]["message"]["content"]

        assert content1 == content2 == "Echo: Hello"


class TestEchoProviderEdgeCases:
    """Test edge cases for EchoProvider."""

    def test_completion_with_special_characters(self):
        """Test completion with special characters in message."""
        provider = EchoProvider()
        special_messages = [
            "Hello! How are you?",
            "Testing... 123",
            "Unicode: 你好世界",
            "Emoji: 😀🎉",
            "Newlines:\nLine 1\nLine 2",
            "Tabs:\tTabbed\tContent",
        ]

        for msg in special_messages:
            messages = [{"role": "user", "content": msg}]
            response = provider.completion(
                model="echo/test",
                messages=messages
            )

            if hasattr(response, 'choices'):
                content = response.choices[0].message.content
            else:
                content = response["choices"][0]["message"]["content"]

            assert content == f"Echo: {msg}"

    def test_completion_with_very_long_message(self):
        """Test completion with very long message."""
        provider = EchoProvider()
        long_message = "word " * 1000  # 1000 words
        messages = [{"role": "user", "content": long_message.strip()}]

        response = provider.completion(
            model="echo/test",
            messages=messages
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]

        assert content == f"Echo: {long_message.strip()}"

    def test_completion_with_missing_content_key(self):
        """Test completion when message dict is missing content."""
        provider = EchoProvider()
        messages = [
            {"role": "user"}  # Missing content
        ]

        response = provider.completion(
            model="echo/test",
            messages=messages
        )

        # Should handle gracefully with empty content
        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]

        assert content == "Echo: "

    def test_completion_with_empty_messages_list(self):
        """Test completion with empty messages list."""
        provider = EchoProvider()
        messages = []

        response = provider.completion(
            model="echo/test",
            messages=messages
        )

        # Should return empty echo
        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response["choices"][0]["message"]["content"]

        assert content == "Echo: "
