"""Tests for response modifier loggers."""

import pytest
import time

from litellm_plugin.examples.response_modifier import (
    ResponseModifier,
    ContentFilter,
    RequestLogger,
)


class TestResponseModifierBasics:
    """Test basic ResponseModifier functionality."""

    def test_initialization_with_default_prefix(self):
        """Test ResponseModifier initializes with default prefix."""
        logger = ResponseModifier()
        assert logger.prefix == "[Verified] "
        assert logger.logger_name == "ResponseModifier"

    def test_initialization_with_custom_prefix(self):
        """Test ResponseModifier initializes with custom prefix."""
        logger = ResponseModifier(prefix="[CUSTOM] ")
        assert logger.prefix == "[CUSTOM] "

    @pytest.mark.asyncio
    async def test_adds_prefix_to_response(
        self,
        mock_user_api_key_dict,
        mock_litellm_call_data,
        mock_model_response
    ):
        """Test ResponseModifier adds prefix to response content."""
        logger = ResponseModifier(prefix="[TEST] ")

        result = await logger.async_post_call_success_hook(
            data=mock_litellm_call_data,
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_model_response
        )

        if hasattr(result, 'choices') and len(result.choices) > 0:
            content = result.choices[0].message.content
            assert content.startswith("[TEST] ")

    @pytest.mark.asyncio
    async def test_preserves_original_content(
        self,
        mock_user_api_key_dict,
        mock_litellm_call_data,
        mock_model_response
    ):
        """Test ResponseModifier preserves original content after prefix."""
        logger = ResponseModifier(prefix="[PREFIX] ")

        # Get original content
        if hasattr(mock_model_response, 'choices'):
            original_content = mock_model_response.choices[0].message.content

            result = await logger.async_post_call_success_hook(
                data=mock_litellm_call_data,
                user_api_key_dict=mock_user_api_key_dict,
                response=mock_model_response
            )

            new_content = result.choices[0].message.content
            assert new_content == f"[PREFIX] {original_content}"

    @pytest.mark.asyncio
    async def test_handles_response_without_content(
        self,
        mock_user_api_key_dict,
        mock_litellm_call_data
    ):
        """Test ResponseModifier handles response without content gracefully."""
        logger = ResponseModifier(prefix="[TEST] ")

        # Mock response without proper structure
        mock_response = type('obj', (object,), {})()

        result = await logger.async_post_call_success_hook(
            data=mock_litellm_call_data,
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_response
        )

        # Should return response unchanged
        assert result == mock_response


class TestResponseModifierLogging:
    """Test ResponseModifier logging functionality."""

    @pytest.mark.asyncio
    async def test_logs_success_event(
        self,
        mock_completion_kwargs,
        mock_model_response,
        capsys
    ):
        """Test ResponseModifier logs successful events."""
        logger = ResponseModifier()
        start = time.time()
        end = start + 0.5

        await logger.async_log_success_event(
            kwargs=mock_completion_kwargs,
            response_obj=mock_model_response,
            start_time=start,
            end_time=end
        )

        captured = capsys.readouterr()
        assert "[ResponseModifier]" in captured.out
        assert "0.50s" in captured.out or "0.5s" in captured.out

    @pytest.mark.asyncio
    async def test_logs_model_name(
        self,
        mock_completion_kwargs,
        mock_model_response,
        capsys
    ):
        """Test ResponseModifier logs model name."""
        logger = ResponseModifier()
        start = time.time()
        end = start + 0.1

        await logger.async_log_success_event(
            kwargs=mock_completion_kwargs,
            response_obj=mock_model_response,
            start_time=start,
            end_time=end
        )

        captured = capsys.readouterr()
        assert mock_completion_kwargs["model"] in captured.out

    @pytest.mark.asyncio
    async def test_logs_token_count(
        self,
        mock_completion_kwargs,
        mock_model_response,
        capsys
    ):
        """Test ResponseModifier logs token count."""
        logger = ResponseModifier()
        start = time.time()
        end = start + 0.1

        await logger.async_log_success_event(
            kwargs=mock_completion_kwargs,
            response_obj=mock_model_response,
            start_time=start,
            end_time=end
        )

        captured = capsys.readouterr()
        assert "Tokens:" in captured.out


class TestContentFilterBasics:
    """Test basic ContentFilter functionality."""

    def test_initialization_without_blocked_words(self):
        """Test ContentFilter initializes with empty blocked words."""
        logger = ContentFilter()
        assert logger.blocked_words == []
        assert logger.logger_name == "ContentFilter"

    def test_initialization_with_blocked_words(self):
        """Test ContentFilter initializes with blocked words."""
        blocked = ["spam", "advertisement", "buy now"]
        logger = ContentFilter(blocked_words=blocked)
        assert logger.blocked_words == blocked

    @pytest.mark.asyncio
    async def test_filters_blocked_word(
        self,
        mock_user_api_key_dict,
        mock_litellm_call_data,
        mock_model_response
    ):
        """Test ContentFilter replaces blocked words."""
        logger = ContentFilter(blocked_words=["test"])

        # Modify mock response to contain blocked word
        if hasattr(mock_model_response, 'choices'):
            mock_model_response.choices[0].message.content = "This is a test response"

            result = await logger.async_post_call_success_hook(
                data=mock_litellm_call_data,
                user_api_key_dict=mock_user_api_key_dict,
                response=mock_model_response
            )

            content = result.choices[0].message.content
            assert "test" not in content
            assert "[FILTERED]" in content
            assert content == "This is a [FILTERED] response"

    @pytest.mark.asyncio
    async def test_filters_multiple_blocked_words(
        self,
        mock_user_api_key_dict,
        mock_litellm_call_data,
        mock_model_response
    ):
        """Test ContentFilter replaces multiple blocked words."""
        logger = ContentFilter(blocked_words=["spam", "buy"])

        if hasattr(mock_model_response, 'choices'):
            mock_model_response.choices[0].message.content = "spam and buy now!"

            result = await logger.async_post_call_success_hook(
                data=mock_litellm_call_data,
                user_api_key_dict=mock_user_api_key_dict,
                response=mock_model_response
            )

            content = result.choices[0].message.content
            assert "spam" not in content
            assert "buy" not in content
            assert "[FILTERED]" in content

    @pytest.mark.asyncio
    async def test_no_filtering_when_no_blocked_words(
        self,
        mock_user_api_key_dict,
        mock_litellm_call_data,
        mock_model_response
    ):
        """Test ContentFilter doesn't filter when no blocked words match."""
        logger = ContentFilter(blocked_words=["spam", "advertisement"])

        if hasattr(mock_model_response, 'choices'):
            original_content = "This is a clean response"
            mock_model_response.choices[0].message.content = original_content

            result = await logger.async_post_call_success_hook(
                data=mock_litellm_call_data,
                user_api_key_dict=mock_user_api_key_dict,
                response=mock_model_response
            )

            content = result.choices[0].message.content
            assert content == original_content
            assert "[FILTERED]" not in content


class TestRequestLoggerBasics:
    """Test basic RequestLogger functionality."""

    def test_initialization_verbose_true(self):
        """Test RequestLogger initializes with verbose=True."""
        logger = RequestLogger(verbose=True)
        assert logger.verbose is True
        assert logger.request_count == 0
        assert logger.logger_name == "RequestLogger"

    def test_initialization_verbose_false(self):
        """Test RequestLogger initializes with verbose=False."""
        logger = RequestLogger(verbose=False)
        assert logger.verbose is False

    @pytest.mark.asyncio
    async def test_tracks_request_count(
        self,
        mock_user_api_key_dict,
        mock_cache,
        mock_litellm_call_data
    ):
        """Test RequestLogger increments request count."""
        logger = RequestLogger(verbose=False)
        assert logger.request_count == 0

        await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=mock_litellm_call_data,
            call_type="completion"
        )

        assert logger.request_count == 1

        await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=mock_litellm_call_data,
            call_type="completion"
        )

        assert logger.request_count == 2

    @pytest.mark.asyncio
    async def test_returns_unmodified_data(
        self,
        mock_user_api_key_dict,
        mock_cache,
        mock_litellm_call_data
    ):
        """Test RequestLogger returns data unmodified."""
        logger = RequestLogger()

        result = await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=mock_litellm_call_data,
            call_type="completion"
        )

        assert result == mock_litellm_call_data


class TestRequestLoggerVerbose:
    """Test RequestLogger verbose logging."""

    @pytest.mark.asyncio
    async def test_verbose_logs_request_details(
        self,
        mock_user_api_key_dict,
        mock_cache,
        mock_litellm_call_data,
        capsys
    ):
        """Test RequestLogger logs details when verbose=True."""
        logger = RequestLogger(verbose=True)

        await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=mock_litellm_call_data,
            call_type="completion"
        )

        captured = capsys.readouterr()
        assert "[RequestLogger]" in captured.out
        assert "Request #1" in captured.out
        assert "Type: completion" in captured.out
        assert mock_litellm_call_data["model"] in captured.out

    @pytest.mark.asyncio
    async def test_verbose_logs_message_count(
        self,
        mock_user_api_key_dict,
        mock_cache,
        capsys
    ):
        """Test RequestLogger logs message count when present."""
        logger = RequestLogger(verbose=True)
        data = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Test 1"},
                {"role": "assistant", "content": "Response 1"},
                {"role": "user", "content": "Test 2"},
            ]
        }

        await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=data,
            call_type="completion"
        )

        captured = capsys.readouterr()
        assert "Messages: 3" in captured.out

    @pytest.mark.asyncio
    async def test_non_verbose_doesnt_log_details(
        self,
        mock_user_api_key_dict,
        mock_cache,
        mock_litellm_call_data,
        capsys
    ):
        """Test RequestLogger doesn't log details when verbose=False."""
        logger = RequestLogger(verbose=False)

        await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=mock_litellm_call_data,
            call_type="completion"
        )

        captured = capsys.readouterr()
        assert "[RequestLogger]" not in captured.out


class TestRequestLoggerSuccess:
    """Test RequestLogger success logging."""

    @pytest.mark.asyncio
    async def test_logs_success_verbose(
        self,
        mock_completion_kwargs,
        mock_model_response,
        capsys
    ):
        """Test RequestLogger logs success when verbose=True."""
        logger = RequestLogger(verbose=True)
        start = time.time()
        end = start + 1.5

        await logger.async_log_success_event(
            kwargs=mock_completion_kwargs,
            response_obj=mock_model_response,
            start_time=start,
            end_time=end
        )

        captured = capsys.readouterr()
        assert "[RequestLogger]" in captured.out
        assert "Success" in captured.out
        assert "1.50s" in captured.out or "1.5s" in captured.out

    @pytest.mark.asyncio
    async def test_logs_success_token_count(
        self,
        mock_completion_kwargs,
        mock_model_response,
        capsys
    ):
        """Test RequestLogger logs token count on success."""
        logger = RequestLogger(verbose=True)
        start = time.time()
        end = start + 0.5

        await logger.async_log_success_event(
            kwargs=mock_completion_kwargs,
            response_obj=mock_model_response,
            start_time=start,
            end_time=end
        )

        captured = capsys.readouterr()
        assert "Tokens:" in captured.out

    @pytest.mark.asyncio
    async def test_doesnt_log_success_non_verbose(
        self,
        mock_completion_kwargs,
        mock_model_response,
        capsys
    ):
        """Test RequestLogger doesn't log success when verbose=False."""
        logger = RequestLogger(verbose=False)
        start = time.time()
        end = start + 0.5

        await logger.async_log_success_event(
            kwargs=mock_completion_kwargs,
            response_obj=mock_model_response,
            start_time=start,
            end_time=end
        )

        captured = capsys.readouterr()
        assert captured.out == ""


class TestRequestLoggerFailure:
    """Test RequestLogger failure logging."""

    @pytest.mark.asyncio
    async def test_logs_failure(
        self,
        mock_completion_kwargs,
        mock_exception,
        capsys
    ):
        """Test RequestLogger logs failures."""
        logger = RequestLogger(verbose=True)
        start = time.time()
        end = start + 0.3

        await logger.async_log_failure_event(
            kwargs=mock_completion_kwargs,
            response_obj=mock_exception,
            start_time=start,
            end_time=end
        )

        captured = capsys.readouterr()
        assert "[RequestLogger]" in captured.out
        assert "Failed" in captured.out
        assert "0.30s" in captured.out or "0.3s" in captured.out

    @pytest.mark.asyncio
    async def test_logs_error_message(
        self,
        mock_completion_kwargs,
        capsys
    ):
        """Test RequestLogger logs error message."""
        logger = RequestLogger(verbose=True)
        error = Exception("Test error message")
        start = time.time()
        end = start + 0.1

        await logger.async_log_failure_event(
            kwargs=mock_completion_kwargs,
            response_obj=error,
            start_time=start,
            end_time=end
        )

        captured = capsys.readouterr()
        assert "Test error message" in captured.out


class TestMultipleLoggers:
    """Test using multiple logger examples together."""

    @pytest.mark.asyncio
    async def test_response_modifier_and_content_filter(
        self,
        mock_user_api_key_dict,
        mock_litellm_call_data,
        mock_model_response
    ):
        """Test ResponseModifier and ContentFilter can be used together."""
        modifier = ResponseModifier(prefix="[PREFIX] ")
        filter_logger = ContentFilter(blocked_words=["bad"])

        # Set content with blocked word
        if hasattr(mock_model_response, 'choices'):
            mock_model_response.choices[0].message.content = "This is bad content"

            # Apply filter first
            filtered = await filter_logger.async_post_call_success_hook(
                data=mock_litellm_call_data,
                user_api_key_dict=mock_user_api_key_dict,
                response=mock_model_response
            )

            # Then apply modifier
            result = await modifier.async_post_call_success_hook(
                data=mock_litellm_call_data,
                user_api_key_dict=mock_user_api_key_dict,
                response=filtered
            )

            content = result.choices[0].message.content
            assert content.startswith("[PREFIX] ")
            assert "[FILTERED]" in content
            assert "bad" not in content

    @pytest.mark.asyncio
    async def test_request_logger_with_response_modifier(
        self,
        mock_user_api_key_dict,
        mock_cache,
        mock_litellm_call_data,
        mock_model_response,
        capsys
    ):
        """Test RequestLogger and ResponseModifier together."""
        request_logger = RequestLogger(verbose=True)
        modifier = ResponseModifier(prefix="[TEST] ")

        # Log request
        await request_logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=mock_litellm_call_data,
            call_type="completion"
        )

        # Modify response
        result = await modifier.async_post_call_success_hook(
            data=mock_litellm_call_data,
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_model_response
        )

        captured = capsys.readouterr()
        assert "[RequestLogger]" in captured.out

        if hasattr(result, 'choices'):
            content = result.choices[0].message.content
            assert content.startswith("[TEST] ")
