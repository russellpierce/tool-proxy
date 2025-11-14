"""Tests for BaseCustomLogger."""

import pytest
import time
from typing import Dict, Any
from unittest.mock import MagicMock

from litellm_plugin.logger import BaseCustomLogger

try:
    from litellm.types.utils import ModelResponse
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    ModelResponse = object


class TestBaseCustomLogger:
    """Test suite for BaseCustomLogger base class."""

    def test_logger_initialization(self):
        """Test that logger initializes correctly."""
        logger = BaseCustomLogger()
        assert logger.logger_name == "BaseCustomLogger"

    def test_logger_name_matches_class_name(self):
        """Test logger_name is set to class name."""
        class CustomNamedLogger(BaseCustomLogger):
            pass

        logger = CustomNamedLogger()
        assert logger.logger_name == "CustomNamedLogger"

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_returns_none_by_default(
        self,
        mock_user_api_key_dict,
        mock_cache,
        mock_litellm_call_data
    ):
        """Test async_pre_call_hook returns None by default."""
        logger = BaseCustomLogger()
        result = await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=mock_litellm_call_data,
            call_type="completion"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_returns_response_unchanged(
        self,
        mock_user_api_key_dict,
        mock_litellm_call_data,
        mock_model_response
    ):
        """Test async_post_call_success_hook returns response unchanged by default."""
        logger = BaseCustomLogger()
        result = await logger.async_post_call_success_hook(
            data=mock_litellm_call_data,
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_model_response
        )
        assert result == mock_model_response

    @pytest.mark.asyncio
    async def test_async_log_success_event_does_nothing_by_default(
        self,
        mock_completion_kwargs,
        mock_model_response
    ):
        """Test async_log_success_event does nothing by default."""
        logger = BaseCustomLogger()
        start = time.time()
        end = start + 1.0

        # Should not raise
        await logger.async_log_success_event(
            kwargs=mock_completion_kwargs,
            response_obj=mock_model_response,
            start_time=start,
            end_time=end
        )

    @pytest.mark.asyncio
    async def test_async_log_failure_event_does_nothing_by_default(
        self,
        mock_completion_kwargs,
        mock_exception
    ):
        """Test async_log_failure_event does nothing by default."""
        logger = BaseCustomLogger()
        start = time.time()
        end = start + 1.0

        # Should not raise
        await logger.async_log_failure_event(
            kwargs=mock_completion_kwargs,
            response_obj=mock_exception,
            start_time=start,
            end_time=end
        )

    @pytest.mark.asyncio
    async def test_async_post_call_streaming_hook_returns_response_unchanged(
        self,
        mock_user_api_key_dict,
        mock_streaming_response
    ):
        """Test async_post_call_streaming_hook returns response unchanged."""
        logger = BaseCustomLogger()
        result = await logger.async_post_call_streaming_hook(
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_streaming_response
        )
        assert result == mock_streaming_response


class TestCustomLoggerImplementations:
    """Test custom logger implementations with modified behavior."""

    class RequestModifierLogger(BaseCustomLogger):
        """Logger that modifies requests."""

        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            """Add a system message to requests."""
            if "messages" in data:
                data["messages"].insert(0, {
                    "role": "system",
                    "content": "You are a helpful assistant."
                })
            return data

    class ResponseModifierLogger(BaseCustomLogger):
        """Logger that modifies responses."""

        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            """Add a prefix to responses."""
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message'):
                    original = response.choices[0].message.content
                    response.choices[0].message.content = f"[VERIFIED] {original}"
            return response

    class EventLoggingLogger(BaseCustomLogger):
        """Logger that tracks events."""

        def __init__(self):
            super().__init__()
            self.success_count = 0
            self.failure_count = 0
            self.success_events = []
            self.failure_events = []

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            """Track successful events."""
            self.success_count += 1
            self.success_events.append({
                "model": kwargs.get("model"),
                "duration": end_time - start_time,
                "response": response_obj
            })

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            """Track failed events."""
            self.failure_count += 1
            self.failure_events.append({
                "model": kwargs.get("model"),
                "duration": end_time - start_time,
                "error": response_obj
            })

    @pytest.mark.asyncio
    async def test_request_modifier_adds_system_message(
        self,
        mock_user_api_key_dict,
        mock_cache
    ):
        """Test RequestModifierLogger adds system message."""
        logger = self.RequestModifierLogger()
        data = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }

        result = await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=data,
            call_type="completion"
        )

        assert result is not None
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_request_modifier_handles_missing_messages(
        self,
        mock_user_api_key_dict,
        mock_cache
    ):
        """Test RequestModifierLogger handles data without messages."""
        logger = self.RequestModifierLogger()
        data = {"model": "text-embedding-ada-002", "input": "test"}

        result = await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=data,
            call_type="embedding"
        )

        assert result is not None
        assert "messages" not in result

    @pytest.mark.asyncio
    async def test_response_modifier_adds_prefix(
        self,
        mock_user_api_key_dict,
        mock_litellm_call_data,
        mock_model_response
    ):
        """Test ResponseModifierLogger adds prefix to response."""
        logger = self.ResponseModifierLogger()

        result = await logger.async_post_call_success_hook(
            data=mock_litellm_call_data,
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_model_response
        )

        assert result is not None
        if hasattr(result, 'choices') and len(result.choices) > 0:
            content = result.choices[0].message.content
            assert content.startswith("[VERIFIED] ")

    @pytest.mark.asyncio
    async def test_event_logger_tracks_success(
        self,
        mock_completion_kwargs,
        mock_model_response
    ):
        """Test EventLoggingLogger tracks successful events."""
        logger = self.EventLoggingLogger()
        start = time.time()
        end = start + 0.5

        await logger.async_log_success_event(
            kwargs=mock_completion_kwargs,
            response_obj=mock_model_response,
            start_time=start,
            end_time=end
        )

        assert logger.success_count == 1
        assert len(logger.success_events) == 1
        assert logger.success_events[0]["model"] == mock_completion_kwargs["model"]
        assert logger.success_events[0]["duration"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_event_logger_tracks_failure(
        self,
        mock_completion_kwargs,
        mock_exception
    ):
        """Test EventLoggingLogger tracks failed events."""
        logger = self.EventLoggingLogger()
        start = time.time()
        end = start + 0.3

        await logger.async_log_failure_event(
            kwargs=mock_completion_kwargs,
            response_obj=mock_exception,
            start_time=start,
            end_time=end
        )

        assert logger.failure_count == 1
        assert len(logger.failure_events) == 1
        assert logger.failure_events[0]["error"] == mock_exception

    @pytest.mark.asyncio
    async def test_event_logger_tracks_multiple_events(
        self,
        mock_completion_kwargs,
        mock_model_response,
        mock_exception
    ):
        """Test EventLoggingLogger tracks multiple events."""
        logger = self.EventLoggingLogger()
        start = time.time()

        # Log 3 successes
        for i in range(3):
            await logger.async_log_success_event(
                kwargs=mock_completion_kwargs,
                response_obj=mock_model_response,
                start_time=start,
                end_time=start + i
            )

        # Log 2 failures
        for i in range(2):
            await logger.async_log_failure_event(
                kwargs=mock_completion_kwargs,
                response_obj=mock_exception,
                start_time=start,
                end_time=start + i
            )

        assert logger.success_count == 3
        assert logger.failure_count == 2
        assert len(logger.success_events) == 3
        assert len(logger.failure_events) == 2


class TestLoggerCallTypes:
    """Test logger behavior with different call types."""

    class CallTypeTracker(BaseCustomLogger):
        """Logger that tracks call types."""

        def __init__(self):
            super().__init__()
            self.call_types = []

        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            """Track the call type."""
            self.call_types.append(call_type)
            return data

    @pytest.mark.asyncio
    async def test_tracks_completion_call_type(
        self,
        mock_user_api_key_dict,
        mock_cache,
        mock_litellm_call_data
    ):
        """Test tracking completion call type."""
        logger = self.CallTypeTracker()
        await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=mock_litellm_call_data,
            call_type="completion"
        )
        assert "completion" in logger.call_types

    @pytest.mark.asyncio
    async def test_tracks_embedding_call_type(
        self,
        mock_user_api_key_dict,
        mock_cache
    ):
        """Test tracking embedding call type."""
        logger = self.CallTypeTracker()
        data = {"model": "text-embedding-ada-002", "input": "test"}
        await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=data,
            call_type="embeddings"
        )
        assert "embeddings" in logger.call_types

    @pytest.mark.asyncio
    async def test_tracks_multiple_call_types(
        self,
        mock_user_api_key_dict,
        mock_cache,
        mock_litellm_call_data
    ):
        """Test tracking multiple different call types."""
        logger = self.CallTypeTracker()

        call_types = ["completion", "embeddings", "image_generation", "moderation"]
        for call_type in call_types:
            await logger.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=mock_cache,
                data=mock_litellm_call_data,
                call_type=call_type
            )

        assert logger.call_types == call_types


class TestLoggerExceptionHandling:
    """Test logger error handling patterns."""

    class SafeLogger(BaseCustomLogger):
        """Logger with exception handling."""

        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            """Safe hook with exception handling."""
            try:
                # Simulate some processing that might fail
                if data.get("should_fail"):
                    raise ValueError("Intentional test error")
                data["processed"] = True
                return data
            except Exception as e:
                # Log error but don't block the request
                print(f"Error in pre_call_hook: {e}")
                return None  # Return None to use original data

    @pytest.mark.asyncio
    async def test_safe_logger_handles_exceptions(
        self,
        mock_user_api_key_dict,
        mock_cache
    ):
        """Test SafeLogger handles exceptions gracefully."""
        logger = self.SafeLogger()
        data = {"model": "test-model", "should_fail": True}

        result = await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=data,
            call_type="completion"
        )

        # Should return None instead of raising
        assert result is None

    @pytest.mark.asyncio
    async def test_safe_logger_processes_normally_when_no_error(
        self,
        mock_user_api_key_dict,
        mock_cache
    ):
        """Test SafeLogger processes normally when no error occurs."""
        logger = self.SafeLogger()
        data = {"model": "test-model", "should_fail": False}

        result = await logger.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=data,
            call_type="completion"
        )

        assert result is not None
        assert result.get("processed") is True
