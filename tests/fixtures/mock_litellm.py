"""Mock LiteLLM objects and utilities for testing."""

import time
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass


@dataclass
class MockMessage:
    """Mock message object."""
    content: str
    role: str = "assistant"

    def dict(self):
        return {"content": self.content, "role": self.role}


@dataclass
class MockUsage:
    """Mock usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class MockChoices:
    """Mock choices object."""
    finish_reason: str = "stop"
    index: int = 0
    message: Optional[MockMessage] = None

    def __post_init__(self):
        if self.message is None:
            self.message = MockMessage(content="Test response", role="assistant")

    def dict(self):
        return {
            "finish_reason": self.finish_reason,
            "index": self.index,
            "message": self.message.dict()
        }


@dataclass
class MockDelta:
    """Mock delta object for streaming."""
    content: str = ""
    role: Optional[str] = None

    def dict(self):
        result = {"content": self.content}
        if self.role:
            result["role"] = self.role
        return result


@dataclass
class MockStreamingChoices:
    """Mock streaming choices object."""
    finish_reason: Optional[str] = None
    index: int = 0
    delta: Optional[MockDelta] = None

    def __post_init__(self):
        if self.delta is None:
            self.delta = MockDelta(content="test")

    def dict(self):
        return {
            "finish_reason": self.finish_reason,
            "index": self.index,
            "delta": self.delta.dict()
        }


class MockModelResponse:
    """Mock ModelResponse object that mimics LiteLLM's ModelResponse."""

    def __init__(
        self,
        id: str = "chatcmpl-test",
        model: str = "test-model",
        object: str = "chat.completion",
        created: Optional[int] = None,
        choices: Optional[List] = None,
        usage: Optional[MockUsage] = None
    ):
        self.id = id
        self.model = model
        self.object = object
        self.created = created or int(time.time())
        self.choices = choices or [MockChoices()]
        self.usage = usage or MockUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )

    def dict(self):
        return {
            "id": self.id,
            "model": self.model,
            "object": self.object,
            "created": self.created,
            "choices": [c.dict() for c in self.choices],
            "usage": self.usage.dict()
        }

    def json(self):
        import json
        return json.dumps(self.dict())


class MockStreamingResponse:
    """Mock streaming response."""

    def __init__(
        self,
        id: str = "chatcmpl-test",
        model: str = "test-model",
        chunks: Optional[List[str]] = None
    ):
        self.id = id
        self.model = model
        self.chunks = chunks or ["Hello", " world", "!"]

    async def __aiter__(self) -> AsyncIterator[MockModelResponse]:
        """Async iterator for streaming chunks."""
        for i, chunk in enumerate(self.chunks):
            response = MockModelResponse(
                id=self.id,
                model=self.model,
                object="chat.completion.chunk",
                choices=[MockStreamingChoices(
                    delta=MockDelta(
                        content=chunk,
                        role="assistant" if i == 0 else None
                    )
                )]
            )
            response.usage = None
            yield response

        # Final chunk with finish_reason
        final_response = MockModelResponse(
            id=self.id,
            model=self.model,
            object="chat.completion.chunk",
            choices=[MockStreamingChoices(
                finish_reason="stop",
                delta=MockDelta(content="")
            )]
        )
        final_response.usage = None
        yield final_response


def create_mock_response(
    content: str = "Test response",
    model: str = "test-model",
    role: str = "assistant"
) -> MockModelResponse:
    """Helper to create a mock response with custom content."""
    return MockModelResponse(
        model=model,
        choices=[MockChoices(
            message=MockMessage(content=content, role=role)
        )]
    )


def create_mock_error_response(
    error_message: str = "Test error",
    error_type: str = "APIError"
) -> Dict[str, Any]:
    """Helper to create a mock error response."""
    return {
        "error": {
            "message": error_message,
            "type": error_type,
            "code": "test_error"
        }
    }


async def create_mock_streaming_response(
    chunks: Optional[List[str]] = None,
    model: str = "test-model"
) -> AsyncIterator[MockModelResponse]:
    """Helper to create a mock streaming response."""
    response = MockStreamingResponse(model=model, chunks=chunks)
    async for chunk in response:
        yield chunk
