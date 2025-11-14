# Development Guide

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Create virtual environment
uv venv

# Install with dev dependencies
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=litellm_plugin --cov-report=html

# Run specific test file
pytest tests/test_base_provider.py -v

# Run specific test
pytest tests/test_base_provider.py::TestBaseCustomProvider::test_provider_initialization -v
```

## Test Structure

- `tests/conftest.py` - Shared fixtures and pytest configuration
- `tests/fixtures/` - Mock objects and sample data
- `tests/test_*.py` - Core module tests
- `tests/test_examples/` - Example implementation tests

**Coverage:** 89% (141 tests passing)

## Key Testing Patterns

- All async hooks tested with `@pytest.mark.asyncio`
- Mock LiteLLM objects for testing without external dependencies
- Exception safety and graceful degradation patterns
- Edge case coverage (unicode, special chars, empty inputs, long messages)

## Quick Commands

```bash
# Format code
ruff format .

# Lint
ruff check .

# Type check
mypy litellm_plugin/
```
