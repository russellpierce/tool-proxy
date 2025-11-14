"""
Example usage of the LiteLLM plugin structure.

This script demonstrates both custom providers and custom loggers.
"""

# Example 1: Using the Echo Provider
def example_echo_provider():
    """Demonstrate the echo provider."""
    print("=" * 60)
    print("Example 1: Echo Provider")
    print("=" * 60)

    try:
        import litellm
        from litellm_plugin.examples.echo_provider import EchoProvider
        from litellm_plugin.registry import register_provider, initialize_plugins

        # Register and initialize the echo provider
        register_provider("echo", EchoProvider)
        initialize_plugins()

        # Use the echo provider
        response = litellm.completion(
            model="echo/test",
            messages=[{"role": "user", "content": "Hello, LiteLLM!"}]
        )

        print(f"Response: {response.choices[0].message.content}")
        print()

    except ImportError:
        print("LiteLLM not installed. Install with: pip install litellm")
        print()


# Example 2: Using the Mock API Provider
def example_mock_api_provider():
    """Demonstrate the mock API provider."""
    print("=" * 60)
    print("Example 2: Mock API Provider")
    print("=" * 60)

    try:
        import litellm
        from litellm_plugin.examples.mock_api_provider import MockAPIProvider
        from litellm_plugin.registry import register_provider, initialize_plugins

        # Register and initialize the mock API provider
        register_provider("mock_api", MockAPIProvider)
        initialize_plugins()

        # Use the mock API provider
        response = litellm.completion(
            model="mock_api/gpt-mock",
            messages=[{"role": "user", "content": "What is the meaning of life?"}],
            api_key="test-key",
            temperature=0.7,
            max_tokens=50
        )

        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        print()

    except ImportError:
        print("LiteLLM not installed. Install with: pip install litellm")
        print()


# Example 3: Using Response Modifier (requires OpenAI key)
def example_response_modifier():
    """Demonstrate the response modifier logger."""
    print("=" * 60)
    print("Example 3: Response Modifier Logger")
    print("=" * 60)
    print("Note: This example requires a valid OpenAI API key.")
    print("Set OPENAI_API_KEY environment variable to test.")
    print()

    try:
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("Skipping - no OPENAI_API_KEY found")
            print()
            return

        import litellm
        from litellm_plugin.examples.response_modifier import ResponseModifier

        # Register the response modifier
        litellm.callbacks = [ResponseModifier(prefix="[AI Response] ")]

        # Use LiteLLM with OpenAI
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello!"}]
        )

        print(f"Modified Response: {response.choices[0].message.content}")
        print()

    except ImportError:
        print("LiteLLM not installed. Install with: pip install litellm")
        print()


# Example 4: Using Request Logger
def example_request_logger():
    """Demonstrate the request logger."""
    print("=" * 60)
    print("Example 4: Request Logger")
    print("=" * 60)

    try:
        import litellm
        from litellm_plugin.examples.echo_provider import EchoProvider
        from litellm_plugin.examples.response_modifier import RequestLogger
        from litellm_plugin.registry import register_provider, initialize_plugins

        # Register echo provider
        register_provider("echo", EchoProvider)
        initialize_plugins()

        # Add request logger
        litellm.callbacks = [RequestLogger(verbose=True)]

        # Make a request - it will be logged
        response = litellm.completion(
            model="echo/test",
            messages=[{"role": "user", "content": "Test logging"}]
        )

        print()

    except ImportError:
        print("LiteLLM not installed. Install with: pip install litellm")
        print()


# Example 5: Combining Multiple Loggers
def example_multiple_loggers():
    """Demonstrate using multiple loggers together."""
    print("=" * 60)
    print("Example 5: Multiple Loggers")
    print("=" * 60)

    try:
        import litellm
        from litellm_plugin.examples.echo_provider import EchoProvider
        from litellm_plugin.examples.response_modifier import (
            ResponseModifier,
            RequestLogger,
        )
        from litellm_plugin.registry import register_provider, initialize_plugins

        # Register echo provider
        register_provider("echo", EchoProvider)
        initialize_plugins()

        # Add multiple loggers
        litellm.callbacks = [
            RequestLogger(verbose=True),
            ResponseModifier(prefix="[FINAL] "),
        ]

        # Make a request - both loggers will process it
        response = litellm.completion(
            model="echo/test",
            messages=[{"role": "user", "content": "Testing multiple loggers"}]
        )

        print(f"\nFinal Response: {response.choices[0].message.content}")
        print()

    except ImportError:
        print("LiteLLM not installed. Install with: pip install litellm")
        print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  LiteLLM Plugin Structure - Usage Examples".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    # Run examples that don't require external API keys
    example_echo_provider()
    example_mock_api_provider()
    example_request_logger()
    example_multiple_loggers()

    # This one requires OpenAI API key
    example_response_modifier()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nFor more information, see the README.md file.")
    print()


if __name__ == "__main__":
    main()
