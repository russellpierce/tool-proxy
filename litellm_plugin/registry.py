"""
Plugin registry for managing custom LiteLLM providers.
"""

from typing import Dict, Optional, Type

from litellm_plugin.base import BaseCustomProvider

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    LITELLM_AVAILABLE = False


class PluginRegistry:
    """
    Registry for managing custom LiteLLM providers.

    This class handles registration and initialization of custom providers,
    making them available to LiteLLM via the custom_provider_map.

    Example:
        registry = PluginRegistry()
        registry.register("my_provider", MyProviderClass)
        registry.initialize()
    """

    def __init__(self):
        """Initialize the plugin registry."""
        self._providers: Dict[str, Type[BaseCustomProvider]] = {}
        self._instances: Dict[str, BaseCustomProvider] = {}

    def register(
        self,
        provider_name: str,
        provider_class: Type[BaseCustomProvider],
    ) -> None:
        """
        Register a custom provider.

        Args:
            provider_name: Unique identifier for the provider
            provider_class: Class implementing BaseCustomProvider

        Raises:
            ValueError: If provider_name is already registered
            TypeError: If provider_class doesn't inherit from BaseCustomProvider
        """
        if provider_name in self._providers:
            raise ValueError(f"Provider '{provider_name}' is already registered")

        if not issubclass(provider_class, BaseCustomProvider):
            raise TypeError(
                f"Provider class must inherit from BaseCustomProvider, "
                f"got {provider_class.__name__}"
            )

        self._providers[provider_name] = provider_class
        print(f"Registered provider: {provider_name}")

    def unregister(self, provider_name: str) -> None:
        """
        Unregister a custom provider.

        Args:
            provider_name: The provider identifier to remove
        """
        if provider_name in self._providers:
            del self._providers[provider_name]
            if provider_name in self._instances:
                del self._instances[provider_name]
            print(f"Unregistered provider: {provider_name}")

    def get_provider(self, provider_name: str) -> Optional[BaseCustomProvider]:
        """
        Get an instance of a registered provider.

        Args:
            provider_name: The provider identifier

        Returns:
            Provider instance or None if not found
        """
        if provider_name not in self._instances:
            if provider_name in self._providers:
                self._instances[provider_name] = self._providers[provider_name]()
        return self._instances.get(provider_name)

    def list_providers(self) -> list:
        """
        List all registered provider names.

        Returns:
            List of provider identifiers
        """
        return list(self._providers.keys())

    def initialize(self) -> None:
        """
        Initialize all registered providers with LiteLLM.

        This method registers all providers with litellm.custom_provider_map
        so they can be used with litellm.completion() and related functions.

        Raises:
            ImportError: If litellm is not installed
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is not installed. Install it with: pip install litellm"
            )

        for provider_name, provider_class in self._providers.items():
            # Instantiate the provider if not already done
            if provider_name not in self._instances:
                self._instances[provider_name] = provider_class()

            # Register with litellm
            litellm.custom_provider_map.append(
                {
                    "provider": provider_name,
                    "custom_handler": self._instances[provider_name],
                }
            )

        print(f"Initialized {len(self._providers)} provider(s) with LiteLLM")

    def initialize_provider(self, provider_name: str) -> None:
        """
        Initialize a specific provider with LiteLLM.

        Args:
            provider_name: The provider to initialize

        Raises:
            ValueError: If provider is not registered
            ImportError: If litellm is not installed
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is not installed. Install it with: pip install litellm"
            )

        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' is not registered")

        # Instantiate the provider if not already done
        if provider_name not in self._instances:
            self._instances[provider_name] = self._providers[provider_name]()

        # Register with litellm
        litellm.custom_provider_map.append(
            {
                "provider": provider_name,
                "custom_handler": self._instances[provider_name],
            }
        )

        print(f"Initialized provider '{provider_name}' with LiteLLM")


# Global registry instance
_global_registry = PluginRegistry()


def register_provider(
    provider_name: str,
    provider_class: Type[BaseCustomProvider],
) -> None:
    """
    Register a provider with the global registry.

    Args:
        provider_name: Unique identifier for the provider
        provider_class: Class implementing BaseCustomProvider
    """
    _global_registry.register(provider_name, provider_class)


def get_provider(provider_name: str) -> Optional[BaseCustomProvider]:
    """
    Get a provider from the global registry.

    Args:
        provider_name: The provider identifier

    Returns:
        Provider instance or None
    """
    return _global_registry.get_provider(provider_name)


def initialize_plugins() -> None:
    """Initialize all registered plugins with LiteLLM."""
    _global_registry.initialize()


def get_registry() -> PluginRegistry:
    """
    Get the global plugin registry.

    Returns:
        The global PluginRegistry instance
    """
    return _global_registry
