"""Tests for PluginRegistry."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from litellm_plugin.registry import (
    PluginRegistry,
    register_provider,
    get_provider,
    initialize_plugins,
    get_registry,
)
from litellm_plugin.base import BaseCustomProvider


# Test provider implementations
class TestProviderA(BaseCustomProvider):
    """First test provider."""

    def completion(self, model: str, messages: list, **kwargs):
        from tests.fixtures.mock_litellm import create_mock_response
        return create_mock_response(content=f"Response from TestProviderA")


class TestProviderB(BaseCustomProvider):
    """Second test provider."""

    def completion(self, model: str, messages: list, **kwargs):
        from tests.fixtures.mock_litellm import create_mock_response
        return create_mock_response(content=f"Response from TestProviderB")


class NotAProvider:
    """Class that doesn't inherit from BaseCustomProvider."""
    pass


class TestPluginRegistryBasics:
    """Test basic PluginRegistry functionality."""

    def test_registry_initialization(self):
        """Test registry initializes with empty providers."""
        registry = PluginRegistry()
        assert len(registry.list_providers()) == 0

    def test_register_provider(self):
        """Test registering a provider."""
        registry = PluginRegistry()
        registry.register("test_provider", TestProviderA)
        assert "test_provider" in registry.list_providers()

    def test_register_multiple_providers(self):
        """Test registering multiple providers."""
        registry = PluginRegistry()
        registry.register("provider_a", TestProviderA)
        registry.register("provider_b", TestProviderB)

        providers = registry.list_providers()
        assert len(providers) == 2
        assert "provider_a" in providers
        assert "provider_b" in providers

    def test_register_duplicate_provider_raises_error(self):
        """Test registering same provider name twice raises ValueError."""
        registry = PluginRegistry()
        registry.register("test_provider", TestProviderA)

        with pytest.raises(ValueError, match="Provider 'test_provider' is already registered"):
            registry.register("test_provider", TestProviderB)

    def test_register_invalid_provider_raises_error(self):
        """Test registering non-BaseCustomProvider class raises TypeError."""
        registry = PluginRegistry()

        with pytest.raises(TypeError, match="Provider class must inherit from BaseCustomProvider"):
            registry.register("invalid", NotAProvider)

    def test_unregister_provider(self):
        """Test unregistering a provider."""
        registry = PluginRegistry()
        registry.register("test_provider", TestProviderA)
        assert "test_provider" in registry.list_providers()

        registry.unregister("test_provider")
        assert "test_provider" not in registry.list_providers()

    def test_unregister_nonexistent_provider_does_nothing(self):
        """Test unregistering a non-existent provider doesn't raise error."""
        registry = PluginRegistry()
        # Should not raise
        registry.unregister("nonexistent")

    def test_list_providers_returns_copy(self):
        """Test list_providers returns a list that can be safely modified."""
        registry = PluginRegistry()
        registry.register("provider_a", TestProviderA)

        providers = registry.list_providers()
        providers.append("fake_provider")

        # Original list should not be modified
        assert "fake_provider" not in registry.list_providers()


class TestProviderInstances:
    """Test provider instance management."""

    def test_get_provider_creates_instance(self):
        """Test get_provider creates and caches instance."""
        registry = PluginRegistry()
        registry.register("test_provider", TestProviderA)

        instance = registry.get_provider("test_provider")
        assert instance is not None
        assert isinstance(instance, TestProviderA)

    def test_get_provider_returns_singleton(self):
        """Test get_provider returns same instance on subsequent calls."""
        registry = PluginRegistry()
        registry.register("test_provider", TestProviderA)

        instance1 = registry.get_provider("test_provider")
        instance2 = registry.get_provider("test_provider")

        assert instance1 is instance2

    def test_get_provider_returns_none_for_unregistered(self):
        """Test get_provider returns None for unregistered provider."""
        registry = PluginRegistry()
        instance = registry.get_provider("nonexistent")
        assert instance is None

    def test_unregister_removes_instance(self):
        """Test unregister also removes cached instance."""
        registry = PluginRegistry()
        registry.register("test_provider", TestProviderA)

        # Create instance
        instance = registry.get_provider("test_provider")
        assert instance is not None

        # Unregister
        registry.unregister("test_provider")

        # Instance should be gone
        assert registry.get_provider("test_provider") is None


class TestInitialization:
    """Test LiteLLM initialization."""

    @patch('litellm_plugin.registry.LITELLM_AVAILABLE', True)
    @patch('litellm_plugin.registry.litellm')
    def test_initialize_registers_with_litellm(self, mock_litellm):
        """Test initialize registers providers with litellm."""
        mock_litellm.custom_provider_map = []

        registry = PluginRegistry()
        registry.register("test_provider", TestProviderA)
        registry.initialize()

        assert len(mock_litellm.custom_provider_map) == 1
        assert mock_litellm.custom_provider_map[0]["provider"] == "test_provider"

    @patch('litellm_plugin.registry.LITELLM_AVAILABLE', True)
    @patch('litellm_plugin.registry.litellm')
    def test_initialize_multiple_providers(self, mock_litellm):
        """Test initialize registers multiple providers."""
        mock_litellm.custom_provider_map = []

        registry = PluginRegistry()
        registry.register("provider_a", TestProviderA)
        registry.register("provider_b", TestProviderB)
        registry.initialize()

        assert len(mock_litellm.custom_provider_map) == 2

    @patch('litellm_plugin.registry.LITELLM_AVAILABLE', False)
    def test_initialize_raises_without_litellm(self):
        """Test initialize raises ImportError when litellm not available."""
        registry = PluginRegistry()
        registry.register("test_provider", TestProviderA)

        with pytest.raises(ImportError, match="litellm is not installed"):
            registry.initialize()

    @patch('litellm_plugin.registry.LITELLM_AVAILABLE', True)
    @patch('litellm_plugin.registry.litellm')
    def test_initialize_creates_instances(self, mock_litellm):
        """Test initialize creates provider instances."""
        mock_litellm.custom_provider_map = []

        registry = PluginRegistry()
        registry.register("test_provider", TestProviderA)

        # Instance shouldn't exist yet
        assert registry._instances.get("test_provider") is None

        registry.initialize()

        # Instance should be created
        assert registry._instances.get("test_provider") is not None
        assert isinstance(registry._instances["test_provider"], TestProviderA)

    @patch('litellm_plugin.registry.LITELLM_AVAILABLE', True)
    @patch('litellm_plugin.registry.litellm')
    def test_initialize_provider_single(self, mock_litellm):
        """Test initialize_provider initializes a single provider."""
        mock_litellm.custom_provider_map = []

        registry = PluginRegistry()
        registry.register("provider_a", TestProviderA)
        registry.register("provider_b", TestProviderB)

        registry.initialize_provider("provider_a")

        # Only provider_a should be initialized
        assert len(mock_litellm.custom_provider_map) == 1
        assert mock_litellm.custom_provider_map[0]["provider"] == "provider_a"

    @patch('litellm_plugin.registry.LITELLM_AVAILABLE', True)
    @patch('litellm_plugin.registry.litellm')
    def test_initialize_provider_nonexistent_raises_error(self, mock_litellm):
        """Test initialize_provider raises ValueError for unregistered provider."""
        mock_litellm.custom_provider_map = []

        registry = PluginRegistry()

        with pytest.raises(ValueError, match="Provider 'nonexistent' is not registered"):
            registry.initialize_provider("nonexistent")

    @patch('litellm_plugin.registry.LITELLM_AVAILABLE', False)
    def test_initialize_provider_raises_without_litellm(self):
        """Test initialize_provider raises ImportError when litellm not available."""
        registry = PluginRegistry()
        registry.register("test_provider", TestProviderA)

        with pytest.raises(ImportError, match="litellm is not installed"):
            registry.initialize_provider("test_provider")


class TestGlobalRegistry:
    """Test global registry functions."""

    def test_get_registry_returns_singleton(self):
        """Test get_registry returns the same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_register_provider_uses_global_registry(self):
        """Test register_provider adds to global registry."""
        # Clean up any existing providers
        registry = get_registry()
        if "global_test_provider" in registry.list_providers():
            registry.unregister("global_test_provider")

        register_provider("global_test_provider", TestProviderA)

        assert "global_test_provider" in get_registry().list_providers()

        # Cleanup
        registry.unregister("global_test_provider")

    def test_get_provider_uses_global_registry(self):
        """Test get_provider retrieves from global registry."""
        registry = get_registry()

        # Clean up
        if "global_test_provider" in registry.list_providers():
            registry.unregister("global_test_provider")

        register_provider("global_test_provider", TestProviderA)
        instance = get_provider("global_test_provider")

        assert instance is not None
        assert isinstance(instance, TestProviderA)

        # Cleanup
        registry.unregister("global_test_provider")

    @patch('litellm_plugin.registry.LITELLM_AVAILABLE', True)
    @patch('litellm_plugin.registry.litellm')
    def test_initialize_plugins_uses_global_registry(self, mock_litellm):
        """Test initialize_plugins initializes global registry."""
        mock_litellm.custom_provider_map = []

        registry = get_registry()

        # Clean up
        if "global_test_provider" in registry.list_providers():
            registry.unregister("global_test_provider")

        register_provider("global_test_provider", TestProviderA)
        initialize_plugins()

        # Should have registered with litellm
        provider_names = [p["provider"] for p in mock_litellm.custom_provider_map]
        assert "global_test_provider" in provider_names

        # Cleanup
        registry.unregister("global_test_provider")


class TestProviderValidation:
    """Test provider validation."""

    def test_register_requires_base_provider_subclass(self):
        """Test that only BaseCustomProvider subclasses can be registered."""
        registry = PluginRegistry()

        # Should work
        registry.register("valid", TestProviderA)

        # Should fail
        with pytest.raises(TypeError):
            registry.register("invalid", NotAProvider)

    def test_provider_instance_has_correct_type(self):
        """Test that get_provider returns correct instance type."""
        registry = PluginRegistry()
        registry.register("provider_a", TestProviderA)
        registry.register("provider_b", TestProviderB)

        instance_a = registry.get_provider("provider_a")
        instance_b = registry.get_provider("provider_b")

        assert isinstance(instance_a, TestProviderA)
        assert isinstance(instance_b, TestProviderB)
        assert not isinstance(instance_a, TestProviderB)
        assert not isinstance(instance_b, TestProviderA)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_provider_name(self):
        """Test registering with empty provider name."""
        registry = PluginRegistry()
        registry.register("", TestProviderA)

        assert "" in registry.list_providers()
        instance = registry.get_provider("")
        assert instance is not None

    def test_special_characters_in_provider_name(self):
        """Test provider names with special characters."""
        registry = PluginRegistry()
        special_names = [
            "test-provider",
            "test_provider",
            "test.provider",
            "test:provider"
        ]

        for name in special_names:
            registry.register(name, TestProviderA)

        assert all(name in registry.list_providers() for name in special_names)

    def test_registry_state_after_multiple_operations(self):
        """Test registry maintains consistent state through multiple operations."""
        registry = PluginRegistry()

        # Register
        registry.register("provider_1", TestProviderA)
        registry.register("provider_2", TestProviderB)
        assert len(registry.list_providers()) == 2

        # Get instances
        inst1 = registry.get_provider("provider_1")
        inst2 = registry.get_provider("provider_2")
        assert inst1 is not None
        assert inst2 is not None

        # Unregister one
        registry.unregister("provider_1")
        assert len(registry.list_providers()) == 1
        assert registry.get_provider("provider_1") is None
        assert registry.get_provider("provider_2") is not None

        # Register new one with same name
        registry.register("provider_1", TestProviderA)
        new_inst1 = registry.get_provider("provider_1")
        assert new_inst1 is not None
        assert new_inst1 is not inst1  # Should be a new instance
