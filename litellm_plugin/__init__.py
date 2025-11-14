"""
LiteLLM Custom Plugin
A basic plugin structure for extending LiteLLM with custom providers and loggers.
"""

from litellm_plugin.base import BaseCustomProvider
from litellm_plugin.logger import BaseCustomLogger
from litellm_plugin.registry import PluginRegistry

__version__ = "0.1.0"
__all__ = ["BaseCustomProvider", "BaseCustomLogger", "PluginRegistry"]
