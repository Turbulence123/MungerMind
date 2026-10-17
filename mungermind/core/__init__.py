"""Core runtime primitives: model access, configuration and messages."""

from .llm import LLM, PROVIDERS, ProviderSpec, detect_provider

__all__ = ["LLM", "PROVIDERS", "ProviderSpec", "detect_provider"]
