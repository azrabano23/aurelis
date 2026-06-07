from aurelis.providers.base import Provider
from aurelis.providers.mock import MockProvider

__all__ = ["Provider", "MockProvider", "get_provider"]


def get_provider(name: str, **kwargs) -> Provider:
    """Resolve a provider by name. Anthropic is imported lazily so the mock
    path never requires the SDK."""
    if name == "mock":
        return MockProvider(**kwargs)
    if name == "anthropic":
        from aurelis.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(**kwargs)
    raise ValueError(f"unknown provider: {name!r} (expected 'anthropic' or 'mock')")
