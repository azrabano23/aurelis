"""Provider interface — the one seam every eval runs through.

An eval never imports a vendor SDK directly; it asks a Provider to `generate`.
That keeps evals model-agnostic and lets the whole test suite run against the
deterministic MockProvider with no network and no API key.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from aurelis.types import GenerationParams, Message, ModelResponse


class Provider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, messages: Sequence[Message], params: GenerationParams) -> ModelResponse:
        """Return a single completion for `messages` under `params`."""

    def generate_text(self, prompt: str, params: GenerationParams) -> ModelResponse:
        return self.generate([Message("user", prompt)], params)
