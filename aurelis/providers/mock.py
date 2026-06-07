"""Deterministic mock provider.

The point of this file: the entire harness — runner, cache, scorers, metrics —
is exercised in CI with no API key and no network. A mock that just returns a
constant would test nothing, so this one is *scriptable*: you hand it rules that
map a substring in the prompt to a canned response. Tests use it to drive every
branch (correct answer, sycophantic flip, refusal, miscalibration) on demand.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from aurelis.providers.base import Provider
from aurelis.types import GenerationParams, Message, ModelResponse


class MockProvider(Provider):
    name = "mock"

    def __init__(
        self,
        rules: list[tuple[str, ModelResponse]] | None = None,
        default: ModelResponse | None = None,
    ) -> None:
        # rules: (regex_pattern, response) — first match on the last user turn wins.
        self._rules = [(re.compile(p, re.I | re.S), r) for p, r in (rules or [])]
        self._default = default or ModelResponse(text="", stop_reason="end_turn")
        self.calls: list[list[Message]] = []

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> ModelResponse:
        self.calls.append(list(messages))
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        for pattern, response in self._rules:
            if pattern.search(last_user):
                return response
        return self._default
