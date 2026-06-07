"""Anthropic provider — the real model backend.

Uses the official `anthropic` SDK. Notes that matter for correctness on
Opus 4.7+ (and which trip people up):
  - adaptive thinking only: thinking={"type": "adaptive"}; budget_tokens is gone
  - no temperature/top_p/top_k (removed — they 400)
  - effort lives under output_config, not top-level
  - stop_reason == "refusal" is a first-class signal, with structured stop_details
  - large max_tokens must stream; we stream above ~16k to dodge HTTP timeouts
"""
from __future__ import annotations

from collections.abc import Sequence

from aurelis.providers.base import Provider
from aurelis.types import GenerationParams, Message, ModelResponse

_STREAM_THRESHOLD = 16_000


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, client=None) -> None:
        # Imported lazily so the package (and its test suite on MockProvider)
        # works without the SDK or an API key installed.
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client

    def _build_kwargs(self, messages: Sequence[Message], params: GenerationParams) -> dict:
        kwargs: dict = {
            "model": params.model,
            "max_tokens": params.max_tokens,
            "messages": [m.to_api() for m in messages if m.role != "system"],
        }
        system = next((m.content for m in messages if m.role == "system"), None)
        if system:
            kwargs["system"] = system
        if params.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if params.effort:
            kwargs["output_config"] = {"effort": params.effort}
        return kwargs

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> ModelResponse:
        kwargs = self._build_kwargs(messages, params)

        if params.max_tokens > _STREAM_THRESHOLD:
            with self._client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        else:
            msg = self._client.messages.create(**kwargs)

        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return ModelResponse(
            text=text,
            stop_reason=msg.stop_reason,
            model=msg.model,
            input_tokens=getattr(msg.usage, "input_tokens", None),
            output_tokens=getattr(msg.usage, "output_tokens", None),
            request_id=getattr(msg, "_request_id", None),
        )
