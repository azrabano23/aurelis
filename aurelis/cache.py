"""Content-addressed response cache.

Reproducibility for LLM evals can't come from a random seed — providers don't
expose one and Opus 4.7+ removed temperature. So we get it the other way: the
*first* run records every (messages, params) -> response, and every rerun reads
the recorded response instead of re-querying. Same inputs -> same outputs,
byte-for-byte, and reruns are free. Delete the cache to force fresh sampling.

The key is a SHA-256 over the canonical (sorted, separators-pinned) JSON of the
inputs, so it's stable across processes and machines.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from aurelis.types import GenerationParams, Message, ModelResponse


def cache_key(messages: Sequence[Message], params: GenerationParams) -> str:
    payload = {
        "messages": [m.to_api() for m in messages],
        "params": params.fingerprint(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, root: str | Path = ".aurelis_cache") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> ModelResponse | None:
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        data = json.loads(path.read_text())
        return ModelResponse(**data)

    def put(self, key: str, response: ModelResponse) -> None:
        data = {
            "text": response.text,
            "stop_reason": response.stop_reason,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "request_id": response.request_id,
        }
        self._path(key).write_text(json.dumps(data, indent=2))
