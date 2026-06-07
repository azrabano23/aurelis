"""Validate a grader on real ACI-Bench clinical notes via section perturbation.

    python scripts/validate_acibench.py --grader checklist --limit 20
    python scripts/validate_acibench.py --grader llm --provider anthropic --limit 10

The checklist grader runs offline (no API key). The LLM grader needs
ANTHROPIC_API_KEY and issues ~(1 + n_perturbations) * n_dims calls per note.
"""
from __future__ import annotations

import argparse
import json

from aurelis.cache import ResponseCache, cache_key
from aurelis.datasets import load_acibench
from aurelis.grading import get_grader
from aurelis.providers import get_provider
from aurelis.rubric import SOAP_RUBRIC
from aurelis.types import GenerationParams
from aurelis.validation import run_perturbation_validation


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grader", default="checklist", choices=["checklist", "llm"])
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    pairs = load_acibench(limit=args.limit)
    grader = get_grader(args.grader)

    generate = None
    if args.grader == "llm":
        provider = get_provider(args.provider)
        params = GenerationParams(model=args.model, max_tokens=1024, thinking=True, effort="high")
        cache = ResponseCache()

        def generate(messages):  # noqa: E306
            key = cache_key(messages, params)
            hit = cache.get(key)
            if hit is not None:
                return hit
            resp = provider.generate(messages, params)
            cache.put(key, resp)
            return resp

    summary = run_perturbation_validation(pairs, grader, SOAP_RUBRIC, generate)
    print(f"grader={args.grader}  notes={len(pairs)}  (ACI-Bench valid)")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
