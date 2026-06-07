"""Experiment store. Every grading run is written as a self-describing JSON
record with enough provenance (model, params, grader, git SHA, timestamp, cache
stats) to reproduce or audit it later — which, for a tool that puts grades in
front of students, is not optional. An append-only index makes runs listable.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class RunRecord:
    run_id: str
    task: str
    grader: str
    provider: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    validation: dict[str, Any]
    created_at: str
    git_sha: str | None
    cache_hits: int
    cache_misses: int
    assessments: list[dict[str, Any]] = field(default_factory=list)


class RunStore:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"

    def save(self, record: RunRecord) -> Path:
        path = self.root / f"{record.run_id}.json"
        path.write_text(json.dumps(asdict(record), indent=2))
        with self.index.open("a") as fh:
            fh.write(json.dumps({
                "run_id": record.run_id,
                "task": record.task,
                "grader": record.grader,
                "model": record.params.get("model"),
                "created_at": record.created_at,
                "metrics": record.metrics,
                "validation": record.validation,
            }) + "\n")
        return path

    def load(self, run_id: str) -> RunRecord:
        return RunRecord(**json.loads((self.root / f"{run_id}.json").read_text()))

    def list_runs(self) -> list[dict]:
        if not self.index.exists():
            return []
        return [json.loads(l) for l in self.index.read_text().splitlines() if l.strip()]
