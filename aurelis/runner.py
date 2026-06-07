"""The grading loop: run a grader over a task's cases, with caching and provenance.

Thin by design — it owns exactly two cross-cutting concerns: the cache (so the
same note graded twice gets the identical grade, and reruns are free) and the
store (so every run is recorded with provenance). Everything domain-specific
lives in the Grader and the Task.
"""
from __future__ import annotations

import secrets
from collections.abc import Sequence
from datetime import datetime, timezone

from aurelis.cache import ResponseCache, cache_key
from aurelis.grading.base import Grader
from aurelis.providers.base import Provider
from aurelis.store import RunRecord, RunStore, git_sha
from aurelis.tasks.base import Task
from aurelis.types import GenerationParams, Message, ModelResponse, NoteAssessment


def _run_id(task_name: str, created_at: datetime) -> str:
    return f"{task_name}-{created_at:%Y%m%dT%H%M%S}-{secrets.token_hex(3)}"


def _assessment_to_dict(a: NoteAssessment) -> dict:
    return {
        "case_id": a.case_id,
        "note_id": a.note_id,
        "total": round(a.total, 2),
        "max_total": a.max_total,
        "percent": round(a.percent, 2),
        "scores": [
            {
                "dimension": s.dimension_key,
                "points": s.points,
                "max_points": s.max_points,
                "feedback": s.feedback,
                "missing": list(s.missing),
                "evidence": list(s.evidence),
            }
            for s in a.scores
        ],
    }


def run(
    task: Task,
    grader: Grader,
    provider: Provider,
    params: GenerationParams,
    *,
    cache: ResponseCache | None = None,
    store: RunStore | None = None,
    limit: int | None = None,
    created_at: datetime | None = None,
) -> RunRecord:
    cache = cache if cache is not None else ResponseCache()
    created_at = created_at or datetime.now(timezone.utc)

    def generate(messages: Sequence[Message]) -> ModelResponse:
        key = cache_key(messages, params)
        hit = cache.get(key)
        if hit is not None:
            return hit
        response = provider.generate(messages, params)
        cache.put(key, response)
        return response

    pairs = task.load()
    if limit is not None:
        pairs = pairs[:limit]

    notes = [note for _, note in pairs]
    assessments = [grader.grade(case, note, task.rubric, generate) for case, note in pairs]

    record = RunRecord(
        run_id=_run_id(task.name, created_at),
        task=task.name,
        grader=grader.name,
        provider=provider.name,
        params=params.fingerprint(),
        metrics=task.aggregate(assessments),
        validation=task.validate(assessments, notes),
        created_at=created_at.isoformat(),
        git_sha=git_sha(),
        cache_hits=cache.hits,
        cache_misses=cache.misses,
        assessments=[_assessment_to_dict(a) for a in assessments],
    )
    if store is not None:
        store.save(record)
    return record
