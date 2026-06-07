"""Core data types.

Two families live here:
  - model I/O types (Message, GenerationParams, ModelResponse) — the contract the
    Provider layer speaks; nothing above it imports a vendor SDK.
  - clinical-documentation types (ClinicalCase, StudentNote, Rubric, ...) — the
    domain Aurelis actually grades.

All frozen dataclasses, so an assessment is fully described by serializable
values — which is what makes a grade reproducible and auditable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "system"]


# ----------------------------- model I/O -----------------------------

@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_api(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class GenerationParams:
    """Knobs that affect the grader model's output. Hashed into the cache key so
    that the same note graded with the same settings returns the same grade —
    a hard requirement for fairness in an educational setting.

    temperature/top_p are intentionally absent (removed on Opus 4.7+); grade
    reproducibility comes from caching the grader's output, not from a seed.
    """

    model: str = "claude-opus-4-8"
    max_tokens: int = 1024
    effort: str | None = None        # low | medium | high | max
    thinking: bool = False           # adaptive thinking on/off

    def fingerprint(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    stop_reason: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


# --------------------------- clinical domain ---------------------------

@dataclass(frozen=True)
class RubricDimension:
    """One graded axis of a clinical note (e.g. 'assessment'). `criteria` is the
    instruction the grader applies; `required_elements` powers deterministic
    checklist scoring for facts that are objectively present-or-absent."""

    key: str
    name: str
    max_points: int
    criteria: str
    required_elements: tuple[str, ...] = ()


@dataclass(frozen=True)
class Rubric:
    id: str
    name: str
    dimensions: tuple[RubricDimension, ...]

    @property
    def max_total(self) -> int:
        return sum(d.max_points for d in self.dimensions)

    def dimension(self, key: str) -> RubricDimension:
        return next(d for d in self.dimensions if d.key == key)


@dataclass(frozen=True)
class ClinicalCase:
    """A patient encounter the student documented. `reference_elements` are the
    clinically salient facts a complete note should capture — the answer key."""

    id: str
    specialty: str
    vignette: str
    reference_elements: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class StudentNote:
    case_id: str
    note_id: str
    text: str
    quality_label: str | None = None          # synthetic-data tag, not used in grading
    human_scores: dict[str, int] = field(default_factory=dict)  # dimension_key -> points (gold)


@dataclass(frozen=True)
class DimensionScore:
    dimension_key: str
    points: float
    max_points: int
    feedback: str = ""
    missing: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class NoteAssessment:
    case_id: str
    note_id: str
    scores: tuple[DimensionScore, ...]
    grader: str
    model: str | None = None

    @property
    def total(self) -> float:
        return sum(s.points for s in self.scores)

    @property
    def max_total(self) -> int:
        return sum(s.max_points for s in self.scores)

    @property
    def percent(self) -> float:
        return 100.0 * self.total / self.max_total if self.max_total else 0.0
