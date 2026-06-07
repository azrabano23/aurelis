"""LLM-as-judge grader.

This is the part that does what a checklist can't: judge clinical *reasoning* —
whether the differential is justified, whether the plan follows from the
assessment, whether the note is internally consistent. It grades one rubric
dimension per model call. That's deliberate: isolating each dimension keeps the
model's attention narrow, makes each grade independently cacheable, and means a
prompt tweak to one dimension doesn't perturb the others — standard rubric-eval
hygiene.

The grader returns, per dimension, a numeric score plus the *evidence it cited
from the note* and the *elements it found missing*, so feedback is actionable and
the grade is defensible rather than a black box. On Anthropic models the JSON
shape can additionally be pinned with output_config.format; here we parse
defensively so the same path works against the mock in tests.
"""
from __future__ import annotations

import json
import re

from aurelis.grading.base import GenerateFn, Grader
from aurelis.types import (
    ClinicalCase,
    DimensionScore,
    Message,
    NoteAssessment,
    Rubric,
    RubricDimension,
    StudentNote,
)

_SYSTEM = (
    "You are an experienced attending physician grading a medical student's "
    "clinical note for one rubric dimension. Grade strictly against the stated "
    "criteria — reward correct clinical reasoning, penalize omissions, fabricated "
    "findings, and internal contradictions. Cite specific phrases from the note "
    "as evidence. Respond with ONLY a JSON object and no other text: "
    '{"points": <number 0..max>, "feedback": "<2-3 sentences>", '
    '"missing": ["..."], "evidence": ["..."]}'
)

_TEMPLATE = """PATIENT ENCOUNTER (what the student saw):
{vignette}

RUBRIC DIMENSION: {dim_name}  (0 to {max_points} points)
Criteria: {criteria}

STUDENT NOTE:
\"\"\"
{note}
\"\"\"

Grade ONLY the {dim_name} dimension against the criteria above."""


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class LLMGrader(Grader):
    name = "llm"

    def _grade_dimension(
        self,
        case: ClinicalCase,
        note: StudentNote,
        dim: RubricDimension,
        generate: GenerateFn,
    ) -> DimensionScore:
        prompt = _TEMPLATE.format(
            vignette=case.vignette,
            dim_name=dim.name,
            max_points=dim.max_points,
            criteria=dim.criteria,
            note=note.text,
        )
        response = generate([Message("system", _SYSTEM), Message("user", prompt)])
        verdict = _extract_json(response.text)

        raw_points = verdict.get("points", 0)
        try:
            points = float(raw_points)
        except (TypeError, ValueError):
            points = 0.0
        points = max(0.0, min(float(dim.max_points), points))

        return DimensionScore(
            dimension_key=dim.key,
            points=round(points, 2),
            max_points=dim.max_points,
            feedback=str(verdict.get("feedback", "")).strip(),
            missing=tuple(str(x) for x in verdict.get("missing", []) or []),
            evidence=tuple(str(x) for x in verdict.get("evidence", []) or []),
        )

    def grade(self, case, note, rubric, generate=None) -> NoteAssessment:
        if generate is None:
            raise ValueError("LLMGrader requires a `generate` callable")
        scores = tuple(
            self._grade_dimension(case, note, dim, generate)
            for dim in rubric.dimensions
        )
        model = None  # populated by the runner from the underlying responses
        return NoteAssessment(case.id, note.note_id, scores, grader=self.name, model=model)
