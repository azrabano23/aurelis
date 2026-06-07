"""Deterministic checklist grader.

Scores each dimension purely by whether the rubric's `required_elements` and the
case's `reference_elements` literally appear in the note. No model in the loop,
so it's instant, free, and perfectly reproducible — the right tool for the
objectively-present-or-absent parts of a note, and a fast baseline to compare the
LLM grader against. It cannot judge clinical *reasoning*; that's what the LLM
grader is for.
"""
from __future__ import annotations

import re

from aurelis.grading.base import Grader
from aurelis.types import (
    ClinicalCase,
    DimensionScore,
    Message,  # noqa: F401  (kept for interface symmetry)
    NoteAssessment,
    Rubric,
    StudentNote,
)


def _present(needle: str, haystack: str) -> bool:
    return needle.lower() in haystack


class ChecklistGrader(Grader):
    name = "checklist"

    def grade(self, case, note, rubric, generate=None) -> NoteAssessment:
        text = re.sub(r"\s+", " ", note.text.lower())
        scores = []
        for dim in rubric.dimensions:
            # The "answer key" for a dimension = its required_elements plus any
            # case-specific reference facts filed under that dimension key.
            checklist = list(dim.required_elements) + case.reference_elements.get(dim.key, [])
            if not checklist:
                # Nothing objectively checkable (e.g. clarity) -> abstain at half credit.
                scores.append(DimensionScore(dim.key, dim.max_points / 2, dim.max_points,
                                             feedback="not checkable deterministically"))
                continue
            hits = [c for c in checklist if _present(c, text)]
            missing = [c for c in checklist if not _present(c, text)]
            frac = len(hits) / len(checklist)
            scores.append(
                DimensionScore(
                    dimension_key=dim.key,
                    points=round(frac * dim.max_points, 2),
                    max_points=dim.max_points,
                    feedback=f"{len(hits)}/{len(checklist)} required elements present",
                    missing=tuple(missing),
                    evidence=tuple(hits),
                )
            )
        return NoteAssessment(case.id, note.note_id, tuple(scores), grader=self.name)
