"""Grader interface.

A Grader turns (case, note, rubric) into a NoteAssessment. The runner hands it a
cached `generate` callable, so a grader that uses a model never touches a vendor
SDK or the cache directly — and a deterministic grader simply ignores it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from aurelis.types import (
    ClinicalCase,
    Message,
    ModelResponse,
    NoteAssessment,
    Rubric,
    StudentNote,
)

GenerateFn = Callable[[Sequence[Message]], ModelResponse]


class Grader(ABC):
    name: str = "base"

    @abstractmethod
    def grade(
        self,
        case: ClinicalCase,
        note: StudentNote,
        rubric: Rubric,
        generate: GenerateFn | None = None,
    ) -> NoteAssessment:
        ...
