"""Aurelis — AI-assisted clinical documentation training and assessment."""
from aurelis.types import (
    ClinicalCase,
    DimensionScore,
    GenerationParams,
    Message,
    ModelResponse,
    NoteAssessment,
    Rubric,
    RubricDimension,
    StudentNote,
)

__version__ = "0.1.0"

__all__ = [
    "ClinicalCase",
    "StudentNote",
    "Rubric",
    "RubricDimension",
    "DimensionScore",
    "NoteAssessment",
    "GenerationParams",
    "Message",
    "ModelResponse",
    "__version__",
]
