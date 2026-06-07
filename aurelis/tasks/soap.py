"""SOAP note assessment task.

Loads clinical cases paired with student write-ups (and, where available, the
human faculty score that lets us validate the grader) from a JSONL dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

from aurelis.rubric import SOAP_RUBRIC
from aurelis.tasks.base import Task, register
from aurelis.types import ClinicalCase, StudentNote

_DATA = Path(__file__).resolve().parent.parent.parent / "data" / "soap_cases.jsonl"


@register
class SOAPTask(Task):
    name = "soap"
    rubric = SOAP_RUBRIC

    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = Path(data_path) if data_path else _DATA

    def load(self) -> list[tuple[ClinicalCase, StudentNote]]:
        pairs = []
        for line in self.data_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            case = ClinicalCase(
                id=row["case_id"],
                specialty=row.get("specialty", ""),
                vignette=row["vignette"],
                reference_elements=row.get("reference_elements", {}),
            )
            note = StudentNote(
                case_id=row["case_id"],
                note_id=row["note_id"],
                text=row["note"],
                quality_label=row.get("quality"),
                human_scores=row.get("human_scores", {}),
            )
            pairs.append((case, note))
        return pairs
