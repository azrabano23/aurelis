"""ACI-Bench loader.

ACI-Bench (Yim et al., 2023) is a public benchmark of doctor-patient encounters
paired with expert-written clinical notes — real clinical documentation, no PHI.
We use the reference notes as gold-standard documentation: the dialogue becomes
the encounter the student "saw", and the expert note is the high-quality write-up
the grader should reward. Paired with the perturbation harness (aurelis.perturb),
this gives objective ground truth for validating the grader on real clinical text.

Source: https://github.com/wyim/aci-bench  (download-on-demand; not vendored).
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

from aurelis.types import ClinicalCase, StudentNote

VALID_URL = "https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data/valid.csv"
_CACHE = Path(__file__).resolve().parent.parent.parent / ".data_cache" / "aci_valid.csv"

# ACI section header -> SOAP rubric dimension(s) it informs.
SECTION_TO_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "CHIEF COMPLAINT": ("subjective",),
    "HISTORY OF PRESENT ILLNESS": ("subjective",),
    "REVIEW OF SYSTEMS": ("subjective",),
    "PHYSICAL EXAMINATION": ("objective",),
    "PHYSICAL EXAM": ("objective",),
    "VITALS": ("objective",),
    "RESULTS": ("objective",),
    "ASSESSMENT AND PLAN": ("assessment", "plan"),
    "ASSESSMENT": ("assessment",),
    "PLAN": ("plan",),
}

_HEADER_RE = re.compile(r"^[A-Z][A-Z /&'\-]{2,}$")


def download(cache: Path = _CACHE) -> Path:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(VALID_URL, cache)  # noqa: S310 (trusted GitHub raw)
    return cache


def split_sections(note: str) -> dict[str, str]:
    """Split a note into {UPPERCASE_HEADER: body_text}."""
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in note.splitlines():
        if _HEADER_RE.match(line.strip()):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line.strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def load_acibench(limit: int | None = None, cache: Path = _CACHE):
    """Yield (ClinicalCase, StudentNote) for ACI-Bench reference notes."""
    import csv

    path = download(cache)
    pairs: list[tuple[ClinicalCase, StudentNote]] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            eid = row["encounter_id"]
            pairs.append((
                ClinicalCase(id=eid, specialty="General", vignette=row["dialogue"].strip()),
                StudentNote(case_id=eid, note_id=eid, text=row["note"].strip(), quality_label="reference"),
            ))
            if limit and len(pairs) >= limit:
                break
    return pairs
