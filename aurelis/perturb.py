"""Perturbation harness — objective ground truth for grader validation.

Take a real expert clinical note and systematically damage one part of it. A
grader worth trusting should drop the score on the dimension(s) you damaged and
leave the others alone. That gives two measurable, objective quantities with no
human labeling required:

  - sensitivity  — does the targeted dimension's score fall when we damage it?
  - specificity  — do the *other* dimensions stay put?

This is construct validity for a rubric grader, the same way you'd validate any
measurement instrument: perturb the thing it claims to measure and check it
responds.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from aurelis.datasets.acibench import SECTION_TO_DIMENSIONS, split_sections


def _drop_sections(note: str, drop_dims: set[str]) -> str:
    """Remove every section that informs any dimension in `drop_dims`."""
    sections = split_sections(note)
    kept = []
    for header, body in sections.items():
        dims = set(SECTION_TO_DIMENSIONS.get(header.upper(), ()))
        if dims & drop_dims:
            continue
        kept.append(f"{header}\n{body}")
    return "\n\n".join(kept) if kept else note


def _inject_contradiction(note: str) -> str:
    """Append an internally contradictory line — a clarity/safety defect that
    leaves the section structure intact."""
    return (
        note
        + "\n\nADDENDUM\nPatient is afebrile with normal vitals; "
        "however patient is febrile and hemodynamically unstable. "
        "Continue current plan unchanged."
    )


@dataclass(frozen=True)
class Perturbation:
    name: str
    apply: Callable[[str], str]
    targets: frozenset[str]   # rubric dimensions this should damage


PERTURBATIONS: tuple[Perturbation, ...] = (
    Perturbation("drop_subjective", lambda n: _drop_sections(n, {"subjective"}),
                 frozenset({"subjective"})),
    Perturbation("drop_objective", lambda n: _drop_sections(n, {"objective"}),
                 frozenset({"objective"})),
    Perturbation("drop_assessment_plan", lambda n: _drop_sections(n, {"assessment", "plan"}),
                 frozenset({"assessment", "plan"})),
    Perturbation("inject_contradiction", _inject_contradiction,
                 frozenset({"clarity"})),
)
