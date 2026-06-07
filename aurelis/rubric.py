"""Grading rubrics for clinical notes.

The default SOAP rubric below mirrors how clerkship faculty actually grade a
write-up: each of the five SOAP-plus-communication axes is scored 0-4 against
explicit criteria, for a 20-point note. The `criteria` strings are what the LLM
grader is held to; `required_elements` feed the deterministic checklist grader
for facts that are objectively present-or-absent.

Rubrics are data, not code paths — define a new one (specialty-specific, OSCE,
discharge summary) and every grader/task works against it unchanged.
"""
from __future__ import annotations

from aurelis.types import Rubric, RubricDimension

SOAP_RUBRIC = Rubric(
    id="soap-v1",
    name="SOAP Clinical Note (clerkship)",
    dimensions=(
        RubricDimension(
            key="subjective",
            name="Subjective",
            max_points=4,
            criteria=(
                "Chief complaint stated. HPI is complete and well-organized "
                "(onset, location, duration, character, aggravating/relieving "
                "factors, timing, severity). Pertinent positives AND negatives "
                "from the review of systems. Relevant PMH, medications, allergies."
            ),
            required_elements=("chief complaint", "history of present illness"),
        ),
        RubricDimension(
            key="objective",
            name="Objective",
            max_points=4,
            criteria=(
                "Vital signs reported. Focused, relevant physical exam documented. "
                "Pertinent labs, imaging, and other data included and accurately "
                "transcribed. No fabricated findings."
            ),
            required_elements=("vital signs", "physical exam"),
        ),
        RubricDimension(
            key="assessment",
            name="Assessment",
            max_points=4,
            criteria=(
                "Clear problem list. A prioritized differential diagnosis with "
                "explicit clinical reasoning that ties the subjective and objective "
                "data to each diagnosis. Most-likely diagnosis justified; dangerous "
                "alternatives considered."
            ),
            required_elements=("differential diagnosis",),
        ),
        RubricDimension(
            key="plan",
            name="Plan",
            max_points=4,
            criteria=(
                "Plan addresses each problem: further diagnostics, treatment with "
                "appropriate specificity, monitoring, disposition/follow-up, and "
                "patient education. Plan is consistent with the assessment."
            ),
            required_elements=("plan",),
        ),
        RubricDimension(
            key="clarity",
            name="Clarity & Structure",
            max_points=4,
            criteria=(
                "Correct SOAP structure. Concise and unambiguous. Standard "
                "terminology and abbreviations. No internal contradictions, no "
                "copy-forward errors, no clinically dangerous ambiguity."
            ),
        ),
    ),
)

REGISTRY = {SOAP_RUBRIC.id: SOAP_RUBRIC}


def get_rubric(rubric_id: str) -> Rubric:
    if rubric_id not in REGISTRY:
        raise ValueError(f"unknown rubric: {rubric_id!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[rubric_id]
