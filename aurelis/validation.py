"""Perturbation-based grader validation.

For each (case, intact note): grade the intact note, then for each perturbation
grade the damaged note and measure how the per-dimension scores moved. Aggregate
into sensitivity (targeted dimensions should drop) and specificity (others
shouldn't) — objective metrics that need no human labels.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from aurelis.grading.base import Grader
from aurelis.perturb import PERTURBATIONS, Perturbation
from aurelis.rubric import Rubric
from aurelis.types import ClinicalCase, Message, ModelResponse, NoteAssessment, StudentNote

GenerateFn = Callable[[Sequence[Message]], ModelResponse]


def _scores_by_dim(a: NoteAssessment) -> dict[str, float]:
    return {s.dimension_key: s.points for s in a.scores}


def run_perturbation_validation(
    pairs: Sequence[tuple[ClinicalCase, StudentNote]],
    grader: Grader,
    rubric: Rubric,
    generate: GenerateFn | None = None,
    perturbations: Sequence[Perturbation] = PERTURBATIONS,
) -> dict:
    all_dims = [d.key for d in rubric.dimensions]
    # accumulators per perturbation
    acc = {
        p.name: {"targeted_drops": [], "offtarget_drops": [], "detected": 0, "n": 0}
        for p in perturbations
    }

    for case, note in pairs:
        intact = _scores_by_dim(grader.grade(case, note, rubric, generate))
        for p in perturbations:
            damaged_note = replace(note, text=p.apply(note.text))
            damaged = _scores_by_dim(grader.grade(case, damaged_note, rubric, generate))
            targeted = [d for d in all_dims if d in p.targets]
            offtarget = [d for d in all_dims if d not in p.targets]

            t_drop = sum(intact[d] - damaged[d] for d in targeted) / max(1, len(targeted))
            o_drop = sum(intact[d] - damaged[d] for d in offtarget) / max(1, len(offtarget))
            acc[p.name]["targeted_drops"].append(t_drop)
            acc[p.name]["offtarget_drops"].append(o_drop)
            acc[p.name]["detected"] += int(t_drop > 0)
            acc[p.name]["n"] += 1

    def _mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    summary = {}
    for name, a in acc.items():
        n = a["n"]
        summary[name] = {
            "n": n,
            "mean_targeted_drop": _mean(a["targeted_drops"]),
            "mean_offtarget_drop": _mean(a["offtarget_drops"]),
            "detection_rate": round(a["detected"] / n, 3) if n else 0.0,
        }
    # headline: a grader is good if targeted drops are large and off-target ~0
    t = [summary[p.name]["mean_targeted_drop"] for p in perturbations]
    o = [summary[p.name]["mean_offtarget_drop"] for p in perturbations]
    summary["_overall"] = {
        "mean_targeted_drop": round(sum(t) / len(t), 3) if t else 0.0,
        "mean_offtarget_drop": round(sum(o) / len(o), 3) if o else 0.0,
        "mean_detection_rate": round(
            sum(summary[p.name]["detection_rate"] for p in perturbations) / len(perturbations), 3
        ),
    }
    return summary
