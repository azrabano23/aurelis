"""A Task bundles a rubric with a dataset of (case, student note) pairs, and
knows how to roll graded assessments into cohort metrics and — when human gold
scores exist — into grader-validation metrics.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from aurelis.metrics import mae, mean, pearson, quadratic_weighted_kappa
from aurelis.rubric import Rubric
from aurelis.types import ClinicalCase, NoteAssessment, StudentNote

_REGISTRY: dict[str, type["Task"]] = {}


def register(cls: type["Task"]) -> type["Task"]:
    _REGISTRY[cls.name] = cls
    return cls


def get_task(name: str, **kwargs) -> "Task":
    if name not in _REGISTRY:
        raise ValueError(f"unknown task: {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


class Task(ABC):
    name: str = "base"
    rubric: Rubric

    @abstractmethod
    def load(self) -> list[tuple[ClinicalCase, StudentNote]]:
        ...

    # ---- cohort-level summary of the grades themselves ----
    def aggregate(self, assessments: Sequence[NoteAssessment]) -> dict:
        if not assessments:
            return {"n": 0}
        per_dim: dict[str, list[float]] = {d.key: [] for d in self.rubric.dimensions}
        for a in assessments:
            for s in a.scores:
                per_dim[s.dimension_key].append(s.points)
        return {
            "n": len(assessments),
            "mean_percent": round(mean([a.percent for a in assessments]), 2),
            "mean_total": round(mean([a.total for a in assessments]), 2),
            "max_total": self.rubric.max_total,
            "per_dimension_mean": {k: round(mean(v), 2) for k, v in per_dim.items()},
        }

    # ---- does the AI grader agree with human faculty? ----
    def validate(
        self,
        assessments: Sequence[NoteAssessment],
        notes: Sequence[StudentNote],
    ) -> dict:
        gold = {n.note_id: n.human_scores for n in notes if n.human_scores}
        ai_pts: list[float] = []
        hu_pts: list[float] = []
        per_dim_pairs: dict[str, tuple[list[int], list[int]]] = {
            d.key: ([], []) for d in self.rubric.dimensions
        }
        for a in assessments:
            if a.note_id not in gold:
                continue
            for s in a.scores:
                h = gold[a.note_id].get(s.dimension_key)
                if h is None:
                    continue
                ai_pts.append(s.points)
                hu_pts.append(float(h))
                per_dim_pairs[s.dimension_key][0].append(int(round(s.points)))
                per_dim_pairs[s.dimension_key][1].append(int(h))

        if not ai_pts:
            return {"validated_notes": 0, "note": "no human gold scores available"}

        max_pt = max(d.max_points for d in self.rubric.dimensions)
        return {
            "validated_notes": len(gold),
            "graded_dimensions": len(ai_pts),
            "qwk": round(
                quadratic_weighted_kappa(
                    [int(round(x)) for x in ai_pts], [int(x) for x in hu_pts],
                    min_rating=0, max_rating=max_pt,
                ),
                4,
            ),
            "pearson_r": round(pearson(ai_pts, hu_pts), 4),
            "mae_points": round(mae(ai_pts, hu_pts), 4),
            "per_dimension_qwk": {
                k: round(
                    quadratic_weighted_kappa(a, h, min_rating=0, max_rating=max_pt), 4
                )
                for k, (a, h) in per_dim_pairs.items()
                if a
            },
        }
