"""Task-level tests: dataset loads, cohort aggregation, and the grader-validation
math that compares AI scores to human faculty gold."""
from aurelis.tasks import SOAPTask
from aurelis.types import DimensionScore, NoteAssessment


def _assessment_from_scores(note, rubric, score_map):
    scores = tuple(
        DimensionScore(d.key, float(score_map[d.key]), d.max_points)
        for d in rubric.dimensions
    )
    return NoteAssessment(note.case_id, note.note_id, scores, grader="test")


def test_dataset_loads():
    pairs = SOAPTask().load()
    assert len(pairs) == 6
    assert all(note.human_scores for _, note in pairs)  # every note has gold


def test_validate_perfect_agreement():
    task = SOAPTask()
    notes = [n for _, n in task.load()]
    # AI reproduces the human scores exactly
    assessments = [_assessment_from_scores(n, task.rubric, n.human_scores) for n in notes]
    v = task.validate(assessments, notes)
    assert v["validated_notes"] == 6
    assert v["qwk"] == 1.0
    assert v["mae_points"] == 0.0


def test_validate_detects_disagreement():
    task = SOAPTask()
    notes = [n for _, n in task.load()]
    # AI is off by one point on every dimension -> imperfect, positive MAE
    assessments = [
        _assessment_from_scores(
            n, task.rubric, {k: max(0, v - 1) for k, v in n.human_scores.items()}
        )
        for n in notes
    ]
    v = task.validate(assessments, notes)
    assert v["mae_points"] > 0.0
    assert v["qwk"] < 1.0


def test_aggregate_reports_cohort_stats():
    task = SOAPTask()
    notes = [n for _, n in task.load()]
    assessments = [_assessment_from_scores(n, task.rubric, n.human_scores) for n in notes]
    agg = task.aggregate(assessments)
    assert agg["n"] == 6
    assert agg["max_total"] == 20
    assert set(agg["per_dimension_mean"]) == {"subjective", "objective", "assessment", "plan", "clarity"}
