"""Grader tests — deterministic checklist and the LLM judge (driven by a scripted
mock, no API key)."""
from aurelis.grading import ChecklistGrader, LLMGrader
from aurelis.providers import MockProvider
from aurelis.tasks import SOAPTask
from aurelis.types import GenerationParams, ModelResponse

PARAMS = GenerationParams()


def _pairs():
    return {n.note_id: (c, n) for c, n in SOAPTask().load()}


def _gen(provider):
    return lambda msgs: provider.generate(msgs, PARAMS)


def test_checklist_ranks_strong_above_weak():
    task = SOAPTask()
    pairs = _pairs()
    g = ChecklistGrader()
    strong = g.grade(*pairs["chestpain-strong"], task.rubric)
    weak = g.grade(*pairs["chestpain-weak"], task.rubric)
    assert strong.total > weak.total
    assert strong.max_total == task.rubric.max_total == 20


def test_llm_grader_parses_and_totals():
    provider = MockProvider(
        default=ModelResponse(text='{"points": 3, "feedback": "solid", "missing": [], "evidence": ["HPI"]}')
    )
    task = SOAPTask()
    case, note = _pairs()["dm2-strong"]
    a = LLMGrader().grade(case, note, task.rubric, _gen(provider))
    assert len(a.scores) == 5
    assert all(s.points == 3 for s in a.scores)
    assert a.total == 15 and a.max_total == 20


def test_llm_grader_clamps_out_of_range_scores():
    provider = MockProvider(default=ModelResponse(text='{"points": 99}'))
    task = SOAPTask()
    case, note = _pairs()["dm2-strong"]
    a = LLMGrader().grade(case, note, task.rubric, _gen(provider))
    assert all(s.points == s.max_points for s in a.scores)  # clamped to dimension max


def test_llm_grader_treats_garbage_as_zero():
    provider = MockProvider(default=ModelResponse(text="not json at all"))
    task = SOAPTask()
    case, note = _pairs()["copd-weak"]
    a = LLMGrader().grade(case, note, task.rubric, _gen(provider))
    assert all(s.points == 0.0 for s in a.scores)
