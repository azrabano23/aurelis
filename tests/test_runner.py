"""End-to-end: grade the SOAP task with the LLM grader over a mock provider,
prove provenance is recorded and reruns are served entirely from cache."""
from aurelis.cache import ResponseCache
from aurelis.grading import LLMGrader
from aurelis.providers.base import Provider
from aurelis.runner import run
from aurelis.store import RunStore
from aurelis.tasks import SOAPTask
from aurelis.types import GenerationParams, ModelResponse


class CountingProvider(Provider):
    name = "counting"

    def __init__(self):
        self.n = 0

    def generate(self, messages, params):
        self.n += 1
        return ModelResponse(
            text='{"points": 3, "feedback": "ok", "missing": [], "evidence": []}',
            stop_reason="end_turn",
        )


def test_runner_records_metrics_validation_and_provenance(tmp_path):
    provider = CountingProvider()
    rec = run(
        SOAPTask(), LLMGrader(), provider, GenerationParams(),
        cache=ResponseCache(tmp_path / "c"), store=RunStore(tmp_path / "r"),
    )
    assert rec.task == "soap" and rec.grader == "llm"
    assert rec.params["model"] == "claude-opus-4-8"
    assert rec.metrics["n"] == 6
    assert len(rec.assessments) == 6
    assert "qwk" in rec.validation                       # human gold present -> validated
    # 5 dimensions x 6 notes = 30 grader calls
    assert provider.n == 30
    assert RunStore(tmp_path / "r").load(rec.run_id).run_id == rec.run_id


def test_rerun_is_served_from_cache(tmp_path):
    provider = CountingProvider()
    cache_dir = tmp_path / "cache"
    run(SOAPTask(), LLMGrader(), provider, GenerationParams(), cache=ResponseCache(cache_dir))
    assert provider.n == 30

    cache2 = ResponseCache(cache_dir)
    run(SOAPTask(), LLMGrader(), provider, GenerationParams(), cache=cache2)
    assert provider.n == 30           # zero new model calls
    assert cache2.hits == 30 and cache2.misses == 0
