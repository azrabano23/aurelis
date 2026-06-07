from aurelis.cache import ResponseCache, cache_key
from aurelis.types import GenerationParams, Message, ModelResponse


def _msgs():
    return [Message("user", "grade this note")]


def test_cache_key_is_stable_and_param_sensitive():
    p1 = GenerationParams(model="claude-opus-4-8", max_tokens=100)
    p2 = GenerationParams(model="claude-opus-4-8", max_tokens=200)
    assert cache_key(_msgs(), p1) == cache_key(_msgs(), p1)
    assert cache_key(_msgs(), p1) != cache_key(_msgs(), p2)


def test_cache_roundtrip_and_hit_accounting(tmp_path):
    cache = ResponseCache(tmp_path / "c")
    key = cache_key(_msgs(), GenerationParams())
    assert cache.get(key) is None
    cache.put(key, ModelResponse(text="hi", stop_reason="end_turn"))
    got = cache.get(key)
    assert got is not None and got.text == "hi"
    assert cache.misses == 1 and cache.hits == 1
