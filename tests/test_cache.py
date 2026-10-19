import json

import httpx

from ragpoison.llm import Ollama, cache_key, request_body


def _transport(calls, thinking=True):
    def handler(request: httpx.Request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion"] + (["thinking"] if thinking else [])})
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"message": {"content": f"Answer: x{len(calls)}"}, "eval_count": 3})
    return httpx.MockTransport(handler)


def test_second_identical_call_hits_cache(tmp_path):
    calls = []
    c = Ollama(cache_path=tmp_path / "c.sqlite", transport=_transport(calls))
    msgs = [{"role": "user", "content": "hi"}]
    a = c.chat("m", msgs)
    b = c.chat("m", msgs)
    assert not a["cached"] and b["cached"] and a["content"] == b["content"]
    assert len(calls) == 1 and c.hits == 1 and c.misses == 1
    assert calls[0]["think"] is False and calls[0]["options"]["temperature"] == 0


def test_cache_survives_reopen_and_rep_changes_key(tmp_path):
    calls = []
    path = tmp_path / "c.sqlite"
    msgs = [{"role": "user", "content": "hi"}]
    Ollama(cache_path=path, transport=_transport(calls)).chat("m", msgs)
    c2 = Ollama(cache_path=path, transport=_transport(calls))
    assert c2.chat("m", msgs)["cached"]
    assert not c2.chat("m", msgs, rep=1)["cached"]
    assert not c2.chat("other", msgs)["cached"]
    assert len(calls) == 3


def test_no_think_field_for_models_without_thinking(tmp_path):
    calls = []
    c = Ollama(cache_path=tmp_path / "c.sqlite", transport=_transport(calls, thinking=False))
    c.chat("m", [{"role": "user", "content": "hi"}])
    assert "think" not in calls[0]


def test_key_depends_on_options():
    m = [{"role": "user", "content": "hi"}]
    assert cache_key(request_body("m", m, {"seed": 1}, False), 0) != cache_key(request_body("m", m, {"seed": 2}, False), 0)
