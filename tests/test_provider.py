"""tests for LLMClient. no real network calls - we monkeypatch _call_openai/_call_anthropic
so we're only testing our retry/fallback/cache/structured-output logic, not the SDKs"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from autoapply.llm import provider as provider_module
from autoapply.llm.cache import ResponseCache
from autoapply.llm.provider import AllProvidersFailedError, LLMClient, ProviderError


class Greeting(BaseModel):
    message: str
    score: int


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """point the module-level cache at a tmp dir so tests don't step on each other"""
    fresh_cache = ResponseCache(directory=str(tmp_path / "cache"))
    monkeypatch.setattr(provider_module, "_cache", fresh_cache)
    yield fresh_cache
    fresh_cache.close()


def test_complete_returns_raw_text(monkeypatch):
    client = LLMClient(provider="openai")
    monkeypatch.setattr(client, "_call_openai", lambda *a, **k: "hello there")

    result = client.complete(system="sys", user="hi", agent_name="test")

    assert result == "hello there"


def test_complete_parses_structured_output(monkeypatch):
    client = LLMClient(provider="openai")
    monkeypatch.setattr(
        client, "_call_openai", lambda *a, **k: '{"message": "hi", "score": 42}'
    )

    result = client.complete(system="sys", user="hi", response_model=Greeting, agent_name="test")

    assert isinstance(result, Greeting)
    assert result.message == "hi"
    assert result.score == 42


def test_invalid_json_triggers_retry_then_raises(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        return "not json at all"

    client = LLMClient(provider="openai")
    monkeypatch.setattr(client, "_call_openai", flaky)

    with pytest.raises(AllProvidersFailedError):
        client.complete(system="sys", user="hi", response_model=Greeting, agent_name="test")

    assert calls["n"] == provider_module.settings.max_retries


def test_retry_then_succeed(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient network error")
        return "recovered"

    client = LLMClient(provider="openai")
    monkeypatch.setattr(client, "_call_openai", flaky)

    result = client.complete(system="sys", user="hi", agent_name="test")

    assert result == "recovered"
    assert calls["n"] == 2


def test_fallback_to_secondary_provider(monkeypatch):
    client = LLMClient(provider="openai")
    client.fallback = "anthropic"
    monkeypatch.setattr(client, "_call_openai", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(client, "_call_anthropic", lambda *a, **k: "from fallback")

    result = client.complete(system="sys", user="hi", agent_name="test")

    assert result == "from fallback"


def test_gemini_as_primary_provider(monkeypatch):
    client = LLMClient(provider="gemini")
    monkeypatch.setattr(client, "_call_gemini", lambda *a, **k: "hi from gemini")

    result = client.complete(system="sys", user="hi", agent_name="test")

    assert result == "hi from gemini"


def test_falls_back_to_gemini(monkeypatch):
    client = LLMClient(provider="openai")
    client.fallback = "gemini"
    monkeypatch.setattr(client, "_call_openai", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(client, "_call_gemini", lambda *a, **k: "from gemini fallback")

    result = client.complete(system="sys", user="hi", agent_name="test")

    assert result == "from gemini fallback"


def test_all_providers_failed_raises(monkeypatch):
    client = LLMClient(provider="openai")
    client.fallback = "anthropic"
    monkeypatch.setattr(client, "_call_openai", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(client, "_call_anthropic", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("also down")))

    with pytest.raises(AllProvidersFailedError):
        client.complete(system="sys", user="hi", agent_name="test")


def test_cache_hit_avoids_second_call(monkeypatch):
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return "cached value"

    client = LLMClient(provider="openai")
    monkeypatch.setattr(client, "_call_openai", counting)

    first = client.complete(system="sys", user="hi", agent_name="test")
    second = client.complete(system="sys", user="hi", agent_name="test")

    assert first == second == "cached value"
    assert calls["n"] == 1


def test_use_cache_false_bypasses_cache(monkeypatch):
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return f"value {calls['n']}"

    client = LLMClient(provider="openai")
    monkeypatch.setattr(client, "_call_openai", counting)

    first = client.complete(system="sys", user="hi", agent_name="test", use_cache=False)
    second = client.complete(system="sys", user="hi", agent_name="test", use_cache=False)

    assert first != second
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_acomplete_returns_raw_text(monkeypatch):
    client = LLMClient(provider="openai")

    async def fake_call(*a, **k):
        return "async hello"

    monkeypatch.setattr(client, "_acall_openai", fake_call)

    result = await client.acomplete(system="sys", user="hi", agent_name="test")

    assert result == "async hello"


@pytest.mark.asyncio
async def test_acomplete_falls_back(monkeypatch):
    client = LLMClient(provider="openai")
    client.fallback = "anthropic"

    async def failing(*a, **k):
        raise RuntimeError("primary down")

    async def working(*a, **k):
        return "async fallback value"

    monkeypatch.setattr(client, "_acall_openai", failing)
    monkeypatch.setattr(client, "_acall_anthropic", working)

    result = await client.acomplete(system="sys", user="hi", agent_name="test")

    assert result == "async fallback value"
