import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nomi-name-search-api"))

import insights_service as service  # noqa: E402


class _FakeAnthropic:
    class Anthropic:
        def __init__(self, api_key):
            self.api_key = api_key


@pytest.fixture(autouse=True)
def clear_cache():
    service._clear_insight_cache()
    yield
    service._clear_insight_cache()


def _prepare(monkeypatch, model_call):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    monkeypatch.setattr(service, "anthropic", _FakeAnthropic)
    monkeypatch.setattr(service, "load_insights_system_prompt", lambda: "prompt-v2")
    monkeypatch.setattr(
        service,
        "_rag_index_cache_token",
        lambda language: f"rag:{language.strip().casefold()}:test-index",
    )
    monkeypatch.setattr(
        service,
        "gather_rag_context",
        lambda *_: (
            "[Semantic_Analysis_of_Igbo_Names.pdf]: Supporting excerpt.",
            ["Semantic_Analysis_of_Igbo_Names.pdf"],
            True,
            "igbo",
        ),
    )
    monkeypatch.setattr(service, "_call_insights_model", model_call)


def test_successful_insight_is_cached_by_normalized_inputs(monkeypatch):
    calls = 0

    def model_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "Adaeze places a daughter within the language of dignity and belonging."

    _prepare(monkeypatch, model_call)
    first = service.generate_insight_paragraph(
        " Adaeze ", " Igbo ", " Daughter of a king "
    )
    second = service.generate_insight_paragraph(
        "adaeze", "igbo", "Daughter of a king"
    )

    assert calls == 1
    assert second == first
    assert first["sources"][0]["excerpt"] == "Supporting excerpt."


def test_transient_generation_failure_is_not_cached(monkeypatch):
    calls = 0

    def model_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary provider failure")
        return "Adaeze places a daughter within the language of dignity and belonging."

    _prepare(monkeypatch, model_call)

    with pytest.raises(RuntimeError, match="temporary provider failure"):
        service.generate_insight_paragraph("Adaeze", "Igbo", "Daughter of a king")

    result = service.generate_insight_paragraph(
        "Adaeze", "Igbo", "Daughter of a king"
    )
    assert calls == 2
    assert result["insight"].startswith("Adaeze")


def test_meaning_change_invalidates_cache_key(monkeypatch):
    calls = 0

    def model_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return f"Insight generation {calls} contains enough useful words to remain valid."

    _prepare(monkeypatch, model_call)
    service.generate_insight_paragraph("Adaeze", "Igbo", "Daughter of a king")
    service.generate_insight_paragraph("Adaeze", "Igbo", "A royal daughter")

    assert calls == 2


def test_rag_index_revision_change_invalidates_cache_key(monkeypatch):
    calls = 0
    token = {"value": "igbo:revision-1"}

    def model_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return f"Insight generation {calls} contains enough useful words to remain valid."

    _prepare(monkeypatch, model_call)
    monkeypatch.setattr(service, "_rag_index_cache_token", lambda _language: token["value"])

    first = service.generate_insight_paragraph(
        "Adaeze", "Igbo", "Daughter of a king"
    )
    token["value"] = "igbo:revision-2"
    second = service.generate_insight_paragraph(
        "Adaeze", "Igbo", "Daughter of a king"
    )

    assert calls == 2
    assert first["insight"] != second["insight"]


def test_store_rejects_ungrounded_payload():
    key = ("v-test", "model", "digest", "rag-index", "name", "lang", "meaning", "")
    service._store_cached_insight(
        key,
        {
            "insight": "Should never be served from cache.",
            "grounded": False,
            "rag_used": False,
            "sources": [],
        },
    )
    assert service._get_cached_insight(key) is None
