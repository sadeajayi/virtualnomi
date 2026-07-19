import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nomi-name-search-api"))
sys.path.insert(0, str(ROOT / "rag"))

import insights_service as service  # noqa: E402
from rag_service import LanguageRAGService  # noqa: E402


class _FakeRag:
    language_key = "yoruba"

    def __init__(self, excerpts):
        self.excerpts = excerpts

    def get_insights_excerpts(self, _name, _meaning):
        return self.excerpts

    def has_name_specific_evidence(self, name, excerpt):
        return LanguageRAGService.has_name_specific_evidence(name, excerpt)


def test_akpofure_without_rag_never_calls_model(monkeypatch):
    service._clear_insight_cache()
    model_called = False

    def model_call(*_args, **_kwargs):
        nonlocal model_called
        model_called = True
        return "This must never be returned."

    monkeypatch.setattr(service, "load_insights_system_prompt", lambda: "prompt")
    monkeypatch.setattr(service, "gather_rag_context", lambda *_: ("", [], False, None))
    monkeypatch.setattr(service, "_call_insights_model", model_call)

    with pytest.raises(service.NoGroundedInsightError):
        service.generate_insight_paragraph(
            "Akpofure",
            "Urhobo",
            "Life is now comfortable",
        )
    assert model_called is False


def test_generic_excerpt_without_name_hit_is_rejected(monkeypatch):
    generic = _FakeRag(
        [{
            "paper": "Yoruba_Praise_Names.pdf",
            "excerpt": "Yoruba naming traditions often express family hopes.",
        }]
    )
    monkeypatch.setattr(
        service,
        "get_rag_service_for_dataset_language",
        lambda *_args, **_kwargs: generic,
    )
    assert service.gather_rag_context("Adunni", "Sweet to have", "Yoruba") == (
        "",
        [],
        False,
        "yoruba",
    )


def test_adunni_alias_hit_produces_grounded_context(monkeypatch):
    grounded = _FakeRag(
        [{
            "paper": "Yoruba_Names_Gender_Markings.pdf",
            "excerpt": "A-din-ni means sweet to possess in the documented name list.",
        }]
    )
    monkeypatch.setattr(
        service,
        "get_rag_service_for_dataset_language",
        lambda *_args, **_kwargs: grounded,
    )
    rag_text, attributions, rag_used, rag_key = service.gather_rag_context(
        "Adunni",
        "Sweet to have",
        "Yoruba",
    )
    assert "A-din-ni" in rag_text
    assert attributions == ["Yoruba_Names_Gender_Markings.pdf"]
    assert rag_used is True
    assert rag_key == "yoruba"
