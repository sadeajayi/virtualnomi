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
        self._proxy = object.__new__(LanguageRAGService)
        self._proxy.morphemes = [
            "ọlá",
            "ola",
            "ade",
            "adé",
            "fola",
            "fọlá",
            "sade",
            "ṣade",
            "temi",
            "tẹmi",
            "ope",
            "ọpẹ",
        ]
        self._proxy.language_key = "yoruba"
        self._proxy.query_suffix = (
            "Yoruba personal name cultural significance morpheme"
        )

    def get_insights_excerpts(self, _name, _meaning):
        return self.excerpts

    def has_name_specific_evidence(self, name, excerpt):
        return LanguageRAGService.has_name_specific_evidence(name, excerpt)

    def is_pattern_relevant_excerpt(self, name, meaning, text, *, score=0.0):
        return LanguageRAGService.is_pattern_relevant_excerpt(
            self._proxy, name, meaning, text, score=score
        )


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


def test_ambiguous_lookup_requires_specific_language(monkeypatch):
    service._clear_insight_cache()
    rag_called = False

    def rag_call(*_args, **_kwargs):
        nonlocal rag_called
        rag_called = True
        return "Excerpt", ["Paper.pdf"], True, "igbo"

    monkeypatch.setattr(service, "gather_rag_context", rag_call)

    with pytest.raises(ValueError, match="matches multiple languages"):
        service.generate_insight_paragraph(
            "Ada",
            "",
            "",
            lookup_name_fn=lambda *_args: [
                {
                    "name": "Ada",
                    "language": "Igbo",
                    "meaning": "First daughter",
                },
                {
                    "name": "Ada",
                    "language": "Yoruba",
                    "meaning": "Royal crown",
                },
            ],
        )

    assert rag_called is False


def test_lookup_prefers_exact_language_before_family_matches(monkeypatch):
    service._clear_insight_cache()
    seen = {}

    monkeypatch.setattr(service, "load_insights_system_prompt", lambda: "prompt")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        service,
        "anthropic",
        type("AnthropicModule", (), {"Anthropic": lambda *args, **kwargs: object()}),
    )
    monkeypatch.setattr(
        service,
        "build_structured_sources",
        lambda *_args, **_kwargs: [{"filename": "Hausa_Names.pdf"}],
    )
    monkeypatch.setattr(
        service,
        "_call_insights_model",
        lambda *_args, **_kwargs: "Ada carries a grounded reading for this name.",
    )

    def rag_call(name, meaning, language):
        seen["name"] = name
        seen["meaning"] = meaning
        seen["language"] = language
        return "Grounded Hausa excerpt", ["Hausa_Names.pdf"], True, "hausa"

    monkeypatch.setattr(service, "gather_rag_context", rag_call)

    payload = service.generate_insight_paragraph(
        "Ada",
        "Hausa",
        "",
        lookup_name_fn=lambda *_args: [
            {
                "name": "Ada",
                "language": "Hausa (Localised Islamic/Arabic)",
                "meaning": "Adornment",
            },
            {
                "name": "Ada",
                "language": "Hausa",
                "meaning": "Noble one",
            },
        ],
    )

    assert seen == {"name": "Ada", "meaning": "Noble one", "language": "Hausa"}
    assert payload["meaning"] == "Noble one"


def test_generic_boilerplate_without_meaning_tie_is_rejected(monkeypatch):
    generic = _FakeRag(
        [{
            "paper": "Yoruba_Praise_Names.pdf",
            "excerpt": "Yoruba naming traditions often express family hopes.",
            "relevance_score": 0.4,
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


def test_morenikeji_generic_excerpts_fail_relevance_gate_before_model(monkeypatch):
    service._clear_insight_cache()
    generic = _FakeRag(
        [
            {
                "paper": "Construction_Morphology_in_Yoruba_names_Schemas_an.pdf",
                "excerpt": "Name formation may obey vowel harmony principles.",
                "relevance_score": 0.2,
            },
            {
                "paper": "Yoruba Naming.pdf",
                "excerpt": "Some African naming traditions reflect belief in reincarnation.",
                "relevance_score": 0.2,
            },
        ]
    )
    model_called = False

    def model_call(*_args, **_kwargs):
        nonlocal model_called
        model_called = True
        return "This unsupported paragraph must never be returned."

    monkeypatch.setattr(service, "load_insights_system_prompt", lambda: "prompt")
    monkeypatch.setattr(
        service,
        "get_rag_service_for_dataset_language",
        lambda *_args, **_kwargs: generic,
    )
    monkeypatch.setattr(service, "_call_insights_model", model_call)

    with pytest.raises(service.NoGroundedInsightError):
        service.generate_insight_paragraph(
            "Morẹ́nikéjì",
            "Yoruba",
            "I have found a companion.",
        )
    assert model_called is False


def test_empty_retrieval_rejects_meaning_only_path(monkeypatch):
    service._clear_insight_cache()
    empty = _FakeRag([])
    model_called = False

    def model_call(*_args, **_kwargs):
        nonlocal model_called
        model_called = True
        return "Meaning-only fiction must never be returned."

    monkeypatch.setattr(service, "load_insights_system_prompt", lambda: "prompt")
    monkeypatch.setattr(
        service,
        "get_rag_service_for_dataset_language",
        lambda *_args, **_kwargs: empty,
    )
    monkeypatch.setattr(service, "_call_insights_model", model_call)

    with pytest.raises(service.NoGroundedInsightError):
        service.generate_insight_paragraph(
            "Folasade",
            "Yoruba",
            "honour confers a crown",
        )
    assert model_called is False


def test_no_index_language_fail_closed(monkeypatch):
    service._clear_insight_cache()
    model_called = False

    def model_call(*_args, **_kwargs):
        nonlocal model_called
        model_called = True
        return "No-index fiction must never be returned."

    monkeypatch.setattr(service, "load_insights_system_prompt", lambda: "prompt")
    monkeypatch.setattr(
        service,
        "get_rag_service_for_dataset_language",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(service, "_call_insights_model", model_call)

    with pytest.raises(service.NoGroundedInsightError):
        service.generate_insight_paragraph(
            "Akpofure",
            "Urhobo",
            "Life is now comfortable",
        )
    assert model_called is False


def test_adunni_alias_hit_produces_grounded_context(monkeypatch):
    grounded = _FakeRag(
        [{
            "paper": "Yoruba_Names_Gender_Markings.pdf",
            "excerpt": "A-din-ni means sweet to possess in the documented name list.",
            "relevance_score": 1.0,
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


def test_folasade_pattern_excerpts_pass_without_exact_name(monkeypatch):
    pattern = _FakeRag(
        [
            {
                "paper": "Construction_Morphology_in_Yoruba_names_Schemas_an.pdf",
                "excerpt": (
                    "These names reflect a sentential-type construction: "
                    "Adéyémí crown-deserve-3sg ‘The crown is entitled to me.’ "
                    "In Yoruba, adé literally means crown."
                ),
                "relevance_score": 0.35,
            },
            {
                "paper": "Yoruba_Names_Communicative_.pdf",
                "excerpt": (
                    "A child’s name is his/her first crown. It is the child’s "
                    "first batch of honour and that is why people painstakingly "
                    "choose these names carefully."
                ),
                "relevance_score": 0.3,
            },
        ]
    )
    monkeypatch.setattr(
        service,
        "get_rag_service_for_dataset_language",
        lambda *_args, **_kwargs: pattern,
    )
    rag_text, attributions, rag_used, rag_key = service.gather_rag_context(
        "Folasade",
        "honour confers a crown",
        "Yoruba",
    )
    assert rag_used is True
    assert "crown" in rag_text.lower()
    assert "Construction_Morphology_in_Yoruba_names_Schemas_an.pdf" in attributions
    assert rag_key == "yoruba"
    assert LanguageRAGService.has_name_specific_evidence("Folasade", rag_text) is False


def test_temitope_gratitude_pattern_excerpts_pass(monkeypatch):
    pattern = _FakeRag(
        [{
            "paper": "Yoruba_Names_Communicative_.pdf",
            "excerpt": (
                "Appreciative Function: Name serves as a means through which "
                "Yoruba show gratitude for what God has done. Examples; "
                "Modupe ----- I thank God Opeyemi ------ I deserve to thank God"
            ),
            "relevance_score": 0.4,
        }]
    )
    monkeypatch.setattr(
        service,
        "get_rag_service_for_dataset_language",
        lambda *_args, **_kwargs: pattern,
    )
    rag_text, attributions, rag_used, _rag_key = service.gather_rag_context(
        "Temitope",
        "mine is worthy of thanks",
        "Yoruba",
    )
    assert rag_used is True
    assert "gratitude" in rag_text.lower() or "thank" in rag_text.lower()
    assert attributions == ["Yoruba_Names_Communicative_.pdf"]


def test_morenikeji_oriki_pattern_excerpts_pass(monkeypatch):
    pattern = _FakeRag(
        [{
            "paper": "Yoruba_ethnopragmatics_personal_names.pdf",
            "excerpt": (
                "These are referred to as oríkì àbísọ ‘attributive personal name’ "
                "(Oyelaran 1976). These names may also express what the child is "
                "to his or her parents, and convey affection or companion praise."
            ),
            "relevance_score": 0.5,
        }]
    )
    monkeypatch.setattr(
        service,
        "get_rag_service_for_dataset_language",
        lambda *_args, **_kwargs: pattern,
    )
    rag_text, attributions, rag_used, _rag_key = service.gather_rag_context(
        "Morẹ́nikéjì",
        "I have found a companion.",
        "Yoruba",
    )
    assert rag_used is True
    assert "oríkì" in rag_text.lower() or "oriki" in rag_text.lower()
    assert attributions == ["Yoruba_ethnopragmatics_personal_names.pdf"]


def test_pattern_relevant_allows_generation(monkeypatch):
    service._clear_insight_cache()
    pattern = _FakeRag(
        [{
            "paper": "Construction_Morphology_in_Yoruba_names_Schemas_an.pdf",
            "excerpt": (
                "Adéfúnké crown-give-1sg-pamper. In Yoruba, adé literally means crown, "
                "and honour compounds place the child under royal favour."
            ),
            "relevance_score": 0.4,
        }]
    )

    def model_call(*_args, **_kwargs):
        return (
            "Folasade belongs with Yoruba crown names: parents place honour "
            "on the child the way a crown marks who is lifted into public regard."
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")

    class _FakeAnthropic:
        class Anthropic:
            def __init__(self, api_key):
                self.api_key = api_key

    monkeypatch.setattr(service, "anthropic", _FakeAnthropic)
    monkeypatch.setattr(service, "load_insights_system_prompt", lambda: "## Role\nprompt")
    monkeypatch.setattr(
        service,
        "get_rag_service_for_dataset_language",
        lambda *_args, **_kwargs: pattern,
    )
    monkeypatch.setattr(service, "_call_insights_model", model_call)
    monkeypatch.setattr(
        service,
        "build_structured_sources",
        lambda *_args, **_kwargs: [
            {
                "filename": "Construction_Morphology_in_Yoruba_names_Schemas_an.pdf",
                "title": "Construction Morphology",
                "excerpt": "adé literally means crown",
            }
        ],
    )

    payload = service.generate_insight_paragraph(
        "Folasade",
        "Yoruba",
        "honour confers a crown",
    )
    assert payload["grounded"] is True
    assert payload["rag_used"] is True
    assert payload["sources"]
    assert "crown" in payload["insight"].lower() or "honour" in payload["insight"].lower()
