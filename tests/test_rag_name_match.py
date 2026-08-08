"""Focused tests for RAG name appearance matching (Unicode fold + OCR forms)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAG_DIR = Path(__file__).resolve().parents[1] / "rag"
sys.path.insert(0, str(RAG_DIR))

from rag_service import LanguageRAGService  # noqa: E402


@pytest.mark.parametrize(
    "name,text",
    [
        ("Adunni", "àdùnní ‘sweet to have’"),
        ("Adunni", "People modify names like Àdùnní for religion"),
        ("Adunni", "Àdùnní appears in the catalog"),
        ("Aduke", "(10d) A-du-ke\nPREF-scramble-pamper"),
        ("Aduke", "names such as Aduke. or Aweke'"),
    ],
)
def test_name_appears_unicode_and_dehyphen(name: str, text: str) -> None:
    assert LanguageRAGService._name_appears_in_text(name, text)


def test_adunni_matches_orie_ocr_via_alias() -> None:
    """Orie appendix OCRs Àdùnní as A-din-ni (din≠dun); alias bridges that gap."""
    text = "A-yin-ld A-din-ni\nPREF-praise-lick PREF-sweet-possess"
    assert LanguageRAGService._name_appears_in_text("Adunni", text)
    assert LanguageRAGService._name_appears_in_text("Àdùnní", text)


def test_dehyphen_alone_does_not_invent_adunni_from_adinni() -> None:
    """Without the alias map, folded dehyphen of A-din-ni is adinni ≠ adunni."""
    forms = LanguageRAGService._text_match_forms("A-din-ni PREF-sweet-possess")
    assert "adinni" in forms
    assert "adunni" not in forms


def test_fold_collapses_combining_marks() -> None:
    assert LanguageRAGService._fold_for_match("Àdùnní") == "adunni"
    assert LanguageRAGService._fold_for_match("àdùnní") == "adunni"


def test_unrelated_name_does_not_match() -> None:
    text = "àdùnní ‘sweet to have’ and A-du-ke typology"
    assert not LanguageRAGService._name_appears_in_text("Olumide", text)


def test_definition_context_uses_folded_name() -> None:
    text = "(43) àdùnní meaning ‘sweet to have’ (44) àríkẹ́"
    svc = object.__new__(LanguageRAGService)
    assert LanguageRAGService._has_name_definition_context(svc, "Adunni", text)


def test_select_insights_excerpts_returns_empty_without_name_grounding() -> None:
    svc = object.__new__(LanguageRAGService)
    results = [
        {
            "id": "generic-come-back",
            "paper": "Yoruba Naming.pdf",
            "text": "Babatunde means father has come back.",
            "similarity": 0.9,
        },
        {
            "id": "generic-joy",
            "paper": "Yoruba_ethnopragmatics_personal_names.pdf",
            "text": "Ayomideji means my joy has doubled.",
            "similarity": 0.8,
        },
    ]

    assert svc._select_insights_excerpts(results, top_k=3, name="Olumide") == []


def test_select_insights_excerpts_filters_name_absent_fillers() -> None:
    svc = object.__new__(LanguageRAGService)
    results = [
        {
            "id": "generic-high-score",
            "paper": "Yoruba Naming.pdf",
            "text": "Babatunde means father has come back.",
            "similarity": 0.95,
        },
        {
            "id": "grounded-low-score",
            "paper": "Olumide paper.pdf",
            "text": "Olumide is glossed as my lord has come.",
            "similarity": 0.2,
        },
        {
            "id": "other-generic",
            "paper": "Yoruba Names.pdf",
            "text": "Names can record family belief and birth circumstances.",
            "similarity": 0.7,
        },
    ]

    chosen = svc._select_insights_excerpts(results, top_k=3, name="Olumide")

    assert [item["id"] for item in chosen] == ["grounded-low-score"]
