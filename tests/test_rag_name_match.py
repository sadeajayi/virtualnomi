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


def test_short_name_does_not_match_inside_unrelated_word() -> None:
    text = "The origin of Yoruba naming carries social meaning."
    assert not LanguageRAGService._name_appears_in_text("Ori", text)


def test_text_search_does_not_boost_substring_only_name_hits() -> None:
    svc = object.__new__(LanguageRAGService)
    svc.chunks = [
        {
            "paper": "false-hit.pdf",
            "text": "The origin of Yoruba naming carries social meaning.",
        },
        {
            "paper": "actual-hit.pdf",
            "text": "Ori means head and also carries spiritual significance.",
        },
    ]
    svc.embeddings = None
    svc.model = None
    svc.query_suffix = "Yoruba personal names"

    results = svc.search("Ori head", top_k=5, name="Ori", meaning="head")

    assert [result["paper"] for result in results] == ["actual-hit.pdf"]


def test_definition_context_uses_folded_name() -> None:
    text = "(43) àdùnní meaning ‘sweet to have’ (44) àríkẹ́"
    svc = object.__new__(LanguageRAGService)
    assert LanguageRAGService._has_name_definition_context(svc, "Adunni", text)
