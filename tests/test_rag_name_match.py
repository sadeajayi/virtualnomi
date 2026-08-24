"""Focused tests for RAG name appearance matching (Unicode fold + OCR forms)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAG_DIR = Path(__file__).resolve().parents[1] / "rag"
sys.path.insert(0, str(RAG_DIR))

import rag_service  # noqa: E402
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


def test_cached_rag_service_reloads_when_index_revision_changes(monkeypatch, tmp_path) -> None:
    index_path = tmp_path / "language_index.json"
    index_path.write_text("{}", encoding="utf-8")
    created_revisions = []

    class FakeRAGService:
        def __init__(self, language_key, quiet=False, text_search_only=False):
            self.language_key = language_key
            self.quiet = quiet
            self.text_search_only = text_search_only
            self.index_file = index_path
            self.index_revision = self._current_index_revision()
            created_revisions.append(self.index_revision)

        def _current_index_revision(self) -> str:
            stat = self.index_file.stat()
            return f"{stat.st_mtime_ns}:{stat.st_size}"

    monkeypatch.setattr(
        rag_service, "dataset_language_to_rag_key", lambda _language: "fake"
    )
    monkeypatch.setattr(rag_service, "LanguageRAGService", FakeRAGService)
    rag_service._rag_instances.clear()

    first = rag_service.get_rag_service_for_dataset_language(
        "Fake", quiet=True, text_search_only=True
    )
    second = rag_service.get_rag_service_for_dataset_language(
        "Fake", quiet=True, text_search_only=True
    )
    index_path.write_text('{"changed": true}', encoding="utf-8")
    third = rag_service.get_rag_service_for_dataset_language(
        "Fake", quiet=True, text_search_only=True
    )

    assert first is second
    assert third is not first
    assert len(created_revisions) == 2

    rag_service._rag_instances.clear()
