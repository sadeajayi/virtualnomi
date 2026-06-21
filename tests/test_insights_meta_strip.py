import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nomi-name-search-api"))

from insights_service import (  # noqa: E402
    _clean_insight_output,
    _contains_meta_source_language,
    _strip_meta_source_sentences,
)


def test_strip_all_meta_sentence_returns_empty():
    bad = (
        "The RAG sources list this name as Obianujuaku, meaning "
        "'she who comes in abundance of wealth or children.'"
    )
    assert _strip_meta_source_sentences(bad) == ""


def test_strip_sources_list_without_rag_returns_empty():
    bad = "The sources list this name as Obianujuaku, meaning abundance."
    assert _strip_meta_source_sentences(bad) == ""


def test_strip_keeps_griot_sentences():
    good = (
        "Obianuju is an Igbo name parents give when a daughter arrives bringing "
        "abundance — wealth, children, or both felt as a gift at her birth."
    )
    assert _strip_meta_source_sentences(good) == good


def test_strip_mixed_paragraph():
    mixed = (
        "Obianuju names abundance at birth. The sources list a longer form. "
        "Parents speak it as a declaration of plenty."
    )
    assert _strip_meta_source_sentences(mixed) == (
        "Obianuju names abundance at birth. Parents speak it as a declaration of plenty."
    )


def test_detects_broader_meta_phrases():
    assert _contains_meta_source_language("According to the RAG sources, Obianuju means abundance.")
    assert _contains_meta_source_language("In the research, this name signals wealth.")
    assert not _contains_meta_source_language(
        "Obianuju is a name for a daughter welcomed with abundance."
    )


def test_clean_insight_output_strips_meta_only_response():
    raw = "The RAG sources list Obianuju as born into abundance."
    assert _clean_insight_output(raw) == ""
