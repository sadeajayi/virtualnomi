import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nomi-name-search-api"))

from insights_service import (  # noqa: E402
    _clean_insight_output,
    _contains_contrast_pedagogy,
    _contains_meta_source_language,
    _cultural_depth_adds_value,
    _strip_contrast_pedagogy_opening,
    _strip_meta_source_sentences,
    build_about_the_name,
    format_attribution,
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


def test_detects_rather_than_in_opening():
    bad = (
        "Amaechi is an Igbo name that asks a question rather than states a fact — "
        "that is not a rhetorical flourish but a whole philosophy."
    )
    assert _contains_contrast_pedagogy(bad)


def test_detects_not_a_but_in_opening():
    bad = "This is not a label but a public declaration about the household."
    assert _contains_contrast_pedagogy(bad)


def test_contrast_detection_ignores_later_sentences():
    text = (
        "Amaechi asks who will lead the household into the next generation. "
        "Parents chose it as a question rather than a statement."
    )
    assert not _contains_contrast_pedagogy(text)


def test_strip_contrast_opening_keeps_direct_lead():
    mixed = (
        "Amaechi is an Igbo name that asks a question rather than states a fact. "
        "Who will carry this family forward is the question parents embed in the name."
    )
    assert _strip_contrast_pedagogy_opening(mixed) == (
        "Who will carry this family forward is the question parents embed in the name."
    )


def test_strip_contrast_opening_only_sentence_returns_empty():
    bad = "Amaechi asks a question rather than states a fact."
    assert _strip_contrast_pedagogy_opening(bad) == ""


def test_clean_insight_output_strips_contrast_opening():
    raw = (
        "Amaechi is an Igbo name that asks a question rather than states a fact. "
        "Who will lead the family is the question at its heart."
    )
    assert _clean_insight_output(raw) == (
        "Who will lead the family is the question at its heart."
    )


def test_direct_opening_passes_contrast_check():
    good = (
        "Amaechi is the question Igbo parents ask when they want a child's name "
        "to carry a whole philosophy in a single sentence."
    )
    assert not _contains_contrast_pedagogy(good)
    assert _strip_contrast_pedagogy_opening(good) == good


def test_build_about_the_name_splits_meaning():
    about = build_about_the_name(
        "Wear wealth like a crown. Parents expect abundance to follow.",
        additional_meaning="Common among Yoruba families.",
    )
    assert about["headline"] == "Wear wealth like a crown."
    assert "Parents expect abundance" in about["body"]
    assert "Common among Yoruba" in about["body"]


def test_cultural_depth_rejects_restatement():
    corpus = (
        "Wear wealth like a crown. Parents expect abundance to follow. "
        "Common among Yoruba families."
    )
    redundant = "Wear wealth like a crown and parents expect abundance to follow."
    novel = (
        "The ọlá root appears across hundreds of Yoruba names and always signals "
        "honour earned in public, witnessed by others."
    )
    assert not _cultural_depth_adds_value(redundant, corpus)
    assert _cultural_depth_adds_value(novel, corpus)


def test_format_attribution_single_and_multiple():
    assert format_attribution(["Yoruba_Praise_Names.pdf"]) == (
        "Source: Yoruba_Praise_Names.pdf"
    )
    assert "Sources:" in format_attribution(["a.pdf", "b.pdf"])
    assert format_attribution([]) == ""


def test_build_user_message_uses_full_insight_prompt():
    from insights_service import build_user_message

    msg = build_user_message("Ikeme", "Igbo", "It is not by our power", "", [])
    assert "NONE" not in msg
    assert "cultural depth ONLY" not in msg
    assert "stay within meaning and precise general knowledge" in msg


def test_build_about_the_name_includes_additional_only():
    about = build_about_the_name(
        "It is not by our power.",
        additional_meaning="Often given when parents acknowledge divine agency.",
    )
    assert about["headline"] == "It is not by our power."
    assert "divine agency" in about["body"]
