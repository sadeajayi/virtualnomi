import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nomi-name-search-api"))
sys.path.insert(0, str(ROOT / "rag"))

from app import InsightsResponse  # noqa: E402
from language_config import LANGUAGE_CONFIG  # noqa: E402
from source_metadata import (  # noqa: E402
    SOURCE_METADATA_REGISTRY,
    build_structured_sources,
    parse_rag_excerpts,
    resolve_source_metadata,
)


def test_registry_covers_every_configured_rag_paper():
    configured = {
        filename
        for config in LANGUAGE_CONFIG.values()
        for filename in config.get("papers", [])
    }
    assert configured == set(SOURCE_METADATA_REGISTRY)


def test_resolves_verified_metadata_without_page_fields():
    source = resolve_source_metadata(
        "The_Sociolinguistics_of_Igbo_Personal_Na.pdf",
        "Naming practices change with social and religious influences.",
    )
    assert source == {
        "filename": "The_Sociolinguistics_of_Igbo_Personal_Na.pdf",
        "title": "The Sociolinguistics of Igbo Personal Names",
        "title_is_fallback": False,
        "author": "Linda Chinelo Nkamigbo",
        "year": "2019",
        "excerpt": "Naming practices change with social and religious influences.",
    }
    assert "page" not in source


def test_parses_multiline_excerpt_blocks():
    parsed = parse_rag_excerpts(
        "[First_Source.pdf]: First line.\nSecond line.\n\n"
        "[Second_Source.pdf]: Another excerpt."
    )
    assert parsed == [
        {
            "filename": "First_Source.pdf",
            "excerpt": "First line.\nSecond line.",
        },
        {
            "filename": "Second_Source.pdf",
            "excerpt": "Another excerpt.",
        },
    ]


def test_sources_preserve_retrieval_order_and_deduplicate():
    sources = build_structured_sources(
        "[Trends_Igbo_Names.pdf]: First retrieved.\n\n"
        "[Semantic_Analysis_of_Igbo_Names.pdf]: Second retrieved.\n\n"
        "[Trends_Igbo_Names.pdf]: Duplicate excerpt.",
        [
            "Semantic_Analysis_of_Igbo_Names.pdf",
            "Unknown_Source.pdf",
        ],
    )
    assert [source["filename"] for source in sources] == [
        "Trends_Igbo_Names.pdf",
        "Semantic_Analysis_of_Igbo_Names.pdf",
        "Unknown_Source.pdf",
    ]
    assert sources[0]["excerpt"] == "First retrieved."


def test_unknown_source_uses_explicit_filename_fallback():
    source = resolve_source_metadata("Unknown_Research-Paper.pdf")
    assert source["title"] == "Unknown Research Paper"
    assert source["title_is_fallback"] is True
    assert "author" not in source
    assert "year" not in source


def test_insights_response_is_additive_and_keeps_legacy_attributions():
    response = InsightsResponse(
        name="Adaeze",
        language="Igbo",
        meaning="Daughter of a king",
        insight="A cultural reading.",
        rag_used=True,
        sources=[resolve_source_metadata("Trends_Igbo_Names.pdf", "Excerpt.")],
        attributions=["Trends_Igbo_Names.pdf"],
    )
    payload = response.model_dump()
    assert payload["sources"][0]["title"].startswith("A Sociolinguistic Study")
    assert payload["attributions"] == ["Trends_Igbo_Names.pdf"]
    assert "page" not in payload["sources"][0]
