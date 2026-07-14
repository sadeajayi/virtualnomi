import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nomi-name-search-api"))

import app as api  # noqa: E402


def _reset_api_state():
    api.ds = None
    api._dataset_lookup = None
    api._audio_column_rows = None
    api._audio_keys_cache = None
    api._audio_bytes_cache = {}
    api._stories_data = {}
    api._stories_lookup = {}
    api._paraphrase_lookup = {}


@pytest.fixture(autouse=True)
def reset_api_state(monkeypatch):
    _reset_api_state()
    monkeypatch.setattr(api, "get_paraphrase_lookup", lambda: {})
    monkeypatch.setattr(api, "load_stories_data", lambda: {})
    monkeypatch.setattr(api, "get_story_from_dataset", lambda name_strip, language: {})
    monkeypatch.setattr(api, "get_name_metadata_from_dataset", lambda name_strip, language: {})
    yield
    _reset_api_state()


def test_query_name_db_matches_display_name_with_diacritics(monkeypatch):
    api.ds = [
        {
            "Name": "Adéọlá",
            "NameStrip": "Adeola",
            "Meaning": "crown of wealth",
            "Language": "Yoruba",
        }
    ]
    monkeypatch.setattr(
        api,
        "ensure_search_components",
        lambda query: pytest.fail("exact display-name lookup should not use semantic search"),
    )

    results = api.query_name_db("Adéọlá", "Yoruba")

    assert len(results) == 1
    assert results[0]["name_strip"] == "Adeola"
    assert results[0]["language"] == "Yoruba"


def test_query_name_db_returns_empty_when_exact_match_is_filtered(monkeypatch):
    api.ds = [
        {
            "Name": "Ada",
            "NameStrip": "Ada",
            "Meaning": "first daughter",
            "Language": "Hausa",
        }
    ]
    monkeypatch.setattr(
        api,
        "ensure_search_components",
        lambda query: pytest.fail("filtered exact-name lookup should not use semantic search"),
    )

    assert api.query_name_db("ada", "Igbo") == []


def test_lookup_name_results_matches_display_name_with_diacritics():
    api.ds = [
        {
            "Name": "Fọláṣade",
            "NameStrip": "Folasade",
            "Meaning": "honor confers a crown",
            "Language": "Yoruba",
        }
    ]
    api._audio_keys_cache = set()

    results = api._lookup_name_results("Fọláṣade", "Yoruba")

    assert len(results) == 1
    assert results[0]["name_strip"] == "Folasade"
    assert results[0]["meaning"] == "honor confers a crown"


def test_dataset_load_failure_is_not_cached(monkeypatch):
    rows = [
        {
            "Name": "Adeola",
            "NameStrip": "Adeola",
            "Meaning": "crown of wealth",
            "Language": "Yoruba",
        }
    ]
    calls = {"count": 0}

    def flaky_read(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary hub failure")
        return rows

    monkeypatch.setattr(api, "_read_parquet_rows", flaky_read)

    assert api.load_dataset_fallback() == []
    assert api.ds is None
    assert api.load_dataset_fallback() == rows
    assert api.ds == rows


def test_audio_key_load_failure_is_not_cached(monkeypatch):
    rows = [
        {
            "NameStrip": "Adeola",
            "Language": "Yoruba",
            "Audio Pronunciation": {"bytes": b"wav"},
        }
    ]
    calls = {"count": 0}

    def flaky_read(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary hub failure")
        return rows

    monkeypatch.setattr(api, "_read_parquet_rows", flaky_read)

    assert api._load_audio_keys() == set()
    assert api._audio_column_rows is None
    assert api._audio_keys_cache is None
    assert api._load_audio_keys() == {("Adeola", "Yoruba")}
