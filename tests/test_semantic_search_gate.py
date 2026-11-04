import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nomi-name-search-api"))

from app import (  # noqa: E402
    SearchUnavailableError,
    coerce_query_embedding,
    local_torch_embeddings_allowed,
    query_name_db,
    query_with_lexical_meaning,
    semantic_search_enabled,
)


def test_coerce_embedding_accepts_flat_vector():
    vec = list(range(384))
    assert coerce_query_embedding(vec) == [float(x) for x in vec]


def test_coerce_embedding_mean_pools_token_matrix():
    token_emb = [[0.0] * 384, [2.0] * 384]
    pooled = coerce_query_embedding(token_emb)
    assert len(pooled) == 384
    assert pooled[0] == 1.0


def test_coerce_embedding_rejects_wrong_size():
    with pytest.raises(SearchUnavailableError):
        coerce_query_embedding([0.1, 0.2, 0.3])


def test_semantic_search_on_when_pinecone_set_even_if_flag_off(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test")
    monkeypatch.setenv("NOMI_SEMANTIC_SEARCH", "0")
    assert semantic_search_enabled() is True
    assert local_torch_embeddings_allowed() is False


def test_semantic_search_off_without_pinecone(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.setenv("NOMI_SEMANTIC_SEARCH", "1")
    assert semantic_search_enabled() is False


def test_lexical_meaning_search_finds_substring(monkeypatch):
    rows = [
        {
            "Name": "Ifẹ́",
            "NameStrip": "ife",
            "Meaning": "love and honor",
            "Language": "Yoruba",
            "Additional meaning": "",
            "cultural_context": "",
            "themes": [],
        },
        {
            "Name": "Ada",
            "NameStrip": "ada",
            "Meaning": "first daughter",
            "Language": "Igbo",
            "Additional meaning": "",
            "cultural_context": "",
            "themes": [],
        },
    ]
    monkeypatch.setattr("app.load_dataset_fallback", lambda: rows)
    monkeypatch.setattr("app.get_name_metadata_from_dataset", lambda *a, **k: {})
    monkeypatch.setattr("app.get_story_from_dataset", lambda *a, **k: {})
    hits = query_with_lexical_meaning("love", "Yoruba")
    assert len(hits) == 1
    assert hits[0]["name_strip"] == "ife"
    assert hits[0]["score"] == 0.8


def test_query_name_db_falls_back_lexically_when_vectors_unavailable(monkeypatch):
    rows = [
        {
            "Name": "Ifẹ́",
            "NameStrip": "ife",
            "Meaning": "love",
            "Language": "Yoruba",
            "Additional meaning": "",
            "cultural_context": "",
            "themes": [],
        }
    ]
    monkeypatch.setattr("app.load_dataset_fallback", lambda: rows)
    monkeypatch.setattr("app.get_name_metadata_from_dataset", lambda *a, **k: {})
    monkeypatch.setattr("app.get_story_from_dataset", lambda *a, **k: {})

    def _unavailable(_query):
        raise SearchUnavailableError("PINECONE_API_KEY is not set")

    monkeypatch.setattr("app.ensure_search_components", _unavailable)
    hits = query_name_db("love", "Yoruba")
    assert hits[0]["name_strip"] == "ife"


def test_query_name_db_falls_back_on_unexpected_vector_error(monkeypatch):
    rows = [
        {
            "Name": "Ifẹ́",
            "NameStrip": "ife",
            "Meaning": "love",
            "Language": "Yoruba",
            "Additional meaning": "",
            "cultural_context": "",
            "themes": [],
        }
    ]
    monkeypatch.setattr("app.load_dataset_fallback", lambda: rows)
    monkeypatch.setattr("app.get_name_metadata_from_dataset", lambda *a, **k: {})
    monkeypatch.setattr("app.get_story_from_dataset", lambda *a, **k: {})
    monkeypatch.setattr("app.ensure_search_components", lambda q: (_ for _ in ()).throw(RuntimeError("pinecone dim")))
    hits = query_name_db("a child who brings joy to the household", "Yoruba")
    assert hits == []
