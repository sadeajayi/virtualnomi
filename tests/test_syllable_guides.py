import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nomi-name-search-api"))

import syllable_store as store  # noqa: E402
from syllable_service import apply_published_phonetic, resolve_syllable_guide  # noqa: E402


@pytest.fixture(autouse=True)
def reset_store(tmp_path, monkeypatch):
    guides = tmp_path / "syllable_guides.jsonl"
    guides.write_text(
        json.dumps(
            {
                "name_strip": "Amarachi",
                "language": "Igbo",
                "flat_phonetic": "ah-mah-RAH-chee",
                "syllables": [
                    {"script": "A", "say": "ah", "anchor": 'the "ah" in "Ah, I see!"', "stress": False},
                    {"script": "ma", "say": "mah", "anchor": 'the "mah" in mama', "stress": False},
                    {"script": "ra", "say": "rah", "anchor": '"rah" — like cheering at a game', "stress": True},
                    {"script": "chi", "say": "chee", "anchor": 'the "chee" in cheese', "stress": False},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(store, "GUIDE_PATHS", (guides,))
    store.reset_syllable_store()
    yield
    store.reset_syllable_store()


def test_published_lookup_returns_anchors_and_stress():
    syllables = store.get_published_syllables("Amarachi", "Igbo")
    assert syllables is not None
    assert [beat["say"] for beat in syllables] == ["ah", "mah", "rah", "chee"]
    assert syllables[2]["stress"] is True
    assert "cheering" in syllables[2]["anchor"]


def test_unreadable_guide_file_is_ignored(tmp_path, monkeypatch):
    bad_guides = tmp_path / "bad_syllable_guides.jsonl"
    bad_guides.write_bytes(b"\xff\xfe\xfa")
    monkeypatch.setattr(store, "GUIDE_PATHS", (bad_guides,))
    store.reset_syllable_store()

    assert store.get_published_syllables("Amarachi", "Igbo") is None


def test_generated_guides_are_remembered_for_later_lookup():
    generated = [
        {"say": "tay", "anchor": 'the "tay" in stay, without the s', "stress": False},
        {"say": "mi", "anchor": "the word “me”", "stress": False},
        {"say": "taw", "anchor": 'the "taw" in talk, without the lk', "stress": True},
        {"say": "kpe", "anchor": '"kpe" — one sound, like a popped p', "stress": False},
    ]
    store.remember_generated("Temitope", "Yoruba", generated)
    assert store.get_published_syllables("Temitope", "Yoruba") == generated


def test_apply_published_phonetic_keeps_beats_and_caps_stress():
    aligned = apply_published_phonetic(
        [
            {"say": "wrong", "anchor": 'the "fuh" in fun', "stress": True},
            {"say": "wrong", "anchor": '"la" — held, like a sung note', "stress": False},
            {"say": "wrong", "anchor": 'the "sha" in shark', "stress": False},
            {"say": "wrong", "anchor": 'the word "day"', "stress": False},
        ],
        "fuh-LAA-sha-day",
    )
    assert [beat["say"] for beat in aligned] == ["fuh", "laa", "sha", "day"]
    assert [beat["stress"] for beat in aligned] == [False, True, False, False]


def test_resolve_uses_published_without_calling_claude(monkeypatch):
    monkeypatch.setattr(
        "syllable_service.generate_syllable_guide",
        lambda **_: (_ for _ in ()).throw(AssertionError("should not generate")),
    )
    payload = resolve_syllable_guide(
        name="Amarachi",
        name_strip="Amarachi",
        language="Igbo",
        phonetic="ah-mah-RAH-chee",
        generate=True,
    )
    assert payload["source"] == "published"
    assert payload["syllables"][2]["stress"] is True


def test_resolve_generates_when_unpublished(monkeypatch):
    beats = [
        {"say": "tay", "anchor": 'the "tay" in stay, without the s', "stress": False},
        {"say": "mi", "anchor": "the word me", "stress": False},
        {"say": "taw", "anchor": 'the "taw" in talk, without the lk', "stress": True},
        {"say": "kpe", "anchor": "kpe — one sound, like a popped p", "stress": False},
    ]
    monkeypatch.setattr(
        "syllable_service.generate_syllable_guide",
        lambda **_: beats,
    )
    payload = resolve_syllable_guide(
        name="Temitope",
        name_strip="Temitope",
        language="Yoruba",
        phonetic="tay-mi-taw-kpe",
        generate=True,
    )
    assert payload["source"] == "generated"
    assert [beat["say"] for beat in payload["syllables"]] == ["tay", "mi", "taw", "kpe"]
    assert store.get_published_syllables("Temitope", "Yoruba") == beats


def test_resolve_regenerates_when_published_beats_do_not_match_phonetic(tmp_path, monkeypatch):
    stale_guides = tmp_path / "stale_syllable_guides.jsonl"
    stale_guides.write_text(
        json.dumps(
            {
                "name_strip": "Temitope",
                "language": "Yoruba",
                "syllables": [
                    {"say": "tay", "anchor": 'the "tay" in stay', "stress": True},
                    {"say": "mi", "anchor": "the word me", "stress": False},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(store, "GUIDE_PATHS", (stale_guides,))
    store.reset_syllable_store()

    generated = [
        {"say": "tay", "anchor": 'the "tay" in stay', "stress": False},
        {"say": "mi", "anchor": "the word me", "stress": False},
        {"say": "taw", "anchor": 'the "taw" in talk', "stress": True},
        {"say": "kpe", "anchor": "kpe — one sound", "stress": False},
    ]
    monkeypatch.setattr(
        "syllable_service.generate_syllable_guide",
        lambda **_: generated,
    )

    payload = resolve_syllable_guide(
        name="Temitope",
        name_strip="Temitope",
        language="Yoruba",
        phonetic="tay-mi-taw-kpe",
        generate=True,
    )

    assert payload["source"] == "generated"
    assert [beat["say"] for beat in payload["syllables"]] == ["tay", "mi", "taw", "kpe"]


def test_syllable_guide_endpoint_returns_published(monkeypatch):
    from fastapi.testclient import TestClient

    import app as api

    monkeypatch.setattr(
        api,
        "_lookup_name_results",
        lambda *_: [
            {
                "name": "Amarachi",
                "name_strip": "Amarachi",
                "language": "Igbo",
                "meaning": "God's grace",
                "phonetic_spelling": "ah-mah-RAH-chee",
            }
        ],
    )
    response = TestClient(api.app).get("/syllable-guide?name=Amarachi&language=Igbo")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "published"
    assert body["syllables"][2]["stress"] is True
    assert "cheering" in body["syllables"][2]["anchor"]


def test_name_lookup_includes_published_syllables(monkeypatch):
    from fastapi.testclient import TestClient

    import app as api

    monkeypatch.setattr(
        api,
        "_lookup_name_results",
        lambda *_: [
            {
                "name": "Amarachi",
                "name_strip": "Amarachi",
                "language": "Igbo",
                "meaning": "God's grace",
                "phonetic_spelling": "ah-mah-RAH-chee",
                "syllables": store.get_published_syllables("Amarachi", "Igbo"),
            }
        ],
    )
    response = TestClient(api.app).get("/name/Amarachi?language=Igbo")
    assert response.status_code == 200
    syllables = response.json()["results"][0]["syllables"]
    assert syllables[0]["say"] == "ah"
    assert syllables[2]["anchor"]
