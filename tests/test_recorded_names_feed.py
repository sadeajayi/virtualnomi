import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nomi-name-search-api"))

import app as api  # noqa: E402

_recorded_names_feed = api._recorded_names_feed


def _lookup():
    return {
        ("Adaeze", "Igbo"): {
            "Name": "Adaeze",
            "Phonetic spelling": "ah-deh-zeh",
            "pronunciation_by": "Chika",
        },
        ("Babangida", "Hausa"): {
            "Name": "Babangida",
            "Phonetic spelling": "bah-bahn-gee-dah",
        },
        ("Lantana", "Hausa (Localised Islamic/Arabic)"): {
            "Name": "Lantana",
        },
        ("Silent", "Igbo"): {
            "Name": "Silent",
        },
    }


def test_recorded_feed_filters_to_audio_keys_and_orders_stably():
    names, total, counts = _recorded_names_feed(
        None,
        10,
        dataset_lookup=_lookup(),
        audio_keys={
            ("Lantana", "Hausa (Localised Islamic/Arabic)"),
            ("Adaeze", "Igbo"),
            ("Babangida", "Hausa"),
        },
    )

    assert [item["name"] for item in names] == ["Adaeze", "Babangida", "Lantana"]
    assert total == 3
    assert counts == {
        "Hausa": 1,
        "Hausa (Localised Islamic/Arabic)": 1,
        "Igbo": 1,
    }
    assert names[0]["audio_url"] == "/audio/Adaeze?language=Igbo"
    assert "Audio Pronunciation" not in names[0]


def test_recorded_feed_supports_language_families_and_bounded_results():
    names, total, _ = _recorded_names_feed(
        "Hausa",
        1,
        dataset_lookup=_lookup(),
        audio_keys={
            ("Lantana", "Hausa (Localised Islamic/Arabic)"),
            ("Adaeze", "Igbo"),
            ("Babangida", "Hausa"),
        },
    )

    assert total == 2
    assert len(names) == 1
    assert names[0]["name"] == "Babangida"

    next_names, next_total, _ = _recorded_names_feed(
        "Hausa",
        1,
        1,
        dataset_lookup=_lookup(),
        audio_keys={
            ("Lantana", "Hausa (Localised Islamic/Arabic)"),
            ("Babangida", "Hausa"),
        },
    )
    assert next_total == 2
    assert [item["name"] for item in next_names] == ["Lantana"]


def test_audio_validation_rejects_empty_or_unknown_blobs():
    assert api._is_valid_audio_bytes(b"RIFF\x00\x00\x00\x00WAVEdata")
    assert not api._is_valid_audio_bytes(b"")
    assert not api._is_valid_audio_bytes(b"not really audio bytes")


def test_audio_endpoint_prefers_exact_language_over_family_match(monkeypatch):
    wav = b"RIFF\x10\x00\x00\x00WAVEdata"
    localised_mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x12payload"

    monkeypatch.setattr(
        api,
        "_load_audio_keys",
        lambda: [
            ("Shared", "Hausa (Localised Islamic/Arabic)"),
            ("Shared", "Hausa"),
        ],
    )
    monkeypatch.setattr(
        api,
        "_fetch_audio_bytes",
        lambda _name, language: {
            "Hausa": wav,
            "Hausa (Localised Islamic/Arabic)": localised_mp3,
        }[language],
    )

    response = TestClient(api.app).get("/audio/Shared?language=Hausa")

    assert response.status_code == 200
    assert response.content == wav
    assert response.headers["content-type"] == "audio/wav"


def test_audio_endpoint_rejects_ambiguous_family_match(monkeypatch):
    monkeypatch.setattr(
        api,
        "_load_audio_keys",
        lambda: {
            ("Shared", "Hausa (Localised Islamic/Arabic)"),
            ("Shared", "Hausa (Traditional)"),
        },
    )

    response = TestClient(api.app).get("/audio/Shared?language=Hausa")

    assert response.status_code == 409
    assert "Pass an exact language" in response.json()["detail"]


def test_audio_endpoint_rejects_ambiguous_missing_language(monkeypatch):
    monkeypatch.setattr(
        api,
        "_load_audio_keys",
        lambda: {
            ("Shared", "Igbo"),
            ("Shared", "Yoruba"),
        },
    )

    response = TestClient(api.app).get("/audio/Shared")

    assert response.status_code == 409
    assert "Igbo" in response.json()["detail"]
    assert "Yoruba" in response.json()["detail"]


def test_audio_endpoint_serves_mp3_with_matching_media_type(monkeypatch):
    mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x12payload"

    monkeypatch.setattr(api, "_load_audio_keys", lambda: {("Adaeze", "Igbo")})
    monkeypatch.setattr(api, "_fetch_audio_bytes", lambda *_args: mp3)

    response = TestClient(api.app).get("/audio/Adaeze?language=Igbo")

    assert response.status_code == 200
    assert response.content == mp3
    assert response.headers["content-type"] == "audio/mpeg"


def test_recorded_names_endpoint_returns_minimal_contract(monkeypatch):
    monkeypatch.setattr(
        api,
        "_recorded_names_feed",
        lambda language, limit, offset: (
            [
                {
                    "name": "Adaeze",
                    "name_strip": "Adaeze",
                    "language": "Igbo",
                    "phonetic_spelling": "ah-deh-zeh",
                    "pronunciation_by": "Chika",
                    "audio_url": "/audio/Adaeze?language=Igbo",
                }
            ],
            1,
            {"Igbo": 1},
        ),
    )
    response = TestClient(api.app).get("/recorded-names?language=Igbo&limit=4&offset=0")
    assert response.status_code == 200
    assert response.json() == {
        "names": [
            {
                "name": "Adaeze",
                "name_strip": "Adaeze",
                "language": "Igbo",
                "phonetic_spelling": "ah-deh-zeh",
                "pronunciation_by": "Chika",
                "audio_url": "/audio/Adaeze?language=Igbo",
            }
        ],
        "total": 1,
        "language": "Igbo",
        "language_counts": {"Igbo": 1},
    }
