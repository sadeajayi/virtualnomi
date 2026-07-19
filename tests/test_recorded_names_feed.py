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
