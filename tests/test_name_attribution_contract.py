import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nomi-name-search-api"))

import app as api  # noqa: E402


def test_name_response_keeps_canonical_meaning_attribution(monkeypatch):
    monkeypatch.setattr(
        api,
        "_lookup_name_results",
        lambda *_: [
            {
                "name": "Adunni",
                "name_strip": "Adunni",
                "language": "Yoruba",
                "meaning": "Sweet to have",
                "attribution": "YorubaNames.com",
            }
        ],
    )
    response = TestClient(api.app).get("/name/Adunni?language=Yoruba")
    assert response.status_code == 200
    assert response.json()["results"][0]["attribution"] == "YorubaNames.com"
