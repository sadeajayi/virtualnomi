import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "paraphrasing"))

import transform_yoruba_meanings as script  # noqa: E402


def test_parse_variations_response_rejects_malformed_text():
    response = "Here are five options:\n- Crown brings joy\n- Joy wears a crown"

    assert script.parse_variations_response(response, "Folasade") == []


def test_parse_variations_response_accepts_fenced_json_array():
    response = '```json\n["Crown brings joy", "Joy wears a crown"]\n```'

    assert script.parse_variations_response(response, "Folasade") == [
        "Crown brings joy",
        "Joy wears a crown",
    ]
