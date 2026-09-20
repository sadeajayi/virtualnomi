import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nomi-name-search-api"))

import app as api  # noqa: E402


PAYLOAD = "</script><script>window.__nomi_xss=1</script>"


def _malicious_result():
    return {
        "name": f"Ada{PAYLOAD}",
        "name_strip": f"ada{PAYLOAD}",
        "language": "Igbo",
        "meaning": f"Daughter of a king {PAYLOAD}",
        "phonetic_spelling": f"ah-deh{PAYLOAD}",
        "audio_url": f"/audio/ada?language=Igbo&x={PAYLOAD}",
        "pronunciation_by": f"Chika{PAYLOAD}",
    }


def test_inline_script_json_escapes_script_breakout_sequences():
    literal = api._json_for_inline_script(PAYLOAD)

    assert "</script>" not in literal
    assert "<script>" not in literal
    assert "\\u003C/script\\u003E\\u003Cscript\\u003E" in literal


def test_full_card_html_does_not_allow_dataset_script_breakout():
    html = api._generate_name_card_html(
        [_malicious_result()],
        "ada",
        base_url="https://example.test",
        mode="full",
    )

    assert html.count("<script>") == 1
    assert html.count("</script>") == 1
    assert "<script>window.__nomi_xss=1</script>" not in html
    assert "\\u003C/script\\u003E\\u003Cscript\\u003E" in html


def test_share_page_html_does_not_allow_namestrip_script_breakout():
    html = api._generate_share_page_html(
        [_malicious_result()],
        "ada",
        base_url="https://example.test",
    )

    assert html.count("<script>") == 1
    assert html.count("</script>") == 1
    assert "<script>window.__nomi_xss=1</script>" not in html
