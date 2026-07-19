import io
import json
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.dataset_updates.deploy_discover_pronunciations import (
    apply_recordings,
    validate_wav_audio,
)


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    samples = (np.sin(np.linspace(0, 8 * np.pi, 4410)) * 8000).astype("<i2")
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(samples.tobytes())
    return buffer.getvalue()


def _manifest():
    return {
        "internal_pronunciation_by": "Folasade Ajayi",
        "recordings": [
            {
                "name_strip": "Temitope",
                "language": "Yoruba",
                "phonetic_spelling": "tay-mi-taw-kpe",
                "audio_filename": "temitope.m4a",
            }
        ],
    }


def _frame():
    return pd.DataFrame(
        [{
            "Name": "Tèmítọ́pẹ́",
            "NameStrip": "Temitope",
            "Language": "Yoruba",
            "Meaning": "Mine is worth celebrating.",
            "Attribution": "YorubaNames.com",
            "Phonetic spelling": "",
            "Audio Pronunciation": None,
            "pronunciation_by": "",
            "Validation_Status": "",
            "validated_by": "",
            "source_notes": "",
        }]
    )


def test_audio_batch_preserves_canonical_evidence_and_sets_internal_credit(tmp_path):
    (tmp_path / "temitope.m4a").write_bytes(b"test source placeholder")
    before = _frame()
    updated, report = apply_recordings(
        before,
        _manifest(),
        tmp_path,
        audio_loader=lambda _path: _wav_bytes(),
    )
    row = updated.iloc[0]
    assert row["Meaning"] == before.iloc[0]["Meaning"]
    assert row["Attribution"] == "YorubaNames.com"
    assert row["Validation_Status"] == ""
    assert row["Phonetic spelling"] == "tay-mi-taw-kpe"
    assert row["pronunciation_by"] == "Folasade Ajayi"
    assert row["Audio Pronunciation"]["bytes"][:4] == b"RIFF"
    assert report[0]["public_attribution_approved"] is False


def test_rejects_missing_or_duplicate_canonical_rows(tmp_path):
    (tmp_path / "temitope.m4a").write_bytes(b"test source placeholder")
    duplicate = pd.concat([_frame(), _frame()], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one canonical row"):
        apply_recordings(
            duplicate,
            _manifest(),
            tmp_path,
            audio_loader=lambda _path: _wav_bytes(),
        )


def test_rejects_silent_wav():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(b"\x00\x00" * 100)
    with pytest.raises(ValueError, match="no audible samples"):
        validate_wav_audio(buffer.getvalue())


def test_adds_new_nomi_sourced_row_with_honest_safe_defaults(tmp_path):
    (tmp_path / "ogunkoya.m4a").write_bytes(b"test source placeholder")
    frame = _frame()
    manifest = {
        "internal_pronunciation_by": "Folasade Ajayi",
        "recordings": [{
            "name_strip": "Ogunkoya",
            "language": "Yoruba",
            "phonetic_spelling": "oh-goon-kuh-yaa",
            "audio_filename": "ogunkoya.m4a",
            "new_canonical_record": {
                "name": "Ògúnkọ̀yà",
                "meaning": "Ògún denounces suffering.",
                "attribution": "Nomi",
            },
        }],
    }
    updated, report = apply_recordings(
        frame,
        manifest,
        tmp_path,
        audio_loader=lambda _path: _wav_bytes(),
    )
    row = updated.loc[updated["NameStrip"] == "Ogunkoya"].iloc[0]
    assert row["Name"] == "Ògúnkọ̀yà"
    assert row["Meaning"] == "Ògún denounces suffering."
    assert row["Attribution"] == "Nomi"
    assert row["Validation_Status"] == ""
    assert row["validated_by"] == ""
    assert report[0]["canonical_row_created"] is True
