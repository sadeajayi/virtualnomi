from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts" / "dataset_updates"),
)

from restore_audio_from_hf_history import _apply_restored_audio  # noqa: E402


def _df(audio: bytes | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "NameStrip": "Ikeme",
                "Language": "Igbo",
                "Audio Pronunciation": {"bytes": audio} if audio is not None else None,
                "pronunciation_by": "Current Speaker",
            }
        ]
    )


def test_restore_skips_existing_audio_without_force() -> None:
    df = _df(b"current-audio")

    changed, message = _apply_restored_audio(
        df,
        "Ikeme",
        "Igbo",
        b"historical-audio",
        force=False,
    )

    assert not changed
    assert "already has" in message
    assert df.loc[0, "Audio Pronunciation"] == {"bytes": b"current-audio"}
    assert df.loc[0, "pronunciation_by"] == "Current Speaker"


def test_restore_fills_missing_audio() -> None:
    df = _df()

    changed, message = _apply_restored_audio(
        df,
        "Ikeme",
        "Igbo",
        b"historical-audio",
        force=False,
    )

    assert changed
    assert "Restored" in message
    assert df.loc[0, "Audio Pronunciation"] == {"bytes": b"historical-audio"}
    assert df.loc[0, "pronunciation_by"] == "Daphne Chiamaka"


def test_restore_force_replaces_existing_audio() -> None:
    df = _df(b"current-audio")

    changed, message = _apply_restored_audio(
        df,
        "Ikeme",
        "Igbo",
        b"historical-audio",
        force=True,
    )

    assert changed
    assert "Restored" in message
    assert df.loc[0, "Audio Pronunciation"] == {"bytes": b"historical-audio"}
    assert df.loc[0, "pronunciation_by"] == "Daphne Chiamaka"
