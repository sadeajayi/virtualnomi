from pathlib import Path

import pandas as pd
import pytest

from scripts.dataset_updates import upload_audio_from_local_file as uploader


WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVEdata"


def _frame(audio=None, pronunciation_by=""):
    return pd.DataFrame(
        [
            {
                "NameStrip": "Folasade",
                "Language": "Yoruba",
                "Audio Pronunciation": audio,
                "pronunciation_by": pronunciation_by,
            }
        ]
    )


def _audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "folasade.wav"
    path.write_bytes(WAV_BYTES)
    return path


def _stub_download_and_frame(monkeypatch, frame):
    monkeypatch.setattr(uploader, "HF_TOKEN", "token")
    monkeypatch.setattr(uploader, "hf_hub_download", lambda **_kwargs: "/tmp/snapshots/source/data/train.parquet")
    monkeypatch.setattr(uploader.pd, "read_parquet", lambda _path: frame.copy())


def test_upload_audio_refuses_existing_slot_without_force(tmp_path, monkeypatch):
    _stub_download_and_frame(
        monkeypatch,
        _frame(audio={"bytes": WAV_BYTES}, pronunciation_by="Existing Recorder"),
    )

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        uploader.upload_audio(
            "Folasade",
            "Yoruba",
            _audio_file(tmp_path),
            pronunciation_by="New Recorder",
        )


def test_upload_audio_preview_only_without_push(tmp_path, monkeypatch):
    _stub_download_and_frame(monkeypatch, _frame())
    pushed = {"called": False}

    class FakeDataset:
        @classmethod
        def from_pandas(cls, _df):
            pushed["called"] = True
            return cls()

    monkeypatch.setattr(uploader, "Dataset", FakeDataset)

    size = uploader.upload_audio(
        "Folasade",
        "Yoruba",
        _audio_file(tmp_path),
        pronunciation_by="Folasade Ajayi",
    )

    assert size == len(WAV_BYTES)
    assert pushed["called"] is False


def test_upload_audio_refuses_stale_source_revision_on_push(tmp_path, monkeypatch):
    _stub_download_and_frame(monkeypatch, _frame())
    monkeypatch.setattr(uploader, "_source_revision_from_download", lambda _path: "old")
    monkeypatch.setattr(uploader, "_current_dataset_revision", lambda: "new")

    with pytest.raises(RuntimeError, match="Refusing to push stale parquet snapshot"):
        uploader.upload_audio(
            "Folasade",
            "Yoruba",
            _audio_file(tmp_path),
            pronunciation_by="Folasade Ajayi",
            push=True,
        )
