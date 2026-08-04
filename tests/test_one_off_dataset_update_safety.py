import pandas as pd
import pytest

from scripts.dataset_updates.safety import require_unique_canonical_row
from scripts.dataset_updates.update_ebube_ebubechukwu import apply_updates
from scripts.dataset_updates import update_morenikeji_meaning


def _row(name_strip: str, language: str, meaning: str = "old") -> dict:
    return {
        "Name": name_strip,
        "NameStrip": name_strip,
        "Language": language,
        "Meaning": meaning,
        "Audio Pronunciation": None,
    }


def test_unique_canonical_row_helper_rejects_duplicates():
    frame = pd.DataFrame(
        [
            _row("Adaora", "Igbo"),
            _row("adaora", "Igbo"),
        ]
    )

    with pytest.raises(ValueError, match="Expected exactly one canonical row"):
        require_unique_canonical_row(frame, "Adaora", "Igbo")


def test_unique_canonical_row_helper_returns_single_index():
    frame = pd.DataFrame(
        [
            _row("Morenikeji", "Yoruba"),
            _row("Adaora", "Igbo"),
        ],
        index=[10, 20],
    )

    assert require_unique_canonical_row(frame, "adaora", "Igbo") == 20


def test_ebube_update_rejects_duplicate_target_rows():
    frame = pd.DataFrame(
        [
            _row("Ebube", "Igbo", "Glory."),
            _row("Ebube", "Igbo", "Another meaning."),
        ]
    )

    with pytest.raises(ValueError, match="Expected exactly one canonical row"):
        apply_updates(frame)


def test_ebube_update_rejects_duplicate_existing_append_target():
    frame = pd.DataFrame(
        [
            _row("Ebube", "Igbo", "Glory."),
            _row("Ebubechukwu", "Igbo", "God's glory."),
            _row("Ebubechukwu", "Igbo", "Duplicate."),
        ]
    )

    with pytest.raises(ValueError, match="Expected at most one canonical row"):
        apply_updates(frame)


def test_morenikeji_update_rejects_duplicates_before_push(monkeypatch):
    frame = pd.DataFrame(
        [
            _row("Morenikeji", "Yoruba", "old one"),
            _row("Morenikeji", "Yoruba", "old two"),
        ]
    )

    monkeypatch.setattr(update_morenikeji_meaning, "HF_TOKEN", "token")
    monkeypatch.setattr(
        update_morenikeji_meaning,
        "hf_hub_download",
        lambda **_kwargs: "downloaded.parquet",
    )
    monkeypatch.setattr(
        update_morenikeji_meaning.pd,
        "read_parquet",
        lambda _path: frame,
    )
    monkeypatch.setattr(
        update_morenikeji_meaning.Dataset,
        "from_pandas",
        lambda *_args, **_kwargs: pytest.fail("duplicate rows must not be pushed"),
    )

    with pytest.raises(ValueError, match="Expected exactly one canonical row"):
        update_morenikeji_meaning.main()
