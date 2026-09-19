"""Safety tests for the Yoruba attribution dataset updater."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "dataset_updates"
    / "update_yoruba_attributions.py"
)

spec = importlib.util.spec_from_file_location("update_yoruba_attributions", SCRIPT_PATH)
assert spec and spec.loader
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)


def test_push_rejects_cached_or_local_parquet_sources() -> None:
    with pytest.raises(SystemExit, match="current Hugging Face dataset"):
        updater._validate_push_source(push=True, from_cache=True, parquet_path=None)

    with pytest.raises(SystemExit, match="current Hugging Face dataset"):
        updater._validate_push_source(
            push=True, from_cache=False, parquet_path="stale.parquet"
        )


def test_nomi_exceptions_are_reported_when_already_correct() -> None:
    df = pd.DataFrame(
        [
            {
                "NameStrip": "Ajobi",
                "Language": "Yoruba",
                "Attribution": updater.NOMI_ATTRIBUTION,
            },
            {
                "NameStrip": "Morohunkeji",
                "Language": "Yoruba",
                "Attribution": "",
            },
            {
                "NameStrip": "Aanu",
                "Language": "Yoruba",
                "Attribution": updater.YORUBANAMES_ATTRIBUTION,
            },
        ]
    )

    updated, report = updater.apply_attributions(df)

    assert updated.loc[0, "Attribution"] == updater.NOMI_ATTRIBUTION
    assert updated.loc[1, "Attribution"] == updater.NOMI_ATTRIBUTION
    assert report["already_nomi"] == ["Ajobi"]
    assert report["set_to_nomi"] == [{"NameStrip": "Morohunkeji", "previous": ""}]
    assert report["counts"]["already_nomi"] == 1
    assert updater._changed_attribution_count(report) == 1


def test_noop_report_has_zero_changed_attributions() -> None:
    df = pd.DataFrame(
        [
            {
                "NameStrip": "Ajobi",
                "Language": "Yoruba",
                "Attribution": updater.NOMI_ATTRIBUTION,
            },
            {
                "NameStrip": "Aanu",
                "Language": "Yoruba",
                "Attribution": updater.YORUBANAMES_ATTRIBUTION,
            },
        ]
    )

    _, report = updater.apply_attributions(df)

    assert report["counts"]["already_nomi"] == 1
    assert report["counts"]["already_yorubanames"] == 1
    assert updater._changed_attribution_count(report) == 0
