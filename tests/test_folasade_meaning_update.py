import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dataset_updates"))

import update_folasade_meaning as updater  # noqa: E402


def test_snapshot_revision_reads_hf_cache_snapshot_path(tmp_path):
    parquet = (
        tmp_path
        / "hub"
        / "datasets--nomi-stories--nomi-names"
        / "snapshots"
        / "abc123"
        / "data"
        / "train-00000-of-00001.parquet"
    )
    parquet.parent.mkdir(parents=True)
    parquet.touch()

    assert updater.snapshot_revision(parquet) == "abc123"


def test_snapshot_revision_rejects_non_snapshot_path(tmp_path):
    with pytest.raises(SystemExit, match="Could not determine HF snapshot revision"):
        updater.snapshot_revision(tmp_path / "data" / "train.parquet")


def test_load_df_returns_dataframe_and_source_revision(tmp_path, monkeypatch):
    parquet = tmp_path / "snapshots" / "source-sha" / "data" / "train.parquet"
    parquet.parent.mkdir(parents=True)
    expected = pd.DataFrame(
        {
            "NameStrip": ["Folasade"],
            "Language": ["Yoruba"],
            "Meaning": ["old"],
        }
    )
    expected.to_parquet(parquet, index=False)
    monkeypatch.setattr(updater, "hf_hub_download", lambda **_: str(parquet))

    actual, revision = updater.load_df("token")

    assert revision == "source-sha"
    pd.testing.assert_frame_equal(actual, expected)


def test_push_dataset_update_guards_upload_with_parent_revision(monkeypatch):
    calls = []

    class FakeHfApi:
        def __init__(self, token):
            self.token = token

        def upload_file(self, **kwargs):
            uploaded = pd.read_parquet(kwargs["path_or_fileobj"])
            calls.append({**kwargs, "uploaded": uploaded})
            return SimpleNamespace(oid="new-sha")

    monkeypatch.setattr(updater, "HfApi", FakeHfApi)
    df = pd.DataFrame({"NameStrip": ["Folasade"], "Meaning": ["new"]})

    commit = updater.push_dataset_update(
        df,
        token="token",
        commit_message="Update Folasade",
        parent_revision="base-sha",
    )

    assert commit.oid == "new-sha"
    assert calls[0]["repo_id"] == updater.DATASET_REPO
    assert calls[0]["path_in_repo"] == updater.PARQUET
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["commit_message"] == "Update Folasade"
    assert calls[0]["parent_commit"] == "base-sha"
    pd.testing.assert_frame_equal(calls[0]["uploaded"], df)
