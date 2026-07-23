from pathlib import Path

import pytest

from scripts.dataset_updates import hf_dataset_uploads


def test_extracts_snapshot_revision_from_hf_download_path(tmp_path):
    path = (
        tmp_path
        / "hub"
        / "datasets--nomi-stories--nomi-names"
        / "snapshots"
        / "abc123"
        / "data"
        / "train-00000-of-00001.parquet"
    )

    assert hf_dataset_uploads.snapshot_revision_from_hf_path(path) == "abc123"


def test_rejects_push_without_immutable_source_revision():
    with pytest.raises(ValueError, match="source_revision"):
        hf_dataset_uploads.push_dataframe_to_hub(
            {"NameStrip": ["Adaora"]},
            "nomi-stories/nomi-names",
            "data/train-00000-of-00001.parquet",
            token="hf_test",
            commit_message="test",
            source_revision="",
        )


def test_push_uses_parent_commit_guard(monkeypatch):
    captured = {}

    class FakeDataset:
        @classmethod
        def from_pandas(cls, frame, preserve_index=False):
            captured["frame"] = frame
            captured["preserve_index"] = preserve_index
            return cls()

        def to_parquet(self, path):
            Path(path).write_bytes(b"parquet bytes")

    class FakeOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class FakeApi:
        def __init__(self, *, token):
            captured["token"] = token

        def create_commit(
            self,
            *,
            repo_id,
            repo_type,
            operations,
            commit_message,
            parent_commit,
        ):
            captured["repo_id"] = repo_id
            captured["repo_type"] = repo_type
            captured["operations"] = operations
            captured["commit_message"] = commit_message
            captured["parent_commit"] = parent_commit
            captured["uploaded_bytes"] = Path(operations[0].path_or_fileobj).read_bytes()
            return {"oid": "new-sha"}

    monkeypatch.setattr(
        hf_dataset_uploads,
        "_require_hf_deps",
        lambda: (FakeDataset, FakeOperationAdd, FakeApi, None),
    )

    result = hf_dataset_uploads.push_dataframe_to_hub(
        {"NameStrip": ["Adaora"]},
        "nomi-stories/nomi-names",
        "data/train-00000-of-00001.parquet",
        token="hf_test",
        commit_message="Add phonetic spelling",
        source_revision="abc123",
    )

    assert result == {"oid": "new-sha"}
    assert captured["token"] == "hf_test"
    assert captured["repo_id"] == "nomi-stories/nomi-names"
    assert captured["repo_type"] == "dataset"
    assert captured["commit_message"] == "Add phonetic spelling"
    assert captured["parent_commit"] == "abc123"
    assert captured["operations"][0].path_in_repo == "data/train-00000-of-00001.parquet"
    assert captured["uploaded_bytes"] == b"parquet bytes"
    assert captured["preserve_index"] is False
