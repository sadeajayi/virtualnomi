"""Helpers for safely replacing the canonical Hugging Face dataset parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class LoadedParquet:
    """A dataset parquet loaded from a specific Hugging Face snapshot."""

    frame: pd.DataFrame
    path: Path
    revision: str


def _require_hf_deps() -> tuple[Any, Any, Any, Any]:
    from datasets import Dataset
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    return Dataset, CommitOperationAdd, HfApi, hf_hub_download


def snapshot_revision_from_hf_path(path: str | Path) -> str:
    """Extract the immutable HF snapshot revision from an hf_hub_download path."""

    parts = Path(path).parts
    try:
        snapshot_idx = parts.index("snapshots")
    except ValueError as exc:
        raise ValueError(f"Cannot find Hugging Face snapshot revision in path: {path}") from exc

    try:
        revision = parts[snapshot_idx + 1]
    except IndexError as exc:
        raise ValueError(f"Cannot find Hugging Face snapshot revision in path: {path}") from exc

    if not revision:
        raise ValueError(f"Cannot find Hugging Face snapshot revision in path: {path}")
    return revision


def download_dataset_parquet(
    repo_id: str,
    filename: str,
    *,
    token: Optional[str],
) -> LoadedParquet:
    """Download and read a dataset parquet, preserving its source revision."""

    import pandas as pd

    _dataset, _operation_add, _api, hf_hub_download = _require_hf_deps()
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            token=token,
        )
    )
    return LoadedParquet(
        frame=pd.read_parquet(path),
        path=path,
        revision=snapshot_revision_from_hf_path(path),
    )


def push_dataframe_to_hub(
    frame: pd.DataFrame,
    repo_id: str,
    filename: str,
    *,
    token: str,
    commit_message: str,
    source_revision: str,
    preserve_index: bool = False,
) -> Any:
    """Replace one parquet file only if the remote dataset still matches the source."""

    if not source_revision or source_revision == "local":
        raise ValueError("source_revision must be an immutable Hugging Face commit SHA")

    Dataset, CommitOperationAdd, HfApi, _hf_hub_download = _require_hf_deps()
    with TemporaryDirectory() as tmp_dir:
        parquet_path = Path(tmp_dir) / Path(filename).name
        Dataset.from_pandas(frame, preserve_index=preserve_index).to_parquet(
            str(parquet_path)
        )
        operation = CommitOperationAdd(
            path_in_repo=filename,
            path_or_fileobj=str(parquet_path),
        )
        return HfApi(token=token).create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[operation],
            commit_message=commit_message,
            parent_commit=source_revision,
        )
