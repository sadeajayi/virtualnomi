#!/usr/bin/env python3
"""Update Folasade / Folashade Meaning on nomi-stories/nomi-names.

Meaning-only wording update. Preserves Name, Attribution, audio, phonetics,
and validation fields. Writes a before/after report under data/dataset_updates/.

Usage (from repo root):
  python scripts/dataset_updates/update_folasade_meaning.py --dry-run
  python scripts/dataset_updates/update_folasade_meaning.py --push
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, HfFolder, hf_hub_download

DATASET_REPO = "nomi-stories/nomi-names"
PARQUET = "data/train-00000-of-00001.parquet"
LANGUAGE = "Yoruba"
NEW_MEANING = "Wear wealth/honour like a crown"

# Orthographic variants of the same name (ṣ vs sh); update both if present.
NAME_STRIPS = ("Folasade", "Folashade")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "data" / "dataset_updates" / "folasade_meaning_update_report.json"

# Fields that must not change during this update.
PRESERVE_FIELDS = (
    "Name",
    "NameStrip",
    "Language",
    "Attribution",
    "Phonetic spelling",
    "Validation_Status",
    "validated_by",
    "pronunciation_by",
    "Additional meaning",
    "cultural_context",
    "themes",
    "transformation_status",
    "source_notes",
)


def audio_bytes_len(val: Any) -> int:
    if isinstance(val, dict):
        b = val.get("bytes")
        return len(b) if b is not None else 0
    if isinstance(val, (bytes, bytearray)):
        return len(val)
    return 0


def row_snapshot(row: pd.Series, columns: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in columns:
        if col not in row.index:
            continue
        val = row[col]
        if col == "Audio Pronunciation":
            out[col] = {"bytes": audio_bytes_len(val)}
            continue
        if val is None:
            out[col] = None
            continue
        if isinstance(val, (list, tuple)):
            out[col] = list(val)
            continue
        if hasattr(val, "tolist") and not isinstance(val, (str, bytes, bytearray)):
            try:
                out[col] = val.tolist()
                continue
            except (TypeError, ValueError):
                pass
        if isinstance(val, (bytes, bytearray)):
            out[col] = f"<{len(val)} bytes>"
            continue
        try:
            if pd.api.types.is_scalar(val) and pd.isna(val):
                out[col] = None
                continue
        except (TypeError, ValueError):
            pass
        out[col] = str(val)
    return out


def load_df(token: Optional[str]) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=PARQUET,
        token=token,
    )
    return pd.read_parquet(path)


def apply_updates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["NameStrip"] = df["NameStrip"].astype(str).str.strip()
    df["Language"] = df["Language"].astype(str).str.strip()

    report: Dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "new_meaning": NEW_MEANING,
        "rows": [],
        "updated_count": 0,
        "already_current_count": 0,
        "missing": [],
    }

    for name_strip in NAME_STRIPS:
        mask = (df["NameStrip"] == name_strip) & (df["Language"] == LANGUAGE)
        count = int(mask.sum())
        if count == 0:
            report["missing"].append({"NameStrip": name_strip, "Language": LANGUAGE})
            print(f"⏭️  Not found: {name_strip} ({LANGUAGE})")
            continue
        if count != 1:
            raise SystemExit(
                f"Expected exactly one row for {name_strip} ({LANGUAGE}); found {count}"
            )

        idx = df.index[mask][0]
        before = row_snapshot(df.loc[idx], list(df.columns))
        old_meaning = str(df.at[idx, "Meaning"] or "").strip()
        audio_before = audio_bytes_len(df.at[idx, "Audio Pronunciation"])

        entry: Dict[str, Any] = {
            "NameStrip": name_strip,
            "Language": LANGUAGE,
            "Name": before.get("Name"),
            "before": before,
            "meaning_changed": False,
        }

        if old_meaning == NEW_MEANING:
            report["already_current_count"] += 1
            print(f"⏭️  {name_strip}: already current")
        else:
            df.at[idx, "Meaning"] = NEW_MEANING
            report["updated_count"] += 1
            entry["meaning_changed"] = True
            print(f'✅ {name_strip}: "{old_meaning}" → "{NEW_MEANING}"')

        after = row_snapshot(df.loc[idx], list(df.columns))
        audio_after = audio_bytes_len(df.at[idx, "Audio Pronunciation"])
        if audio_before != audio_after:
            raise SystemExit(
                f"{name_strip} audio bytes changed ({audio_before} -> {audio_after})"
            )

        for field in PRESERVE_FIELDS:
            if field not in before:
                continue
            if before.get(field) != after.get(field):
                raise SystemExit(
                    f"{name_strip}: preserved field {field!r} changed "
                    f"({before.get(field)!r} -> {after.get(field)!r})"
                )

        entry["after"] = after
        report["rows"].append(entry)

    if report["missing"]:
        raise SystemExit(f"Missing expected rows: {report['missing']}")

    return df, report


def write_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote report: {REPORT_PATH}")


def run(*, dry_run: bool, push: bool) -> int:
    if dry_run and push:
        raise SystemExit("Use either --dry-run or --push, not both.")
    if not dry_run and not push:
        raise SystemExit("Pass --dry-run (preview) or --push (update HF).")

    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        raise SystemExit("HF_TOKEN not set. Run: export HF_TOKEN=... or huggingface-cli login")

    print("Downloading current dataset from Hugging Face...")
    df = load_df(token)
    print(f"Loaded {len(df)} rows")

    df_out, report = apply_updates(df)
    write_report(report)

    if report["updated_count"] == 0:
        print("No meaning changes needed.")
        return 0

    if dry_run:
        print("Dry run: no push to Hugging Face.")
        return 0

    commit_message = (
        "Update Folasade/Folashade meaning: Wear wealth/honour like a crown"
    )
    print(f"Pushing to Hugging Face: {commit_message}")
    Dataset.from_pandas(df_out, preserve_index=False).push_to_hub(
        DATASET_REPO,
        token=token,
        commit_message=commit_message,
    )

    api = HfApi(token=token)
    info = api.dataset_info(DATASET_REPO)
    report["hf_revision"] = info.sha
    report["hf_commit_message"] = commit_message
    write_report(report)
    print(f"✅ HF push complete. revision={info.sha}")
    print(
        "Note: nomi-name-search-api caches parquet in-process; "
        "Render may need a restart to pick up the new meaning."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview + write report without pushing",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Apply Meaning updates and push to Hugging Face",
    )
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run, push=args.push))


if __name__ == "__main__":
    main()
