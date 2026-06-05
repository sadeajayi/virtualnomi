#!/usr/bin/env python3
"""
Enrich Ebube (Igbo) meaning and append Ebubechukwu if missing.

- Updates Ebube Meaning only (preserves Audio Pronunciation bytes).
- Appends Ebubechukwu when (NameStrip, Language=Igbo) is not present.
- Does not set audio on new rows unless the name already exists with audio.

Usage (from repo root):
  python scripts/dataset_updates/update_ebube_ebubechukwu.py --dry-run
  python scripts/dataset_updates/update_ebube_ebubechukwu.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfFolder, hf_hub_download

DATASET_REPO = "nomi-stories/nomi-names"
PARQUET = "data/train-00000-of-00001.parquet"
LANGUAGE = "Igbo"

EBUBE_NAME_STRIP = "Ebube"
EBUBE_NEW_MEANING = "Glory (short form of Ebubechukwu — God's glory)"

EBUBECHUKWU_ROW = {
    "Name": "Ebubechukwu",
    "NameStrip": "Ebubechukwu",
    "Meaning": "God's glory",
    "Language": LANGUAGE,
}


def sample_audio_size(
    df: pd.DataFrame, name_strip: str, language: str
) -> Optional[int]:
    rows = df[
        (df["NameStrip"].astype(str).str.strip().str.lower() == name_strip.lower())
        & (df["Language"].astype(str).str.strip().str.lower() == language.lower())
    ]
    if rows.empty:
        return None
    audio = rows.iloc[0]["Audio Pronunciation"]
    if isinstance(audio, dict):
        b = audio.get("bytes")
        return len(b) if b else 0
    return 0


def row_snapshot(row: pd.Series, columns: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in columns:
        val = row[col]
        if col == "Audio Pronunciation":
            if isinstance(val, dict):
                b = val.get("bytes")
                out[col] = {"bytes": len(b) if b is not None else 0}
            else:
                out[col] = val
        elif val is None:
            out[col] = None
        else:
            try:
                if pd.isna(val):
                    out[col] = None
                    continue
            except (TypeError, ValueError):
                pass
            out[col] = str(val)
    return out


def load_dataset(token: Optional[str]) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=PARQUET,
        token=token,
    )
    return pd.read_parquet(path)


def build_new_row(df: pd.DataFrame, name_data: Dict[str, str]) -> Dict[str, Any]:
    new_row: Dict[str, Any] = {c: None for c in df.columns}
    new_row["Name"] = name_data["Name"]
    new_row["NameStrip"] = name_data["NameStrip"]
    new_row["Meaning"] = name_data["Meaning"]
    new_row["Language"] = name_data["Language"]
    if "Phonetic spelling" in df.columns:
        new_row["Phonetic spelling"] = ""
    if "Audio Pronunciation" in df.columns:
        new_row["Audio Pronunciation"] = None
    if "Additional meaning" in df.columns:
        new_row["Additional meaning"] = None
    if "Attribution" in df.columns:
        new_row["Attribution"] = ""
    if "Validation_Status" in df.columns:
        new_row["Validation_Status"] = ""
    if "validated_by" in df.columns:
        new_row["validated_by"] = ""
    if "pronunciation_by" in df.columns:
        new_row["pronunciation_by"] = ""
    if "cultural_context" in df.columns:
        new_row["cultural_context"] = ""
    if "themes" in df.columns:
        new_row["themes"] = []
    if "transformation_status" in df.columns:
        new_row["transformation_status"] = ""
    if "source_notes" in df.columns:
        new_row["source_notes"] = ""
    return new_row


def apply_updates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["NameStrip"] = df["NameStrip"].astype(str).str.strip()
    df["Language"] = df["Language"].astype(str).str.strip()

    report = {
        "ebube_found": False,
        "ebube_before": None,
        "ebube_after": None,
        "ebube_meaning_changed": False,
        "ebubechukwu_added": False,
        "ebubechukwu_already_present": False,
    }

    ebube_mask = (df["NameStrip"] == EBUBE_NAME_STRIP) & (df["Language"] == LANGUAGE)
    if not ebube_mask.any():
        raise SystemExit(f"Not found: {EBUBE_NAME_STRIP} ({LANGUAGE})")

    report["ebube_found"] = True
    ebube_idx = df.index[ebube_mask][0]
    report["ebube_before"] = row_snapshot(df.loc[ebube_idx], list(df.columns))

    old_meaning = str(df.loc[ebube_idx, "Meaning"] or "").strip()
    if old_meaning != EBUBE_NEW_MEANING.strip():
        df.loc[ebube_idx, "Meaning"] = EBUBE_NEW_MEANING
        report["ebube_meaning_changed"] = True
        print(
            f'✅ Ebube meaning: "{old_meaning}" → "{EBUBE_NEW_MEANING}"'
        )
    else:
        print(f"⏭️  Ebube meaning already: {EBUBE_NEW_MEANING!r}")

    report["ebube_after"] = row_snapshot(df.loc[ebube_idx], list(df.columns))

    chukwu_mask = (
        df["NameStrip"].str.lower() == EBUBECHUKWU_ROW["NameStrip"].lower()
    ) & (df["Language"] == LANGUAGE)
    if chukwu_mask.any():
        report["ebubechukwu_already_present"] = True
        print(f"⏭️  {EBUBECHUKWU_ROW['NameStrip']} ({LANGUAGE}) already in dataset")
    else:
        new_row = build_new_row(df, EBUBECHUKWU_ROW)
        new_df = pd.DataFrame([new_row])
        for col in df.columns:
            if col not in new_df.columns:
                new_df[col] = None
        new_df = new_df[df.columns]
        df = pd.concat([df, new_df], ignore_index=True)
        report["ebubechukwu_added"] = True
        print(
            f"✅ Added {EBUBECHUKWU_ROW['NameStrip']} ({LANGUAGE}): "
            f"Meaning = {EBUBECHUKWU_ROW['Meaning']!r}"
        )

    return df, report


def run(dry_run: bool) -> int:
    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token and not dry_run:
        print("HF_TOKEN not set. Re-run with --dry-run to preview planned changes.")
        return 1

    if token:
        print("Downloading current dataset from Hugging Face...")
        df = load_dataset(token)
    else:
        print("No HF_TOKEN — cannot load live dataset without --dry-run preview only.")
        print("Set HF_TOKEN or huggingface-cli login, then re-run without --dry-run.")
        return 1

    rows_before = len(df)
    audio_before = sample_audio_size(df, EBUBE_NAME_STRIP, LANGUAGE)
    print(f"Pre-update: rows={rows_before}, Ebube audio bytes={audio_before}")

    df_out, report = apply_updates(df)

    audio_after = sample_audio_size(df_out, EBUBE_NAME_STRIP, LANGUAGE)
    rows_after = len(df_out)

    if audio_before != audio_after:
        raise SystemExit(
            f"Ebube audio changed ({audio_before} -> {audio_after}). Aborting."
        )

    print("\n--- Summary ---")
    print(json.dumps(report, indent=2, default=str))

    changed = report["ebube_meaning_changed"] or report["ebubechukwu_added"]
    if not changed:
        print("No changes needed.")
        return 0

    if dry_run:
        print("\nDry run: no push to Hugging Face.")
        return 0

    parts = []
    if report["ebube_meaning_changed"]:
        parts.append("enrich Ebube meaning")
    if report["ebubechukwu_added"]:
        parts.append("add Ebubechukwu (Igbo)")
    commit_message = "Update Igbo names: " + ", ".join(parts)

    print(f"\nPushing to Hugging Face: {commit_message}")
    Dataset.from_pandas(df_out, preserve_index=False).push_to_hub(
        DATASET_REPO,
        token=token,
        commit_message=commit_message,
    )
    print("✅ HF push complete.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without pushing to Hugging Face",
    )
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
