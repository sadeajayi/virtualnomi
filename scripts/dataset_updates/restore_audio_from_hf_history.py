#!/usr/bin/env python3
"""
Restore missing Audio Pronunciation bytes from an older nomi-names HF revision.

Used for Ikeme and Ogechukwukanma (Daphne Chiamaka, Nov 2025) after Feb 2026
dataset restore re-added rows without embedded audio.
"""

import os
from typing import Dict, Optional, Tuple

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download

try:
    from huggingface_hub import get_token
except ImportError:  # huggingface_hub<0.20 compatibility
    from huggingface_hub import HfFolder

    get_token = HfFolder.get_token

HF_TOKEN = os.getenv("HF_TOKEN") or get_token()
REPO = "nomi-stories/nomi-names"
PARQUET = "data/train-00000-of-00001.parquet"
PRONUNCIATION_BY = "Daphne Chiamaka"

# (NameStrip, Language) -> HF commit revision with audio bytes
RESTORE_FROM: Dict[Tuple[str, str], str] = {
    ("Ikeme", "Igbo"): "ed074f7e",  # 2025-11-23 Approve Igbo_Ikeme_65bb61c1.wav
    ("Ogechukwukanma", "Igbo"): "2485ecb1",  # 2025-11-23 Approve Igbo_Ogechukwukanma
}


def _audio_bytes(df: pd.DataFrame, name_strip: str, language: str) -> Optional[bytes]:
    rows = df[(df["NameStrip"] == name_strip) & (df["Language"] == language)]
    if rows.empty:
        return None
    audio = rows.iloc[0]["Audio Pronunciation"]
    if isinstance(audio, dict):
        return audio.get("bytes") or None
    return None


def main():
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN not set")

    print("Downloading current parquet...")
    current_path = hf_hub_download(
        repo_id=REPO, repo_type="dataset", filename=PARQUET, token=HF_TOKEN
    )
    df = pd.read_parquet(current_path)
    df["NameStrip"] = df["NameStrip"].astype(str).str.strip()
    df["Language"] = df["Language"].astype(str).str.strip()
    if "pronunciation_by" not in df.columns:
        df["pronunciation_by"] = ""

    updated = 0
    for (name_strip, language), revision in RESTORE_FROM.items():
        print(f"\nFetching {name_strip} audio from revision {revision[:8]}...")
        old_path = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=PARQUET,
            token=HF_TOKEN,
            revision=revision,
        )
        old_df = pd.read_parquet(old_path, columns=["NameStrip", "Language", "Audio Pronunciation"])
        old_df["NameStrip"] = old_df["NameStrip"].astype(str).str.strip()
        old_df["Language"] = old_df["Language"].astype(str).str.strip()
        audio_bytes = _audio_bytes(old_df, name_strip, language)
        if not audio_bytes:
            print(f"  ⚠️  No audio bytes in revision for {name_strip}")
            continue

        mask = (df["NameStrip"] == name_strip) & (df["Language"] == language)
        if not mask.any():
            print(f"  ⚠️  {name_strip} not in current dataset")
            continue

        df.loc[mask, "Audio Pronunciation"] = [{"bytes": audio_bytes} for _ in range(mask.sum())]
        df.loc[mask, "pronunciation_by"] = PRONUNCIATION_BY
        updated += int(mask.sum())
        print(f"  ✅ Restored {len(audio_bytes)} bytes for {name_strip}")

    if updated == 0:
        print("Nothing to push.")
        return

    print(f"\nPushing {updated} row(s) to Hugging Face...")
    Dataset.from_pandas(df).push_to_hub(
        REPO,
        token=HF_TOKEN,
        commit_message=(
            "Restore Ikeme and Ogechukwukanma pronunciations from Nov 2025 HF history "
            "(Daphne Chiamaka)"
        ),
    )
    print("✅ Done.")


if __name__ == "__main__":
    main()
