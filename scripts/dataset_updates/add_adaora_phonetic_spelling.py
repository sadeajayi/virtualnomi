#!/usr/bin/env python3
"""Add phonetic spelling for Adaora (Igbo) in nomi-names."""

import os

from huggingface_hub import HfFolder

try:
    from .hf_dataset_uploads import download_dataset_parquet, push_dataframe_to_hub
except ImportError:
    from hf_dataset_uploads import download_dataset_parquet, push_dataframe_to_hub

HF_TOKEN = os.getenv("HF_TOKEN") or HfFolder.get_token()
DATASET_REPO = "nomi-stories/nomi-names"
PARQUET = "data/train-00000-of-00001.parquet"
NAME_STRIP = "Adaora"
LANGUAGE = "Igbo"
PHONETIC_SPELLING = "ah-daw-rah"


def main() -> None:
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN not set")

    loaded = download_dataset_parquet(DATASET_REPO, PARQUET, token=HF_TOKEN)
    df = loaded.frame

    if "Phonetic spelling" not in df.columns:
        df["Phonetic spelling"] = ""

    mask = (
        df["NameStrip"].astype(str).str.strip().str.lower() == NAME_STRIP.lower()
    ) & (df["Language"].astype(str).str.strip() == LANGUAGE)

    if not mask.any():
        raise SystemExit(f"Could not find {NAME_STRIP} ({LANGUAGE})")

    current = str(df.loc[mask, "Phonetic spelling"].iloc[0] or "").strip()
    if current == PHONETIC_SPELLING:
        print(f"Already set: {PHONETIC_SPELLING}")
        return

    df.loc[mask, "Phonetic spelling"] = PHONETIC_SPELLING
    push_dataframe_to_hub(
        df,
        DATASET_REPO,
        PARQUET,
        token=HF_TOKEN,
        commit_message="Add phonetic spelling for Adaora (Igbo): ah-daw-rah",
        source_revision=loaded.revision,
    )
    print(f"Updated {NAME_STRIP} → {PHONETIC_SPELLING}")


if __name__ == "__main__":
    main()
