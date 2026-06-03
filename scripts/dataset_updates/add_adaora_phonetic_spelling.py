#!/usr/bin/env python3
"""Add phonetic spelling for Adaora (Igbo) in nomi-names."""

import os

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfFolder, hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN") or HfFolder.get_token()
DATASET_REPO = "nomi-stories/nomi-names"
NAME_STRIP = "Adaora"
LANGUAGE = "Igbo"
PHONETIC_SPELLING = "ah-daw-rah"


def main() -> None:
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN not set")

    parquet_path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename="data/train-00000-of-00001.parquet",
        token=HF_TOKEN,
    )
    df = pd.read_parquet(parquet_path)

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
    Dataset.from_pandas(df, preserve_index=False).push_to_hub(
        DATASET_REPO,
        token=HF_TOKEN,
        commit_message="Add phonetic spelling for Adaora (Igbo): ah-daw-rah",
    )
    print(f"Updated {NAME_STRIP} → {PHONETIC_SPELLING}")


if __name__ == "__main__":
    main()
