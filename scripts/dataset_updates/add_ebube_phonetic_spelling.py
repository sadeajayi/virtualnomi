#!/usr/bin/env python3
"""
Add a phonetic spelling for Ebube so the demo has ten curated names with
meaning, phonetic text, and a published story.
"""

import os

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download

try:
    from huggingface_hub import get_token
except ImportError:  # huggingface_hub<0.20 compatibility
    from huggingface_hub import HfFolder

    get_token = HfFolder.get_token


HF_TOKEN = os.getenv("HF_TOKEN") or get_token()
DATASET_REPO = "nomi-stories/nomi-names"
NAME_STRIP = "Ebube"
LANGUAGE = "Igbo"
PHONETIC_SPELLING = "ay-boo-bay"


def main() -> None:
    if not HF_TOKEN:
        raise SystemExit(
            "HF_TOKEN not set. Run: export HF_TOKEN=... or huggingface-cli login"
        )

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
        raise SystemExit(f"Could not find {NAME_STRIP} ({LANGUAGE}) in {DATASET_REPO}")

    current = str(df.loc[mask, "Phonetic spelling"].iloc[0] or "").strip()
    if current == PHONETIC_SPELLING:
        print(f"{NAME_STRIP} already has phonetic spelling: {PHONETIC_SPELLING}")
        return

    df.loc[mask, "Phonetic spelling"] = PHONETIC_SPELLING

    updated_dataset = Dataset.from_pandas(df, preserve_index=False)
    updated_dataset.push_to_hub(
        DATASET_REPO,
        token=HF_TOKEN,
        commit_message="Add phonetic spelling for Ebube demo story",
    )

    print(f"Updated {NAME_STRIP} ({LANGUAGE}) phonetic spelling to {PHONETIC_SPELLING}")


if __name__ == "__main__":
    main()
