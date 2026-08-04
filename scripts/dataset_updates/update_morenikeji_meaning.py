#!/usr/bin/env python3
"""Update Morenikeji meaning in the nomi-names Hugging Face dataset."""

import os
import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download, HfFolder

from scripts.dataset_updates.safety import require_unique_canonical_row

HF_TOKEN = os.getenv("HF_TOKEN") or HfFolder.get_token()
DATASET_REPO = "nomi-stories/nomi-names"

NAME_STRIP = "Morenikeji"
LANGUAGE = "Yoruba"
NEW_MEANING = "I have found a companion."


def main():
    if not HF_TOKEN:
        raise SystemExit(
            "HF_TOKEN not set. Run: export HF_TOKEN=... or huggingface-cli login"
        )

    print("Downloading current dataset...")
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename="data/train-00000-of-00001.parquet",
        token=HF_TOKEN,
    )
    df = pd.read_parquet(path)
    df["NameStrip"] = df["NameStrip"].astype(str).str.strip()
    df["Language"] = df["Language"].astype(str).str.strip()

    index = require_unique_canonical_row(
        df,
        NAME_STRIP,
        LANGUAGE,
        case_insensitive_name=False,
        case_insensitive_language=False,
    )
    old = df.at[index, "Meaning"]
    df.at[index, "Meaning"] = NEW_MEANING
    print(f'✅ Updated {NAME_STRIP}: "{old}" → "{NEW_MEANING}"')

    print("Pushing to Hugging Face...")
    Dataset.from_pandas(df).push_to_hub(
        DATASET_REPO,
        token=HF_TOKEN,
        commit_message="Update Morenikeji meaning: I have found a companion.",
    )
    print("✅ Done.")


if __name__ == "__main__":
    main()
