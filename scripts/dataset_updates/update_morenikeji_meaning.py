#!/usr/bin/env python3
"""Update Morenikeji meaning in the nomi-names Hugging Face dataset."""

import os
from huggingface_hub import HfFolder

try:
    from .hf_dataset_uploads import download_dataset_parquet, push_dataframe_to_hub
except ImportError:
    from hf_dataset_uploads import download_dataset_parquet, push_dataframe_to_hub

HF_TOKEN = os.getenv("HF_TOKEN") or HfFolder.get_token()
DATASET_REPO = "nomi-stories/nomi-names"
PARQUET = "data/train-00000-of-00001.parquet"

NAME_STRIP = "Morenikeji"
LANGUAGE = "Yoruba"
NEW_MEANING = "I have found a companion."

if not HF_TOKEN:
    raise SystemExit("HF_TOKEN not set. Run: export HF_TOKEN=... or huggingface-cli login")


def main():
    print("Downloading current dataset...")
    loaded = download_dataset_parquet(DATASET_REPO, PARQUET, token=HF_TOKEN)
    df = loaded.frame
    df["NameStrip"] = df["NameStrip"].astype(str).str.strip()
    df["Language"] = df["Language"].astype(str).str.strip()

    mask = (df["NameStrip"] == NAME_STRIP) & (df["Language"] == LANGUAGE)
    if not mask.any():
        raise SystemExit(f"Not found: {NAME_STRIP} ({LANGUAGE})")

    old = df.loc[mask, "Meaning"].iloc[0]
    df.loc[mask, "Meaning"] = NEW_MEANING
    print(f'✅ Updated {NAME_STRIP}: "{old}" → "{NEW_MEANING}"')

    print("Pushing to Hugging Face...")
    push_dataframe_to_hub(
        df,
        DATASET_REPO,
        PARQUET,
        token=HF_TOKEN,
        commit_message="Update Morenikeji meaning: I have found a companion.",
        source_revision=loaded.revision,
    )
    print("✅ Done.")


if __name__ == "__main__":
    main()
