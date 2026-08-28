#!/usr/bin/env python3
"""
Embed a local audio file into nomi-names parquet (Audio Pronunciation column).

Converts non-WAV inputs (e.g. m4a, mp3) to WAV via ffmpeg so nomi-name-search-api
can serve bytes as audio/wav. Updates pronunciation_by when provided.

Usage (from repo root):
  export HF_TOKEN=...
  python scripts/dataset_updates/upload_audio_from_local_file.py \\
    --name-strip Folasade --language Yoruba \\
    --audio-file "/path/to/recording.m4a" \\
    --pronunciation-by "Folasade Ajayi" \\
    --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download

try:
    from huggingface_hub import get_token
except ImportError:  # huggingface_hub<0.20 compatibility
    from huggingface_hub import HfFolder

    get_token = HfFolder.get_token

HF_TOKEN = os.getenv("HF_TOKEN") or get_token()
MAIN_REPO = "nomi-stories/nomi-names"
PARQUET = "data/train-00000-of-00001.parquet"
WAV_SUFFIXES = {".wav"}


def _load_wav_bytes(audio_path: Path) -> bytes:
    suffix = audio_path.suffix.lower()
    if suffix in WAV_SUFFIXES:
        return audio_path.read_bytes()

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ffmpeg = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
    if ffmpeg.returncode != 0:
        raise SystemExit(
            "ffmpeg is required to convert non-WAV audio. Install ffmpeg or provide a .wav file."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "1",
                out_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")
        data = Path(out_path).read_bytes()
        if not data:
            raise RuntimeError("ffmpeg produced empty WAV output")
        if data[:4] != b"RIFF":
            raise RuntimeError("Converted file is not a valid WAV (missing RIFF header)")
        return data
    finally:
        Path(out_path).unlink(missing_ok=True)


def upload_audio(
    name_strip: str,
    language: str,
    audio_path: Path,
    pronunciation_by: Optional[str] = None,
    commit_message: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN not set. Run: export HF_TOKEN=... or huggingface-cli login")

    audio_bytes = _load_wav_bytes(audio_path)
    name_strip = name_strip.strip()
    language = language.strip()

    print(f"📥 Downloading current {MAIN_REPO} parquet...")
    parquet_path = hf_hub_download(
        repo_id=MAIN_REPO,
        repo_type="dataset",
        filename=PARQUET,
        token=HF_TOKEN,
    )
    df = pd.read_parquet(parquet_path)
    df["NameStrip"] = df["NameStrip"].astype(str).str.strip()
    df["Language"] = df["Language"].astype(str).str.strip()

    mask = (df["NameStrip"] == name_strip) & (df["Language"] == language)
    if not mask.any():
        raise SystemExit(f"No row found for NameStrip={name_strip!r} Language={language!r}")

    row_count = int(mask.sum())
    print(
        f"✅ Found {row_count} row(s) for {name_strip} ({language}); "
        f"embedding {len(audio_bytes)} WAV bytes from {audio_path.name}"
    )

    if pronunciation_by:
        if "pronunciation_by" not in df.columns:
            df["pronunciation_by"] = ""
        df.loc[mask, "pronunciation_by"] = pronunciation_by.strip()
        print(f"   pronunciation_by -> {pronunciation_by.strip()!r}")

    df.loc[mask, "Audio Pronunciation"] = [{"bytes": audio_bytes} for _ in range(row_count)]

    if dry_run:
        print("Dry run: no push to Hugging Face.")
        return len(audio_bytes)

    msg = commit_message or (
        f"Add pronunciation for {name_strip} ({language}) from local file"
    )
    if pronunciation_by:
        msg += f" ({pronunciation_by.strip()})"

    print(f"\n💾 Pushing to Hugging Face: {msg}")
    Dataset.from_pandas(df).push_to_hub(MAIN_REPO, token=HF_TOKEN, commit_message=msg)
    print("✅ Done.")
    return len(audio_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-strip", required=True, help="NameStrip key (e.g. Folasade)")
    parser.add_argument("--language", required=True, help="Language (e.g. Yoruba)")
    parser.add_argument("--audio-file", required=True, type=Path, help="Local audio file path")
    parser.add_argument(
        "--pronunciation-by",
        default="",
        help="Contributor for pronunciation_by column (optional)",
    )
    parser.add_argument("--commit-message", default="", help="Override HF commit message")
    parser.add_argument("--dry-run", action="store_true", help="Prepare update without pushing")
    args = parser.parse_args()

    size = upload_audio(
        name_strip=args.name_strip,
        language=args.language,
        audio_path=args.audio_file.expanduser().resolve(),
        pronunciation_by=args.pronunciation_by or None,
        commit_message=args.commit_message or None,
        dry_run=args.dry_run,
    )
    print(f"Audio byte size: {size}")


if __name__ == "__main__":
    main()
