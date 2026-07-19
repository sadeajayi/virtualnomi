#!/usr/bin/env python3
"""Apply the approved Discover pronunciation batch to one canonical HF snapshot.

The batch combines three founder-provided Yoruba recordings with the four
approved Ibibio phonetic-only updates. It verifies exact row uniqueness,
preserves meaning/attribution and validation fields, converts audio to mono
44.1 kHz PCM WAV, and only uploads when --push is explicit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfFolder, hf_hub_download

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from update_ibibio_phonetics import (
    EXPECTED_UPDATES as IBIBIO_UPDATES,
    apply_updates as apply_ibibio_updates,
)
from upload_audio_from_local_file import _load_wav_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_REPO = "nomi-stories/nomi-names"
DATASET_FILE = "data/train-00000-of-00001.parquet"
DEFAULT_MANIFEST = (
    REPO_ROOT / "data" / "dataset_updates" / "founder_pronunciations_2026-07-19.json"
)
DEFAULT_REPORT = (
    REPO_ROOT / "data" / "dataset_updates" / "discover_pronunciation_deploy_report.json"
)


def validate_wav_audio(data: bytes) -> dict[str, Any]:
    if not data or data[:4] != b"RIFF":
        raise ValueError("Converted audio is empty or not RIFF WAV")
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        handle.write(data)
        handle.flush()
        with wave.open(handle.name, "rb") as audio:
            frames = audio.readframes(audio.getnframes())
            sample_width = audio.getsampwidth()
            frame_count = audio.getnframes()
            duration = frame_count / audio.getframerate()
            if sample_width != 2:
                raise ValueError(f"Expected 16-bit PCM WAV; got {sample_width * 8}-bit")
            samples = np.frombuffer(frames, dtype="<i2")
            peak = int(np.max(np.abs(samples.astype(np.int32)))) if samples.size else 0
            if frame_count <= 0 or duration <= 0 or peak <= 0:
                raise ValueError("Decoded WAV contains no audible samples")
            return {
                "bytes": len(data),
                "duration_seconds": round(duration, 3),
                "sample_rate": audio.getframerate(),
                "channels": audio.getnchannels(),
                "peak_amplitude": peak,
            }


def _exact_row_index(frame: pd.DataFrame, name_strip: str, language: str) -> Any:
    mask = (
        frame["NameStrip"].astype(str).str.strip() == name_strip
    ) & (frame["Language"].astype(str).str.strip() == language)
    count = int(mask.sum())
    if count != 1:
        raise ValueError(
            f"Expected exactly one canonical row for {name_strip} ({language}); found {count}"
        )
    return frame.index[mask][0]


def _append_canonical_row(
    frame: pd.DataFrame,
    *,
    name_strip: str,
    language: str,
    canonical: dict,
) -> tuple[pd.DataFrame, Any]:
    text_defaults = {
        "Name",
        "NameStrip",
        "Meaning",
        "Phonetic spelling",
        "Language",
        "Additional meaning",
        "Attribution",
        "Validation_Status",
        "validated_by",
        "pronunciation_by",
        "cultural_context",
        "transformation_status",
        "source_notes",
    }
    row = {
        column: ("" if column in text_defaults else None)
        for column in frame.columns
    }
    row.update(
        {
            "Name": str(canonical["name"]),
            "NameStrip": name_strip,
            "Meaning": str(canonical["meaning"]),
            "Language": language,
            "Attribution": str(canonical["attribution"]),
        }
    )
    updated = pd.concat([frame, pd.DataFrame([row], columns=frame.columns)], ignore_index=True)
    return updated, updated.index[-1]


def apply_recordings(
    frame: pd.DataFrame,
    manifest: dict,
    audio_dir: Path,
    audio_loader: Callable[[Path], bytes] = _load_wav_bytes,
) -> tuple[pd.DataFrame, list[dict]]:
    updated = frame.copy()
    reports = []
    recorder = str(manifest["internal_pronunciation_by"]).strip()

    for item in manifest["recordings"]:
        name_strip = str(item["name_strip"]).strip()
        language = str(item["language"]).strip()
        phonetic = str(item["phonetic_spelling"]).strip()
        source = audio_dir / str(item["audio_filename"])
        if not source.is_file():
            raise FileNotFoundError(f"Audio file not found: {source}")

        mask = (
            updated["NameStrip"].astype(str).str.strip() == name_strip
        ) & (updated["Language"].astype(str).str.strip() == language)
        match_count = int(mask.sum())
        created = False
        if match_count == 0 and item.get("new_canonical_record"):
            updated, index = _append_canonical_row(
                updated,
                name_strip=name_strip,
                language=language,
                canonical=item["new_canonical_record"],
            )
            created = True
        else:
            index = _exact_row_index(updated, name_strip, language)
        before_meaning = updated.at[index, "Meaning"]
        before_attribution = updated.at[index, "Attribution"]
        before_validation = {
            field: updated.at[index, field]
            for field in ("Validation_Status", "validated_by", "source_notes")
            if field in updated.columns
        }
        wav_bytes = audio_loader(source)
        audio_audit = validate_wav_audio(wav_bytes)

        updated.at[index, "Phonetic spelling"] = phonetic
        updated.at[index, "Audio Pronunciation"] = {"bytes": wav_bytes}
        updated.at[index, "pronunciation_by"] = recorder

        if updated.at[index, "Meaning"] != before_meaning:
            raise AssertionError("Meaning changed during pronunciation update")
        if updated.at[index, "Attribution"] != before_attribution:
            raise AssertionError("Attribution changed during pronunciation update")
        for field, before in before_validation.items():
            if updated.at[index, field] != before:
                raise AssertionError(f"{field} changed during pronunciation update")

        reports.append(
            {
                "name": str(updated.at[index, "Name"]).strip(),
                "name_strip": name_strip,
                "language": language,
                "phonetic_spelling": phonetic,
                "audio_source": source.name,
                "audio": audio_audit,
                "meaning_preserved": True,
                "attribution_preserved": True,
                "internal_pronunciation_by": recorder,
                "public_attribution_approved": False,
                "canonical_row_created": created,
            }
        )
    return updated, reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-parquet", type=Path)
    parser.add_argument("--push", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
    source = args.source_parquet or Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=DATASET_FILE,
            token=token,
        )
    )
    frame = pd.read_parquet(source)
    with_recordings, recording_report = apply_recordings(
        frame, manifest, args.audio_dir.expanduser().resolve()
    )
    updated, ibibio_report = apply_ibibio_updates(
        with_recordings, IBIBIO_UPDATES
    )

    report: dict[str, Any] = {
        "dataset_repo": DATASET_REPO,
        "dataset_file": DATASET_FILE,
        "source_revision": source.parents[1].name
        if not args.source_parquet
        else "local",
        "uploaded": False,
        "recordings": recording_report,
        "phonetic_only": ibibio_report,
        "excluded": manifest.get("excluded", []),
    }

    if args.push:
        if not token:
            raise SystemExit("HF_TOKEN is required for --push")
        commit = Dataset.from_pandas(updated, preserve_index=False).push_to_hub(
            DATASET_REPO,
            token=token,
            commit_message=(
                "Add founder Yoruba recordings and approved Ibibio phonetics"
            ),
        )
        report["uploaded"] = True
        report["hf_commit_url"] = getattr(commit, "commit_url", None)
        report["hf_revision"] = getattr(commit, "oid", None)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
