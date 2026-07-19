#!/usr/bin/env python3
"""Audit and stage founder-provided Ibibio phonetic spellings without uploading.

The pending upload artifact is data/phonetic_updates.csv. By default this
script downloads the current canonical parquet, verifies exactly one matching
row per (NameStrip, Language), and prints a before/after report. It never
pushes to Hugging Face.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT / "data" / "dataset_updates" / "ibibio_phonetics_2026-07-19.json"
)
DATASET_REPO = "nomi-stories/nomi-names"
DATASET_FILE = "data/train-00000-of-00001.parquet"
LANGUAGE = "Ibibio"
PROVENANCE = "Founder-provided learner spellings, 2026-07-19"
EXPECTED_UPDATES: Dict[Tuple[str, str], str] = {
    ("Abasi", LANGUAGE): "aa-baa-see",
    ("Adiaha", LANGUAGE): "ah-dee-aa-ha",
    ("Ekaete", LANGUAGE): "eh-kai-tay",
    ("Eno", LANGUAGE): "eh-naw",
}


def load_manifest(path: Path) -> Dict[Tuple[str, str], str]:
    updates: Dict[Tuple[str, str], str] = {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("updates", []):
        key = (
            (row.get("name_strip") or "").strip(),
            (row.get("language") or "").strip(),
        )
        phonetic = (row.get("phonetic_spelling") or "").strip()
        if key in updates:
            raise ValueError(f"Duplicate update in manifest: {key}")
        updates[key] = phonetic

    if updates != EXPECTED_UPDATES:
        raise ValueError(
            f"Ibibio manifest does not match approved updates: {updates!r}"
        )
    return updates


def apply_updates(
    frame: pd.DataFrame,
    updates: Dict[Tuple[str, str], str],
) -> tuple[pd.DataFrame, list[dict]]:
    required = {"NameStrip", "Language", "Phonetic spelling"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    updated = frame.copy()
    report = []
    stripped_names = updated["NameStrip"].astype(str).str.strip()
    stripped_languages = updated["Language"].astype(str).str.strip()

    for (name_strip, language), phonetic in updates.items():
        mask = (stripped_names == name_strip) & (stripped_languages == language)
        match_count = int(mask.sum())
        if match_count != 1:
            raise ValueError(
                f"Expected exactly one row for {name_strip} ({language}); "
                f"found {match_count}"
            )

        index = updated.index[mask][0]
        current = updated.at[index, "Phonetic spelling"]
        before = "" if pd.isna(current) else str(current).strip()
        updated.at[index, "Phonetic spelling"] = phonetic
        report.append(
            {
                "Name": str(updated.at[index, "Name"]).strip()
                if "Name" in updated.columns
                else name_strip,
                "NameStrip": name_strip,
                "Language": language,
                "before": before,
                "after": phonetic,
                "changed": before != phonetic,
            }
        )

    return updated, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source-parquet",
        type=Path,
        help="Use a local canonical parquet instead of downloading the HF dataset.",
    )
    parser.add_argument(
        "--write-local-parquet",
        type=Path,
        help="Optionally write a full updated local parquet. Never uploads it.",
    )
    parser.add_argument("--report", type=Path, help="Write the JSON audit report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_parquet or Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=DATASET_FILE,
        )
    )
    updates = load_manifest(args.manifest)
    updated, rows = apply_updates(pd.read_parquet(source), updates)

    payload = {
        "dataset_repo": DATASET_REPO,
        "dataset_file": DATASET_FILE,
        "source": (
            {"kind": "local", "path": str(source)}
            if args.source_parquet
            else {"kind": "huggingface", "revision": source.parents[1].name}
        ),
        "manifest": str(args.manifest.relative_to(REPO_ROOT))
        if args.manifest.is_relative_to(REPO_ROOT)
        else str(args.manifest),
        "provenance": PROVENANCE,
        "uploaded": False,
        "rows": rows,
    }

    if args.write_local_parquet:
        args.write_local_parquet.parent.mkdir(parents=True, exist_ok=True)
        updated.to_parquet(args.write_local_parquet, index=False)
        payload["local_parquet"] = str(args.write_local_parquet)

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
