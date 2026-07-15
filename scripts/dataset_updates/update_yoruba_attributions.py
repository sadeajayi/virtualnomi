#!/usr/bin/env python3
"""
Set Yoruba meaning Attribution on nomi-stories/nomi-names.

Default (inherited catalog):
  Attribution = "YorubaNames.com"

Nomi-sourced exceptions (newer additions, not YorubaNames catalog lineage):
  Ajobi, Morohunkeji — added via add_missing_story_names.py
  Erukubami — newer HF-only row (not in AllNames0618 / local catalog);
              treated as Nomi-sourced; weaker documentary trail than the two above

Preserve existing academic Attribution (e.g. Olatunji et al.).

Does NOT push unless --push is passed. Prefer review with:
  python3 scripts/dataset_updates/update_yoruba_attributions.py --from-cache
then review data/dataset_updates/yoruba_attribution_report.json. To push, omit
--from-cache / --parquet so the script downloads the current HF dataset first.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import HfFolder, hf_hub_download

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_OUT_DIR = _REPO_ROOT / "data" / "dataset_updates"
DATASET_REPO = "nomi-stories/nomi-names"
PARQUET_NAME = "data/train-00000-of-00001.parquet"

YORUBANAMES_ATTRIBUTION = "YorubaNames.com"
NOMI_ATTRIBUTION = "Nomi"

# Exact NameStrip values as stored on HF (Unicode-sensitive).
NOMI_SOURCED_NAMESTRIPS = {
    "Ajobi",
    "Morohunkeji",
    "Erukubami",
}

# Academic / non-YorubaNames attributions we keep as-is (substring match, casefold).
PRESERVE_ATTRIBUTION_SUBSTRINGS = (
    "olatunji",
    "journal of pan african",
)


def _is_empty(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    s = str(val).strip()
    return not s or s.lower() in {"nan", "none", "null"}


def _should_preserve(attribution: str) -> bool:
    low = attribution.casefold()
    if "yorubanames" in low:
        return False  # normalize spelling below
    return any(s in low for s in PRESERVE_ATTRIBUTION_SUBSTRINGS)


def _normalize_existing(attribution: str) -> str:
    """Map known YorubaNames spellings to the canonical credit string."""
    if "yorubanames" in attribution.casefold():
        return YORUBANAMES_ATTRIBUTION
    return attribution.strip()


def _validate_push_source(push: bool, from_cache: bool, parquet_path: str | None) -> None:
    if push and (from_cache or parquet_path):
        raise SystemExit(
            "--push must load the current Hugging Face dataset directly. "
            "Omit --from-cache / --parquet for push runs; use those flags only "
            "for local report review."
        )


def _changed_attribution_count(report: dict) -> int:
    counts = report["counts"]
    return int(counts["set_to_yorubanames"]) + int(counts["set_to_nomi"])


def find_cached_parquet() -> Path | None:
    """Prefer HF hub cache snapshot pointed at by refs/main."""
    hub = Path.home() / ".cache/huggingface/hub/datasets--nomi-stories--nomi-names"
    main_ref = hub / "refs" / "main"
    if main_ref.is_file():
        snap = hub / "snapshots" / main_ref.read_text().strip()
        pq = snap / PARQUET_NAME
        if pq.is_file():
            return pq
    # Fallback: newest parquet under hub cache
    parquets = list(hub.rglob("train-00000-of-00001.parquet")) if hub.exists() else []
    if not parquets:
        return None
    return max(parquets, key=lambda p: p.stat().st_mtime)


def load_dataframe(
    from_cache: bool, parquet_path: str | None, *, force_download: bool = False
) -> pd.DataFrame:
    if parquet_path:
        path = Path(parquet_path)
        print(f"Loading parquet from --parquet: {path}")
        return pd.read_parquet(path)

    if from_cache:
        cached = find_cached_parquet()
        if not cached:
            raise FileNotFoundError(
                "No cached nomi-names parquet found. Pass --parquet or omit --from-cache "
                "to download (requires HF login / HF_TOKEN)."
            )
        print(f"Loading from HF cache: {cached}")
        return pd.read_parquet(cached)

    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        raise ValueError(
            "HF_TOKEN / huggingface login required unless --from-cache or --parquet is set."
        )
    print("Downloading parquet from Hugging Face...")
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=PARQUET_NAME,
        repo_type="dataset",
        token=token,
        force_download=force_download,
    )
    print(f"Loading: {path}")
    return pd.read_parquet(path)


def apply_attributions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if "Attribution" not in df.columns:
        df["Attribution"] = ""

    df["NameStrip"] = df["NameStrip"].astype(str).str.strip()
    df["Language"] = df["Language"].astype(str).str.strip()
    # Keep Attribution as string-ish for writes
    df["Attribution"] = df["Attribution"].apply(
        lambda v: "" if _is_empty(v) else str(v).strip()
    )

    yoruba_mask = df["Language"] == "Yoruba"
    yoruba = df[yoruba_mask]
    report = {
        "dataset_total": int(len(df)),
        "yoruba_total": int(yoruba_mask.sum()),
        "set_to_yorubanames": [],
        "set_to_nomi": [],
        "already_nomi": [],
        "preserved": [],
        "already_yorubanames": [],
        "skipped_non_empty_other": [],
    }

    for idx in yoruba.index:
        name_strip = df.at[idx, "NameStrip"]
        current = df.at[idx, "Attribution"]
        current = "" if _is_empty(current) else str(current).strip()

        if name_strip in NOMI_SOURCED_NAMESTRIPS:
            if current != NOMI_ATTRIBUTION:
                df.at[idx, "Attribution"] = NOMI_ATTRIBUTION
                report["set_to_nomi"].append(
                    {"NameStrip": name_strip, "previous": current}
                )
            else:
                report["already_nomi"].append(name_strip)
            continue

        if current and _should_preserve(current):
            report["preserved"].append(
                {"NameStrip": name_strip, "Attribution": current}
            )
            continue

        if current and "yorubanames" in current.casefold():
            normalized = _normalize_existing(current)
            if normalized != current:
                df.at[idx, "Attribution"] = normalized
                report["set_to_yorubanames"].append(
                    {
                        "NameStrip": name_strip,
                        "previous": current,
                        "note": "normalized spelling",
                    }
                )
            else:
                report["already_yorubanames"].append(name_strip)
            continue

        if current and current != YORUBANAMES_ATTRIBUTION:
            # Unexpected non-empty attribution — do not overwrite blindly
            report["skipped_non_empty_other"].append(
                {"NameStrip": name_strip, "Attribution": current}
            )
            continue

        if current != YORUBANAMES_ATTRIBUTION:
            df.at[idx, "Attribution"] = YORUBANAMES_ATTRIBUTION
            report["set_to_yorubanames"].append(
                {"NameStrip": name_strip, "previous": current}
            )
        else:
            report["already_yorubanames"].append(name_strip)

    report["counts"] = {
        "set_to_yorubanames": len(report["set_to_yorubanames"]),
        "set_to_nomi": len(report["set_to_nomi"]),
        "already_nomi": len(report["already_nomi"]),
        "preserved": len(report["preserved"]),
        "already_yorubanames": len(report["already_yorubanames"]),
        "skipped_non_empty_other": len(report["skipped_non_empty_other"]),
        "nomi_sourced_namestrips": sorted(NOMI_SOURCED_NAMESTRIPS),
        "canonical_yorubanames_string": YORUBANAMES_ATTRIBUTION,
        "canonical_nomi_string": NOMI_ATTRIBUTION,
    }
    return df, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Load parquet from local Hugging Face hub cache (no download).",
    )
    parser.add_argument(
        "--parquet",
        default=None,
        help="Path to a local nomi-names parquet file.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push updated dataset to Hugging Face (default: local write only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute report only; do not write parquet or push.",
    )
    args = parser.parse_args()

    _validate_push_source(args.push, args.from_cache, args.parquet)

    df = load_dataframe(args.from_cache, args.parquet, force_download=args.push)
    print(f"Dataset rows: {len(df)}")
    rows_before = len(df)

    df, report = apply_attributions(df)
    counts = report["counts"]
    rows_after = len(df)
    if rows_after != rows_before:
        raise SystemExit(
            f"Row count changed unexpectedly ({rows_before} -> {rows_after}). Aborting."
        )
    print("\nSummary:")
    print(f"  Yoruba rows: {report['yoruba_total']}")
    print(f"  → YorubaNames.com: {counts['set_to_yorubanames']}")
    print(f"  → Nomi: {counts['set_to_nomi']}")
    print(f"  Already Nomi: {counts['already_nomi']}")
    print(f"  Preserved (academic etc.): {counts['preserved']}")
    print(f"  Already YorubaNames.com: {counts['already_yorubanames']}")
    print(f"  Skipped other non-empty: {counts['skipped_non_empty_other']}")
    print(f"  Nomi-sourced left as Nomi: {counts['nomi_sourced_namestrips']}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _OUT_DIR / "yoruba_attribution_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote report: {report_path}")

    if args.dry_run:
        print("Dry run — no parquet write / push.")
        return

    out_parquet = _OUT_DIR / "nomi_names_with_yoruba_attributions.parquet"
    # Drop pandas index column artifact if present
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])
    df.to_parquet(out_parquet, index=False)
    print(f"Wrote local parquet: {out_parquet}")

    if args.push:
        if _changed_attribution_count(report) == 0:
            print("No attribution changes detected; skipping Hugging Face push.")
            return

        from datasets import Dataset

        token = os.getenv("HF_TOKEN") or HfFolder.get_token()
        if not token:
            raise ValueError("HF_TOKEN required for --push")
        print("Pushing to Hugging Face...")
        Dataset.from_pandas(df, preserve_index=False).push_to_hub(
            DATASET_REPO,
            token=token,
            commit_message=(
                f"Set Yoruba Attribution to {YORUBANAMES_ATTRIBUTION} "
                f"(Nomi exceptions: {', '.join(sorted(NOMI_SOURCED_NAMESTRIPS))})"
            ),
        )
        print("✅ Pushed to nomi-stories/nomi-names")
    else:
        print(
            "\nNot pushed. When ready:\n"
            "  HF_TOKEN=... python3 scripts/dataset_updates/update_yoruba_attributions.py "
            "--from-cache --push\n"
            "or upload the local parquet to the dataset."
        )


if __name__ == "__main__":
    main()
