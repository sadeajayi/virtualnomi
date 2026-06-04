#!/usr/bin/env python3
"""
Add Igbo names from Semantic_Analysis_of_Igbo_Names.pdf (Onumajuru) to nomi-names.

Append-only: new rows only where (NameStrip, Language=Igbo) is not already present.
Never modifies existing Audio Pronunciation bytes.

Usage (from repo root):
  python scripts/dataset_updates/add_igbo_names_from_semantic_analysis.py --dry-run
  python scripts/dataset_updates/add_igbo_names_from_semantic_analysis.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfFolder, hf_hub_download
from unidecode import unidecode

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = REPO_ROOT / "Research papers" / "Semantic_Analysis_of_Igbo_Names.pdf"
CANDIDATES_CSV = REPO_ROOT / "data" / "igbo_semantic_analysis_candidates.csv"
DATASET_REPO = "nomi-stories/nomi-names"
PARQUET = "data/train-00000-of-00001.parquet"
LANGUAGE = "Igbo"
ATTRIBUTION = "Semantic_Analysis_of_Igbo_Names.pdf (Onumajuru)"
VALIDATION_STATUS = "Research source"

# Pages 5–12 (0-based indices 4–11)
PAGE_START = 4
PAGE_END = 12

SKIP_NAME_TOKENS = {
    "pragmatically",
    "semantically",
    "literally",
    "pro",
    "god",
    "child",
    "hand",
    "war",
    "beauty",
    "patience",
    "woman",
    "death",
    "road",
    "way",
    "does",
    "let",
    "name",
    "names",
    "igbo",
    "the",
    "and",
    "for",
    "etc",
    "note",
    "take",
    "starting",
    "similar",
}

ARROW_RE = re.compile(
    r"^([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]{2,30})\s*→\s*.+?\(([^)]+)\)",
    re.MULTILINE,
)
ARROW_INLINE_RE = re.compile(
    r"([A-Z][a-z]{2,28})\s*→\s*[^(\n]+\(([^)]+)\)"
)
SEM_MEANS_RE = re.compile(
    r"([A-Z][a-z]{3,28})\s+semantically means\s+['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
INJUNCTION_RE = re.compile(r"-\s*([A-Za-z]{4,28})\s*→\s*([^.\n]+)")
SENTENTIAL_RE = re.compile(
    r"([A-Z][a-z]{4,24})\s*\(cid:\d+\)([A-Za-z][^(\n]{5,70}?)\(cid:859\)"
)

# OCR-compressed sentential / interrogative names (pages 6–8) and related glosses
MANUAL_STRUCTURAL: List[Tuple[str, str]] = [
    ("Aghadinuno", "War is in the house"),
    ("Akanegbu", "Hand is killing"),
    ("Akobundu", "Wisdom is life"),
    ("Anagboso", "Land does not run"),
    ("Azubuike", "Support is strength"),
    ("Chibueze", "God is road/way"),
    ("Chibuzo", "God leads the way"),
    ("Chijioke", "God apportions gifts"),
    ("Mmagwulaku", "Beauty that exhausted wealth"),
    ("Ndidiamaka", "Patience is very good"),
    ("Nwaanyiaba", "A woman has come"),
    ("Nwabugo", "Child is glory"),
    ("Nwakaego", "Child surpasses money"),
    ("Nwanneka", "Relation surpasses all"),
    ("Obiageli", "She comes to enjoy"),
    ("Obianuju", "She comes in abundance"),
    ("Uzamaka", "The road is very good"),
    ("Afelechiaanya", "Does one see God?"),
    ("Amandaneze", "Does one know those to avoid?"),
    ("Onyebechi", "Who is God?"),
    ("Onyekachi", "Who is greater than God?"),
    ("Onyedikachukwu", "Who is like God?"),
    ("Onyekozulu", "Who is self-sufficient?"),
    ("Onyemaechi", "Who knows tomorrow?"),
    ("Onyenweewa", "Who owns the world?"),
    ("Afuluenuanya", "Does one see heaven?"),
    ("Agaegbu", "Does one need to kill?"),
    ("Amauche", "Does one know the mind of God?"),
    ("Obummneme", "Am I the creator?"),
    ("Oledimma", "How many are good?"),
    ("Olekamma", "How many are better?"),
    ("Emenanjo", "Don't do evil"),
    ("Ekwutosi", "Don't talk evil"),
    ("Ekwunife", "Don't say anything"),
    ("Chekwubechukwu", "Trust in God"),
    ("Nebeolisa", "Look up to God"),
    ("Ikeegbunam", "Let forces not kill me"),
    ("Onwuegbunam", "Let death not kill me"),
    ("Kaosisichukwu", "As it pleases God"),
    ("Nkemjika", "The one I hold is greater"),
    ("Mmesomachukwu", "The kindness of God"),
    ("Ogbugbunam", "Let charity not kill me"),
    ("Uzodimma", "The road is fine"),
    ("Onyeaghananwanne", "Let no one abandon his/her relation"),
    ("Omanukwue", "Let the one who knows speak"),
    ("Odirachukwumma", "Once it pleases God"),
    ("Anagbogu", "Let land intervene in the war"),
    ("Igwebuike", "There is strength in cordial relationship"),
    ("Ibebuike", "There is strength in cordial relationship"),
]

# Paper uses Ifeedi in structural list and Ifedi in semantic discussion
SEMANTIC_ALIASES: List[Tuple[str, str]] = [
    ("Ifedi", "Something exists"),
]

# Multi-line / split gloss fixes
MEANING_OVERRIDES: Dict[str, str] = {
    "Umekwulu": "Let untimely death cease",
    "Chekwube Chukwu": "Trust in God",
    "Chekwubechukwu": "Trust in God",
}


def ascii_name_strip(name: str) -> str:
    """ASCII NameStrip from extracted surface form."""
    return unidecode(name).strip()


def clean_meaning(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace("(cid:859)", "").replace("(cid:858)", "")
    text = re.sub(r"\(cid:\d+\)", "", text)
    if not text:
        return text
    if text[0].islower():
        text = text[0].upper() + text[1:]
    return text.rstrip(".,; ")


def _should_skip_name(name: str) -> bool:
    if not name or len(name) < 3:
        return True
    low = name.lower()
    if low in SKIP_NAME_TOKENS:
        return True
    if low.startswith("pragmat"):
        return True
    if not name[0].isupper():
        return True
    return False


def _add_candidate(
    store: Dict[str, Tuple[str, str, str]],
    name: str,
    meaning: str,
    source: str,
) -> None:
    name = re.sub(r"\s+", " ", name.strip())
    if _should_skip_name(name):
        return
    meaning = clean_meaning(MEANING_OVERRIDES.get(name, meaning))
    if not meaning or len(meaning) < 3:
        return
    key = ascii_name_strip(name)
    if not key:
        return
    if key not in store or len(meaning) > len(store[key][0]):
        store[key] = (meaning, name, source)


def extract_candidates_from_pdf(pdf_path: Path) -> Dict[str, Tuple[str, str, str]]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise SystemExit("pdfplumber is required. pip install pdfplumber") from exc

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    store: Dict[str, Tuple[str, str, str]] = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx in range(PAGE_START, min(PAGE_END, len(pdf.pages))):
            text = pdf.pages[page_idx].extract_text() or ""
            page_no = page_idx + 1

            for m in ARROW_RE.finditer(text):
                _add_candidate(store, m.group(1), m.group(2), f"p{page_no}_arrow")
            for m in ARROW_INLINE_RE.finditer(text):
                _add_candidate(store, m.group(1), m.group(2), f"p{page_no}_arrow_inline")

            for m in SEM_MEANS_RE.finditer(text):
                _add_candidate(store, m.group(1), m.group(2), f"p{page_no}_semantic")

            for m in SENTENTIAL_RE.finditer(text):
                eng = re.sub(r"\(cid:\d+\)", "", m.group(2))
                eng = re.sub(r"[^A-Za-z0-9 ,'\-?./]", " ", eng)
                eng = " ".join(eng.split())
                if eng:
                    _add_candidate(store, m.group(1), eng, f"p{page_no}_sentential")

            if page_no == 9:
                for m in INJUNCTION_RE.finditer(text):
                    gloss = m.group(2).strip().rstrip(".")
                    _add_candidate(store, m.group(1), gloss, "p9_injunction")

    for name, meaning in MANUAL_STRUCTURAL + SEMANTIC_ALIASES:
        _add_candidate(store, name, meaning, "manual_structural")

    return store


def count_audio_rows(df: pd.DataFrame) -> int:
    if "Audio Pronunciation" not in df.columns:
        return 0
    count = 0
    for val in df["Audio Pronunciation"]:
        if isinstance(val, dict) and val.get("bytes"):
            count += 1
    return count


def sample_audio_size(
    df: pd.DataFrame, name_strip: str, language: str
) -> Optional[int]:
    rows = df[
        (df["NameStrip"].astype(str).str.strip().str.lower() == name_strip.lower())
        & (df["Language"].astype(str).str.strip().str.lower() == language.lower())
    ]
    if rows.empty:
        return None
    audio = rows.iloc[0]["Audio Pronunciation"]
    if isinstance(audio, dict):
        b = audio.get("bytes")
        return len(b) if b else 0
    return 0


def load_dataset_parquet(token: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=PARQUET,
        token=token,
    )
    return pd.read_parquet(path)


def build_candidates_df(
    extracted: Dict[str, Tuple[str, str, str]],
    existing_strips: set,
) -> pd.DataFrame:
    rows = []
    for strip, (meaning, surface_name, source) in sorted(extracted.items()):
        rows.append(
            {
                "Name": surface_name,
                "NameStrip": strip,
                "Meaning": meaning,
                "Language": LANGUAGE,
                "source_page": source,
                "in_dataset_already": strip.lower() in existing_strips,
            }
        )
    return pd.DataFrame(rows)


def append_new_rows(df: pd.DataFrame, to_add: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col not in to_add.columns:
            to_add[col] = None

    new_rows = []
    for _, row in to_add.iterrows():
        new_row = {c: None for c in df.columns}
        new_row["Name"] = row["Name"]
        new_row["NameStrip"] = row["NameStrip"]
        new_row["Meaning"] = row["Meaning"]
        new_row["Language"] = LANGUAGE
        new_row["Attribution"] = ATTRIBUTION
        new_row["Validation_Status"] = VALIDATION_STATUS
        if "validated_by" in df.columns:
            new_row["validated_by"] = ""
        if "Phonetic spelling" in df.columns:
            new_row["Phonetic spelling"] = ""
        if "Audio Pronunciation" in df.columns:
            new_row["Audio Pronunciation"] = None
        if "Additional meaning" in df.columns:
            new_row["Additional meaning"] = None
        new_rows.append(new_row)

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def run(dry_run: bool) -> int:
    token = __import__("os").getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        raise SystemExit("HF_TOKEN not set. Run: export HF_TOKEN=... or huggingface-cli login")

    extracted = extract_candidates_from_pdf(PDF_PATH)
    print(f"Extracted {len(extracted)} candidate name(s) from PDF")

    df = load_dataset_parquet(token)
    df["NameStrip"] = df["NameStrip"].astype(str).str.strip()
    df["Language"] = df["Language"].astype(str).str.strip()

    igbo_strips = set(
        df[df["Language"].str.lower() == "igbo"]["NameStrip"].str.lower()
    )

    candidates_df = build_candidates_df(extracted, igbo_strips)
    CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(CANDIDATES_CSV, index=False)
    print(f"Wrote review CSV: {CANDIDATES_CSV}")

    to_add = candidates_df[~candidates_df["in_dataset_already"]].copy()
    skipped = int(candidates_df["in_dataset_already"].sum())

    rows_before = len(df)
    audio_before = count_audio_rows(df)
    samples_before = {
        "Folasade/Yoruba": sample_audio_size(df, "Folasade", "Yoruba"),
        "Ebube/Igbo": sample_audio_size(df, "Ebube", "Igbo"),
        "Ikeme/Igbo": sample_audio_size(df, "Ikeme", "Igbo"),
    }

    print(f"\nPre-update: rows={rows_before}, audio_rows={audio_before}")
    print(f"  Sample audio sizes: {samples_before}")
    print(f"  Would add: {len(to_add)}, skip (already in dataset): {skipped}")

    if to_add.empty:
        print("No new rows to add.")
        return 0

    if len(to_add) > 80:
        raise SystemExit(
            f"Refusing to add {len(to_add)} rows (>80). Check extraction."
        )

    if dry_run:
        print("\nDry run: no push to Hugging Face.")
        return len(to_add)

    df_out = append_new_rows(df, to_add)
    rows_after = len(df_out)
    audio_after = count_audio_rows(df_out)
    samples_after = {
        "Folasade/Yoruba": sample_audio_size(df_out, "Folasade", "Yoruba"),
        "Ebube/Igbo": sample_audio_size(df_out, "Ebube", "Igbo"),
        "Ikeme/Igbo": sample_audio_size(df_out, "Ikeme", "Igbo"),
    }

    if audio_after != audio_before:
        raise SystemExit(
            f"Audio row count changed {audio_before} -> {audio_after}. Aborting."
        )
    if rows_after != rows_before + len(to_add):
        raise SystemExit(
            f"Row count mismatch: before={rows_before} + add={len(to_add)} != after={rows_after}"
        )
    for key, before in samples_before.items():
        after = samples_after[key]
        if before != after:
            raise SystemExit(
                f"Sample audio size changed for {key}: {before} -> {after}"
            )

    names_added = ", ".join(to_add["NameStrip"].head(12).tolist())
    if len(to_add) > 12:
        names_added += f", ... (+{len(to_add) - 12} more)"
    commit_message = (
        f"Add {len(to_add)} Igbo names from Semantic_Analysis_of_Igbo_Names.pdf "
        f"(Onumajuru): {names_added}"
    )

    print(f"\nPost-check OK: rows={rows_after}, audio_rows={audio_after}")
    print(f"  Sample audio sizes unchanged: {samples_after}")
    print(f"\nPushing to Hugging Face: {commit_message[:200]}...")
    Dataset.from_pandas(df_out).push_to_hub(
        DATASET_REPO,
        token=token,
        commit_message=commit_message,
    )
    print("HF push complete.")
    return len(to_add)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and write CSV only; do not push to Hugging Face",
    )
    args = parser.parse_args()
    added = run(dry_run=args.dry_run)
    print(f"\nSummary: added={added} (0 if dry-run preview only)")


if __name__ == "__main__":
    main()
