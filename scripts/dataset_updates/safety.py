"""Shared safety checks for one-off dataset update scripts."""

from __future__ import annotations

import pandas as pd


def canonical_row_mask(
    frame: pd.DataFrame,
    name_strip: str,
    language: str,
    *,
    case_insensitive_name: bool = True,
    case_insensitive_language: bool = True,
) -> pd.Series:
    """Return the canonical-row mask for a normalized name/language pair."""
    required = {"NameStrip", "Language"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    names = frame["NameStrip"].astype(str).str.strip()
    languages = frame["Language"].astype(str).str.strip()
    expected_name = name_strip.strip()
    expected_language = language.strip()

    if case_insensitive_name:
        names = names.str.lower()
        expected_name = expected_name.lower()
    if case_insensitive_language:
        languages = languages.str.lower()
        expected_language = expected_language.lower()

    return (names == expected_name) & (languages == expected_language)


def require_unique_canonical_row(
    frame: pd.DataFrame,
    name_strip: str,
    language: str,
    *,
    case_insensitive_name: bool = True,
    case_insensitive_language: bool = True,
) -> int:
    """Return the matching row index, or fail before a broad dataset mutation."""
    mask = canonical_row_mask(
        frame,
        name_strip,
        language,
        case_insensitive_name=case_insensitive_name,
        case_insensitive_language=case_insensitive_language,
    )
    match_count = int(mask.sum())
    if match_count != 1:
        raise ValueError(
            f"Expected exactly one canonical row for {name_strip} ({language}); "
            f"found {match_count}"
        )
    return int(frame.index[mask][0])
