"""Published syllable guides (anchor + stress) for the name API.

Stage 5 lite: serve reviewed pipeline JSON to Discover without a new HF column.
Later files overwrite earlier ones; last record in a file wins.

- `phonetic-pipeline-demo/data/approved_phonetics.jsonl` (Stage 2 approvals)
- `nomi-name-search-api/data/syllable_guides.jsonl` (hand-published overlay)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_API_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _API_DIR.parent

GUIDE_PATHS = (
    _REPO_ROOT / "phonetic-pipeline-demo" / "data" / "approved_phonetics.jsonl",
    _API_DIR / "data" / "syllable_guides.jsonl",
)

_published: Optional[Dict[str, List[Dict[str, Any]]]] = None
_generated: Dict[str, List[Dict[str, Any]]] = {}


def practice_key(name_strip: str, language: str) -> str:
    family = ""
    stripped = (language or "").strip().lower()
    if stripped:
        family = stripped.split()[0].split("(")[0]
    return f"{(name_strip or '').strip().lower()}|{family}"


def phonetic_parts(phonetic: str) -> List[str]:
    return [part.strip() for part in (phonetic or "").split("-") if part.strip()]


def stressed_index_from_phonetic(phonetic: str) -> Optional[int]:
    parts = phonetic_parts(phonetic)
    caps = [
        i
        for i, part in enumerate(parts)
        if part.isupper() and any(ch.isalpha() for ch in part)
    ]
    if len(caps) == 1:
        return caps[0]
    return None


def normalize_syllable(beat: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(beat, dict):
        return None
    say = str(beat.get("say") or "").strip()
    if not say:
        return None
    out: Dict[str, Any] = {"say": say.lower()}
    script = str(beat.get("script") or "").strip()
    anchor = str(beat.get("anchor") or "").strip()
    if script:
        out["script"] = script
    if anchor:
        out["anchor"] = anchor
    stress = beat.get("stress")
    if isinstance(stress, bool):
        out["stress"] = stress
    elif isinstance(stress, str) and stress.strip().lower() in {"true", "false"}:
        out["stress"] = stress.strip().lower() == "true"
    return out


def normalize_syllables(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    syllables = [item for item in (normalize_syllable(beat) for beat in raw) if item]
    return syllables


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def _load_published() -> Dict[str, List[Dict[str, Any]]]:
    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for path in GUIDE_PATHS:
        for record in _read_jsonl(path):
            syllables = normalize_syllables(record.get("syllables"))
            if not syllables:
                continue
            name_strip = str(record.get("name_strip") or record.get("name") or "").strip()
            language = str(record.get("language") or "").strip()
            key = practice_key(name_strip, language)
            if not name_strip or "|" not in key or key.endswith("|"):
                continue
            lookup[key] = syllables
    return lookup


def published_lookup() -> Dict[str, List[Dict[str, Any]]]:
    global _published
    if _published is None:
        _published = _load_published()
    return _published


def get_published_syllables(name_strip: str, language: str) -> Optional[List[Dict[str, Any]]]:
    key = practice_key(name_strip, language)
    generated = _generated.get(key)
    if generated:
        return generated
    return published_lookup().get(key)


def remember_generated(name_strip: str, language: str, syllables: List[Dict[str, Any]]) -> None:
    key = practice_key(name_strip, language)
    if key in published_lookup():
        return
    _generated[key] = syllables


def reset_syllable_store() -> None:
    global _published, _generated
    _published = None
    _generated = {}
