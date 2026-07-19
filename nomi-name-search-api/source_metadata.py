"""Structured metadata for research papers used by the RAG indexes."""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Dict, Iterable, List, Optional


# Exact filenames mirror rag/language_config.py. Empty entries intentionally use
# filename-derived display titles until publication metadata is verified.
SOURCE_METADATA_REGISTRY: Dict[str, Dict[str, str]] = {
    "Construction_Morphology_in_Yoruba_names_Schemas_an.pdf": {
        "title": "Construction Morphology in Yoruba Names: Schemas and Processes",
        "author": "Taiwo O. Ehineni",
        "year": "2022",
    },
    "Yoruba_Traditional_Names_and_the_Transmi.pdf": {
        "title": "Yoruba Traditional Names and the Transmission of Cultural Knowledge",
        "author": "F. Niyi Akinnaso",
        "year": "1983",
    },
    "Yoruba_Praise_Names.pdf": {},
    "Endangerment_of_Yoruba_Individual_Names.pdf": {},
    "Yoruba_Names_Communicative_.pdf": {},
    "Yoruba Naming.pdf": {
        "title": "Yoruba Names as a Reflection of People’s Cultural Heritage",
        "author": "Noah Yusuf",
        "year": "2014",
    },
    "YORUBA-CRITICAL-ANALYSIS-OF-PERSONAL-NAME.pdf": {},
    "Yoruba_Construction_schemas.pdf": {},
    "Yoruba_ethnopragmatics_personal_names.pdf": {},
    "Yoruba_Ifa_Personal_Names.pdf": {
        "title": (
            "Names as Message Vectors in Communication: Oduological Analysis "
            "of Traditional Yoruba Personal Names from Ifa"
        ),
        "author": "Agboola Odesanya, Oloruntola Sunday, and Kunle Akinjogbin",
        "year": "2017",
    },
    "Yoruba_Ilaje_names.pdf": {},
    "Yoruba_Names_Gender_Markings.pdf": {},
    "Yoruba-names-Modupe-Oduyoye.pdf": {},
    "Naming_in_Igbo_Land_A_Linguistic_and_Cul.pdf": {},
    "Religious_Significance_Igbo_Names.pdf": {},
    "Semantic_Analysis_of_Igbo_Names.pdf": {
        "title": "A Semantic & Pragmatic Analysis of Igbo Names",
        "author": "V.C. Onumajuru",
    },
    "The_Sociolinguistics_of_Igbo_Personal_Na.pdf": {
        "title": "The Sociolinguistics of Igbo Personal Names",
        "author": "Linda Chinelo Nkamigbo",
        "year": "2019",
    },
    "Trends_Igbo_Names.pdf": {
        "title": "A Sociolinguistic Study of the Emerging Trends in Igbo Personal Names",
        "author": "Geraldine Ifesinachi Nnamdi-Eruchalu",
        "year": "2018",
    },
    "Decolonizing_Hausa_Naming.pdf": {},
    "Hausa Names.pdf": {},
    "Hausa Names (Ghana).pdf": {},
    "HausaNamesandNamingTraditions.pdf": {},
    "Hausa_Naming_Patternspdf.pdf": {},
    "Hausa_Naming_Practices_Modern.pdf": {},
    "Edo Personal Names_World View - Ota Ogie.pdf": {},
    "Edo.pdf": {},
    "Trends_Edo_Names.pdf": {},
    "Ibibio_Name_Structure.pdf": {},
    "Ibibio_Names.pdf": {},
    "Ibibio_Emotionreferencing_Names (1).pdf": {},
    "Ibibio_MensahEnglishisation.pdf": {},
    "Akan_Personal_Names(Ghana).pdf": {},
    "Trends_Ewe_Names(Ghana).pdf": {},
    "Urhobo-Names-and-Their-Meanings.pdf": {},
    "Igala_names_and_meanings.pdf": {},
    "KANURI_PERSONAL_NAMES.pdf": {},
    "Bakossi_namingculture.pdf": {},
    "Bukusu names (Kenya).pdf": {},
    "Siswati_names.pdf": {},
}


_EXCERPT_HEADER = re.compile(r"(?m)^\[([^\]\r\n]+)\]:\s*")


def humanize_source_filename(filename: str) -> str:
    """Create a display-only title without asserting publication metadata."""
    basename = PurePath((filename or "").replace("\\", "/")).name
    stem = re.sub(r"\.[^.]+$", "", basename)
    words = re.sub(r"[_-]+", " ", stem)
    words = re.sub(r"\s+", " ", words).strip()
    return words.title() if words else "Untitled Research Source"


def resolve_source_metadata(filename: str, excerpt: Optional[str] = None) -> Dict:
    basename = PurePath((filename or "").replace("\\", "/")).name
    verified = SOURCE_METADATA_REGISTRY.get(basename, {})
    source = {
        "filename": basename,
        "title": verified.get("title") or humanize_source_filename(basename),
        "title_is_fallback": not bool(verified.get("title")),
    }
    for field in ("author", "year"):
        if verified.get(field):
            source[field] = verified[field]
    if excerpt and excerpt.strip():
        source["excerpt"] = excerpt.strip()
    return source


def parse_rag_excerpts(rag_excerpts: str) -> List[Dict[str, str]]:
    """Parse ``[filename]: excerpt`` blocks without assuming page metadata."""
    text = (rag_excerpts or "").strip()
    if not text:
        return []
    matches = list(_EXCERPT_HEADER.finditer(text))
    parsed: List[Dict[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parsed.append(
            {
                "filename": match.group(1).strip(),
                "excerpt": text[match.end() : end].strip(),
            }
        )
    return parsed


def build_structured_sources(
    rag_excerpts: str,
    attributions: Iterable[str],
) -> List[Dict]:
    """Deduplicate by filename while preserving RAG retrieval order."""
    ordered: List[Dict] = []
    seen = set()

    for item in parse_rag_excerpts(rag_excerpts):
        filename = item["filename"]
        key = filename.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(resolve_source_metadata(filename, item.get("excerpt")))

    for filename in attributions or []:
        filename = (filename or "").strip()
        key = filename.casefold()
        if not filename or key in seen:
            continue
        seen.add(key)
        ordered.append(resolve_source_metadata(filename))

    return ordered
