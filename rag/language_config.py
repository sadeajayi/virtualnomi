"""
Per-language RAG configuration: paper lists, index paths, and search query hints.
Maps Nomi dataset `Language` values to a RAG language key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_PAPERS_DIR = _REPO_ROOT / "Research papers"
INDEX_DIR = _REPO_ROOT / "research_papers_index"

# Dataset Language value (lowercase substring) -> RAG key
LANGUAGE_ALIASES: Dict[str, str] = {
    "yoruba": "yoruba",
    "igbo": "igbo",
    "hausa": "hausa",
    "edo": "edo",
    "ibibio": "ibibio",
    "akan": "akan",
    "ewe": "ewe",
    "urhobo": "urhobo",
    "igala": "igala",
    "kanuri": "kanuri",
    "wolof": "wolof",
    "bakossi": "bakossi",
    "bukusu": "bukusu",
    "siswati": "siswati",
}

LANGUAGE_CONFIG: Dict[str, Dict] = {
    "yoruba": {
        "display_name": "Yoruba",
        "index_file": "yoruba_papers_index.json",
        "query_suffix": "Yoruba personal name cultural significance morpheme",
        "papers": [
            "Construction_Morphology_in_Yoruba_names_Schemas_an.pdf",
            # Excluded: Semantics_Yoruba_Names.pdf — image-only scan, 0 extractable text chunks (needs OCR)
            "Yoruba_Traditional_Names_and_the_Transmi.pdf",
            "Yoruba_Praise_Names.pdf",
            "Endangerment_of_Yoruba_Individual_Names.pdf",
            "Yoruba_Names_Communicative_.pdf",
            "Yoruba Naming.pdf",
            "YORUBA-CRITICAL-ANALYSIS-OF-PERSONAL-NAME.pdf",
            "Yoruba_Construction_schemas.pdf",
            "Yoruba_ethnopragmatics_personal_names.pdf",
            "Yoruba_Ifa_Personal_Names.pdf",
            "Yoruba_Ilaje_names.pdf",
            "Yoruba_Names_Gender_Markings.pdf",
        ],
        "morphemes": [
            "ọlá", "ola", "ọla", "olá",
            "ade", "adé",
            "oluwa", "olúwa", "olu",
            "tunde", "túndé",
            "baba", "bàbá",
            "iya", "ìyá", "yeye",
            "ori", "orí",
            "ife", "ifẹ",
            "ifá", "ifa", "Ifa", "Ifá",
            "tolu", "tólú",
            "fola", "fọlá",
            "sade", "ṣade",
        ],
    },
    "igbo": {
        "display_name": "Igbo",
        "index_file": "igbo_papers_index.json",
        "query_suffix": "Igbo personal name cultural significance naming",
        "papers": [
            "Naming_in_Igbo_Land_A_Linguistic_and_Cul.pdf",
            "Religious_Significance_Igbo_Names.pdf",
            "Semantic_Analysis_of_Igbo_Names.pdf",
            "The_Sociolinguistics_of_Igbo_Personal_Na.pdf",
            "Trends_Igbo_Names.pdf",
        ],
        "morphemes": [],
    },
    "hausa": {
        "display_name": "Hausa",
        "index_file": "hausa_papers_index.json",
        "query_suffix": "Hausa personal name cultural significance naming",
        "papers": [
            # Excluded: 1001_Hausa_Names.pdf — name/meaning list only, not research insights
            "Decolonizing_Hausa_Naming.pdf",
            "Hausa Names.pdf",
            "Hausa Names (Ghana).pdf",
            "HausaNamesandNamingTraditions.pdf",
            "Hausa_Naming_Patternspdf.pdf",
            "Hausa_Naming_Practices_Modern.pdf",
        ],
        "morphemes": [],
    },
    "edo": {
        "display_name": "Edo",
        "index_file": "edo_papers_index.json",
        "query_suffix": "Edo personal name cultural significance",
        "papers": [
            "Edo Personal Names_World View - Ota Ogie.pdf",
            "Edo.pdf",
            "Trends_Edo_Names.pdf",
        ],
        "morphemes": [],
    },
    "ibibio": {
        "display_name": "Ibibio",
        "index_file": "ibibio_papers_index.json",
        "query_suffix": "Ibibio personal name cultural significance",
        "papers": [
            "Ibibio_Name_Structure.pdf",
            "Ibibio_Names.pdf",
            "Ibibio_Emotionreferencing_Names (1).pdf",
            "Ibibio_MensahEnglishisation.pdf",
        ],
        "morphemes": [],
    },
    "akan": {
        "display_name": "Akan",
        "index_file": "akan_papers_index.json",
        "query_suffix": "Akan personal name cultural significance Ghana",
        "papers": [
            "Akan_Personal_Names(Ghana).pdf",
        ],
        "morphemes": [],
    },
    "ewe": {
        "display_name": "Ewe",
        "index_file": "ewe_papers_index.json",
        "query_suffix": "Ewe personal name cultural significance Ghana",
        "papers": [
            "Trends_Ewe_Names(Ghana).pdf",
        ],
        "morphemes": [],
    },
    "urhobo": {
        "display_name": "Urhobo",
        "index_file": "urhobo_papers_index.json",
        "query_suffix": "Urhobo personal name meaning",
        "papers": [
            "Urhobo-Names-and-Their-Meanings.pdf",
        ],
        "morphemes": [],
    },
    "igala": {
        "display_name": "Igala",
        "index_file": "igala_papers_index.json",
        "query_suffix": "Igala personal name meaning",
        "papers": [
            "Igala_names_and_meanings.pdf",
        ],
        "morphemes": [],
    },
    "kanuri": {
        "display_name": "Kanuri",
        "index_file": "kanuri_papers_index.json",
        "query_suffix": "Kanuri personal name meaning",
        "papers": [
            "KANURI_PERSONAL_NAMES.pdf",
        ],
        "morphemes": [],
    },
    "bakossi": {
        "display_name": "Bakossi",
        "index_file": "bakossi_papers_index.json",
        "query_suffix": "Bakossi naming culture personal name",
        "papers": [
            "Bakossi_namingculture.pdf",
        ],
        "morphemes": [],
    },
    "bukusu": {
        "display_name": "Bukusu",
        "index_file": "bukusu_papers_index.json",
        "query_suffix": "Bukusu personal name Kenya",
        "papers": [
            "Bukusu names (Kenya).pdf",
        ],
        "morphemes": [],
    },
    "siswati": {
        "display_name": "Siswati",
        "index_file": "siswati_papers_index.json",
        "query_suffix": "Siswati personal name meaning",
        "papers": [
            "Siswati_names.pdf",
        ],
        "morphemes": [],
    },
    "wolof": {
        "display_name": "Wolof",
        "index_file": "wolof_papers_index.json",
        "query_suffix": "Wolof personal name meaning",
        "papers": [],
        "morphemes": [],
    },
}


def dataset_language_to_rag_key(language: str) -> Optional[str]:
    """Map a Nomi dataset Language string to a RAG config key, if supported."""
    lang_lower = (language or "").strip().lower()
    if not lang_lower:
        return None
    for needle, key in LANGUAGE_ALIASES.items():
        if needle in lang_lower:
            return key
    return None


def get_language_config(rag_key: str) -> Dict:
    if rag_key not in LANGUAGE_CONFIG:
        raise KeyError(f"Unknown RAG language key: {rag_key}")
    cfg = dict(LANGUAGE_CONFIG[rag_key])
    cfg["rag_key"] = rag_key
    cfg["index_path"] = str(INDEX_DIR / cfg["index_file"])
    return cfg


def list_rag_languages() -> List[str]:
    return sorted(LANGUAGE_CONFIG.keys())
