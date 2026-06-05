#!/usr/bin/env python3
"""
Index Yoruba research papers from the list in language_config.py.

Prefer running from repo root:
  python rag/index_language_papers.py yoruba

This module remains as a legacy entry point for docs and tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "rag") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "rag"))

from index_language_papers import index_language  # noqa: E402


def index_yoruba_papers() -> bool:
    return index_language("yoruba")


if __name__ == "__main__":
    ok = index_yoruba_papers()
    sys.exit(0 if ok else 1)
