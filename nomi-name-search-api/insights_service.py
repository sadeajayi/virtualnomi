"""
Generate Nomi name insights: RAG retrieval + Claude synthesis.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RAG_DIR = _REPO_ROOT / "rag"
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

try:
    from rag_service import get_rag_service_for_dataset_language
except ImportError:
    get_rag_service_for_dataset_language = None  # type: ignore

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

INSIGHTS_MODEL = os.environ.get("NOMI_INSIGHTS_MODEL", "claude-sonnet-4-20250514")
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "nomi_insights_system_prompt.md"
_system_prompt_cache: Optional[str] = None


def load_insights_system_prompt() -> str:
    global _system_prompt_cache
    if _system_prompt_cache is not None:
        return _system_prompt_cache
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Insights system prompt not found: {_PROMPT_PATH}")
    _system_prompt_cache = _PROMPT_PATH.read_text(encoding="utf-8")
    return _system_prompt_cache


def gather_rag_context(
    name: str,
    meaning: str,
    language: str,
) -> Tuple[str, List[str], bool, Optional[str]]:
    """
    Returns (rag_excerpts_text, attributions, rag_used, rag_language_key).
    """
    if not get_rag_service_for_dataset_language:
        return "", [], False, None

    rag = get_rag_service_for_dataset_language(
        language, quiet=True, text_search_only=True
    )
    if rag is None:
        return "", [], False, None

    context = rag.get_cultural_context(name, meaning)
    excerpts = rag.get_relevant_excerpts(f"{name} {meaning}", max_excerpts=4)

    parts: List[str] = []
    if context.strip():
        parts.append(context.strip())
    for item in excerpts:
        parts.append(f"[{item['paper']}]: {item['excerpt'][:500]}")

    attributions = sorted({item["paper"] for item in excerpts if item.get("paper")})
    rag_text = "\n\n".join(parts).strip()
    return rag_text, attributions, bool(rag_text), rag.language_key


def build_user_message(
    name: str,
    language: str,
    meaning: str,
    rag_excerpts: str,
    attributions: List[str],
) -> str:
    attr_text = ", ".join(attributions) if attributions else "(none)"
    rag_block = rag_excerpts if rag_excerpts else "(none — stay within meaning and precise general knowledge)"
    return (
        f"Name: {name}\n"
        f"Language: {language}\n"
        f"Meaning: {meaning}\n"
        f"RAG context (if available): {rag_block}\n"
        f"Source attributions (if available): {attr_text}"
    )


def _clean_insight_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^(here is the insight:?\s*)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    return cleaned.strip()


def generate_insight_paragraph(
    name: str,
    language: str,
    meaning: str,
    *,
    lookup_name_fn: Optional[Callable[[str, Optional[str]], list]] = None,
) -> Dict:
    """
    Full pipeline: optional dataset lookup, RAG, Claude.
    `lookup_name_fn` should match `_lookup_name_results` signature when called from the API.
    """
    resolved_name = name
    resolved_meaning = (meaning or "").strip()
    resolved_language = (language or "").strip()

    if lookup_name_fn and (not resolved_meaning or not resolved_language):
        matches = lookup_name_fn(name, language or None)
        if not matches:
            raise ValueError(f"Name '{name}' not found in dataset")
        primary = matches[0]
        resolved_name = primary.get("name") or name
        resolved_meaning = primary.get("meaning") or resolved_meaning
        resolved_language = primary.get("language") or resolved_language

    if not resolved_meaning:
        raise ValueError("Meaning is required (provide meaning= or a name in the dataset)")

    rag_excerpts, attributions, rag_used, rag_key = gather_rag_context(
        resolved_name, resolved_meaning, resolved_language
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed")

    raw_prompt = load_insights_system_prompt()
    role_idx = raw_prompt.find("## Role")
    system_prompt = raw_prompt[role_idx:] if role_idx >= 0 else raw_prompt
    user_message = build_user_message(
        resolved_name,
        resolved_language,
        resolved_meaning,
        rag_excerpts,
        attributions,
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=INSIGHTS_MODEL,
        max_tokens=320,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw += block.text

    insight = _clean_insight_output(raw)
    if not insight:
        raise RuntimeError("Claude returned an empty insight")

    return {
        "name": resolved_name,
        "language": resolved_language,
        "meaning": resolved_meaning,
        "insight": insight,
        "rag_used": rag_used,
        "rag_excerpts": rag_excerpts,
        "rag_language_key": rag_key,
        "attributions": attributions,
        "model": INSIGHTS_MODEL,
    }
