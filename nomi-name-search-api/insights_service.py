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

INSIGHTS_MODEL = os.environ.get("NOMI_INSIGHTS_MODEL", "claude-sonnet-4-6")
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "nomi_insights_system_prompt.md"
_system_prompt_cache: Optional[str] = None
_PROMPT_MTIME: Optional[float] = None


def load_insights_system_prompt() -> str:
    global _system_prompt_cache, _PROMPT_MTIME
    mtime = _PROMPT_PATH.stat().st_mtime if _PROMPT_PATH.exists() else None
    if _system_prompt_cache is not None and mtime == _PROMPT_MTIME:
        return _system_prompt_cache
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Insights system prompt not found: {_PROMPT_PATH}")
    _system_prompt_cache = _PROMPT_PATH.read_text(encoding="utf-8")
    _PROMPT_MTIME = mtime
    return _system_prompt_cache


def _normalize_additional_meaning(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return ""
    return text


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

    excerpts = rag.get_insights_excerpts(name, meaning)

    parts: List[str] = []
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
    additional_meaning: str = "",
) -> str:
    rag_block = rag_excerpts if rag_excerpts else "(none — stay within meaning and precise general knowledge)"
    lines = [
        f"Name: {name}",
        f"Language: {language}",
        f"Meaning: {meaning}",
    ]
    if additional_meaning:
        lines.append(f"Additional meaning: {additional_meaning}")
    lines.append(
        f"Background notes (internal — never mention in your paragraph): {rag_block}"
    )
    return "\n".join(lines)


_META_SOURCE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\brag\b"),
    re.compile(
        r"(?i)\b(?:the\s+)?(?:rag\s+)?(?:sources?|excerpts?|background\s+notes?|"
        r"research\s+papers?|attributions?|literature|source\s+material)\s+"
        r"(?:list|lists|show|shows|say|says|describe|describes|note|notes|give|gives|"
        r"mention|mentions|indicate|indicates|suggest|suggests|state|states|reveal|reveals)\b"
    ),
    re.compile(
        r"(?i)\b(?:according to|based on|drawing on)\s+"
        r"(?:the\s+)?(?:rag\s+)?(?:sources?|excerpts?|research|literature|"
        r"background\s+notes?|notes?|papers?)\b"
    ),
    re.compile(
        r"(?i)(?:^|[.!?]\s+)(?:in (?:the )?(?:research|literature|sources?|notes?|excerpts?)|"
        r"research shows?|the literature)\b"
    ),
    re.compile(r"(?i)\b(?:one|a)\s+source\s+(?:give|gives|list|lists|note|notes|mention|mentions)\b"),
    re.compile(r"(?i)\blisted alongside\b"),
    re.compile(
        r"(?i)\b(?:researchers?|studies|papers?)\s+"
        r"(?:show|shows|note|notes|describe|describes|list|lists|suggest|suggests)\b"
    ),
    re.compile(r"(?i)\b(?:meaning field|gloss|dataset|index|academic literature)\b"),
)


def _sentence_has_meta_source(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in _META_SOURCE_PATTERNS)


def _contains_meta_source_language(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    parts = re.split(r"(?<=[.!?])\s+", stripped)
    return any(_sentence_has_meta_source(part) for part in parts if part)


def _strip_meta_source_sentences(text: str) -> str:
    """Drop sentences that report what background material says instead of the name."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [part for part in parts if part and not _sentence_has_meta_source(part)]
    if not kept:
        return ""
    cleaned = " ".join(kept)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _clean_insight_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^(here is the insight:?\s*)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    cleaned = _strip_meta_source_sentences(cleaned)
    return cleaned.strip()


def _extract_response_text(response) -> str:
    raw = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw += block.text
    return raw


def _call_insights_model(
    client: "anthropic.Anthropic",
    system_prompt: str,
    user_message: str,
    *,
    retry_note: str = "",
) -> str:
    content = user_message if not retry_note else f"{user_message}\n\n{retry_note}"
    response = client.messages.create(
        model=INSIGHTS_MODEL,
        max_tokens=320,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    return _extract_response_text(response)


_META_RETRY_NOTE = (
    "Important: Do not mention RAG, sources, excerpts, background notes, research, "
    "literature, or attributions. Write only as a griot about the name."
)


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
    resolved_additional_meaning = ""

    if lookup_name_fn:
        matches = lookup_name_fn(name, language or None)
        if not resolved_meaning or not resolved_language:
            if not matches:
                raise ValueError(f"Name '{name}' not found in dataset")
        if matches:
            primary = matches[0]
            resolved_name = primary.get("name") or name
            if not resolved_meaning:
                resolved_meaning = primary.get("meaning") or resolved_meaning
            if not resolved_language:
                resolved_language = primary.get("language") or resolved_language
            resolved_additional_meaning = _normalize_additional_meaning(
                primary.get("additional_meaning")
            )

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
        additional_meaning=resolved_additional_meaning,
    )

    client = anthropic.Anthropic(api_key=api_key)
    raw = _call_insights_model(client, system_prompt, user_message)
    insight = _clean_insight_output(raw)
    if not insight or _contains_meta_source_language(insight):
        raw = _call_insights_model(
            client,
            system_prompt,
            user_message,
            retry_note=_META_RETRY_NOTE,
        )
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
