"""
Generate Nomi name insights: RAG retrieval + Claude synthesis.
"""

from __future__ import annotations

import os
import re
import sys
import time
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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

from source_metadata import build_structured_sources

INSIGHTS_MODEL = os.environ.get("NOMI_INSIGHTS_MODEL", "claude-sonnet-5")
INSIGHTS_CACHE_SCHEMA_VERSION = os.environ.get(
    "NOMI_INSIGHTS_CACHE_VERSION", "v9-post-gate-avoid-ai-writing"
)
INSIGHTS_CACHE_TTL_SECONDS = int(
    os.environ.get("NOMI_INSIGHTS_CACHE_TTL_SECONDS", str(7 * 24 * 60 * 60))
)
INSIGHTS_CACHE_MAX_ENTRIES = int(os.environ.get("NOMI_INSIGHTS_CACHE_MAX_ENTRIES", "512"))
# Production voice: Cursor rules (e.g. avoid-ai-writing.mdc) do not apply on Render.
# Keep the system prompt story-first; enforce high-signal bans in the post-generation gate.
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "nomi_insights_system_prompt.md"
_system_prompt_cache: Optional[str] = None
_PROMPT_MTIME: Optional[float] = None
_insight_result_cache: Dict[Tuple[str, ...], Tuple[float, Dict[str, Any]]] = {}


class NoGroundedInsightError(ValueError):
    """No relevant research excerpts for this name's language/meaning/patterns."""


class OffVoiceInsightError(RuntimeError):
    """The model failed the deterministic Nomi voice contract after one retry."""


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


def _normalized_cache_text(value: str, *, casefold: bool = False) -> str:
    normalized = " ".join((value or "").split())
    return normalized.casefold() if casefold else normalized


def _insight_cache_key(
    name: str,
    language: str,
    meaning: str,
    additional_meaning: str,
    system_prompt: str,
) -> Tuple[str, ...]:
    """Versioned key: name/language/meaning plus every prompt input that changes output."""
    prompt_digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    return (
        INSIGHTS_CACHE_SCHEMA_VERSION,
        INSIGHTS_MODEL,
        prompt_digest,
        _normalized_cache_text(name, casefold=True),
        _normalized_cache_text(language, casefold=True),
        _normalized_cache_text(meaning),
        _normalized_cache_text(additional_meaning),
    )


def _get_cached_insight(key: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    cached = _insight_result_cache.get(key)
    if cached is None:
        return None
    stored_at, payload = cached
    if time.monotonic() - stored_at >= INSIGHTS_CACHE_TTL_SECONDS:
        _insight_result_cache.pop(key, None)
        return None
    if (
        payload.get("grounded") is not True
        or not payload.get("rag_used")
        or not payload.get("sources")
    ):
        _insight_result_cache.pop(key, None)
        return None
    return deepcopy(payload)


def _store_cached_insight(key: Tuple[str, ...], payload: Dict[str, Any]) -> None:
    # Never persist ungrounded or sources-less payloads (failures must not cache).
    if (
        payload.get("grounded") is not True
        or not payload.get("rag_used")
        or not payload.get("sources")
        or not str(payload.get("insight") or "").strip()
    ):
        return
    if len(_insight_result_cache) >= INSIGHTS_CACHE_MAX_ENTRIES:
        oldest_key = min(_insight_result_cache, key=lambda item: _insight_result_cache[item][0])
        _insight_result_cache.pop(oldest_key, None)
    _insight_result_cache[key] = (time.monotonic(), deepcopy(payload))


def _clear_insight_cache() -> None:
    """Test/admin helper; production invalidation uses prompt digest or cache version."""
    _insight_result_cache.clear()


def _lookup_language(value: Any) -> str:
    return str(value or "").strip()


def _select_lookup_match(
    name: str,
    matches: List[Dict[str, Any]],
    requested_language: str,
) -> Optional[Dict[str, Any]]:
    if not matches:
        return None

    normalized_requested = _lookup_language(requested_language).casefold()
    if normalized_requested:
        exact_matches = [
            match
            for match in matches
            if _lookup_language(match.get("language")).casefold() == normalized_requested
        ]
        if exact_matches:
            return exact_matches[0]

    languages = sorted(
        {
            _lookup_language(match.get("language"))
            for match in matches
            if _lookup_language(match.get("language"))
        },
        key=str.casefold,
    )
    if len(languages) > 1:
        raise ValueError(
            f"Name '{name}' matches multiple languages ({', '.join(languages)}); "
            "provide a specific language"
        )
    return matches[0]


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

    Fail closed unless RAG returns excerpts that pass the pattern-relevance
    gate (exact name hit optional; meaning/morphology/pattern ties required).
    """
    if not get_rag_service_for_dataset_language:
        return "", [], False, None

    rag = get_rag_service_for_dataset_language(
        language, quiet=True, text_search_only=True
    )
    if rag is None:
        return "", [], False, None

    excerpts = [
        item
        for item in rag.get_insights_excerpts(name, meaning)
        if rag.is_pattern_relevant_excerpt(
            name,
            meaning,
            item.get("excerpt", ""),
            score=float(item.get("relevance_score") or 0),
        )
    ]

    parts: List[str] = []
    for item in excerpts:
        parts.append(f"[{item['paper']}]: {item['excerpt'][:500]}")

    attributions = list(
        dict.fromkeys(item["paper"] for item in excerpts if item.get("paper"))
    )
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
    rag_block = (
        rag_excerpts
        if rag_excerpts
        else "(none — do not invent research; this request should have been blocked)"
    )
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
    if attributions:
        lines.append(
            "Note: Research illuminates patterns and relations for this name. "
            "Only claim the notes discuss this exact personal name when that name "
            "appears in the notes."
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


def _split_sentences(text: str) -> List[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def _join_sentences(parts: List[str]) -> str:
    if not parts:
        return ""
    cleaned = " ".join(parts)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _strip_meta_source_sentences(text: str) -> str:
    """Drop sentences that report what background material says instead of the name."""
    kept = [part for part in _split_sentences(text) if not _sentence_has_meta_source(part)]
    return _join_sentences(kept)


_CONTRAST_PEDAGOGY_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\brather than\b"),
    re.compile(r"(?i)\bnot\b[^.!?]{0,120}\bbut\b"),
    re.compile(r"(?i)\bnot a\b[^.!?]{0,120}\bbut\b"),
    re.compile(
        r"(?i)\bit(?:'s| is| was| isn\'t| is not| wasn\'t| was not)\b"
        r"[^.!?]{0,120}\b(?:but|it(?:'s| is))\b"
    ),
    re.compile(r"(?i)\bless\b[^.!?]{0,60}\bthan\b"),
)

# High-precision avoid-ai-writing Tier-1 / structural bans. Prefer precision over
# coverage — legitimate cultural prose must not trip these.
_STYLE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("em dash", re.compile(r"[\u2014\u2e3a]")),
    ("en dash as pause", re.compile(r"\u2013")),
    ("double-hyphen dash", re.compile(r"(?<!\d)--(?!\d)")),
    ("not X but Y", re.compile(r"(?i)\bnot\b[^.!?]{0,120}\bbut\b")),
    ("not only X", re.compile(r"(?i)\bnot\s+only\b")),
    (
        "negative corrective parallelism",
        re.compile(
            r"(?i)\b(?:isn['’]t|is not|wasn['’]t|was not)\s+"
            r"(?:just|merely|only)\b"
        ),
    ),
    ("rather than", re.compile(r"(?i)\brather\s+than\b")),
    ("less X than Y", re.compile(r"(?i)\bless\b[^.!?]{0,80}\bthan\b")),
    (
        "hollow intensifier / hedge phrase",
        re.compile(
            r"(?i)\b(?:it(?:['’]s| is) worth noting(?: that)?|to be clear|"
            r"quite frankly|to be honest|let['’]s be clear|"
            r"could potentially|at its core)\b"
        ),
    ),
    (
        "significance inflation",
        re.compile(
            r"(?i)\b(?:testament to|stands as a testament|pivotal moment|"
            r"watershed moment|marking a pivotal|game[- ]changer|"
            r"game[- ]changing|cutting[- ]edge)\b"
        ),
    ),
    (
        "AI Tier-1 filler",
        re.compile(
            r"(?i)\b(?:delve(?:s|d|ing)?(?:\s+into)?|tapestry|paradigm|"
            r"embark(?:s|ed|ing)?|beacon|seamless(?:ly)?|"
            r"utili[sz]e(?:s|d|ing)?|nestled|deep dive|dive into|"
            r"unpack(?:s|ed|ing)?|synerg(?:y|ies)|"
            r"rich cultural heritage|deeply rooted)\b"
        ),
    ),
    (
        "brochure / transition AI-ism",
        re.compile(
            r"(?i)(?:(?:^|[.!?]\s+)(?:moreover|furthermore|additionally)\b|"
            r"\b(?:in today['’]s|in an era where|at the end of the day|"
            r"only time will tell|the future looks bright|"
            r"here['’]s the thing|the catch\?|plot twist|"
            r"let['’]s (?:explore|dive|break this down|take a look))\b)"
        ),
    ),
)


def insight_style_violations(text: str) -> List[str]:
    """Return deterministic whole-paragraph voice violations (avoid-ai-writing gate)."""
    return [
        label
        for label, pattern in _STYLE_PATTERNS
        if pattern.search(text or "")
    ]


def avoid_ai_writing_gate(text: str) -> List[str]:
    """Post-generation avoid-ai-writing check; alias of insight_style_violations."""
    return insight_style_violations(text)


def _sentence_has_contrast_pedagogy(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in _CONTRAST_PEDAGOGY_PATTERNS)


def _contains_contrast_pedagogy(text: str) -> bool:
    parts = _split_sentences(text)
    if not parts:
        return False
    return _sentence_has_contrast_pedagogy(parts[0])


def _strip_contrast_pedagogy_opening(text: str) -> str:
    """Drop a contrast-setup first sentence (not X but Y, rather than, etc.)."""
    parts = _split_sentences(text)
    if parts and _sentence_has_contrast_pedagogy(parts[0]):
        parts = parts[1:]
    return _join_sentences(parts)


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

_AVOID_AI_RETRY_NOTE = (
    "Important: Rewrite this Reading to remove the listed AI-writing patterns. "
    "State every claim directly. Use no em dashes, en dashes as pauses, or "
    "double-hyphen dashes. Do not use negative contrast or corrective parallelism. "
    "Cut hollow intensifiers, significance-inflation phrases, and brochure filler."
)


def _build_insight_retry_note(insight: str) -> str:
    notes: List[str] = []
    if not insight or _contains_meta_source_language(insight):
        notes.append(_META_RETRY_NOTE)
    violations = avoid_ai_writing_gate(insight)
    if violations:
        notes.append(
            f"{_AVOID_AI_RETRY_NOTE} Detected violations: {', '.join(violations)}."
        )
    return "\n\n".join(notes)


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
            primary = _select_lookup_match(name, matches, language)
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

    raw_prompt = load_insights_system_prompt()
    cache_key = _insight_cache_key(
        resolved_name,
        resolved_language,
        resolved_meaning,
        resolved_additional_meaning,
        raw_prompt,
    )
    cached = _get_cached_insight(cache_key)
    if cached is not None:
        return cached

    rag_excerpts, attributions, rag_used, rag_key = gather_rag_context(
        resolved_name, resolved_meaning, resolved_language
    )
    if not rag_used or not rag_excerpts or not attributions:
        raise NoGroundedInsightError(
            f"No research-grounded reading is available for '{resolved_name}'"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed")

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
    retry_note = _build_insight_retry_note(insight)
    if retry_note:
        raw = _call_insights_model(
            client,
            system_prompt,
            user_message,
            retry_note=retry_note,
        )
        insight = _clean_insight_output(raw)
    if not insight:
        raise RuntimeError("Claude returned an empty insight")
    final_violations = avoid_ai_writing_gate(insight)
    if final_violations:
        raise OffVoiceInsightError(
            "Generated Reading failed avoid-ai-writing gate after one retry: "
            + ", ".join(final_violations)
        )

    payload = {
        "name": resolved_name,
        "language": resolved_language,
        "meaning": resolved_meaning,
        "insight": insight,
        "grounded": True,
        "rag_used": rag_used,
        "rag_excerpts": rag_excerpts,
        "rag_language_key": rag_key,
        "sources": build_structured_sources(rag_excerpts, attributions),
        # Deprecated: retained temporarily for existing personal-card clients.
        "attributions": attributions,
        "model": INSIGHTS_MODEL,
    }
    _store_cached_insight(cache_key, payload)
    return payload
