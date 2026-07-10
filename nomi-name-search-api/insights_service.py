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

INSIGHTS_MODEL = os.environ.get("NOMI_INSIGHTS_MODEL", "claude-sonnet-5")
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


def _first_sentence(text: str) -> str:
    parts = _split_sentences(text.strip())
    return parts[0] if parts else text.strip()


def _rest_after_first_sentence(text: str) -> str:
    parts = _split_sentences(text.strip())
    if len(parts) <= 1:
        return ""
    return _join_sentences(parts[1:])


def build_about_the_name(
    meaning: str,
    *,
    additional_meaning: str = "",
    cultural_context: str = "",
) -> Dict[str, str]:
    """
    Dataset-only copy for know-me: headline (poetic gloss) + body (fuller context).
    No RAG, no attribution, no model call.
    """
    meaning = (meaning or "").strip()
    additional_meaning = _normalize_additional_meaning(additional_meaning)
    cultural_context = (cultural_context or "").strip()

    headline = _first_sentence(meaning) if meaning else ""
    body_parts: List[str] = []

    rest_meaning = _rest_after_first_sentence(meaning)
    if rest_meaning:
        body_parts.append(rest_meaning)
    if additional_meaning:
        body_parts.append(additional_meaning)
    if cultural_context:
        body_parts.append(cultural_context)

    body = _join_sentences(body_parts)
    if not body and meaning and headline != meaning:
        body = meaning

    return {"headline": headline, "body": body}


def _normalize_for_overlap(text: str) -> set:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    stop = {
        "a", "an", "the", "and", "or", "of", "to", "in", "for", "is", "it",
        "this", "that", "with", "as", "at", "by", "from", "on", "be", "are",
        "was", "were", "name", "names", "meaning", "means",
    }
    return {t for t in tokens if len(t) > 2 and t not in stop}


def _cultural_depth_adds_value(cultural_depth: str, dataset_corpus: str) -> bool:
    """True when RAG text is not mostly restating dataset meaning/about copy."""
    depth = cultural_depth.strip()
    if not depth or depth.upper() == "NONE":
        return False

    depth_tokens = _normalize_for_overlap(depth)
    corpus_tokens = _normalize_for_overlap(dataset_corpus)
    if not depth_tokens:
        return False
    if not corpus_tokens:
        return True

    overlap = len(depth_tokens & corpus_tokens) / len(depth_tokens)
    if overlap >= 0.72:
        return False

    depth_lower = depth.lower()
    corpus_lower = dataset_corpus.lower()
    if len(depth) > 40 and depth_lower in corpus_lower:
        return False

    return True


def format_attribution(attributions: List[str]) -> str:
    papers = [p.strip() for p in attributions if p and str(p).strip()]
    if not papers:
        return ""
    if len(papers) == 1:
        return f"Source: {papers[0]}"
    return "Sources: " + "; ".join(papers)


def build_user_message(
    name: str,
    language: str,
    meaning: str,
    rag_excerpts: str,
    attributions: List[str],
    additional_meaning: str = "",
    about_the_name: Optional[Dict[str, str]] = None,
) -> str:
    rag_block = rag_excerpts if rag_excerpts else "(none)"
    about = about_the_name or {}
    about_headline = (about.get("headline") or "").strip()
    about_body = (about.get("body") or "").strip()
    lines = [
        f"Name: {name}",
        f"Language: {language}",
        f"Meaning: {meaning}",
    ]
    if additional_meaning:
        lines.append(f"Additional meaning: {additional_meaning}")
    if about_headline or about_body:
        lines.append(f"About the name (already shown to the reader): {about_headline}")
        if about_body:
            lines.append(f"About the name body: {about_body}")
    lines.extend(
        [
            "Task: Write 2–4 sentences of cultural depth ONLY if background notes add "
            "something the about-the-name text does not already say. Do not restate the "
            "gloss, meaning, or about copy. If notes add nothing new, return exactly: NONE",
            f"Background notes (internal — never mention in your paragraph): {rag_block}",
        ]
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
    cleaned = _strip_contrast_pedagogy_opening(cleaned)
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

_CONTRAST_RETRY_NOTE = (
    "Important: The first sentence must state the insight directly — no rather-than "
    "or not-but constructions. Lead with what the name is or asks."
)


def _build_insight_retry_note(insight: str) -> str:
    notes: List[str] = []
    if not insight or _contains_meta_source_language(insight):
        notes.append(_META_RETRY_NOTE)
    if _contains_contrast_pedagogy(insight):
        notes.append(_CONTRAST_RETRY_NOTE)
    return "\n\n".join(notes)


def generate_insight_paragraph(
    name: str,
    language: str,
    meaning: str,
    *,
    lookup_name_fn: Optional[Callable[[str, Optional[str]], list]] = None,
) -> Dict:
    """
    Full pipeline: dataset about-the-name + optional RAG cultural depth.
    `lookup_name_fn` should match `_lookup_name_results` signature when called from the API.
    """
    resolved_name = name
    resolved_meaning = (meaning or "").strip()
    resolved_language = (language or "").strip()
    resolved_additional_meaning = ""
    resolved_cultural_context = ""

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
            resolved_cultural_context = (primary.get("cultural_context") or "").strip()

    if not resolved_meaning:
        raise ValueError("Meaning is required (provide meaning= or a name in the dataset)")

    about_the_name = build_about_the_name(
        resolved_meaning,
        additional_meaning=resolved_additional_meaning,
        cultural_context=resolved_cultural_context,
    )

    rag_excerpts, attributions, rag_used, rag_key = gather_rag_context(
        resolved_name, resolved_meaning, resolved_language
    )

    cultural_depth: Optional[str] = None
    attribution: Optional[str] = None
    dataset_corpus = " ".join(
        part
        for part in (
            resolved_meaning,
            resolved_additional_meaning,
            resolved_cultural_context,
            about_the_name.get("headline", ""),
            about_the_name.get("body", ""),
        )
        if part
    )

    if rag_used and rag_excerpts:
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
            about_the_name=about_the_name,
        )

        client = anthropic.Anthropic(api_key=api_key)
        raw = _call_insights_model(client, system_prompt, user_message)
        depth_candidate = _clean_insight_output(raw)
        retry_note = _build_insight_retry_note(depth_candidate)
        if retry_note:
            raw = _call_insights_model(
                client,
                system_prompt,
                user_message,
                retry_note=retry_note,
            )
            depth_candidate = _clean_insight_output(raw)

        if depth_candidate and depth_candidate.upper() != "NONE":
            if _cultural_depth_adds_value(depth_candidate, dataset_corpus):
                formatted = format_attribution(attributions)
                if formatted:
                    cultural_depth = depth_candidate
                    attribution = formatted

    return {
        "name": resolved_name,
        "language": resolved_language,
        "meaning": resolved_meaning,
        "about_the_name": about_the_name,
        "cultural_depth": cultural_depth,
        "attribution": attribution,
        "rag_used": rag_used,
        "rag_excerpts": rag_excerpts,
        "rag_language_key": rag_key,
        "attributions": attributions,
        "model": INSIGHTS_MODEL if cultural_depth else None,
    }
