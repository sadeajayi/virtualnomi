"""Expand a published phonetic spelling into learner anchors + stress."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

from syllable_store import (
    get_generated_syllables,
    get_published_syllables,
    normalize_syllables,
    phonetic_parts,
    remember_generated,
    stressed_index_from_phonetic,
)

SYLLABLE_MODEL = os.environ.get("NOMI_PHONETIC_MODEL") or os.environ.get(
    "NOMI_INSIGHTS_MODEL", "claude-sonnet-5"
)
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "syllable_guide_system_prompt.md"
_system_prompt_cache: Optional[str] = None
TOOL_NAME = "submit_syllable_guide"

TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "language": {"type": "string"},
        "flat_phonetic": {"type": "string"},
        "syllables": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "script": {"type": "string"},
                    "say": {"type": "string"},
                    "anchor": {"type": "string"},
                    "stress": {"type": "boolean"},
                },
                "required": ["say", "anchor", "stress"],
            },
        },
    },
    "required": ["syllables"],
}


class SyllableGuideError(ValueError):
    """Generation or validation failed."""


class SyllableGuideConfigError(RuntimeError):
    """Missing API key or SDK."""


def load_system_prompt() -> str:
    global _system_prompt_cache
    if _system_prompt_cache is None:
        _system_prompt_cache = _PROMPT_PATH.read_text(encoding="utf-8")
    return _system_prompt_cache


def apply_published_phonetic(
    syllables: List[Dict[str, Any]], phonetic: str
) -> List[Dict[str, Any]]:
    parts = phonetic_parts(phonetic)
    if len(syllables) != len(parts):
        raise SyllableGuideError(
            f"Expected {len(parts)} syllable(s) for '{phonetic}', got {len(syllables)}"
        )
    aligned: List[Dict[str, Any]] = []
    for beat, part in zip(syllables, parts):
        item = dict(beat)
        item["say"] = part.lower()
        if not str(item.get("anchor") or "").strip():
            raise SyllableGuideError(f"Missing anchor for beat '{part}'")
        aligned.append(item)

    forced = stressed_index_from_phonetic(phonetic)
    if forced is not None:
        for index, item in enumerate(aligned):
            item["stress"] = index == forced
    else:
        stressed = [item for item in aligned if item.get("stress") is True]
        if len(stressed) != 1:
            raise SyllableGuideError("Exactly one syllable must have stress: true")
    return aligned


def _require_anthropic() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SyllableGuideConfigError("ANTHROPIC_API_KEY is not set")
    if anthropic is None:
        raise SyllableGuideConfigError("anthropic package is not installed")


def _parse_tool_result(response: Any) -> Dict[str, Any]:
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != TOOL_NAME:
            continue
        tool_input = getattr(block, "input", None)
        if isinstance(tool_input, dict):
            return tool_input
    return {}


def _call_claude(system_prompt: str, user_message: str) -> Dict[str, Any]:
    _require_anthropic()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=SYLLABLE_MODEL,
        max_tokens=1200,
        system=system_prompt,
        tools=[
            {
                "name": TOOL_NAME,
                "description": "Submit the syllable guide. stress must be a boolean.",
                "input_schema": TOOL_SCHEMA,
            }
        ],
        tool_choice={"type": "tool", "name": TOOL_NAME},
        messages=[{"role": "user", "content": user_message}],
    )
    parsed = _parse_tool_result(response)
    if not parsed:
        raise SyllableGuideError("Claude did not return a syllable guide")
    return parsed


def generate_syllable_guide(
    *,
    name: str,
    name_strip: str,
    language: str,
    phonetic: str,
) -> List[Dict[str, Any]]:
    parts = phonetic_parts(phonetic)
    if len(parts) < 2:
        raise SyllableGuideError("Need a hyphenated phonetic spelling to build a guide")

    user_message = "\n".join(
        [
            f"Name: {name or name_strip}",
            f"NameStrip: {name_strip}",
            f"Language: {language}",
            f"Published phonetic (do not change the beats): {phonetic}",
            f"Beats in order: {', '.join(parts)}",
            "Write one anchor per beat and mark exactly one stress.",
        ]
    )
    parsed = _call_claude(load_system_prompt(), user_message)
    syllables = normalize_syllables(parsed.get("syllables"))
    return apply_published_phonetic(syllables, phonetic)


def resolve_syllable_guide(
    *,
    name: str,
    name_strip: str,
    language: str,
    phonetic: str,
    generate: bool = True,
) -> Optional[Dict[str, Any]]:
    published = get_published_syllables(name_strip, language)
    if published:
        return {"syllables": published, "source": "published", "phonetic": phonetic}

    generated = get_generated_syllables(name_strip, language)
    if generated:
        return {"syllables": generated, "source": "generated", "phonetic": phonetic}

    if not generate:
        return None
    if len(phonetic_parts(phonetic)) < 2:
        return None

    syllables = generate_syllable_guide(
        name=name,
        name_strip=name_strip,
        language=language,
        phonetic=phonetic,
    )
    remember_generated(name_strip, language, syllables)
    return {"syllables": syllables, "source": "generated", "phonetic": phonetic}
