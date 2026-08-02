#!/usr/bin/env python3
"""
RAG service for querying indexed African naming research papers by language.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from language_config import dataset_language_to_rag_key, get_language_config

# Hyphenated morpheme spellings in indexed text (Orie-style A-du-ke).
_HYPHENATED_NAME_FORM = re.compile(r"(?<![a-z])[a-z]+(?:-[a-z]+){1,}(?![a-z])")
# Letter runs after fold (keeps Hausa hooks ɓɗƙƴ that do not NFKD to ASCII).
_LETTER_RUN = re.compile(r"[^\W\d_]+", re.UNICODE)

# Folded keys → extra folded forms when OCR/appendix spelling diverges after
# dehyphenation (Orie Gender Markings: A-din-ni → adinni vs Àdùnní → adunni).
_NAME_MATCH_ALIASES: Dict[str, frozenset[str]] = {
    "adunni": frozenset({"adinni"}),
}

TOP_K = 5
TOP_K_DIVERSIFY = 20
MAX_CHUNKS_PER_PAPER = 2
INSIGHTS_TOP_K = 6
INSIGHTS_MAX_CHUNKS_PER_PAPER = 1
INSIGHTS_MAX_CHUNKS_PER_PAPER_PATTERN = 2
_PARALLEL_NAME_COLON = re.compile(r"[A-ZÀ-Ỹ][\w\u00C0-\u024F\u1E00-\u1EFF]+:\s*[\(\[]")
_NUMBERED_NAME_ENTRY = re.compile(r"\d+\.\s+[A-ZÀ-Ỹ][\w\u00C0-\u024F\u1E00-\u1EFF]+\s")

_NAME_DEFINITION_HINTS = (
    "meaning",
    "translation",
    "literally",
    "literal",
    "signify",
    "denote",
    "gloss",
    "etymology",
    "morpheme",
    "semantic",
    "translated",
    "consumer",
    "named",
)
_GENERIC_THEME_WORDS = frozenset(
    {
        "personal",
        "name",
        "names",
        "naming",
        "cultural",
        "significance",
        "tradition",
        "traditions",
        "child",
        "children",
        "born",
        "birth",
        "male",
        "female",
        "abundance",
        "prosperity",
        "harvest",
        "season",
        "plentiful",
        "wealth",
        "satisfied",
        "satisfaction",
        "community",
        "communal",
    }
)
# Single-token meaning overlaps that often hitchhike on acknowledgements / noise.
_WEAK_SINGLE_OVERLAP_TOKENS = frozenset(
    {
        "thanks",
        "thank",
        "mine",
        "found",
        "have",
        "worthy",
        "people",
        "person",
        "given",
        "great",
        "good",
        "love",
        "life",
        "world",
        "house",
        "family",
    }
)
# Expand meaning cues so "thanks" ties to gratitude-name excerpts, etc.
_RELATED_MEANING_GROUPS: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "thank",
            "thanks",
            "thankful",
            "gratitude",
            "grateful",
            "appreciate",
            "appreciative",
            "modupe",
            "opeyemi",
        }
    ),
    frozenset({"crown", "royal", "royalty", "king", "ade"}),
    frozenset({"honour", "honor", "dignity", "prestige"}),
    frozenset(
        {
            "companion",
            "oriki",
            "attributive",
            "abiso",
            "praise",
            "affection",
            "endear",
            "endearment",
        }
    ),
)
_NAMING_PATTERN_HINTS = (
    "oriki",
    "abiso",
    "attributive",
    "amutorunwa",
    "construction",
    "compound",
    "schema",
    "morpheme",
    "theophoric",
    "sentential",
    "appreciative",
    "gratitude",
    "birth circumstance",
)
_ACKNOWLEDGEMENT_MARKERS = (
    "thanks also to",
    "special thanks to",
    "i would like to thank",
    "i am especially grateful",
    "grateful to the late",
    "for comments on an earlier",
    "acknowledgement",
    "acknowledgment",
    "acknowledgments",
    "acknowledgements",
)
_MEANING_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "one",
        "who",
        "when",
        "was",
        "were",
        "is",
        "are",
        "into",
        "with",
        "for",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "from",
        "that",
        "this",
        "has",
        "have",
        "had",
        "be",
        "been",
        "being",
    }
)
# Minimum reranked similarity for score-only relevance (with non-generic evidence).
INSIGHTS_RELEVANCE_MIN_SCORE = 0.15

_rag_instances: Dict[str, "LanguageRAGService"] = {}


class LanguageRAGService:
    """Retrieve excerpts and cultural context from a per-language paper index."""

    def __init__(
        self,
        language_key: str,
        index_file: Optional[str] = None,
        quiet: bool = False,
        text_search_only: bool = False,
    ):
        self.language_key = language_key
        cfg = get_language_config(language_key)
        self.display_name = cfg["display_name"]
        self.query_suffix = cfg["query_suffix"]
        self.morphemes = cfg.get("morphemes") or []
        self.index_file = Path(index_file or cfg["index_path"])
        self.quiet = quiet
        self.text_search_only = text_search_only
        self.index_data = None
        self.model = None
        self.chunks: List[Dict] = []
        self.embeddings = None

        if not self.index_file.exists():
            raise FileNotFoundError(
                f"Index file not found: {self.index_file}\n"
                f"Run: python rag/index_language_papers.py {language_key}"
            )

        self._load_index()
        if not text_search_only:
            self._load_model()

    def _log(self, message: str) -> None:
        if not self.quiet:
            print(message)

    def _load_index(self) -> None:
        self._log(f"📥 Loading {self.language_key} index from {self.index_file}...")
        with open(self.index_file, encoding="utf-8") as fh:
            self.index_data = json.load(fh)
        self.chunks = self.index_data.get("chunks", [])
        if self.chunks and "embedding" in self.chunks[0]:
            self.embeddings = np.array([chunk["embedding"] for chunk in self.chunks])
        else:
            self.embeddings = None
        meta = self.index_data.get("metadata", {})
        self._log(
            f"✅ {len(self.chunks)} chunks from {meta.get('total_papers', '?')} papers"
        )

    def _load_model(self) -> None:
        if self.embeddings is None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        *,
        name: Optional[str] = None,
        meaning: Optional[str] = None,
    ) -> List[Dict]:
        if not self.chunks:
            return []
        if self.embeddings is not None and self.model is not None:
            return self._semantic_search(query, top_k, name=name, meaning=meaning)
        return self._text_search(query, top_k, name=name, meaning=meaning)

    @staticmethod
    def _fold_for_match(text: str) -> str:
        """NFKD-fold: strip combining marks and lowercase for Unicode-robust compare."""
        decomposed = unicodedata.normalize("NFKD", text or "")
        return "".join(
            c for c in decomposed if unicodedata.category(c) != "Mn"
        ).lower()

    @classmethod
    def _name_match_keys(cls, name: str) -> Set[str]:
        """Folded forms of a query name, including known OCR/appendix aliases."""
        folded = cls._fold_for_match(name).strip()
        if not folded:
            return set()
        keys = {folded, folded.replace("-", "")}
        keys.update(_NAME_MATCH_ALIASES.get(folded, ()))
        return {k for k in keys if k}

    @classmethod
    def _text_match_forms(cls, text: str) -> Set[str]:
        """Folded letter-runs plus dehyphenated compounds (A-du-ke → aduke)."""
        folded = cls._fold_for_match(text)
        forms = set(_LETTER_RUN.findall(folded))
        for compound in _HYPHENATED_NAME_FORM.findall(folded):
            forms.add(compound.replace("-", ""))
        return forms

    @staticmethod
    def _text_tokens(text: str) -> Set[str]:
        """Tokenize text; fold diacritics; split slash/hyphen compounds."""
        folded = LanguageRAGService._fold_for_match(text)
        words = set(_LETTER_RUN.findall(folded))
        for part in re.split(r"[/\-]", folded):
            part = part.strip()
            if len(part) >= 2:
                words.update(_LETTER_RUN.findall(part))
        words.update(LanguageRAGService._text_match_forms(text))
        return words

    @classmethod
    def _name_appears_in_text(cls, name: str, text: str) -> bool:
        keys = cls._name_match_keys(name)
        if not keys:
            return False
        text_folded = cls._fold_for_match(text)
        if any(key in text_folded for key in keys):
            return True
        text_forms = cls._text_match_forms(text)
        if keys & text_forms:
            return True
        return bool(keys & cls._text_tokens(text))

    @classmethod
    def has_name_specific_evidence(cls, name: str, text: str) -> bool:
        """True when the queried name (or alias) appears in the excerpt."""
        return cls._name_appears_in_text(name, text)

    def _meaning_content_tokens(self, meaning: str) -> Set[str]:
        tokens = self._text_tokens(meaning or "")
        return {
            t
            for t in tokens
            if len(t) >= 4 and t not in _MEANING_STOPWORDS and t not in _GENERIC_THEME_WORDS
        }

    def _expanded_meaning_tokens(self, meaning: str) -> Set[str]:
        """Meaning content tokens plus related pattern cues (thanks↔gratitude, etc.)."""
        base = self._meaning_content_tokens(meaning)
        raw = self._text_tokens(meaning or "")
        expanded = set(base)
        for group in _RELATED_MEANING_GROUPS:
            if base & group or raw & group:
                expanded |= group
        return expanded

    def _meaning_overlap_tokens(self, meaning: str, text: str) -> Set[str]:
        return self._expanded_meaning_tokens(meaning) & self._text_tokens(text)

    def _has_naming_pattern_context(self, text: str) -> bool:
        text_folded = self._fold_for_match(text)
        return any(hint in text_folded for hint in _NAMING_PATTERN_HINTS)

    def _morphemes_in_text(self, name: str, text: str) -> List[str]:
        text_folded = self._fold_for_match(text)
        text_tokens = self._text_tokens(text)
        found: List[str] = []
        for morpheme in self._extract_morphemes(name):
            mf = self._fold_for_match(morpheme)
            if len(mf) < 3:
                continue
            if mf in text_tokens:
                found.append(morpheme)
                continue
            # Substring only with letter boundaries (avoid ope⊂people, ade⊂trade).
            if re.search(rf"(?<![a-z]){re.escape(mf)}(?![a-z])", text_folded):
                found.append(morpheme)
        return found

    def _is_acknowledgement_boilerplate(self, text: str) -> bool:
        folded = self._fold_for_match(text)
        return any(marker in folded for marker in _ACKNOWLEDGEMENT_MARKERS)

    def is_pattern_relevant_excerpt(
        self,
        name: str,
        meaning: str,
        text: str,
        *,
        score: float = 0.0,
    ) -> bool:
        """
        Relevance gate for insights: require a real tie to this name's
        structure/meaning/pattern family. Exact name hits are sufficient but
        not required. Pure boilerplate ("names are important") fails.
        """
        excerpt = (text or "").strip()
        if not excerpt:
            return False

        # Author thank-yous must not satisfy gratitude-name queries.
        if self._is_acknowledgement_boilerplate(excerpt):
            return False

        # Prefer exact / alias name hits when present.
        if self._name_appears_in_text(name, excerpt):
            return True

        text_tokens = self._text_tokens(excerpt)
        base_overlap = self._meaning_content_tokens(meaning) & text_tokens
        meaning_overlap = self._expanded_meaning_tokens(meaning) & text_tokens
        morph_hits = self._morphemes_in_text(name, excerpt)
        pattern_family = self._is_pattern_family_chunk(excerpt)
        naming_context = self._has_naming_pattern_context(excerpt)
        structural_cue = bool(morph_hits or pattern_family or naming_context)
        strong_overlap = meaning_overlap - _WEAK_SINGLE_OVERLAP_TOKENS

        if pattern_family and meaning_overlap:
            return True

        if morph_hits:
            if meaning_overlap or naming_context:
                return True
            text_folded = self._fold_for_match(excerpt)
            if any(hint in text_folded for hint in _NAME_DEFINITION_HINTS):
                return True

        if len(strong_overlap) >= 2:
            return True

        if len(strong_overlap) == 1:
            tok = next(iter(strong_overlap))
            # Direct meaning content (e.g. crown, companion, honour) is enough.
            if tok in base_overlap and len(tok) >= 5:
                return True
            # Related-pattern cues (gratitude, oriki) need a structural cue.
            if structural_cue:
                return True
            return False

        # Weak-only overlaps (mine, found, thanks) need morphology/pattern support.
        if meaning_overlap and structural_cue:
            return True

        if (
            score >= INSIGHTS_RELEVANCE_MIN_SCORE
            and strong_overlap
            and naming_context
            and not self._is_generic_boilerplate(excerpt, meaning)
        ):
            return True

        return False

    def _is_generic_boilerplate(self, text: str, meaning: str) -> bool:
        """True for tropes that do not tie to this name's meaning/structure."""
        if self._meaning_overlap_tokens(meaning, text):
            return False
        text_tokens = self._text_tokens(text)
        content = {
            t
            for t in text_tokens
            if len(t) >= 4 and t not in _MEANING_STOPWORDS
        }
        if content and content <= _GENERIC_THEME_WORDS:
            return True
        folded = self._fold_for_match(text)
        markers = (
            "names are important",
            "naming traditions often",
            "cultural significance of names",
            "names carry deep",
            "deep meaning",
        )
        return any(marker in folded for marker in markers)

    def _has_name_definition_context(self, name: str, text: str) -> bool:
        if not self._name_appears_in_text(name, text):
            return False
        text_folded = self._fold_for_match(text)
        if any(hint in text_folded for hint in _NAME_DEFINITION_HINTS):
            return True
        for key in self._name_match_keys(name):
            if re.search(
                rf"{re.escape(key)}.{{0,120}}(mean|translat|literal|gloss|signif|denot)",
                text_folded,
                re.DOTALL,
            ):
                return True
        if re.search(r"name\s+literal\s+meaning", text_folded):
            return True
        return False

    def _insights_rerank_score(
        self,
        name: str,
        meaning: str,
        chunk: Dict,
        base_sim: float,
        query_words: Set[str],
    ) -> float:
        text = chunk["text"]
        text_tokens = self._text_tokens(text)
        score = base_sim
        name_present = self._name_appears_in_text(name, text)
        meaning_overlap = self._meaning_overlap_tokens(meaning, text)

        if name_present:
            score += 0.55
            if self._has_name_definition_context(name, text):
                score += 0.35
        else:
            overlap_words = query_words & text_tokens
            generic_only = overlap_words and overlap_words <= (
                _GENERIC_THEME_WORDS | set(self.query_suffix.lower().split())
            )
            # Penalize boilerplate-only hits; do not blanket-penalize all
            # non-name pattern/meaning excerpts (pattern-based insights).
            if generic_only:
                score -= 0.3

        if meaning_overlap:
            score += 0.2 if name_present else 0.25
            if len(meaning_overlap) >= 2:
                score += 0.1

        if self._morphemes_in_text(name, text):
            score += 0.2

        name_tokens = self._text_tokens(name)
        related_in_chunk = sum(
            1 for t in name_tokens if t in text_tokens and t != name.lower()
        )
        if name_present and related_in_chunk:
            score += min(0.15 * related_in_chunk, 0.25)

        if self._is_pattern_family_chunk(text):
            score += 0.2
            if meaning_overlap:
                score += 0.25

        return score

    @staticmethod
    def _is_pattern_family_chunk(text: str) -> bool:
        """Detect parallel name lists (Omuma-style or numbered gloss tables)."""
        if len(_PARALLEL_NAME_COLON.findall(text)) >= 2:
            return True
        if len(_NUMBERED_NAME_ENTRY.findall(text)) >= 3:
            return True
        return False

    @staticmethod
    def _insights_query_expansions(name: str, meaning: str) -> List[str]:
        """Add secondary search terms for rhetorical / pattern name families."""
        del name  # reserved for future name-structure expansions
        meaning_lower = (meaning or "").strip().lower()
        if not meaning_lower:
            return []
        expansions: List[str] = []
        if "who knows" in meaning_lower or "tomorrow" in meaning_lower:
            expansions.append(
                "Onyemaechi Omuma sentential rhetorical question names"
            )
        if "?" in meaning_lower or meaning_lower.startswith("who "):
            expansions.append("interrogative phrasal names rhetorical question")
        if any(
            cue in meaning_lower
            for cue in ("thank", "gratitude", "grateful", "appreciate")
        ):
            expansions.append(
                "appreciative gratitude names Modupe Opeyemi thank God"
            )
        if any(cue in meaning_lower for cue in ("crown", "honour", "honor")):
            expansions.append("adé crown compound construction names")
        if any(
            cue in meaning_lower
            for cue in ("companion", "found a friend", "found a companion")
        ):
            expansions.append("oríkì àbísọ attributive personal names")
        return expansions

    def _select_insights_excerpts(
        self,
        results: List[Dict],
        top_k: int,
        name: str,
    ) -> List[Dict]:
        """Pick diversified excerpts; allow a second chunk per paper for pattern lists."""
        results.sort(
            key=lambda r: (
                0 if self._name_appears_in_text(name, r["text"]) else 1,
                -r.get("similarity", 0),
            )
        )
        chosen: List[Dict] = []
        per_paper: Dict[str, int] = {}
        chosen_keys: Set[str] = set()

        def chunk_key(result: Dict) -> str:
            return str(result.get("id") or result["text"][:120])

        for result in results:
            if len(chosen) >= top_k:
                break
            paper = result.get("paper") or "unknown"
            if per_paper.get(paper, 0) >= INSIGHTS_MAX_CHUNKS_PER_PAPER:
                continue
            key = chunk_key(result)
            if key in chosen_keys:
                continue
            chosen.append(result)
            chosen_keys.add(key)
            per_paper[paper] = per_paper.get(paper, 0) + 1

        for result in results:
            if len(chosen) >= top_k:
                break
            paper = result.get("paper") or "unknown"
            if per_paper.get(paper, 0) >= INSIGHTS_MAX_CHUNKS_PER_PAPER_PATTERN:
                continue
            if per_paper.get(paper, 0) < 1:
                continue
            if not self._is_pattern_family_chunk(result["text"]):
                continue
            key = chunk_key(result)
            if key in chosen_keys:
                continue
            chosen.append(result)
            chosen_keys.add(key)
            per_paper[paper] = per_paper.get(paper, 0) + 1

        return chosen

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        *,
        name: Optional[str] = None,
        meaning: Optional[str] = None,
    ) -> List[Dict]:
        query_embedding = self.model.encode([query])[0]
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        pool_k = max(top_k * 4, TOP_K_DIVERSIFY) if (name or meaning) else top_k
        top_indices = np.argsort(similarities)[::-1][:pool_k]
        results = []
        query_words = set(query.lower().split())
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            base_sim = float(similarities[idx])
            if name:
                base_sim = self._insights_rerank_score(
                    name, meaning or "", chunk, base_sim, query_words
                )
            chunk["similarity"] = base_sim
            chunk.pop("embedding", None)
            results.append(chunk)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _diversify_by_paper(
        self,
        results: List[Dict],
        top_k: int = TOP_K,
        max_per_paper: int = MAX_CHUNKS_PER_PAPER,
    ) -> List[Dict]:
        if not results or max_per_paper < 1:
            return results[:top_k]
        chosen = []
        per_paper: Dict[str, int] = {}
        for result in results:
            if len(chosen) >= top_k:
                break
            paper = result.get("paper") or "unknown"
            if per_paper.get(paper, 0) >= max_per_paper:
                continue
            chosen.append(result)
            per_paper[paper] = per_paper.get(paper, 0) + 1
        return chosen

    def _text_search(
        self,
        query: str,
        top_k: int,
        *,
        name: Optional[str] = None,
        meaning: Optional[str] = None,
    ) -> List[Dict]:
        query_words = self._text_tokens(query)
        scored_chunks = []
        for chunk in self.chunks:
            text_tokens = self._text_tokens(chunk["text"])
            overlap = len(query_words & text_tokens)
            base_sim = overlap / len(query_words) if query_words else 0.0
            if name and self._name_appears_in_text(name, chunk["text"]):
                base_sim = max(base_sim, 0.12)
            if base_sim <= 0:
                continue
            chunk_copy = chunk.copy()
            if name:
                chunk_copy["similarity"] = self._insights_rerank_score(
                    name, meaning or "", chunk, base_sim, query_words
                )
            else:
                chunk_copy["similarity"] = base_sim
            chunk_copy.pop("embedding", None)
            scored_chunks.append(chunk_copy)
        scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_chunks[:top_k]

    def _extract_morphemes(self, name: str) -> List[str]:
        name_folded = self._fold_for_match(name)
        found: List[str] = []
        seen: Set[str] = set()
        for morpheme in self.morphemes:
            mf = self._fold_for_match(morpheme)
            if len(mf) < 3 or mf in seen:
                continue
            if mf in name_folded:
                found.append(morpheme)
                seen.add(mf)
        return found

    def get_cultural_context(self, name: str, meaning: str) -> str:
        """Format research excerpts for paraphrasing (same retrieval path as /insights)."""
        morphemes = self._extract_morphemes(name)
        excerpts = self.get_insights_excerpts(name, meaning)
        if not excerpts:
            return ""

        context_parts: List[str] = []
        morpheme_excerpts: List[Dict] = []
        morpheme_keys: Set[str] = set()

        if morphemes:
            for item in excerpts:
                text_lower = item["excerpt"].lower()
                for morpheme in morphemes:
                    if morpheme.lower() in text_lower:
                        semantic_keywords = (
                            "mean",
                            "meaning",
                            "signify",
                            "denote",
                            "semantic",
                            "morpheme",
                        )
                        if any(kw in text_lower for kw in semantic_keywords):
                            key = str(item.get("paper") or "") + item["excerpt"][:120]
                            if key not in morpheme_keys:
                                morpheme_excerpts.append(item)
                                morpheme_keys.add(key)
                            break
            if morpheme_excerpts:
                context_parts.append("Morpheme Analysis:")
                for item in morpheme_excerpts[:2]:
                    context_parts.append(
                        f"[From {item['paper']}]: {item['excerpt'][:400]}..."
                    )
                context_parts.append("")

        general_excerpts = [
            item
            for item in excerpts
            if (str(item.get("paper") or "") + item["excerpt"][:120]) not in morpheme_keys
        ][:3]
        if general_excerpts:
            if morphemes:
                context_parts.append("General Cultural Context:")
            for item in general_excerpts:
                context_parts.append(
                    f"[From {item['paper']}]: {item['excerpt'][:300]}..."
                )

        return "\n\n".join(context_parts)

    def get_relevant_excerpts(
        self,
        query: str,
        max_excerpts: int = 3,
        *,
        name: Optional[str] = None,
        meaning: Optional[str] = None,
    ) -> List[Dict]:
        results = self.search(
            query,
            top_k=max_excerpts,
            name=name,
            meaning=meaning,
        )
        return [
            {
                "paper": result["paper"],
                "excerpt": result["text"],
                "relevance_score": result.get("similarity", 0),
            }
            for result in results
        ]

    def get_insights_excerpts(
        self,
        name: str,
        meaning: str,
        *,
        top_k: int = INSIGHTS_TOP_K,
        max_per_paper: int = INSIGHTS_MAX_CHUNKS_PER_PAPER,
    ) -> List[Dict]:
        """Retrieve capped, per-paper-diversified excerpts for the insights endpoint."""
        del max_per_paper  # selection uses INSIGHTS_MAX_* constants
        morphemes = self._extract_morphemes(name)
        morph_part = " ".join(morphemes[:4])
        primary = f"{name} {meaning} {morph_part}".strip()
        queries = [primary]
        for expansion in self._insights_query_expansions(name, meaning):
            queries.append(f"{expansion} {meaning}".strip())

        merged_by_key: Dict[str, Dict] = {}
        for query in queries:
            for result in self.search(
                query,
                top_k=TOP_K_DIVERSIFY,
                name=name,
                meaning=meaning,
            ):
                key = str(result.get("id") or result["text"][:120])
                prev = merged_by_key.get(key)
                if prev is None or float(result.get("similarity") or 0) > float(
                    prev.get("similarity") or 0
                ):
                    merged_by_key[key] = result

        merged = list(merged_by_key.values())
        results = self._select_insights_excerpts(merged, top_k=top_k, name=name)
        return [
            {
                "paper": result["paper"],
                "excerpt": result["text"],
                "relevance_score": result.get("similarity", 0),
            }
            for result in results
        ]


class YorubaRAGService(LanguageRAGService):
    """Backward-compatible alias for Yoruba-only callers."""

    def __init__(
        self,
        index_file: Optional[str] = None,
        quiet: bool = False,
        text_search_only: bool = False,
    ):
        super().__init__(
            "yoruba",
            index_file=index_file,
            quiet=quiet,
            text_search_only=text_search_only,
        )


def get_rag_service_for_dataset_language(
    language: str, quiet: bool = True, text_search_only: bool = False
) -> Optional[LanguageRAGService]:
    """Return a cached RAG service for a Nomi dataset language, or None if unavailable."""
    rag_key = dataset_language_to_rag_key(language)
    if not rag_key:
        return None
    cache_key = f"{rag_key}:text" if text_search_only else rag_key
    if cache_key in _rag_instances:
        return _rag_instances[cache_key]
    try:
        service = LanguageRAGService(
            rag_key, quiet=quiet, text_search_only=text_search_only
        )
    except FileNotFoundError:
        return None
    _rag_instances[cache_key] = service
    return service


def main() -> None:
    print("=" * 80)
    print("🧪 TESTING LANGUAGE RAG SERVICE (yoruba)")
    print("=" * 80)
    try:
        rag = YorubaRAGService()
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return
    context = rag.get_cultural_context("Folasade", "crown me with honour")
    print(context or "(no context)")


if __name__ == "__main__":
    main()
