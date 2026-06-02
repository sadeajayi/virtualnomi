#!/usr/bin/env python3
"""
RAG service for querying indexed African naming research papers by language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from language_config import dataset_language_to_rag_key, get_language_config

TOP_K = 5
TOP_K_DIVERSIFY = 20
MAX_CHUNKS_PER_PAPER = 2

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
        if not ST_AVAILABLE or self.embeddings is None:
            return
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        if not self.chunks:
            return []
        if self.embeddings is not None and self.model is not None:
            return self._semantic_search(query, top_k)
        return self._text_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        query_embedding = self.model.encode([query])[0]
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["similarity"] = float(similarities[idx])
            chunk.pop("embedding", None)
            results.append(chunk)
        return results

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

    def _text_search(self, query: str, top_k: int) -> List[Dict]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored_chunks = []
        for chunk in self.chunks:
            text_lower = chunk["text"].lower()
            text_words = set(text_lower.split())
            overlap = len(query_words & text_words)
            score = overlap / len(query_words) if query_words else 0
            if score > 0:
                chunk_copy = chunk.copy()
                chunk_copy["similarity"] = score
                chunk_copy.pop("embedding", None)
                scored_chunks.append(chunk_copy)
        scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_chunks[:top_k]

    def _extract_morphemes(self, name: str) -> List[str]:
        name_lower = name.lower()
        found = []
        for morpheme in self.morphemes:
            if morpheme.lower() in name_lower:
                found.append(morpheme)
        return found

    def get_cultural_context(self, name: str, meaning: str) -> str:
        morphemes = self._extract_morphemes(name)
        base_query = f"{name} {meaning} {self.query_suffix}"
        if morphemes:
            morpheme_query = " ".join(f"{m} morpheme meaning semantic" for m in morphemes)
            query = f"{base_query} {morpheme_query}"
        else:
            query = base_query

        raw_results = self.search(query, top_k=TOP_K_DIVERSIFY)
        results = self._diversify_by_paper(raw_results, top_k=5, max_per_paper=MAX_CHUNKS_PER_PAPER)
        if not results:
            return ""

        context_parts: List[str] = []
        morpheme_results: List[Dict] = []

        if morphemes:
            for result in results:
                text_lower = result["text"].lower()
                for morpheme in morphemes:
                    if morpheme.lower() in text_lower:
                        semantic_keywords = [
                            "mean",
                            "meaning",
                            "signify",
                            "denote",
                            "semantic",
                            "morpheme",
                        ]
                        if any(kw in text_lower for kw in semantic_keywords):
                            morpheme_results.append(result)
                            break
            if morpheme_results:
                context_parts.append("Morpheme Analysis:")
                for result in morpheme_results[:2]:
                    context_parts.append(
                        f"[From {result['paper']}]: {result['text'][:400]}..."
                    )
                context_parts.append("")

        general_results = [r for r in results if r not in morpheme_results][:3]
        if general_results:
            if morphemes:
                context_parts.append("General Cultural Context:")
            for result in general_results:
                context_parts.append(
                    f"[From {result['paper']}]: {result['text'][:300]}..."
                )

        return "\n\n".join(context_parts)

    def get_relevant_excerpts(self, query: str, max_excerpts: int = 3) -> List[Dict]:
        results = self.search(query, top_k=max_excerpts)
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

    def __init__(self, index_file: Optional[str] = None, quiet: bool = False):
        super().__init__("yoruba", index_file=index_file, quiet=quiet)


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
