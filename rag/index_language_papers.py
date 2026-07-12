#!/usr/bin/env python3
"""
Build a RAG index for one or more languages from PDFs in Research papers/.

Usage (from repo root):
  python rag/index_language_papers.py yoruba
  python rag/index_language_papers.py --all
  python rag/index_language_papers.py igbo hausa edo

Hausa: `1001_Hausa_Names.pdf` is not in `language_config` (names/meanings only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "rag") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "rag"))

from language_config import (  # noqa: E402
    INDEX_DIR,
    LANGUAGE_CONFIG,
    RESEARCH_PAPERS_DIR,
    get_language_config,
    list_rag_languages,
)

CHUNK_SIZE = 350
CHUNK_OVERLAP = 75
# Truncate very large extractions to avoid OOM on oversized PDFs
MAX_EXTRACTED_CHARS = 2_000_000

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    try:
        import PyPDF2

        PDF2_AVAILABLE = True
    except ImportError:
        PDF2_AVAILABLE = False


def extract_text_from_pdf(pdf_path: Path) -> str:
    text = ""
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as exc:
            print(f"   ⚠️  pdfplumber error: {exc}")
    if PDF2_AVAILABLE:
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as exc:
            print(f"   ❌ PyPDF2 error: {exc}")
    return ""


def chunk_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text] if text.strip() else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            for punct in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
                last_punct = text.rfind(punct, start, end)
                if last_punct != -1:
                    end = last_punct + len(punct)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def index_language(rag_key: str) -> bool:
    cfg = get_language_config(rag_key)
    papers = cfg.get("papers") or []
    if not papers:
        print(f"⏭️  {rag_key}: no papers configured")
        return False

    print(f"\n{'=' * 72}\n📚 Indexing {cfg['display_name']} ({rag_key})\n{'=' * 72}")
    papers_dir = RESEARCH_PAPERS_DIR
    if not papers_dir.exists():
        print(f"❌ Missing directory: {papers_dir}")
        return False

    paper_paths: list[Path] = []
    for paper_name in papers:
        path = papers_dir / paper_name
        if path.exists():
            paper_paths.append(path)
        else:
            print(f"⚠️  Not found: {paper_name}")

    if not paper_paths:
        print("❌ No PDFs found for this language")
        return False

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_file = INDEX_DIR / cfg["index_file"]
    jsonl_file = INDEX_DIR / f".{rag_key}_chunks.jsonl"
    if jsonl_file.exists():
        jsonl_file.unlink()

    paper_metadata = []
    total_chunks = 0

    for paper_path in paper_paths:
        print(f"📖 {paper_path.name}")
        text = extract_text_from_pdf(paper_path)
        if not text or len(text.strip()) < 100:
            print("   ⚠️  No usable text\n")
            continue
        if len(text) > MAX_EXTRACTED_CHARS:
            print(
                f"   ⚠️  Truncating extraction ({len(text):,} chars → {MAX_EXTRACTED_CHARS:,})"
            )
            text = text[:MAX_EXTRACTED_CHARS]
        chunks = chunk_text(text)
        print(f"   ✅ {len(text)} chars → {len(chunks)} chunks")
        paper_chunk_count = 0
        with open(jsonl_file, "a", encoding="utf-8") as fh:
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{rag_key}_{paper_path.name}_{i}".encode()
                ).hexdigest()
                json.dump(
                    {
                        "id": chunk_id,
                        "paper": paper_path.name,
                        "paper_path": str(paper_path),
                        "chunk_index": i,
                        "text": chunk,
                        "char_count": len(chunk),
                        "language": rag_key,
                    },
                    fh,
                    ensure_ascii=False,
                )
                fh.write("\n")
                paper_chunk_count += 1
        total_chunks += paper_chunk_count
        paper_metadata.append(
            {
                "name": paper_path.name,
                "chunks": paper_chunk_count,
                "total_chars": len(text),
            }
        )
        del text, chunks
        print(f"   💾 {paper_chunk_count} chunks on disk\n")

    if total_chunks == 0 or not jsonl_file.exists():
        print("❌ No chunks produced")
        return False

    index_file = INDEX_DIR / cfg["index_file"]
    metadata = {
        "language": rag_key,
        "display_name": cfg["display_name"],
        "total_papers": len(paper_paths),
        "total_chunks": total_chunks,
        "papers": [p.name for p in paper_paths],
        "paper_details": paper_metadata,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "has_embeddings": False,
    }
    _write_index_streaming(index_file, metadata, jsonl_file)
    jsonl_file.unlink(missing_ok=True)
    size_mb = index_file.stat().st_size / 1024 / 1024
    print(f"✅ Wrote {index_file} ({total_chunks} chunks, {size_mb:.2f} MB)")
    return True


def _write_index_streaming(index_file: Path, metadata: dict, jsonl_file: Path) -> None:
    """Write index JSON without loading all chunks into memory."""
    temp_file = index_file.with_name(f".{index_file.name}.tmp")
    try:
        with open(temp_file, "w", encoding="utf-8") as out:
            out.write('{"metadata":')
            json.dump(metadata, out, ensure_ascii=False)
            out.write(',"chunks":[')
            first = True
            with open(jsonl_file, encoding="utf-8") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    if not first:
                        out.write(",")
                    out.write(line)
                    first = False
            out.write("]}")
            out.flush()
            os.fsync(out.fileno())
        temp_file.replace(index_file)
        try:
            dir_fd = os.open(index_file.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        temp_file.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Index research papers per language")
    parser.add_argument(
        "languages",
        nargs="*",
        help="RAG keys (e.g. yoruba igbo). Use --all for every configured language.",
    )
    parser.add_argument("--all", action="store_true", help="Index all languages with papers")
    args = parser.parse_args()

    if not PDFPLUMBER_AVAILABLE and not PDF2_AVAILABLE:
        print("❌ Install pdfplumber or PyPDF2 to extract PDF text")
        sys.exit(1)

    keys = list_rag_languages() if args.all else (args.languages or ["yoruba"])
    if args.all and args.languages:
        keys = args.languages

    ok = 0
    for key in keys:
        if key not in LANGUAGE_CONFIG:
            print(f"❌ Unknown language key: {key}")
            continue
        if index_language(key):
            ok += 1
    print(f"\nDone: {ok}/{len(keys)} languages indexed.")


if __name__ == "__main__":
    main()
