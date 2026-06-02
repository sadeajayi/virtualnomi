#!/usr/bin/env python3
"""
Index Yoruba research papers for RAG system (memory-efficient version).
Extracts text from PDFs and chunks them. Embeddings can be added later.
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import hashlib

# PDF parsing libraries
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

RESEARCH_PAPERS_DIR = "Research papers"
OUTPUT_DIR = "research_papers_index"
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using available library"""
    
    text = ""
    
    # Try pdfplumber first (better quality)
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"⚠️  pdfplumber failed: {e}")
    
    # Fallback to PyPDF2
    if PDF2_AVAILABLE:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"❌ PyPDF2 failed: {e}")
            return ""
    
    raise ImportError("No PDF parsing library available")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last_punct = text.rfind(punct, start, end)
                if last_punct != -1:
                    end = last_punct + len(punct)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def index_yoruba_papers():
    """Index all Yoruba research papers - one at a time to save memory"""
    
    print("=" * 80)
    print("📚 INDEXING YORUBA RESEARCH PAPERS (Memory-Efficient)")
    print("=" * 80)
    print()
    
    # Find all Yoruba-related PDFs
    papers_dir = Path(RESEARCH_PAPERS_DIR)
    if not papers_dir.exists():
        print(f"❌ Directory '{RESEARCH_PAPERS_DIR}' not found")
        return
    
    yoruba_papers = []
    for pdf_file in papers_dir.glob("*.pdf"):
        filename_lower = pdf_file.name.lower()
        if "yoruba" in filename_lower:
            yoruba_papers.append(pdf_file)
    
    if not yoruba_papers:
        print(f"❌ No Yoruba papers found")
        return
    
    print(f"📄 Found {len(yoruba_papers)} Yoruba research papers\n")
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # Process and save each paper separately
    all_chunks = []
    total_chunks = 0
    
    for paper_idx, paper_path in enumerate(yoruba_papers, 1):
        print(f"📖 Processing ({paper_idx}/{len(yoruba_papers)}): {paper_path.name}...")
        
        try:
            # Extract text
            text = extract_text_from_pdf(str(paper_path))
            
            if not text or len(text.strip()) < 100:
                print(f"   ⚠️  Little or no text extracted")
                continue
            
            # Chunk text
            chunks = chunk_text(text)
            print(f"   ✅ Extracted {len(text)} chars, created {len(chunks)} chunks")
            
            # Create chunk metadata
            paper_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(f"{paper_path.name}_{i}".encode()).hexdigest()
                chunk_data = {
                    "id": chunk_id,
                    "paper": paper_path.name,
                    "paper_path": str(paper_path),
                    "chunk_index": i,
                    "text": chunk,
                    "char_count": len(chunk)
                }
                paper_chunks.append(chunk_data)
            
            all_chunks.extend(paper_chunks)
            total_chunks += len(paper_chunks)
            
            # Clear memory immediately
            del text, chunks, paper_chunks
            
            print(f"   💾 Total chunks: {total_chunks}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    print(f"\n✅ Total chunks created: {total_chunks}")
    
    # Save indexed data (without embeddings for now)
    index_file = output_path / "yoruba_papers_index.json"
    print(f"\n💾 Saving index to {index_file}...")
    
    index_data = {
        "metadata": {
            "total_papers": len(yoruba_papers),
            "total_chunks": len(all_chunks),
            "papers": [p.name for p in yoruba_papers],
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embeddings": False  # Will add later
        },
        "chunks": all_chunks
    }
    
    # Save in chunks to avoid memory issues
    print("   Writing chunks to file...")
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Index saved to {index_file}")
    print(f"   Note: Embeddings will be generated on-demand by RAG service")
    
    print("\n" + "=" * 80)
    print("✅ INDEXING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    index_yoruba_papers()

