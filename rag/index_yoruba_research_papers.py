#!/usr/bin/env python3
"""
Index Yoruba research papers - process one paper at a time, save incrementally.
"""

import os
import json
from pathlib import Path
import hashlib

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

SELECTED_PAPERS = [
    "Construction_Morphology_in_Yoruba_names_Schemas_an.pdf",
    "Semantics_Yoruba_Names.pdf",
    "Yoruba_Traditional_Names_and_the_Transmi.pdf",
    "Yoruba_Praise_Names.pdf"
]

RESEARCH_PAPERS_DIR = "Research papers"
OUTPUT_DIR = "research_papers_index"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 75

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF"""
    text = ""
    
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"⚠️  Error: {e}")
            return ""
    
    if PDF2_AVAILABLE:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"❌ Error: {e}")
            return ""
    
    return ""

def chunk_text(text: str) -> list:
    """Split text into chunks"""
    if len(text) <= CHUNK_SIZE:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last_punct = text.rfind(punct, start, end)
                if last_punct != -1:
                    end = last_punct + len(punct)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break
    
    return chunks

def index_yoruba_papers():
    """Index papers one at a time"""
    
    print("=" * 80)
    print("📚 INDEXING YORUBA RESEARCH PAPERS")
    print("=" * 80)
    print()
    
    papers_dir = Path(RESEARCH_PAPERS_DIR)
    if not papers_dir.exists():
        print(f"❌ Directory not found")
        return
    
    yoruba_papers = []
    for paper_name in SELECTED_PAPERS:
        paper_path = papers_dir / paper_name
        if paper_path.exists():
            yoruba_papers.append(paper_path)
        else:
            print(f"⚠️  Not found: {paper_name}")
    
    if not yoruba_papers:
        print(f"❌ No papers found")
        return
    
    print(f"📄 Processing {len(yoruba_papers)} papers\n")
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # Use JSONL to save incrementally
    jsonl_file = output_path / "chunks.jsonl"
    if jsonl_file.exists():
        jsonl_file.unlink()
    
    paper_metadata = []
    total_chunks = 0
    
    # Process one paper at a time
    for paper_idx, paper_path in enumerate(yoruba_papers, 1):
        print(f"📖 ({paper_idx}/{len(yoruba_papers)}) {paper_path.name}")
        
        try:
            # Extract
            text = extract_text_from_pdf(str(paper_path))
            if not text or len(text.strip()) < 100:
                print(f"   ⚠️  No text extracted\n")
                continue
            
            # Chunk
            chunks = chunk_text(text)
            print(f"   ✅ {len(text)} chars → {len(chunks)} chunks")
            
            # Save immediately to JSONL
            paper_chunk_count = 0
            with open(jsonl_file, 'a', encoding='utf-8') as f:
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
                    json.dump(chunk_data, f, ensure_ascii=False)
                    f.write('\n')
                    paper_chunk_count += 1
            
            total_chunks += paper_chunk_count
            paper_metadata.append({
                "name": paper_path.name,
                "chunks": paper_chunk_count,
                "total_chars": len(text)
            })
            
            # Clear memory
            del text, chunks
            
            print(f"   💾 Saved {paper_chunk_count} chunks (Total: {total_chunks})\n")
        
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            continue
    
    # Load from JSONL and create final index
    print(f"💾 Creating final index...")
    all_chunks = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_chunks.append(json.loads(line))
    
    # Save final index
    index_file = output_path / "yoruba_papers_index.json"
    index_data = {
        "metadata": {
            "total_papers": len(yoruba_papers),
            "total_chunks": len(all_chunks),
            "papers": [p.name for p in yoruba_papers],
            "paper_details": paper_metadata,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "has_embeddings": False
        },
        "chunks": all_chunks
    }
    
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    # Clean up
    if jsonl_file.exists():
        jsonl_file.unlink()
    
    print(f"✅ Index saved: {os.path.getsize(index_file) / 1024 / 1024:.2f} MB")
    print(f"\n📊 Summary:")
    print(f"   Papers: {len(yoruba_papers)}")
    print(f"   Chunks: {total_chunks}")
    print("\n✅ COMPLETE")

if __name__ == "__main__":
    index_yoruba_papers()
