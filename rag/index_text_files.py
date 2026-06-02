#!/usr/bin/env python3
"""
Index extracted text files - minimal memory version.
"""

import os
import json
from pathlib import Path
import hashlib

SELECTED_PAPERS = [
    "Construction_Morphology_in_Yoruba_names_Schemas_an",
    "Yoruba_Traditional_Names_and_the_Transmi",
    "Yoruba_Praise_Names"
]

EXTRACTED_DIR = "Research papers/extracted"
OUTPUT_DIR = "research_papers_index"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 75

def chunk_text(text: str) -> list:
    """Split text into overlapping chunks"""
    if len(text) <= CHUNK_SIZE:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last = text.rfind(punct, start, end)
                if last != -1:
                    end = last + len(punct)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break
    
    return chunks

def index_text_files():
    """Index text files"""
    
    print("=" * 80)
    print("📚 INDEXING TEXT FILES")
    print("=" * 80)
    print()
    
    extracted_path = Path(EXTRACTED_DIR)
    if not extracted_path.exists():
        print(f"❌ Directory not found")
        return
    
    # Find text files
    text_files = []
    for paper_name in SELECTED_PAPERS:
        txt_file = extracted_path / f"{paper_name}.txt"
        if txt_file.exists():
            text_files.append((paper_name, txt_file))
    
    if not text_files:
        print(f"❌ No text files found")
        return
    
    print(f"📄 Found {len(text_files)} text files\n")
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # Build index
    all_chunks = []
    paper_metadata = []
    
    for paper_name, txt_file in text_files:
        print(f"📖 Processing: {paper_name}")
        
        try:
            # Read file
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text or len(text.strip()) < 100:
                print(f"   ⚠️  File too short\n")
                continue
            
            # Chunk
            chunks = chunk_text(text)
            print(f"   ✅ {len(text)} chars → {len(chunks)} chunks")
            
            # Create chunk data
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(f"{paper_name}_{i}".encode()).hexdigest()
                all_chunks.append({
                    "id": chunk_id,
                    "paper": f"{paper_name}.pdf",
                    "paper_path": str(txt_file),
                    "chunk_index": i,
                    "text": chunk,
                    "char_count": len(chunk)
                })
            
            paper_metadata.append({
                "name": f"{paper_name}.pdf",
                "chunks": len(chunks),
                "total_chars": len(text)
            })
            
            del text, chunks
            print(f"   💾 Added {len(chunks)} chunks\n")
        
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            continue
    
    # Save index
    print(f"💾 Saving index...")
    index_file = output_path / "yoruba_papers_index.json"
    
    index_data = {
        "metadata": {
            "total_papers": len(text_files),
            "total_chunks": len(all_chunks),
            "papers": [f"{p[0]}.pdf" for p in text_files],
            "paper_details": paper_metadata,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "has_embeddings": False
        },
        "chunks": all_chunks
    }
    
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    index_size_mb = os.path.getsize(index_file) / 1024 / 1024
    
    print("=" * 80)
    print("✅ INDEXING COMPLETE")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Papers: {len(text_files)}")
    print(f"   Chunks: {len(all_chunks)}")
    print(f"   Index size: {index_size_mb:.2f} MB")
    print(f"\nNext: Test with test_rag_index.py")

if __name__ == "__main__":
    index_text_files()
