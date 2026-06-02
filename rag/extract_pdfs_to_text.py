#!/usr/bin/env python3
"""
Extract text from selected Yoruba research papers to .txt files.
Processes one PDF at a time, page-by-page, to minimize memory usage.
"""

import os
from pathlib import Path

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

# Selected papers for morphological patterns
SELECTED_PAPERS = [
    "Construction_Morphology_in_Yoruba_names_Schemas_an.pdf",
    "Semantics_Yoruba_Names.pdf",
    "Yoruba_Traditional_Names_and_the_Transmi.pdf",
    "Yoruba_Praise_Names.pdf"
]

RESEARCH_PAPERS_DIR = "Research papers"
EXTRACTED_DIR = "Research papers/extracted"

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF page-by-page"""
    text = ""
    
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    # Progress indicator
                    if i % 10 == 0 or i == total_pages:
                        print(f"      Page {i}/{total_pages}...")
            return text
        except Exception as e:
            print(f"⚠️  pdfplumber error: {e}")
            return ""
    
    if PDF2_AVAILABLE:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                for i, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    if i % 10 == 0 or i == total_pages:
                        print(f"      Page {i}/{total_pages}...")
            return text
        except Exception as e:
            print(f"⚠️  PyPDF2 error: {e}")
            return ""
    
    raise ImportError("No PDF parsing library available")

def extract_pdfs_to_text():
    """Extract all selected PDFs to text files"""
    
    print("=" * 80)
    print("📄 EXTRACTING PDFs TO TEXT FILES")
    print("=" * 80)
    print()
    
    # Check for PDF libraries
    if not PDFPLUMBER_AVAILABLE and not PDF2_AVAILABLE:
        print("❌ No PDF parsing library available")
        print("   Install: pip install pdfplumber")
        return
    
    papers_dir = Path(RESEARCH_PAPERS_DIR)
    if not papers_dir.exists():
        print(f"❌ Directory '{RESEARCH_PAPERS_DIR}' not found")
        return
    
    # Create output directory
    extracted_path = Path(EXTRACTED_DIR)
    extracted_path.mkdir(parents=True, exist_ok=True)
    
    # Find selected papers
    yoruba_papers = []
    for paper_name in SELECTED_PAPERS:
        paper_path = papers_dir / paper_name
        if paper_path.exists():
            yoruba_papers.append(paper_path)
        else:
            print(f"⚠️  Paper not found: {paper_name}")
    
    if not yoruba_papers:
        print(f"❌ No selected papers found")
        return
    
    print(f"📄 Processing {len(yoruba_papers)} papers:")
    for paper in yoruba_papers:
        print(f"   • {paper.name}")
    print()
    
    # Process one paper at a time
    successful = 0
    failed = 0
    
    for paper_idx, paper_path in enumerate(yoruba_papers, 1):
        print(f"📖 ({paper_idx}/{len(yoruba_papers)}) {paper_path.name}")
        
        try:
            # Extract text page-by-page
            text = extract_text_from_pdf(str(paper_path))
            
            if not text or len(text.strip()) < 100:
                print(f"   ⚠️  Little or no text extracted")
                failed += 1
                continue
            
            # Save to text file
            txt_filename = paper_path.stem + ".txt"
            txt_path = extracted_path / txt_filename
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            file_size = os.path.getsize(txt_path) / 1024  # KB
            print(f"   ✅ Extracted {len(text)} chars → {txt_filename} ({file_size:.1f} KB)")
            successful += 1
            
            # Clear memory
            del text
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
        
        print()
    
    # Summary
    print("=" * 80)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Successful: {successful}/{len(yoruba_papers)}")
    print(f"   Failed: {failed}")
    print(f"\n📁 Text files saved to: {extracted_path}")
    print(f"\nNext step: Run index_text_files.py to create the index")

if __name__ == "__main__":
    extract_pdfs_to_text()


