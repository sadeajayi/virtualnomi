#!/usr/bin/env python3
"""
Search Yoruba research papers for morpheme analysis and semantic mappings.
Specifically looks for deeper cultural meanings of morphemes like ọlá, ade, oluwa, etc.
"""

import os
import pdfplumber
import re
from pathlib import Path
from collections import defaultdict

RESEARCH_PAPERS_DIR = "Research papers"

# Common Yoruba name morphemes to search for
MORPHEMES_TO_SEARCH = [
    "ọlá", "ola", "ọla",
    "ade", "adé",
    "oluwa", "olúwa", "olu",
    "tunde", "túndé",
    "baba", "bàbá",
    "iya", "ìyá", "yeye",
    "ori", "orí",
    "ife", "ifẹ",
    "tolu", "tólú"
]

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"  ⚠️  Error extracting from {pdf_path.name}: {e}")
        return ""

def find_morpheme_context(text, morpheme):
    """Find sentences/paragraphs discussing a morpheme"""
    # Normalize morpheme for search (handle variations)
    search_patterns = [
        morpheme,
        morpheme.replace("ọ", "o").replace("á", "a"),
        morpheme.replace("o", "ọ").replace("a", "á"),
    ]
    
    contexts = []
    
    # Split into sentences
    sentences = re.split(r'[.!?]\s+', text)
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Check if sentence mentions morpheme and has semantic/meaning words
        if any(pattern.lower() in sentence_lower for pattern in search_patterns):
            # Look for semantic discussion keywords
            semantic_keywords = [
                "mean", "meaning", "signify", "denote", "represent", 
                "understand", "interpret", "cultural", "tradition",
                "wealth", "honour", "honor", "prestige", "status",
                "morpheme", "semantic", "etymology", "origin"
            ]
            
            if any(keyword in sentence_lower for keyword in semantic_keywords):
                contexts.append(sentence.strip())
    
    return contexts

def search_papers_for_morphemes():
    """Search all Yoruba papers for morpheme analysis"""
    
    print("=" * 80)
    print("🔍 SEARCHING YORUBA RESEARCH PAPERS FOR MORPHEME ANALYSIS")
    print("=" * 80)
    print()
    
    papers_dir = Path(RESEARCH_PAPERS_DIR)
    if not papers_dir.exists():
        print(f"❌ Directory '{RESEARCH_PAPERS_DIR}' not found")
        return
    
    # Find Yoruba papers
    yoruba_papers = []
    for pdf_file in papers_dir.glob("*.pdf"):
        if "yoruba" in pdf_file.name.lower():
            yoruba_papers.append(pdf_file)
    
    print(f"📄 Found {len(yoruba_papers)} Yoruba research papers\n")
    
    # Store findings by morpheme
    morpheme_findings = defaultdict(lambda: defaultdict(list))
    
    # Process each paper
    for paper_path in yoruba_papers:
        print(f"📖 Processing: {paper_path.name}")
        
        text = extract_text_from_pdf(str(paper_path))
        
        if not text:
            print(f"  ⚠️  Could not extract text")
            continue
        
        print(f"  ✅ Extracted {len(text)} characters")
        
        # Search for each morpheme
        for morpheme in MORPHEMES_TO_SEARCH:
            contexts = find_morpheme_context(text, morpheme)
            if contexts:
                morpheme_findings[morpheme][paper_path.name].extend(contexts)
                print(f"    ✅ Found {len(contexts)} contexts for '{morpheme}'")
        
        print()
    
    # Display findings
    print("=" * 80)
    print("📊 MORPHEME ANALYSIS FINDINGS")
    print("=" * 80)
    print()
    
    for morpheme in MORPHEMES_TO_SEARCH:
        if morpheme in morpheme_findings:
            print(f"\n🔤 Morpheme: '{morpheme}'")
            print("-" * 80)
            
            for paper_name, contexts in morpheme_findings[morpheme].items():
                print(f"\n  📄 From: {paper_name}")
                for i, context in enumerate(contexts[:5], 1):  # Show first 5
                    print(f"    {i}. {context[:300]}...")
                if len(contexts) > 5:
                    print(f"    ... and {len(contexts) - 5} more contexts")
    
    # Save findings to JSON
    output_file = "morpheme_analysis_findings.json"
    
    # Convert to serializable format
    findings_dict = {}
    for morpheme, papers in morpheme_findings.items():
        findings_dict[morpheme] = {
            paper: contexts
            for paper, contexts in papers.items()
        }
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "morphemes_searched": MORPHEMES_TO_SEARCH,
            "findings": findings_dict
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Findings saved to {output_file}")
    
    # Special focus on ọlá
    print("\n" + "=" * 80)
    print("🔍 DEEP DIVE: ọlá MORPHEME")
    print("=" * 80)
    
    ola_variants = ["ọlá", "ola", "ọla"]
    ola_contexts = []
    
    for paper_path in yoruba_papers:
        text = extract_text_from_pdf(str(paper_path))
        if text:
            for variant in ola_variants:
                contexts = find_morpheme_context(text, variant)
                for ctx in contexts:
                    if ctx not in ola_contexts:
                        ola_contexts.append(ctx)
    
    if ola_contexts:
        print(f"\nFound {len(ola_contexts)} relevant contexts about ọlá:\n")
        for i, ctx in enumerate(ola_contexts[:10], 1):
            print(f"{i}. {ctx}")
            print()
    else:
        print("\n⚠️  No detailed morpheme analysis found for ọlá")
        print("   Papers may discuss names containing ọlá but not the morpheme itself")

if __name__ == "__main__":
    search_papers_for_morphemes()


