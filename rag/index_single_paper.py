#!/usr/bin/env python3
"""Test indexing a single PDF to debug memory issues"""

import pdfplumber
import json
from pathlib import Path

paper = Path("Research papers/Yoruba_Praise_Names.pdf")
print(f"Testing with: {paper.name}")

try:
    with pdfplumber.open(str(paper)) as pdf:
        print(f"Pages: {len(pdf.pages)}")
        text = ""
        for i, page in enumerate(pdf.pages[:3]):  # Just first 3 pages
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            print(f"Page {i+1}: {len(page_text) if page_text else 0} chars")
        print(f"Total text: {len(text)} chars")
        print("✅ Success!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

