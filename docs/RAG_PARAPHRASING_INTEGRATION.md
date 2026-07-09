# RAG Integration with Paraphrasing Pipeline

## Overview

The paraphrasing pipeline (`transform_yoruba_meanings.py`) **already has RAG integrated** to enhance paraphrasing with cultural context from research papers.

## How It Works

### 1. Automatic RAG Initialization

When you run `transform_yoruba_meanings.py`, it automatically:

```python
# Initialize RAG service if available
rag_service = None
try:
    from rag_service import YorubaRAGService
    rag_service = YorubaRAGService()
    print("✅ RAG service loaded - will use research papers for cultural context")
except (ImportError, FileNotFoundError) as e:
    print("⚠️  RAG service not available")
    # Continues without RAG (still works, just no cultural context)
```

### 2. Cultural Context Retrieval

For each name being paraphrased, the pipeline:

1. **Uses the same retrieval path as `/insights`** via `get_insights_excerpts()` (name-aware reranking, query expansion, per-paper diversity)
2. **Extracts morphemes** from the name (ọlá, ade, oluwa, etc.) and surfaces morpheme-focused excerpts when found
3. **Includes context** in the LLM prompt

```python
# Get cultural context from RAG if available (delegates to get_insights_excerpts)
cultural_context = ""
if rag_service:
    cultural_context = rag_service.get_cultural_context(name, meaning)
    if cultural_context:
        print("📚 Found cultural context from research papers")
```

**Paraphrase vs insights:** Both call `rag/rag_service.py`. Insights passes excerpts to Claude for synthesis; paraphrase formats them as "Morpheme Analysis" + "General Cultural Context" in the paraphrase prompt. Retrieval is shared; only the downstream use differs.

### 3. Enhanced Prompting

The cultural context is included in the paraphrasing prompt:

```python
def create_paraphrase_prompt(name: str, meaning: str, cultural_context: str = ""):
    if cultural_context:
        context_section = f"""
Cultural Context from Research:
{cultural_context}

Use this cultural context to ensure your paraphrases are accurate and culturally authentic."""
```

This means the LLM receives:
- The original name and meaning
- **Cultural context from research papers**
- **Morpheme-specific information** (if morphemes found)
- Instructions to use this context for authentic paraphrases

## Benefits

### Without RAG:
- Paraphrases based on general LLM knowledge
- May miss cultural nuances
- Less context about morpheme meanings

### With RAG:
- ✅ **Cultural authenticity**: Uses actual research on Yoruba names
- ✅ **Morpheme insights**: Understands deeper meanings (e.g., ọlá = wealth/honor)
- ✅ **Contextual accuracy**: References real academic sources
- ✅ **Better paraphrases**: More nuanced and culturally appropriate

## Usage

### Single name: Folasade (end-to-end)

From repo root, with `ANTHROPIC_API_KEY` set:

```bash
# 1. (Optional) Refresh identification queue — Folasade is already in the file
python3 scripts/paraphrasing/identify_yoruba_transformations.py

# 2. Paraphrase one name (Claude Sonnet 5 by default)
export ANTHROPIC_API_KEY='your-key'
export PARAPHRASE_MODEL='claude'   # optional; auto also picks Claude when key is set
python3 scripts/paraphrasing/transform_yoruba_meanings.py --name Folasade

# If Folasade is already in output, regenerate:
python3 scripts/paraphrasing/transform_yoruba_meanings.py --name Folasade --rephrase
```

**View results (pick one):**

```bash
# CLI — print variations from JSON
python3 -c "
import json
from pathlib import Path
p = Path('data/paraphrasing/yoruba_paraphrased_meanings.json')
data = json.loads(p.read_text(encoding='utf-8'))
r = next(x for x in data['results'] if x['name'].lower() == 'folasade')
print('Original:', r['original_meaning'])
print('RAG used:', r.get('rag_context_used'))
if r.get('rag_context_preview'):
    print('RAG preview:', r['rag_context_preview'][:200], '...')
for i, v in enumerate(r.get('variations') or [], 1):
    print(f'{i}. {v}')
"

# Gradio lookup UI
python3 scripts/paraphrasing/paraphrase_lookup.py
# Open http://127.0.0.1:7860 and search "Folasade"
```

### Running the full pipeline

Simply run the paraphrasing pipeline — RAG is integrated automatically:

```bash
# Set your API key
export ANTHROPIC_API_KEY='your-key'
export PARAPHRASE_MODEL='claude'  # uses claude-sonnet-5 (override with PARAPHRASE_CLAUDE_MODEL)

# Run the pipeline
python3 scripts/paraphrasing/transform_yoruba_meanings.py
```

The pipeline will:
1. ✅ Load RAG service automatically
2. ✅ Fetch cultural context for each name
3. ✅ Include context in paraphrasing prompts
4. ✅ Generate culturally-enhanced paraphrases

### What You'll See

When RAG is working, you'll see:
```
✅ RAG service loaded - will use research papers for cultural context

📝 Paraphrasing: Abiola - "Born into wealth/success/nobility...."
     📚 Found cultural context from research papers
     ✅ Generated 5 variations (2.3s)
```

### Without RAG

If RAG isn't available, it still works:
```
⚠️  RAG service not available: ...
   Continuing without research paper context...

📝 Paraphrasing: Abiola - "Born into wealth/success/nobility...."
     ✅ Generated 5 variations (2.1s)
```

### Output JSON: RAG flag

The saved file `yoruba_paraphrased_meanings.json` includes a **summary** that records whether RAG was used for that run:

- **`rag_used`** (boolean): `true` if the pipeline had the RAG service loaded and used research paper context for each name; `false` otherwise.
- **`rag_note`** (string): Short note, e.g. *"Paraphrases generated with cultural context from Yoruba research papers (RAG)."* or *"Paraphrases generated without RAG; no research paper context was used."*

When you open the file or use it for display (e.g. Option B in the API/Gradio), check `summary.rag_used` to know whether the paraphrases in that file were generated with cultural context.

## Example: How RAG Enhances Paraphrasing

### Name: "Abiola" (meaning: "Born into wealth/success/nobility")

**Without RAG:**
- Paraphrases: "Born into prosperity", "Came from wealth", etc.
- Based on general understanding

**With RAG:**
- **Morpheme found**: "ola" (ọlá = wealth/honor)
- **Cultural context retrieved**: 
  - Information about ọlá morpheme from research papers
  - Context about wealth/honor in Yoruba culture
  - Traditional naming practices
- **Enhanced paraphrases**: 
  - "Came into existence amidst affluence and prominence, reflecting the Yoruba tradition of naming children based on their birth circumstances and the family's social standing."
  - More culturally nuanced and authentic

## Testing the Integration

Test RAG integration with paraphrasing:

```bash
python3 test_rag_paraphrasing_integration.py
```

This will:
- Show morpheme extraction
- Display cultural context retrieved
- Generate paraphrases with RAG context
- Demonstrate the enhancement

## Current Status

✅ **RAG Integration**: Complete and working
✅ **Automatic Loading**: RAG loads automatically if available
✅ **Cultural Context**: Retrieved for each name
✅ **Morpheme Analysis**: Extracted and used
✅ **Fallback**: Works without RAG (just less context)

## Next Steps

The paraphrasing pipeline is ready to use! Just run:

```bash
python3 transform_yoruba_meanings.py
```

It will automatically use RAG for enhanced, culturally-authentic paraphrases.


