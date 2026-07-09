#!/usr/bin/env python3
"""
AI-assisted paraphrasing pipeline for Yoruba name meanings.
Generates multiple paraphrased variations while preserving semantic meaning and cultural accuracy.
"""

import argparse
import os
import sys
import json
import time
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_PARAPHRASING = _REPO_ROOT / "data" / "paraphrasing"
sys.path.insert(0, str(_REPO_ROOT / "rag"))


def _load_env_files() -> None:
    """Load repo-root .env so API keys work without manual export."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_REPO_ROOT / ".env")

# Try to import different LLM providers
OPENAI_AVAILABLE = False
ANTHROPIC_AVAILABLE = False



try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None  # type: ignore
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
# Type hints for optional imports
if TYPE_CHECKING:
    from anthropic import Anthropic as AnthropicType
    from openai import OpenAI as OpenAIType
else:
    AnthropicType = type(None)
    OpenAIType = type(None)

_load_env_files()

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
INPUT_FILE = str(_DATA_PARAPHRASING / "yoruba_names_ab_for_transformation.json")
OUTPUT_FILE = str(_DATA_PARAPHRASING / "yoruba_paraphrased_meanings.json")
BATCH_SIZE = 10  # Process names in batches
MAX_VARIATIONS = 5  # Number of paraphrased variations to generate
RAG_PREVIEW_MAX_CHARS = 350  # Store this many chars of RAG context per result for audit (0 = don't store)

# Model preferences (in order of preference)
# Options: "claude" (best quality), "gpt4" (high quality), "gpt35" (cost-effective)
# Default: Try Claude first, fallback to GPT-4, then GPT-3.5
PREFERRED_MODEL = os.environ.get("PARAPHRASE_MODEL", "auto")  # "auto", "claude", "gpt4", "gpt35"
PARAPHRASE_CLAUDE_MODEL = os.environ.get("PARAPHRASE_CLAUDE_MODEL", "claude-sonnet-5")

def get_llm_client():
    """Initialize and return the best available LLM client"""
    anthropic_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()

    # Auto mode: Try Claude first, then GPT-4, then GPT-3.5
    if PREFERRED_MODEL in ("auto", "claude"):
        if ANTHROPIC_AVAILABLE:
            if anthropic_key:
                return Anthropic(api_key=anthropic_key), "claude"
            if PREFERRED_MODEL == "claude":
                print("⚠️  ANTHROPIC_API_KEY not set, trying OpenAI...")
        elif anthropic_key:
            print("⚠️  ANTHROPIC_API_KEY is set but anthropic is not installed (pip install anthropic)")

    if OPENAI_AVAILABLE:
        if openai_key:
            client = OpenAI(api_key=openai_key)
            # Determine which OpenAI model to use
            if PREFERRED_MODEL == "gpt4":
                model_name = "gpt-4"
            elif PREFERRED_MODEL == "gpt35":
                model_name = "gpt-3.5-turbo"
            else:  # auto mode - default to GPT-4 for quality
                model_name = "gpt-4"
            return client, model_name
    elif openai_key:
        print("⚠️  OPENAI_API_KEY is set but openai is not installed (pip install openai)")

    hints = []
    if PREFERRED_MODEL in ("gpt4", "gpt35") and anthropic_key and not openai_key:
        hints.append(
            f"PARAPHRASE_MODEL={PREFERRED_MODEL!r} requires OPENAI_API_KEY "
            "(Anthropic key is ignored in this mode; use PARAPHRASE_MODEL=claude or auto)"
        )
    if PREFERRED_MODEL == "claude" and openai_key and not anthropic_key:
        hints.append("PARAPHRASE_MODEL='claude' requires ANTHROPIC_API_KEY")
    if not anthropic_key and not openai_key:
        hints.append(
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "(export in shell, inline on the command, or add to .env at repo root)"
        )
    if not ANTHROPIC_AVAILABLE and not OPENAI_AVAILABLE:
        hints.append("Install a provider SDK: pip install -r requirements/requirements_paraphrasing.txt")

    detail = "\n".join(f"  • {hint}" for hint in hints) if hints else ""
    raise ValueError(
        "No LLM API key found. Set either ANTHROPIC_API_KEY or OPENAI_API_KEY.\n"
        "For Claude: export ANTHROPIC_API_KEY='your-key'\n"
        "For OpenAI: export OPENAI_API_KEY='your-key'"
        + (f"\n\nLikely issue:\n{detail}" if detail else "")
    )

def create_paraphrase_prompt(name: str, meaning: str, cultural_context: str = "", language: str = "Yoruba") -> str:
    """Create a detailed prompt for paraphrasing Yoruba name meanings"""
    
    context_section = ""
    if cultural_context:
        context_section = f"""

Cultural Context from Research:
{cultural_context}

Use this cultural context to ensure your paraphrases are accurate and culturally authentic."""
    
    prompt = f"""You are an expert in {language} language and culture, specializing in name meanings and cultural significance.

Task: Paraphrase the meaning of the {language} name "{name}" while staying faithful to the original.

Original meaning: "{meaning}"{context_section}

Rules (follow strictly):
1. Preserve concrete, literal elements. If the original mentions specific things (e.g. crown, horse, birth, water), keep them in your paraphrases. Do not replace concrete words with symbolic interpretations (e.g. do not turn "horse" into "warrior" or "champion" unless the original clearly uses horse as a metaphor).
2. Stay within the same semantic scope. The paraphrases must express the same idea as the original—same subject, same relationship. For example, "birthed by the crown" can become "born of royalty" or "birthed by royalty," but not "crowned by the divine" or "divinely bestowed with the crown," which change the meaning.
3. Prefer clear, natural English. Avoid ornate or flowery language. Do not add ideas that are not in the original (e.g. do not introduce "divine" or "heavens" unless the original meaning includes that).
4. Vary only the wording: synonyms (crown/royalty, birthed/born of), rephrasing, simpler or slightly different sentence structure. Each variation should still be recognizably the same meaning.
5. Be suitable for a name database and culturally accurate. Use research context only to avoid errors; do not invent or embellish.

Generate {MAX_VARIATIONS} distinct paraphrased variations. Each must keep the essence and key elements of the original while using different wording.

Format your response as a JSON array of strings, where each string is one paraphrased variation.
Example format: ["variation 1", "variation 2", "variation 3", "variation 4", "variation 5"]

Only return the JSON array, no additional text."""
    
    return prompt

def paraphrase_with_claude(client, name: str, meaning: str, cultural_context: str = "") -> List[str]:
    """Generate paraphrases using Claude"""
    
    prompt = create_paraphrase_prompt(name, meaning, cultural_context)
    
    models_to_try = [
        PARAPHRASE_CLAUDE_MODEL,
        "claude-3-haiku-20240307",  # fallback if primary model unavailable
    ]
    
    last_error = None
    for model_name in models_to_try:
        try:
            message = client.messages.create(
                model=model_name,
                max_tokens=2000,
                temperature=0.3,  # Lower = more faithful to original, less ornate drift
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            if model_name != models_to_try[0]:
                print(f"     ⚠️  Using {model_name} (preferred model not available)")
            break
        except Exception as e:
            last_error = e
            if "not_found" in str(e).lower() and model_name != models_to_try[-1]:
                continue  # Try next model
            else:
                raise  # Re-raise if it's the last model or a different error
    
    try:
        response_text = message.content[0].text.strip()
        
        # Try to parse JSON from response
        # Sometimes Claude adds markdown formatting
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        variations = json.loads(response_text)
        
        if isinstance(variations, list) and len(variations) > 0:
            return variations[:MAX_VARIATIONS]
        else:
            return [variations] if isinstance(variations, str) else []
            
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parse error for {name}: {e}")
        # Try to extract variations from text
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        return lines[:MAX_VARIATIONS]
    except Exception as e:
        print(f"❌ Error paraphrasing {name} with Claude: {e}")
        return []

def paraphrase_with_openai(client, model: str, name: str, meaning: str, cultural_context: str = "") -> List[str]:
    """Generate paraphrases using OpenAI"""
    
    prompt = create_paraphrase_prompt(name, meaning, cultural_context)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Yoruba language and culture. Paraphrase name meanings faithfully: keep concrete terms (e.g. horse, crown), stay within the same semantic scope, use clear natural English, and do not add ornate or invented ideas. Only reword—same meaning, different phrasing."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        variations = json.loads(response_text)
        
        if isinstance(variations, list) and len(variations) > 0:
            return variations[:MAX_VARIATIONS]
        else:
            return [variations] if isinstance(variations, str) else []
            
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parse error for {name}: {e}")
        # Try to extract variations from text
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        return lines[:MAX_VARIATIONS]
    except Exception as e:
        print(f"❌ Error paraphrasing {name} with OpenAI: {e}")
        return []

def paraphrase_name(client, model_type: str, name: str, meaning: str, rag_service=None) -> Dict:
    """Paraphrase a single name meaning"""
    
    if not meaning or not meaning.strip():
        return {
            "name": name,
            "original_meaning": meaning,
            "variations": [],
            "error": "Empty meaning"
        }
    
    print(f"  📝 Paraphrasing: {name} - \"{meaning[:50]}...\"")
    
    # Get cultural context from RAG if available
    cultural_context = ""
    if rag_service:
        try:
            cultural_context = rag_service.get_cultural_context(name, meaning)
            if cultural_context:
                print(f"     📚 Found cultural context from research papers")
        except Exception as e:
            print(f"     ⚠️  RAG context error: {e}")
    
    start_time = time.time()
    
    llm_model = PARAPHRASE_CLAUDE_MODEL if model_type == "claude" else model_type
    if model_type == "claude":
        variations = paraphrase_with_claude(client, name, meaning, cultural_context)
    else:
        variations = paraphrase_with_openai(client, model_type, name, meaning, cultural_context)
    
    elapsed = time.time() - start_time
    
    result = {
        "name": name,
        "original_meaning": meaning,
        "variations": variations,
        "num_variations": len(variations),
        "processing_time": round(elapsed, 2),
        "llm_model": llm_model,
        "rag_context_used": bool(cultural_context and cultural_context.strip()),
    }
    if RAG_PREVIEW_MAX_CHARS and cultural_context:
        result["rag_context_preview"] = (cultural_context.strip()[:RAG_PREVIEW_MAX_CHARS] + ("..." if len(cultural_context) > RAG_PREVIEW_MAX_CHARS else ""))
    
    if variations:
        print(f"     ✅ Generated {len(variations)} variations ({elapsed:.2f}s)")
    else:
        print(f"     ⚠️  No variations generated")
        result["error"] = "Failed to generate variations"
    
    return result


def load_existing_paraphrases() -> Tuple[List[Dict], set]:
    """Load existing yoruba_paraphrased_meanings.json if present. Returns (list of existing results, set of name_strip lower)."""
    existing_results = []
    existing_names = set()
    if not os.path.exists(OUTPUT_FILE):
        return existing_results, existing_names
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        existing_results = data.get("results", [])
        existing_names = {str(r.get("name", "")).strip().lower() for r in existing_results if r.get("name")}
    except Exception:
        pass
    return existing_results, existing_names


def load_names_to_transform(name_filter: Optional[str] = None) -> List[Dict]:
    """Load names from the identification JSON file. Optionally filter to one NameStrip."""
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file '{INPUT_FILE}' not found. Run identify_yoruba_transformations.py first."
        )
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prioritize high priority names first
    high_priority = data.get("high_priority_names", [])
    all_names = data.get("all_names", [])
    
    # Start with high priority, then add others
    names_to_process = []
    
    # Add high priority names
    for name_data in high_priority:
        names_to_process.append({
            "name": name_data["name"],
            "meaning": name_data["meaning"],
            "priority": "high",
            "attribution": name_data.get("attribution", ""),
            "has_yorubanames": name_data.get("has_yorubanames_attribution", False)
        })
    
    # Add remaining names (avoid duplicates)
    processed_names = {item["name"] for item in names_to_process}
    for name_data in all_names:
        if name_data["name"] not in processed_names:
            names_to_process.append({
                "name": name_data["name"],
                "meaning": name_data["meaning"],
                "priority": "medium",
                "attribution": name_data.get("attribution", ""),
                "has_yorubanames": name_data.get("has_yorubanames_attribution", False)
            })
    
    if name_filter:
        key = name_filter.strip().lower()
        matches = [n for n in names_to_process if n["name"].strip().lower() == key]
        if not matches:
            raise ValueError(
                f"Name '{name_filter.strip()}' not found in {INPUT_FILE}. "
                "Run identify_yoruba_transformations.py to refresh the queue, "
                "or check NameStrip spelling/diacritics."
            )
        return matches
    
    return names_to_process

def process_batch(client, model_type: str, batch: List[Dict], results: List[Dict], rag_service=None):
    """Process a batch of names"""
    
    for name_data in batch:
        result = paraphrase_name(
            client,
            model_type,
            name_data["name"],
            name_data["meaning"],
            rag_service
        )
        
        # Add metadata
        result["priority"] = name_data.get("priority", "medium")
        result["attribution"] = name_data.get("attribution", "")
        result["has_yorubanames"] = name_data.get("has_yorubanames", False)
        
        results.append(result)
        
        # Rate limiting - be nice to the API
        time.sleep(0.5)  # Small delay between requests

def main():
    """Main paraphrasing pipeline"""
    parser = argparse.ArgumentParser(description="Paraphrase Yoruba name meanings.")
    parser.add_argument(
        "--rephrase-all",
        action="store_true",
        help="Re-run for all names from the identification file (ignore existing JSON; overwrite output).",
    )
    parser.add_argument(
        "--name",
        type=str,
        metavar="NAME",
        help="Process only this Yoruba NameStrip (case-insensitive), e.g. Folasade.",
    )
    parser.add_argument(
        "--rephrase",
        action="store_true",
        help="With --name: regenerate even if this name is already in the output JSON.",
    )
    args = parser.parse_args()
    rephrase_all = args.rephrase_all
    single_name = (args.name or "").strip() or None
    rephrase_single = args.rephrase

    print("=" * 80)
    print("🔄 YORUBA NAME MEANING PARAPHRASING PIPELINE")
    if single_name:
        print(f"   (Single name: {single_name})")
    if rephrase_all:
        print("   (Re-phrase all: ignoring existing paraphrases, will overwrite output)")
    elif rephrase_single and single_name:
        print(f"   (Re-phrase: will replace existing entry for {single_name})")
    print("=" * 80)
    print()

    # Initialize LLM client
    try:
        client, model_type = get_llm_client()
        model_label = PARAPHRASE_CLAUDE_MODEL if model_type == "claude" else model_type
        print(f"✅ Using {model_label} model")
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Initialize RAG service if available (same retrieval path as /insights)
    rag_service = None
    try:
        from rag_service import get_rag_service_for_dataset_language
        rag_service = get_rag_service_for_dataset_language(
            "Yoruba", quiet=False, text_search_only=True
        )
        if rag_service is None:
            raise FileNotFoundError("Yoruba RAG index not found")
        print(
            "✅ RAG service loaded — insights-style retrieval "
            "(morpheme reranking, query expansion, per-paper diversity)\n"
        )
    except (ImportError, FileNotFoundError) as e:
        print(f"⚠️  RAG service not available: {e}")
        print(f"   Continuing without research paper context...\n")
    
    # Load names to transform
    print(f"📥 Loading names from {INPUT_FILE}...")
    try:
        names_to_process = load_names_to_transform(name_filter=single_name)
        if single_name:
            print(f"✅ Found {names_to_process[0]['name']} in identification file")
        else:
            print(f"✅ Loaded {len(names_to_process)} names from identification file")
    except Exception as e:
        print(f"❌ Error loading names: {e}")
        return

    # Resume: skip names already in output file (unless --rephrase-all or --name --rephrase)
    if rephrase_all:
        existing_results = []
        existing_names = set()
        print(f"   📌 Re-phrase all: processing all {len(names_to_process)} names (output will be overwritten)\n")
    else:
        existing_results, existing_names = load_existing_paraphrases()
        if single_name:
            key = single_name.lower()
            if key in existing_names and not rephrase_single:
                print(
                    f"✅ {single_name} is already in {OUTPUT_FILE}. "
                    "Use --rephrase to regenerate, or paraphrase_lookup.py to review."
                )
                return
        if existing_names and not (single_name and rephrase_single):
            before = len(names_to_process)
            names_to_process = [n for n in names_to_process if n["name"].strip().lower() not in existing_names]
            skipped = before - len(names_to_process)
            if skipped:
                print(f"   📌 Resume: {skipped} names already in {OUTPUT_FILE} (skipped); {len(names_to_process)} remaining\n")
        elif not existing_names:
            print(f"   (No existing output file — processing from scratch)\n")

    if not names_to_process:
        if single_name:
            print(f"❌ Nothing to process for {single_name}.")
        else:
            print("✅ Nothing left to process. All names from the identification file are already paraphrased.")
        return

    # Ask user how many to process (batch mode only)
    print(f"📊 Remaining names to process: {len(names_to_process)}")
    print(f"   • High priority: {sum(1 for n in names_to_process if n.get('priority') == 'high')}")
    print(f"   • Medium priority: {sum(1 for n in names_to_process if n.get('priority') == 'medium')}")
    print()
    
    if single_name:
        limit = len(names_to_process)
    else:
        try:
            limit_input = input(f"How many names to process? (Enter number or 'all' for {len(names_to_process)}): ").strip()
            if limit_input.lower() == 'all':
                limit = len(names_to_process)
            else:
                limit = int(limit_input)
            limit = min(limit, len(names_to_process))
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input or cancelled")
            return
    
    names_to_process = names_to_process[:limit]
    
    print(f"\n🚀 Processing {len(names_to_process)} names in batches of {BATCH_SIZE}...\n")
    
    # Process in batches
    results = []
    total_batches = (len(names_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(names_to_process))
        batch = names_to_process[start_idx:end_idx]
        
        print(f"📦 Batch {batch_num + 1}/{total_batches} ({len(batch)} names)")
        process_batch(client, model_type, batch, results, rag_service)
        print()
    
    # Merge with existing results (resume: keep existing, add new). With --rephrase-all, only this run's results.
    if rephrase_all:
        merged_results = results
    else:
        new_names_lower = {r.get("name", "").strip().lower() for r in results if r.get("name")}
        merged_results = [r for r in existing_results if r.get("name", "").strip().lower() not in new_names_lower]
        merged_results.extend(results)

    # Save results
    print(f"💾 Saving results to {OUTPUT_FILE}...")
    
    rag_used = rag_service is not None
    names_with_rag_context = sum(1 for r in results if r.get("rag_context_used"))
    output_data = {
        "summary": {
            "total_in_file": len(merged_results),
            "processed_this_run": len(results),
            "successful_this_run": sum(1 for r in results if r.get("variations")),
            "failed_this_run": sum(1 for r in results if not r.get("variations")),
            "model_used": (
                PARAPHRASE_CLAUDE_MODEL if model_type == "claude" else model_type
            ),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rag_used": rag_used,
            "rag_names_this_run": names_with_rag_context if rag_used else None,
            "rag_note": (
                "Paraphrases generated with cultural context from Yoruba research papers (RAG)."
                if rag_used
                else "Paraphrases generated without RAG; no research paper context was used."
            ),
        },
        "results": merged_results
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to {OUTPUT_FILE} ({len(merged_results)} names total, +{len(results)} this run)")
    if rag_used:
        print(f"   📚 RAG: {names_with_rag_context}/{len(results)} names had research-paper context injected (see summary.rag_names_this_run and per-result rag_context_used)")
    else:
        print(f"   ⚠️  Summary includes rag_used: false (run with RAG available to use research paper context)")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Processed this run: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.get('variations'))}")
    print(f"Failed: {sum(1 for r in results if not r.get('variations'))}")
    print(f"Total names in file: {len(merged_results)}")
    if rag_used and names_with_rag_context is not None:
        print(f"Names with RAG context this run: {names_with_rag_context}/{len(results)}")
    print(f"\nNext steps:")
    print(f"1. Review {OUTPUT_FILE} to see all paraphrased variations")
    print(f"2. Use the review interface to select best variations")
    print(f"3. Update the dataset with selected paraphrases")

if __name__ == "__main__":
    main()

