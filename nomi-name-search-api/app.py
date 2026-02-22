#!/usr/bin/env python3
"""
FastAPI backend for Nomi Name Search
Exposes semantic search functionality as REST API for frontend use
"""

import io
import math
import os
import re as _re
import html as html_mod
import urllib.request as _urllib
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
import numpy as np
from pinecone import Pinecone
from huggingface_hub import hf_hub_download
import json

# Try to import datasets, but make it optional
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    # Fallback: use requests to load data directly
    import requests

# Optional OpenAI for hybrid embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="Nomi Name Search API", version="1.0.0")

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy loading
ds = None
pc = None
index = None
model = None
openai_client = None
_stories_data = None
_stories_lookup = None
_dataset_lookup = None
_paraphrase_lookup = None

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PARAPHRASE_FILE = _REPO_ROOT / "data" / "paraphrasing" / "yoruba_paraphrased_meanings.json"


def get_paraphrase_lookup() -> Dict[str, str]:
    """Load yoruba_paraphrased_meanings.json and return name_strip (lower) -> first variation."""
    global _paraphrase_lookup
    if _paraphrase_lookup is not None:
        return _paraphrase_lookup
    _paraphrase_lookup = {}
    if not _PARAPHRASE_FILE.exists():
        return _paraphrase_lookup
    try:
        with open(_PARAPHRASE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("results", []):
            name = (item.get("name") or "").strip().lower()
            variations = item.get("variations") or []
            if name and variations:
                _paraphrase_lookup[name] = (variations[0] or "").strip()
    except Exception:
        pass
    return _paraphrase_lookup


def display_meaning_for_result(language: str, name_strip: str, canonical_meaning: str) -> str:
    """For Yoruba names, use first paraphrase when available; else canonical."""
    if language != "Yoruba":
        return canonical_meaning
    lookup = get_paraphrase_lookup()
    key = (name_strip or "").strip().lower()
    return lookup.get(key, canonical_meaning)


# Response models
class NameResult(BaseModel):
    name: str
    name_strip: str
    meaning: str
    language: str
    phonetic_spelling: Optional[str] = None
    pronunciation_url: Optional[str] = None
    pronunciation_by: Optional[str] = None
    validated_by: Optional[str] = None
    validation_status: Optional[str] = None
    cultural_context: Optional[str] = None
    themes: Optional[List[str]] = None
    story: Optional[Dict[str, Any]] = None
    score: float
    audio_url: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[NameResult]
    total: int
    language_filter: Optional[str] = None
    stories_only: bool = False

class NameCardData(BaseModel):
    name: str
    name_strip: str
    language: str
    meaning: str
    phonetic_spelling: Optional[str] = None
    audio_url: Optional[str] = None
    pronunciation_by: Optional[str] = None
    cultural_context: Optional[str] = None
    themes: Optional[List[str]] = None
    story: Optional[Dict[str, Any]] = None

class NameLookupResponse(BaseModel):
    name_strip: str
    results: List[NameCardData]
    total: int

def load_dataset_fallback():
    """Load dataset using HuggingFace Hub API directly (lighter weight)"""
    global ds
    if ds is None:
        if DATASETS_AVAILABLE:
            print("Loading dataset using datasets library...")
            ds = load_dataset("nomi-stories/nomi-names", split="train", token=HF_TOKEN)
        else:
            print("Loading dataset using HuggingFace Hub API...")
            # Use HuggingFace Hub API to download parquet file
            try:
                parquet_path = hf_hub_download(
                    repo_id="nomi-stories/nomi-names",
                    repo_type="dataset",
                    filename="data/train-00000-of-00001.parquet",
                    token=HF_TOKEN
                )
                # Read parquet using pyarrow (lighter than pandas)
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(parquet_path)
                    # Convert to list of dicts
                    ds = [dict(zip(table.column_names, row)) for row in table.to_pylist()]
                except ImportError:
                    # Fallback to pandas if pyarrow not available
                    import pandas as pd
                    df = pd.read_parquet(parquet_path)
                    ds = df.to_dict('records')
            except Exception as e:
                print(f"Error loading dataset: {e}")
                ds = []
    return ds

# Initialize components
def initialize_components():
    """Lazy initialization of all components"""
    global ds, pc, index, model, openai_client
    
    if ds is None:
        load_dataset_fallback()
    
    if pc is None and PINECONE_API_KEY:
        print("Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("nomi-name-encoder")
    
    if model is None:
        print("Loading embedding model...")
        from sentence_transformers import SentenceTransformer  # lazy: torch is ~400MB
        model = SentenceTransformer("fajayi/nomi-name-encoder")
    
    if openai_client is None and OPENAI_AVAILABLE and OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

def _build_lookup_from_rows(rows):
    """Build (name_strip, language) -> row dict from an iterable of row dicts."""
    lookup = {}
    for row in rows:
        name_strip = str(row.get("NameStrip", "")).strip()
        language = str(row.get("Language", "")).strip()
        if name_strip and language:
            key = (name_strip, language)
            if key not in lookup or row.get("Audio Pronunciation"):
                lookup[key] = row
    return lookup


def get_dataset_lookup():
    """Create O(1) lookup dictionary for dataset.

    Uses a sentinel of None = not yet loaded, {} = loaded but empty.
    Only caches non-empty result so a transient failure is retried next request.
    """
    global _dataset_lookup
    if _dataset_lookup is not None:
        return _dataset_lookup

    dataset = load_dataset_fallback()
    lookup = {}

    if isinstance(dataset, list):
        # Already a list of dicts (pyarrow/pandas fallback path)
        print(f"[dataset_lookup] list path: {len(dataset)} rows")
        lookup = _build_lookup_from_rows(dataset)
    else:
        # datasets.Dataset — try multiple access strategies to avoid torchcodec
        # Strategy 1: dataset.data.to_pydict() (PyArrow table, avoids audio decoder)
        try:
            table_dict = dataset.data.to_pydict()
            n = len(next(iter(table_dict.values()), []))
            rows = [{col: table_dict[col][i] for col in table_dict} for i in range(n)]
            lookup = _build_lookup_from_rows(rows)
            print(f"[dataset_lookup] strategy 1 (data.to_pydict): {len(lookup)} entries")
        except Exception as e1:
            print(f"[dataset_lookup] strategy 1 failed: {e1}")

        # Strategy 2: dataset._data.to_pydict() (private attr, more stable across versions)
        if not lookup:
            try:
                table_dict = dataset._data.to_pydict()
                n = len(next(iter(table_dict.values()), []))
                rows = [{col: table_dict[col][i] for col in table_dict} for i in range(n)]
                lookup = _build_lookup_from_rows(rows)
                print(f"[dataset_lookup] strategy 2 (_data.to_pydict): {len(lookup)} entries")
            except Exception as e2:
                print(f"[dataset_lookup] strategy 2 failed: {e2}")

        # Strategy 3: pandas (bypasses audio decoder entirely)
        if not lookup:
            try:
                df = dataset.to_pandas()
                lookup = _build_lookup_from_rows(df.to_dict("records"))
                print(f"[dataset_lookup] strategy 3 (to_pandas): {len(lookup)} entries")
            except Exception as e3:
                print(f"[dataset_lookup] strategy 3 failed: {e3}")

        # Strategy 4: strip audio columns then iterate (last resort, loses audio data)
        if not lookup:
            try:
                audio_cols = [c for c in dataset.column_names if "audio" in c.lower() or "pronunciation" in c.lower()]
                stripped = dataset.remove_columns(audio_cols) if audio_cols else dataset
                lookup = _build_lookup_from_rows(stripped)
                print(f"[dataset_lookup] strategy 4 (remove_columns+iterate): {len(lookup)} entries")
            except Exception as e4:
                print(f"[dataset_lookup] strategy 4 failed: {e4}")

    if lookup:
        _dataset_lookup = lookup
    else:
        print("[dataset_lookup] WARNING: all strategies failed or returned empty — will retry next request")

    return lookup

def get_name_metadata_from_dataset(name_strip: str, language: str) -> Dict[str, Any]:
    """Get metadata for a name from the dataset"""
    dataset_lookup = get_dataset_lookup()
    match = dataset_lookup.get((name_strip.strip(), language.strip()))
    if not match:
        return {}
    
    audio_url = ""
    audio_val = match.get("Audio Pronunciation")
    if audio_val:
        try:
            if isinstance(audio_val, dict):
                # Audio bytes are embedded in the parquet — serve via local endpoint
                if audio_val.get("bytes"):
                    audio_url = f"/audio/{name_strip}?language={language}"
        except Exception:
            pass
    
    return {
        "phonetic_spelling": match.get("Phonetic Spelling", ""),
        "pronunciation_url": audio_url,
        "pronunciation_by": match.get("pronunciation_by", ""),
        "validated_by": match.get("Validated_By", ""),
        "validation_status": match.get("Validation_Status", ""),
        "audio_url": audio_url,
    }

def load_stories_data():
    """Load stories from HuggingFace dataset"""
    global _stories_data, _stories_lookup
    if _stories_data is None:
        try:
            if DATASETS_AVAILABLE:
                stories_ds = load_dataset("nomi-stories/nomi-stories", split="train", token=HF_TOKEN)
            else:
                # Use HuggingFace Hub API
                stories_path = hf_hub_download(
                    repo_id="nomi-stories/nomi-stories",
                    repo_type="dataset",
                    filename="data/train-00000-of-00001.parquet",
                    token=HF_TOKEN
                )
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(stories_path)
                    stories_ds = [dict(zip(table.column_names, row)) for row in table.to_pylist()]
                except ImportError:
                    import pandas as pd
                    df = pd.read_parquet(stories_path)
                    stories_ds = df.to_dict('records')
            
            _stories_data = {}
            _stories_lookup = {}
            for row in stories_ds:
                if row.get('story') and row.get('story', {}).get('status') == 'published':
                    name_strip = str(row.get('name_strip', '')).strip()
                    language = str(row.get('language', '')).strip()
                    if name_strip and language:
                        key_tuple = (name_strip, language)
                        story = row['story']
                        _stories_lookup[key_tuple] = story
        except Exception:
            _stories_data = {}
            _stories_lookup = {}
    return _stories_data

def get_story_from_dataset(name_strip: str, language: str) -> Dict[str, Any]:
    """Get story for a name"""
    global _stories_lookup
    if _stories_lookup is None:
        load_stories_data()
    key = (str(name_strip).strip(), str(language).strip())
    return _stories_lookup.get(key, {})

def is_complex_query(query: str) -> bool:
    """Determine if query is complex (multi-word phrases)"""
    words = query.strip().split()
    return len(words) > 2

def query_with_sentence_transformer(query: str, lang_filter: str) -> List[Dict[str, Any]]:
    """Query using Sentence Transformer and Pinecone"""
    global model, index, ds
    
    if not model or not index:
        raise HTTPException(status_code=500, detail="Search service not initialized")
    
    query_embedding = model.encode([query])[0].tolist()
    
    filter_dict = {}
    if lang_filter != "All":
        filter_dict = {"language": lang_filter}
    
    results = index.query(
        vector=query_embedding,
        top_k=20,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )
    
    hits = []
    dataset_lookup = get_dataset_lookup()
    
    for match in results.matches:
        metadata = match.metadata
        name_strip = metadata.get("name_strip", "")
        language = metadata.get("language", "")
        
        row = dataset_lookup.get((name_strip, language))
        if not row:
            continue
        
        name_metadata = get_name_metadata_from_dataset(name_strip, language)
        story = get_story_from_dataset(name_strip, language)
        
        name_data = {
            "name": row.get("Name", name_strip),
            "name_strip": name_strip,
            "meaning": row.get("Meaning", ""),
            "language": language,
            "cultural_context": row.get("cultural_context", ""),
            "themes": row.get("themes", []),
        }
        
        result = build_result_dict(name_data, name_metadata, story, match.score)
        hits.append(result)
    
    return hits

def query_with_openai(query: str, lang_filter: str) -> List[Dict[str, Any]]:
    """Query using OpenAI embeddings (for complex queries)"""
    global openai_client, index, ds
    
    if not openai_client or not index:
        return query_with_sentence_transformer(query, lang_filter)
    
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding
    
    filter_dict = {}
    if lang_filter != "All":
        filter_dict = {"language": lang_filter}
    
    results = index.query(
        vector=query_embedding,
        top_k=20,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )
    
    hits = []
    dataset_lookup = get_dataset_lookup()
    
    for match in results.matches:
        metadata = match.metadata
        name_strip = metadata.get("name_strip", "")
        language = metadata.get("language", "")
        
        row = dataset_lookup.get((name_strip, language))
        if not row:
            continue
        
        name_metadata = get_name_metadata_from_dataset(name_strip, language)
        story = get_story_from_dataset(name_strip, language)
        
        name_data = {
            "name": row.get("Name", name_strip),
            "name_strip": name_strip,
            "meaning": row.get("Meaning", ""),
            "language": language,
            "cultural_context": row.get("cultural_context", ""),
            "themes": row.get("themes", []),
        }
        
        result = build_result_dict(name_data, name_metadata, story, match.score)
        hits.append(result)
    
    return hits

def build_result_dict(name_data: Dict, metadata: Dict, story: Dict, score: float) -> Dict[str, Any]:
    """Build a result dictionary. For Yoruba, display meaning uses first paraphrase when available."""
    lang = name_data.get("language", "")
    name_strip = name_data.get("name_strip", "")
    canonical = name_data.get("meaning", "")
    meaning = display_meaning_for_result(lang, name_strip, canonical)
    return {
        "name": name_data.get("name", ""),
        "name_strip": name_data.get("name_strip", ""),
        "meaning": meaning,
        "language": lang,
        "phonetic_spelling": metadata.get("phonetic_spelling", ""),
        "pronunciation_url": metadata.get("pronunciation_url", ""),
        "pronunciation_by": metadata.get("pronunciation_by", ""),
        "validated_by": metadata.get("validated_by", ""),
        "validation_status": metadata.get("validation_status", ""),
        "cultural_context": name_data.get("cultural_context", ""),
        "themes": name_data.get("themes", []),
        "story": story,
        "score": score,
        "audio_url": metadata.get("audio_url", ""),
    }

def query_name_db(query: str, lang_filter: str) -> List[Dict[str, Any]]:
    """Main query function - hybrid search"""
    global ds, openai_client
    
    dataset = load_dataset_fallback()
    # Handle both list and dataset object
    if isinstance(dataset, list):
        rows = dataset
    else:
        rows = dataset
    
    # 1) Direct name lookup (exact match)
    match = next((row for row in rows if row.get("NameStrip", "").strip().lower() == query.strip().lower()), None)
    if match:
        name_strip = str(match.get("NameStrip", "")).strip()
        lang = str(match.get("Language", "")).strip()
        name = match.get("Name", name_strip)
        meaning = match.get("Meaning", "")
        
        metadata = get_name_metadata_from_dataset(name_strip, lang)
        story = get_story_from_dataset(name_strip, lang)
        
        name_data = {
            "name": name,
            "name_strip": name_strip,
            "meaning": meaning,
            "language": lang,
            "cultural_context": match.get("cultural_context", ""),
            "themes": match.get("themes", []),
        }
        
        return [build_result_dict(name_data, metadata, story, 1.0)]
    
    # 2) Semantic search
    use_openai = is_complex_query(query) and openai_client is not None
    
    if use_openai:
        return query_with_openai(query, lang_filter)
    else:
        return query_with_sentence_transformer(query, lang_filter)

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Nomi Name Search API",
        "version": "1.0.0"
    }

@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    language: Optional[str] = Query("All", description="Language filter"),
    stories_only: bool = Query(False, description="Only return names with published stories")
):
    """Search for names by meaning, theme, or exact name"""
    initialize_components()
    
    if stories_only:
        load_stories_data()
        dataset_lookup = get_dataset_lookup()
        hits = []
        
        global _stories_lookup
        if _stories_lookup is None:
            load_stories_data()
        for (name_strip, lang), story_data in _stories_lookup.items():
            if language != "All" and language != lang:
                continue
            
            match = dataset_lookup.get((name_strip, lang))
            if match:
                metadata = get_name_metadata_from_dataset(name_strip, lang)
                name_data = {
                    "name": match["Name"],
                    "name_strip": name_strip,
                    "meaning": match["Meaning"],
                    "language": lang,
                    "cultural_context": match.get("cultural_context", ""),
                    "themes": match.get("themes", []),
                }
                story = get_story_from_dataset(name_strip, lang)
                hits.append(build_result_dict(name_data, metadata, story, 1.0))
        
        results = hits
    elif not q or q.strip() == "":
        return SearchResponse(
            query=q,
            results=[],
            total=0,
            language_filter=language,
            stories_only=stories_only
        )
    else:
        results = query_name_db(q, language)
    
    return SearchResponse(
        query=q,
        results=[NameResult(**r) for r in results],
        total=len(results),
        language_filter=language,
        stories_only=stories_only
    )

@app.get("/languages")
async def get_languages():
    """Get list of available languages"""
    initialize_components()
    dataset = load_dataset_fallback()
    languages = set()
    
    if isinstance(dataset, list):
        rows = dataset
    else:
        rows = dataset
    
    for row in rows:
        lang = row.get("Language", "").strip()
        if lang:
            languages.add(lang)
    return {"languages": sorted(list(languages))}

# ── Name card HTML helpers ────────────────────────────────────────────────────
# Aesthetic spec: docs/Nomi_Aesthetics_System_Prompt.md (v2)
# Variant A — Cream Stationery: page #FBF7F0, card #FFFFFF, accent from palette.
# Card must read as object ON TOP OF page; never same colour as background.
# One brushstroke (underswash), warm shadows, Livvic/Sen, no emojis.

# Dominant accent colours (hex only); card uses Cream Stationery so text is Ink/Stone.
_PALETTE_HEX = [
    "#5B3FD9",   # Ink Violet
    "#E8557A",   # Bloom Pink
    "#F2C94C",   # Sunwash Yellow
    "#3A9BDC",   # Sky Cerulean
    "#5BA07A",   # Sage Leaf
    "#E8845C",   # Dusty Coral
    "#9B59B6",   # Soft Plum
    "#F4B8A0",   # Blush Peach (pastel)
    "#C5B8F0",   # Lavender Mist (pastel)
    "#B8D8C4",   # Sage Wash (pastel)
]

PAGE_BG_CREAM = "#FBF7F0"
CARD_BG_WHITE = "#FFFFFF"
INK = "#1A1A2E"
STONE = "#4A4A6A"

def _pick_accent(name_strip: str) -> str:
    idx = sum(ord(c) for c in (name_strip or "").lower()) % len(_PALETTE_HEX)
    return _PALETTE_HEX[idx]


# ── OG image generation (1200×630 PNG for social sharing previews) ────────────

_pil_font_cache: Dict[str, Any] = {}


def _download_font_ttf(family: str, weight: int) -> Optional[str]:
    """Return path to a bundled TTF font file for the given family/weight."""
    # Bundled fonts live in fonts/ next to app.py
    bundled = Path(__file__).parent / "fonts" / f"{family.replace(' ', '')}_{weight}.ttf"
    if bundled.exists():
        return str(bundled)
    return None


def _pil_font(family: str, weight: int, size: int) -> Any:
    key = f"{family}_{weight}_{size}"
    if key not in _pil_font_cache:
        path = _download_font_ttf(family, weight)
        try:
            _pil_font_cache[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
        except Exception:
            _pil_font_cache[key] = ImageFont.load_default()
    return _pil_font_cache[key]


def _blend(fg: tuple, bg: tuple, a: float) -> tuple:
    """Alpha-blend fg onto bg (a=0→bg, a=1→fg). Returns RGB tuple."""
    return tuple(int(fg[i] * a + bg[i] * (1 - a)) for i in range(3))


def _draw_brush_stroke(draw: Any, x0: float, y: float, x1: float,
                       text_rgb: tuple, bg_rgb: tuple, opacity: float) -> None:
    """Variable-width organic brush stroke underswash using cubic bezier."""
    color = _blend(text_rgb, bg_rgb, opacity)
    cx1, cx2 = x0 + (x1 - x0) * 0.3, x0 + (x1 - x0) * 0.7
    cy1, cy2 = y - 5, y + 5
    for i in range(200):
        t = i / 200
        px = (1-t)**3*x0 + 3*(1-t)**2*t*cx1 + 3*(1-t)*t**2*cx2 + t**3*x1
        py = (1-t)**3*y + 3*(1-t)**2*t*cy1 + 3*(1-t)*t**2*cy2 + t**3*y
        w = 13 * math.sin(math.pi * t) + 3
        r = w / 2
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)


def _wrap_text(draw: Any, text: str, font: Any, max_px: int) -> list:
    words = text.split()
    lines, line = [], ""
    for word in words:
        test = (line + " " + word).strip()
        if draw.textlength(test, font=font) <= max_px:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _generate_og_image(results: list, name_strip: str) -> bytes:
    """
    Render a 1200×630 PNG for og:image meta tag.
    Uses Vivid Field layout (solid accent bg) — boldest for social previews.
    """
    W, H, PAD = 1200, 630, 80

    ns      = results[0].get("name_strip", name_strip) if results else name_strip
    accent  = _pick_accent(ns)
    r_bg, g_bg, b_bg = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)
    bg      = (r_bg, g_bg, b_bg)
    # Sunwash Yellow needs dark text; all others use white
    dark    = accent in ("#F2C94C",)
    text    = (26, 26, 46) if dark else (255, 255, 255)

    img  = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    f_name     = _pil_font("Livvic", 800, 118)
    f_phonetic = _pil_font("Sen",    400,  30)
    f_meaning  = _pil_font("Livvic", 400,  32)
    f_label    = _pil_font("Sen",    700,  14)
    f_brand    = _pil_font("Sen",    400,  20)

    # NOMI — top left
    draw.text((PAD, 52), "NOMI", font=f_label, fill=_blend(text, bg, 0.45))

    primary      = results[0] if results else {}
    display_name = (primary.get("name") or name_strip)
    phonetic     = (primary.get("phonetic_spelling") or "").strip()
    language     = (primary.get("language") or "").strip()
    meaning      = (primary.get("meaning") or "").strip()

    # Name hero
    name_y = 130
    draw.text((PAD, name_y), display_name, font=f_name, fill=text)
    bbox   = draw.textbbox((PAD, name_y), display_name, font=f_name)
    name_w = bbox[2] - bbox[0]
    name_h = bbox[3] - bbox[1]

    # Brush stroke underswash
    stroke_y = name_y + name_h + 14
    _draw_brush_stroke(draw, PAD, stroke_y,
                       PAD + min(name_w, W - PAD * 2 - 80),
                       text, bg, 0.28 if not dark else 0.15)

    cy = stroke_y + 30

    # Phonetic chip
    if phonetic:
        pad_x, chip_h = 20, 52
        tw       = int(draw.textlength(phonetic, font=f_phonetic))
        chip_w   = tw + pad_x * 2
        chip_bg  = _blend(text, bg, 0.16)
        draw.rounded_rectangle([PAD, cy, PAD + chip_w, cy + chip_h],
                                radius=chip_h // 2, fill=chip_bg)
        draw.text((PAD + pad_x, cy + 11), phonetic, font=f_phonetic, fill=text)

        # Language tag (outline pill, right of chip)
        if language:
            lx  = PAD + chip_w + 12
            lw  = int(draw.textlength(language.upper(), font=f_label)) + 28
            draw.rounded_rectangle([lx, cy, lx + lw, cy + chip_h],
                                   radius=chip_h // 2,
                                   outline=_blend(text, bg, 0.3), width=2)
            draw.text((lx + 14, cy + 19), language.upper(),
                      font=f_label, fill=_blend(text, bg, 0.7))
        cy += chip_h + 28

    elif language:
        lw = int(draw.textlength(language.upper(), font=f_label)) + 28
        draw.rounded_rectangle([PAD, cy, PAD + lw, cy + 44],
                               radius=22, outline=_blend(text, bg, 0.3), width=2)
        draw.text((PAD + 14, cy + 14), language.upper(),
                  font=f_label, fill=_blend(text, bg, 0.7))
        cy += 56

    # Meaning (up to 3 lines)
    if meaning:
        lines = _wrap_text(draw, f'"{meaning}"', f_meaning, W - PAD * 2 - 80)
        for line in lines[:3]:
            draw.text((PAD, cy), line, font=f_meaning,
                      fill=_blend(text, bg, 0.88))
            cy += 46

    # nomistories.com — bottom right
    bw = int(draw.textlength("nomistories.com", font=f_brand))
    draw.text((W - PAD - bw, H - 52), "nomistories.com",
              font=f_brand, fill=_blend(text, bg, 0.4))

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


# Static CSS — Variant A (Cream Stationery). Page #FBF7F0, card #FFF, accent for hero/brush/tag/button.
# Shadows: warm, card sits on top of page (aesthetics doc). Motion: 350ms ease-in-out, 8px→0.
_CARD_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--page-bg);font-family:'Livvic',system-ui,sans-serif;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px 16px}
.card{background:var(--card-bg);border:1px solid var(--card-border);border-radius:20px;padding:48px 36px 40px;max-width:420px;width:100%;position:relative;overflow:hidden}
.card{box-shadow:0 2px 8px rgba(26,26,46,.08),0 8px 24px rgba(26,26,46,.10),0 1px 2px rgba(26,26,46,.04)}
.logo{font-family:'Sen',sans-serif;font-size:11px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--ink);opacity:.45;margin-bottom:40px;display:block}
.name-hero{font-family:'Livvic',sans-serif;font-size:clamp(56px,15vw,84px);font-weight:800;color:var(--accent);line-height:.92;letter-spacing:-.02em;position:relative;z-index:1;text-transform:lowercase}
.name-hero:first-letter{text-transform:uppercase}
.brush-wrap{display:block;width:100%;margin-top:6px;margin-bottom:22px;overflow:visible;line-height:0}
.phonetic-chip{display:inline-flex;align-items:center;background:var(--stone);border-radius:999px;padding:8px 18px;font-family:'Sen',sans-serif;font-size:15px;font-weight:400;color:var(--card-bg);letter-spacing:.04em;margin-bottom:10px;min-height:14px}
.lang-tag{display:inline-flex;align-items:center;background:var(--tag-bg);border:1px solid var(--accent);border-radius:999px;padding:5px 14px;font-family:'Sen',sans-serif;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin-bottom:28px;margin-left:8px}
.divider-wave{display:block;width:100%;margin-bottom:20px}
.meaning-label{font-family:'Sen',sans-serif;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--ink);opacity:.45;margin-bottom:8px}
.meaning-text{font-family:'Livvic',sans-serif;font-size:19px;font-weight:400;color:var(--ink);line-height:1.55;margin-bottom:30px}
.audio-row{display:flex;align-items:center;gap:16px;margin-bottom:22px}
.play-btn{width:54px;height:54px;border-radius:50%;background:var(--accent);border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 2px 10px rgba(26,26,46,.15);transition:transform .35s ease-in-out,box-shadow .35s ease-in-out}
.play-btn:hover{transform:translateY(-1px);box-shadow:0 4px 18px rgba(26,26,46,.2)}
.play-btn svg path{fill:var(--card-bg)}
.play-btn.playing{animation:breathe .85s ease-in-out infinite}
.audio-meta{flex:1}
.audio-label{font-family:'Sen',sans-serif;font-size:14px;font-weight:600;color:var(--ink)}
.audio-sub{font-family:'Sen',sans-serif;font-size:12px;color:var(--ink);opacity:.5;margin-top:2px}
.no-audio{font-family:'Sen',sans-serif;font-size:14px;color:var(--ink);opacity:.6;margin-bottom:22px}
.no-audio a{color:var(--accent);font-weight:700;text-decoration:underline;text-underline-offset:2px}
.share-row{display:flex;gap:10px}
.btn{flex:1;display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:13px;border-radius:12px;font-family:'Sen',sans-serif;font-size:14px;font-weight:600;cursor:pointer;border:none;transition:transform .35s ease-in-out,box-shadow .35s ease-in-out,background .15s ease}
.btn-copy{background:transparent;border:1.5px solid var(--accent);color:var(--accent)}
.btn-copy:hover{background:var(--tag-bg);transform:translateY(-1px)}
.btn-primary{background:var(--accent);color:var(--card-bg)}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(26,26,46,.18)}
.other-lang{font-family:'Sen',sans-serif;font-size:13px;color:var(--ink);opacity:.7;margin-bottom:6px;line-height:1.4}
.other-lang strong{opacity:.9}
.footer{text-align:center;margin-top:20px}
.footer a{font-family:'Sen',sans-serif;font-size:12px;color:var(--stone);text-decoration:none;letter-spacing:.04em}
.footer a:hover{color:var(--ink)}
.toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%) translateY(80px);background:var(--ink);color:var(--page-bg);padding:10px 22px;border-radius:12px;font-family:'Sen',sans-serif;font-size:14px;font-weight:600;transition:transform .35s ease-in-out;z-index:100}
.toast.show{transform:translateX(-50%) translateY(0)}
.animate-in{animation:fadeUp .35s ease-in-out forwards;opacity:0}
.s1{animation-delay:0ms}.s2{animation-delay:80ms}.s3{animation-delay:160ms}.s4{animation-delay:240ms}.s5{animation-delay:320ms}.s6{animation-delay:400ms}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes breathe{0%,100%{transform:scale(1)}50%{transform:scale(1.02)}}
@media(max-width:400px){.card{padding:36px 22px 32px}}
"""


def _css_vars_cream_stationery(accent: str) -> str:
    """Variant A: page cream, card white, 10% accent border, tag 18% accent fill."""
    r, g, b = (int(accent[i : i + 2], 16) for i in (1, 3, 5))
    border = f"rgba({r},{g},{b},0.1)"
    tag_bg = f"rgba({r},{g},{b},0.18)"
    return (
        f"--page-bg:{PAGE_BG_CREAM};"
        f"--card-bg:{CARD_BG_WHITE};"
        f"--accent:{accent};"
        f"--ink:{INK};"
        f"--stone:{STONE};"
        f"--card-border:{border};"
        f"--tag-bg:{tag_bg};"
    )


def _not_found_html(name_strip: str) -> str:
    esc = html_mod.escape(name_strip)
    accent = _pick_accent(name_strip)
    vars_css = _css_vars_cream_stationery(accent)
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc} — Nomi</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Livvic:wght@400;700;800&family=Sen:wght@400;600;700&display=swap" rel="stylesheet">
<style>
{_CARD_CSS}
.add-btn{{display:inline-flex;align-items:center;justify-content:center;padding:14px 28px;background:var(--accent);color:var(--card-bg);border-radius:12px;font-family:'Sen',sans-serif;font-size:15px;font-weight:600;text-decoration:none;transition:transform .35s ease-in-out}}
.add-btn:hover{{transform:translateY(-1px)}}
p{{font-family:'Sen',sans-serif;font-size:16px;color:var(--ink);opacity:.7;line-height:1.6;margin-bottom:28px}}
</style></head>
<body style="{vars_css}">
<div class="card" style="text-align:center">
  <span class="logo animate-in s1">Nomi</span>
  <div class="name-hero animate-in s2" style="margin-bottom:20px">{esc}</div>
  <p class="animate-in s3">We don't have this name yet.<br>Help us add it.</p>
  <a class="add-btn animate-in s4" href="https://huggingface.co/spaces/nomi-stories/nomi-pronunciation-inbox">Add this name →</a>
</div>
<div class="footer" style="margin-top:20px">
  <a href="https://nomistories.com">nomistories.com</a>
</div>
</body></html>"""


def _generate_name_card_html(results: list, name_strip: str, base_url: str = "") -> str:
    if not results:
        return _not_found_html(name_strip)

    primary = results[0]
    others = results[1:]

    ns = primary.get("name_strip", name_strip)
    accent = _pick_accent(ns)
    vars_css = _css_vars_cream_stationery(accent)

    display_name = html_mod.escape(primary.get("name", name_strip))
    phonetic     = html_mod.escape(primary.get("phonetic_spelling", "") or "")
    language     = html_mod.escape(primary.get("language", ""))
    meaning      = html_mod.escape(primary.get("meaning", ""))
    audio_url    = primary.get("audio_url", "") or ""
    pron_by      = html_mod.escape(primary.get("pronunciation_by", "") or "")

    og_desc = html_mod.escape(
        (f"{phonetic} · {meaning[:100]}" if phonetic else meaning[:120])
    )

    # Divider: subtle wave in accent at low opacity (section divider stroke)
    r, g, b = (int(accent[i : i + 2], 16) for i in (1, 3, 5))
    divider_stroke = f"rgba({r},{g},{b},0.2)"

    # Brushstroke — underswash: double-path (thick + thin offset) per aesthetics doc
    brush_svg = (
        '<svg class="brush-wrap" viewBox="0 0 620 28" height="28" preserveAspectRatio="xMidYMid meet" aria-hidden="true">'
        f'<path d="M 20 14 C 100 6 220 20 340 12 C 460 4 540 18 600 14" '
        f'stroke="{accent}" stroke-width="12" stroke-linecap="round" fill="none" opacity="0.9"/>'
        f'<path d="M 22 18 C 102 10 222 24 342 16 C 462 8 542 22 598 18" '
        f'stroke="{accent}" stroke-width="4" stroke-linecap="round" fill="none" opacity="0.4"/>'
        "</svg>"
    )

    divider_svg = (
        f'<svg class="divider-wave" viewBox="0 0 400 18" height="18" '
        f'preserveAspectRatio="none" aria-hidden="true">'
        f'<path d="M 0 9 Q 100 2 200 9 Q 300 16 400 9" '
        f'stroke="{divider_stroke}" stroke-width="2" fill="none" stroke-linecap="round"/>'
        f'</svg>'
    )

    phonetic_html = f'<div class="phonetic-chip animate-in s2">{phonetic}</div>' if phonetic else ""
    lang_html     = f'<div class="lang-tag animate-in s3">{language}</div>' if language else ""

    # Audio section
    if audio_url:
        safe_audio = html_mod.escape(audio_url)
        by_note = f'<div class="audio-sub">Contributed by {pron_by}</div>' if pron_by else ""
        audio_html = (
            f'<div class="audio-row animate-in s4">'
            f'<button class="play-btn" id="play-btn" onclick="playAudio()" aria-label="Play pronunciation">'
            f'<svg width="22" height="22" viewBox="0 0 22 22" fill="none">'
            f'<path id="play-icon" d="M8 5.5L18 11L8 16.5V5.5Z" fill="currentColor"/>'
            f'</svg></button>'
            f'<div class="audio-meta">'
            f'<div class="audio-label" id="audio-label">Hear it pronounced</div>'
            f'{by_note}</div></div>'
        )
        audio_js = (
            f'const _aud=new Audio("{safe_audio}");let _pl=false;\n'
            f'function playAudio(){{\n'
            f'  const btn=document.getElementById("play-btn");\n'
            f'  const lbl=document.getElementById("audio-label");\n'
            f'  if(_pl){{_aud.pause();_aud.currentTime=0;_pl=false;btn.classList.remove("playing");lbl.textContent="Hear it pronounced";}}\n'
            f'  else{{_aud.play();_pl=true;btn.classList.add("playing");lbl.textContent="Playing...";\n'
            f'    _aud.onended=()=>{{_pl=false;btn.classList.remove("playing");lbl.textContent="Hear it pronounced";}};}};\n'
            f'}}'
        )
    else:
        audio_html = (
            '<div class="no-audio animate-in s4">No audio yet — '
            '<a href="https://huggingface.co/spaces/nomi-stories/nomi-pronunciation-inbox">'
            'contribute one</a></div>'
        )
        audio_js = "function playAudio(){}"

    # Other language variants (max 2, compact)
    others_html = ""
    if others:
        items = ""
        for o in others[:2]:
            o_lang    = html_mod.escape(o.get("language", ""))
            o_meaning = html_mod.escape(o.get("meaning", ""))
            o_ph      = html_mod.escape(o.get("phonetic_spelling", "") or "")
            ph_bit    = f" · {o_ph}" if o_ph else ""
            snippet   = o_meaning[:80] + ("…" if len(o_meaning) > 80 else "")
            items += (
                f'<div class="other-lang">'
                f'<strong>{o_lang}{ph_bit}</strong><br>{snippet}'
                f'</div>'
            )
        others_html = items

    display_name_js = display_name.replace("'", "\\'")
    phonetic_js     = phonetic.replace("'", "\\'")
    share_text_js   = (
        f"Here\\'s how to say my name: {display_name_js} ({phonetic_js})"
        if phonetic_js else
        f"Here\\'s how to say my name: {display_name_js}"
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{display_name} — Nomi</title>
<meta name="description" content="{og_desc}">
<meta property="og:title" content="{display_name}">
<meta property="og:description" content="{og_desc}">
<meta property="og:image" content="{base_url}/card-image/{ns}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:site_name" content="Nomi">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{display_name} — Nomi">
<meta name="twitter:description" content="{og_desc}">
<meta name="twitter:image" content="{base_url}/card-image/{ns}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Livvic:wght@400;700;800&family=Sen:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{_CARD_CSS}</style>
</head>
<body style="{vars_css}">
<div class="card">
  <span class="logo animate-in s1">Nomi</span>
  <div class="name-hero animate-in s1">{display_name}</div>
  {brush_svg}
  {phonetic_html}
  {lang_html}
  {divider_svg}
  <div class="meaning-label">Meaning</div>
  <div class="meaning-text animate-in s4">"{meaning}"</div>
  {others_html}
  {audio_html}
  <div class="share-row animate-in s5">
    <button class="btn btn-copy" onclick="copyLink()">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <path d="M4.5 2H9.5C10.33 2 11 2.67 11 3.5V8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        <rect x="2.5" y="5" width="7" height="7" rx="1.5" stroke="currentColor" stroke-width="1.5"/>
      </svg>
      Copy link
    </button>
    <button class="btn btn-primary" onclick="shareCard()">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <path d="M7 1.5v7M7 1.5L4.5 4M7 1.5L9.5 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        <path d="M2.5 8v4h9V8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
      Share
    </button>
  </div>
</div>
<div class="footer">
  <a href="https://nomistories.com">nomistories.com</a>
</div>
<div class="toast" id="toast">Copied!</div>
<script>
{audio_js}
function copyLink(){{
  const u=window.location.href;
  navigator.clipboard.writeText(u)
    .then(()=>showToast('Link copied!'))
    .catch(()=>{{const e=document.createElement('input');e.value=u;document.body.appendChild(e);e.select();document.execCommand('copy');document.body.removeChild(e);showToast('Link copied!');}});
}}
function shareCard(){{
  if(navigator.share){{
    navigator.share({{title:'{display_name_js} — Nomi',text:'{share_text_js}',url:window.location.href}}).catch(()=>{{}});
  }}else{{copyLink();}}
}}
function showToast(m){{
  const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),2200);
}}
</script>
</body></html>"""


# ── Name lookup & card endpoints ──────────────────────────────────────────────

def _lookup_name_results(name_strip: str, language: Optional[str]) -> list:
    """Shared logic: find all dataset rows matching name_strip (case-insensitive)."""
    load_dataset_fallback()
    load_stories_data()
    dataset_lookup = get_dataset_lookup()
    needle = name_strip.lower().strip()
    results = []
    for (ns, lang), row in dataset_lookup.items():
        if ns.lower() != needle:
            continue
        if language and lang.lower() != language.lower():
            continue
        metadata = get_name_metadata_from_dataset(ns, lang)
        story = get_story_from_dataset(ns, lang)
        canonical_meaning = row.get("Meaning", "")
        meaning = display_meaning_for_result(lang, ns, canonical_meaning)
        results.append({
            "name": row.get("Name", ns),
            "name_strip": ns,
            "language": lang,
            "meaning": meaning,
            "phonetic_spelling": metadata.get("phonetic_spelling") or None,
            "audio_url": metadata.get("audio_url") or None,
            "pronunciation_by": metadata.get("pronunciation_by") or None,
            "cultural_context": row.get("cultural_context") or None,
            "themes": row.get("themes") or None,
            "story": story if story else None,
        })
    return results


@app.get("/audio/{name_strip}")
async def get_audio(
    name_strip: str,
    language: Optional[str] = Query(None, description="Language to fetch audio for")
):
    """Serve embedded audio bytes for a name directly from the dataset parquet."""
    dataset_lookup = get_dataset_lookup()
    needle = name_strip.strip()
    # Find the matching row (prefer the requested language, else first with audio)
    row = None
    for (ns, lang), r in dataset_lookup.items():
        if ns.lower() != needle.lower():
            continue
        if language and lang.lower() != language.lower():
            continue
        audio_val = r.get("Audio Pronunciation")
        if isinstance(audio_val, dict) and audio_val.get("bytes"):
            row = r
            break
    if row is None:
        raise HTTPException(status_code=404, detail="No audio for this name")
    audio_bytes = row["Audio Pronunciation"]["bytes"]
    return Response(content=audio_bytes, media_type="audio/wav")


@app.get("/name/{name_strip}", response_model=NameLookupResponse)
async def get_name(
    name_strip: str,
    language: Optional[str] = Query(None, description="Filter by language")
):
    """
    Direct name lookup — no ML inference, dataset-only, fast.
    Returns meaning, phonetic spelling, audio URL, and cultural context.
    Supports all languages in the dataset; ?language= to filter.
    """
    results = _lookup_name_results(name_strip, language)
    if not results:
        raise HTTPException(status_code=404, detail=f"Name '{name_strip}' not found in dataset")
    return NameLookupResponse(
        name_strip=name_strip,
        results=[NameCardData(**r) for r in results],
        total=len(results),
    )


@app.get("/card/{name_strip}", response_class=HTMLResponse)
async def name_card(
    request: Request,
    name_strip: str,
    language: Optional[str] = Query(None, description="Filter by language")
):
    """
    Shareable HTML name card. Designed to be linked from email signatures,
    LinkedIn bios, Slack profiles, or anywhere you want people to learn
    how to say your name correctly.
    """
    results = _lookup_name_results(name_strip, language)
    base_url = str(request.base_url).rstrip("/")
    return HTMLResponse(content=_generate_name_card_html(results, name_strip, base_url=base_url))


@app.get("/card-image/{name_strip}")
async def name_card_image(
    name_strip: str,
    language: Optional[str] = Query(None, description="Filter by language")
):
    """
    1200×630 PNG for og:image social sharing previews.
    Vivid Field layout: solid accent background, bold name, phonetic, meaning.
    """
    if not PIL_AVAILABLE:
        raise HTTPException(status_code=501, detail="Pillow not installed")
    results = _lookup_name_results(name_strip, language)
    img_bytes = _generate_og_image(results, name_strip)
    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
