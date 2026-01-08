#!/usr/bin/env python3
"""
FastAPI backend for Nomi Name Search
Exposes semantic search functionality as REST API for frontend use
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from huggingface_hub import hf_hub_download
import pandas as pd

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
# In production, specify your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend domain in production
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
_audio_df = None
_audio_lookup = None
_stories_data = None
_stories_lookup = None
_dataset_lookup = None

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

# Initialize components
def initialize_components():
    """Lazy initialization of all components"""
    global ds, pc, index, model, openai_client
    
    if ds is None:
        print("Loading HuggingFace dataset...")
        ds = load_dataset("nomi-stories/nomi-names", split="train", token=HF_TOKEN)
    
    if pc is None and PINECONE_API_KEY:
        print("Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("nomi-name-encoder")
    
    if model is None:
        print("Loading embedding model...")
        model = SentenceTransformer("fajayi/nomi-name-encoder")
    
    if openai_client is None and OPENAI_AVAILABLE and OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_dataset_lookup():
    """Create O(1) lookup dictionary for dataset"""
    global _dataset_lookup, ds
    if _dataset_lookup is None:
        _dataset_lookup = {}
        for row in ds:
            name_strip = str(row.get("NameStrip", "")).strip()
            language = str(row.get("Language", "")).strip()
            if name_strip and language:
                key = (name_strip, language)
                if key not in _dataset_lookup or row.get("Audio Pronunciation"):
                    _dataset_lookup[key] = row
    return _dataset_lookup

def get_name_metadata_from_dataset(name_strip: str, language: str) -> Dict[str, Any]:
    """Get metadata for a name from the dataset"""
    dataset_lookup = get_dataset_lookup()
    match = dataset_lookup.get((name_strip.strip(), language.strip()))
    if not match:
        return {}
    
    # Extract audio URL if available
    audio_url = ""
    audio_val = match.get("Audio Pronunciation")
    if audio_val:
        try:
            if isinstance(audio_val, dict):
                audio_path = audio_val.get("path")
                if audio_path:
                    audio_url = f"https://huggingface.co/datasets/nomi-stories/nomi-names/resolve/main/{audio_path}"
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
            stories_ds = load_dataset("nomi-stories/nomi-stories", split="train", token=HF_TOKEN)
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
    
    # Generate query embedding
    query_embedding = model.encode([query])[0].tolist()
    
    # Query Pinecone
    filter_dict = {}
    if lang_filter != "All":
        filter_dict = {"language": lang_filter}
    
    results = index.query(
        vector=query_embedding,
        top_k=20,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )
    
    # Process results
    hits = []
    dataset_lookup = get_dataset_lookup()
    
    for match in results.matches:
        metadata = match.metadata
        name_strip = metadata.get("name_strip", "")
        language = metadata.get("language", "")
        
        # Get full data from dataset
        row = dataset_lookup.get((name_strip, language))
        if not row:
            continue
        
        # Get additional metadata
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
    
    # Generate OpenAI embedding
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding
    
    # Query Pinecone
    filter_dict = {}
    if lang_filter != "All":
        filter_dict = {"language": lang_filter}
    
    results = index.query(
        vector=query_embedding,
        top_k=20,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )
    
    # Process results (same as sentence transformer)
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
    """Build a result dictionary"""
    return {
        "name": name_data.get("name", ""),
        "name_strip": name_data.get("name_strip", ""),
        "meaning": name_data.get("meaning", ""),
        "language": name_data.get("language", ""),
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
    
    # 1) Direct name lookup (exact match)
    match = next((row for row in ds if row.get("NameStrip", "").strip().lower() == query.strip().lower()), None)
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
    language: Optional[str] = Query("All", description="Language filter (e.g., 'Yoruba', 'Igbo', 'Hausa')"),
    stories_only: bool = Query(False, description="Only return names with published stories")
):
    """
    Search for names by meaning, theme, or exact name
    
    - **q**: Search query (e.g., "love", "strength", "joy", or exact name)
    - **language**: Filter by language (default: "All")
    - **stories_only**: Only return names with published stories (default: False)
    """
    initialize_components()
    
    if stories_only:
        # Return names with stories
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
    languages = set()
    for row in ds:
        lang = row.get("Language", "").strip()
        if lang:
            languages.add(lang)
    return {"languages": sorted(list(languages))}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
