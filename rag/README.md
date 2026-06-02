# RAG (research papers)

Retrieval over indexed naming research papers, by language. Powers paraphrasing, HF Space “Research Insights,” and the **`/insights`** API endpoint.

## Layout

| File | Purpose |
| --- | --- |
| `language_config.py` | Per-language PDF lists, index paths, morphemes (Yoruba), dataset language mapping |
| `rag_service.py` | `LanguageRAGService`, `YorubaRAGService` (alias), `get_rag_service_for_dataset_language()` |
| `index_language_papers.py` | Build `{language}_papers_index.json` under `research_papers_index/` |
| `query_rag.py` | Interactive CLI |
| Legacy `index_yoruba_*.py` | Older Yoruba-only indexers |

**PDFs:** `Research papers/` at repo root.  
**Indexes:** `research_papers_index/{language}_papers_index.json`.

## Build indexes

From repo root (requires `pdfplumber` or `PyPDF2`):

```bash
python rag/index_language_papers.py yoruba
python rag/index_language_papers.py igbo hausa edo
python rag/index_language_papers.py --all
```

Deploy the API from **repo root** (or ship `research_papers_index/` with the service) so indexes resolve.

## API: `/insights`

`nomi-name-search-api` exposes:

- `GET /insights?name=Folasade&language=Yoruba` — Claude paragraph grounded in RAG when an index exists. Response includes `rag_excerpts` (text sent to the model) and `attributions` for auditability.
- `GET /insights/languages` — which language indexes are present on the server

Requires `ANTHROPIC_API_KEY`. System prompt: `nomi-name-search-api/prompts/nomi_insights_system_prompt.md`.

## Supported RAG languages (config)

Yoruba, Igbo, Hausa, Edo, Ibibio, Akan, Ewe, Urhobo, Igala, Kanuri, Bakossi, Bukusu, Siswati (Wolof: no papers listed yet). Nomi dataset languages map via substring (e.g. `Hausa (Localised Islamic/Arabic)` → Hausa). Hausa indexing intentionally skips `1001_Hausa_Names.pdf` (names/meanings only).

If no index exists for a language, `/insights` still runs but uses meaning + careful model knowledge only (per prompt).
