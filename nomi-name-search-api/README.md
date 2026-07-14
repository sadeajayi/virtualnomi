# Nomi Name Search API

FastAPI backend for Nomi semantic name search. Exposes search functionality as REST API for frontend use.

## Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export HF_TOKEN="your-huggingface-token"
export PINECONE_API_KEY="your-pinecone-api-key"
export OPENAI_API_KEY="your-openai-api-key"  # Optional
```

3. Run the server:
```bash
python app.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Health check
- `GET /search?q=love&language=Yoruba` - Search for names
- `GET /name/{name_strip}` - Direct name lookup (card data)
- `GET /card/{name_strip}` - **Phase 0 wedge** shareable HTML card (default `mode=share`)
- `GET /insights?name={name_strip}&language={Lang}` - Griot cultural insight (RAG + Claude)
- `GET /languages` - Get available languages

See `DEPLOYMENT.md` for deployment instructions (Railway, Render, Fly.io).

## Phase 0 wedge — shareable name cards

Paste a link in Slack, iMessage, or a calendar invite. Recipients get a clean card: name, human audio, phonetic, meaning, optional story preview. No search, no “create yours” funnel.

**Example URLs** (production API on Render):

- [Fọláṣadé](https://nomi-name-search-api.onrender.com/card/folasade)
- [Adaora](https://nomi-name-search-api.onrender.com/card/adaora)

Local dev: `http://localhost:8000/card/folasade`

| Query | Purpose |
| --- | --- |
| `mode=share` | Default wedge layout (calendar / link unfurl) |
| `mode=full` | Owner tools: share buttons, teacher mailto, name lookup footer |
| `note=...` | Optional personal note on the card |
| `language=Yoruba` | Filter when a name exists in multiple languages |

Owner prep flow (add note, copy link): `GET /share/{name_strip}`

## Usage in v0 (nomistories.com)

Wire insights into the conference demo per **[docs/v0_insights_integration.md](../docs/v0_insights_integration.md)**.

Copy-paste helpers: `v0-insights-fetch.ts` (`fetchInsights`, `loadNameWithInsight`, 8s timeout).

```typescript
import { loadNameWithInsight } from '@/lib/nomi-api';

const { card, insight } = await loadNameWithInsight('folasade', 'Yoruba');
// Display card.name (not name_strip); Stage 5 uses insight (hide stage if null)
```
