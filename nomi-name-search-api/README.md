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
- `GET /languages` - Get available languages

See `DEPLOYMENT.md` for deployment instructions (Railway, Render, Fly.io).

## Usage in v0

Once deployed, use your API URL in v0:

```typescript
const response = await fetch('https://your-api-url.com/search?q=love&language=Yoruba');
const data = await response.json();
// data.results contains the name results
```
