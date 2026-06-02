# Deployment Guide

## Option 1: Railway (Recommended - Easiest)

### Prerequisites
- GitHub account
- Railway account (free tier available)

### Steps

1. **Install Railway CLI** (optional, but helpful):
```bash
npm i -g @railway/cli
railway login
```

2. **Deploy via Railway Dashboard**:
   - Go to https://railway.app
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Select the `nomi-name-search-api` folder
   - Railway will auto-detect Python and install dependencies

3. **Set Environment Variables**:
   - In Railway dashboard, go to your project
   - Click on "Variables" tab
   - Add these variables:
     - `HF_TOKEN` = your HuggingFace token
     - `PINECONE_API_KEY` = your Pinecone API key
     - `OPENAI_API_KEY` = your OpenAI API key (optional)

4. **Deploy**:
   - Railway will automatically deploy
   - You'll get a URL like: `https://your-project.railway.app`

### Deploy via CLI (Alternative)
```bash
cd nomi-name-search-api
railway init
railway up
railway variables set HF_TOKEN=your-token
railway variables set PINECONE_API_KEY=your-key
railway variables set OPENAI_API_KEY=your-key
```

---

## Option 2: Render

### Steps

1. **Go to Render Dashboard**:
   - https://render.com
   - Sign up/login

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

3. **Configure**:
   - **Name**: `nomi-name-search-api`
   - **Root Directory**: leave **blank** (repo root) so `data/paraphrasing/yoruba_paraphrased_meanings.json` is available for Yoruba paraphrase display
   - **Environment**: `Python 3`
   - **Build Command**: `cd nomi-name-search-api && pip install -r requirements.txt`
   - **Start Command**: `cd nomi-name-search-api && uvicorn app:app --host 0.0.0.0 --port $PORT`

4. **Set Environment Variables**:
   - Scroll down to "Environment Variables"
   - Add:
     - `HF_TOKEN`
     - `PINECONE_API_KEY`
     - `OPENAI_API_KEY` (optional)

5. **Deploy**:
   - Click "Create Web Service"
   - Render will build and deploy
   - You'll get a URL like: `https://nomi-name-search-api.onrender.com`

6. **Instance size (memory)**:
   - Render's free/starter **512 MB** plan cannot run **PyTorch + sentence-transformers** (~400 MB+ RAM on first semantic `/search`). The default build uses **`requirements.txt` only** (no torch) so the service stays up for exact-name routes, cards, and insights.
   - Set **`NOMI_SEMANTIC_SEARCH=0`** (recommended on 512 MB) to disable meaning-based search and return **503** with a clear message instead of OOM.
   - **`GET /`** reports `"semantic_search": true|false`. When `false`, `/search?q=folasade` (exact name) still works; `/search?q=love` (meaning query) returns 503.
   - For full semantic search: upgrade to **at least 1 GB RAM**, set build to `pip install -r requirements-semantic.txt`, set **`NOMI_SEMANTIC_SEARCH=1`**, and keep **`PINECONE_API_KEY`** set.
   - **`GET /insights`** uses text-only RAG (no torch) and needs **`ANTHROPIC_API_KEY`**; it fits 512 MB once indexes are deployed from repo root.
   - If the instance shows **"Ran out of memory (used over 512MB)"**, check Logs for the first heavy request (usually semantic `/search` or an old deploy that still installed torch). Upgrade or disable semantic search as above.

---

## Option 3: Fly.io

### Prerequisites
- Fly.io account
- Fly CLI installed

### Steps

1. **Install Fly CLI**:
```bash
curl -L https://fly.io/install.sh | sh
```

2. **Login**:
```bash
fly auth login
```

3. **Initialize**:
```bash
cd nomi-name-search-api
fly launch
# Follow prompts:
# - App name: nomi-name-search-api (or choose your own)
# - Region: choose closest to you
# - PostgreSQL: No
# - Redis: No
```

4. **Set Secrets**:
```bash
fly secrets set HF_TOKEN=your-token
fly secrets set PINECONE_API_KEY=your-key
fly secrets set OPENAI_API_KEY=your-key
```

5. **Deploy**:
```bash
fly deploy
```

6. **Get URL**:
```bash
fly status
# You'll see your app URL
```

---

## Option 4: Local Development (for testing)

1. **Install dependencies**:
```bash
cd nomi-name-search-api
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
export HF_TOKEN="your-token"
export PINECONE_API_KEY="your-key"
export OPENAI_API_KEY="your-key"  # optional
```

3. **Run**:
```bash
python app.py
# Or: uvicorn app:app --reload
```

4. **Test**:
```bash
curl "http://localhost:8000/search?q=love&language=Yoruba"
```

---

## Testing Your Deployment

Once deployed, test your API:

```bash
export API_BASE="https://nomi-name-search-api.onrender.com"  # replace if Render assigned a different hostname

# Health check
curl "$API_BASE/"

# Search
curl "$API_BASE/search?q=love&language=Yoruba"

# Get languages
curl "$API_BASE/languages"
```

### Current Render smoke test

The expected Render hostname from `render.yaml` is `https://nomi-name-search-api.onrender.com`. A smoke test from this workspace on 2026-05-26 timed out after 60 seconds for both:

- `GET https://nomi-name-search-api.onrender.com/`
- `GET https://nomi-name-search-api.onrender.com/search?q=love&language=Yoruba`

That means the demo should keep `API_BASE` configurable until the Render dashboard confirms the live hostname and a successful `GET /search` response.

### Insights endpoint

```bash
export ANTHROPIC_API_KEY="your-key"
curl "$API_BASE/insights?name=Folasade&language=Yoruba"
curl "$API_BASE/insights/languages"
```

Deploy from **repo root** so `research_papers_index/` and `rag/` are available. Build indexes with `python rag/index_language_papers.py yoruba` (and other languages as needed).

---

## Troubleshooting

### Common Issues

1. **Port binding error**:
   - Make sure you're using `$PORT` environment variable
   - Railway/Render/Fly.io provide this automatically

2. **Module not found**:
   - Check that `requirements.txt` has all dependencies
   - Verify the build completed successfully

3. **API key errors**:
   - Double-check environment variables are set correctly
   - Make sure variable names match exactly (case-sensitive)

4. **CORS errors**:
   - Update CORS settings in `app.py` to allow your frontend domain
   - For development, `allow_origins=["*"]` is fine
   - For production, specify your frontend URL

---

## Next Steps

After deployment:
1. Copy your API URL
2. Use it in v0 frontend code
3. Test the API endpoints
4. Update CORS settings for production

