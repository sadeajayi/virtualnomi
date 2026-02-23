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
# Health check
curl https://your-api-url.com/

# Search
curl "https://your-api-url.com/search?q=love&language=Yoruba"

# Get languages
curl https://your-api-url.com/languages
```

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

