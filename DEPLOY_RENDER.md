# Deploy to Render (Easiest Free Option)

## Why Render?
- ✅ **Free tier**: 750 hours/month (enough for 24/7)
- ✅ **No Docker needed**: Native Python support
- ✅ **Git push deploy**: Just connect GitHub repo
- ✅ **No API enablement**: No complex IAM policies
- ✅ **PostgreSQL included**: Free tier database

## Step 1: Create Render Account
1. Go to https://render.com
2. Sign up with GitHub
3. No credit card needed for free tier

## Step 2: Create `render.yaml`

This file tells Render how to build and run your app:

```yaml
services:
  - type: web
    name: multi-agent-genai
    runtime: python
    plan: free
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: LLM_PROVIDER
        value: groq
      - key: GROQ_API_KEY
        sync: false  # Set manually in Render dashboard
      - key: GROQ_MODEL
        value: llama-3.1-8b-instant
      - key: API_BEARER_TOKEN
        value: ""
      - key: RATE_LIMIT_PER_MINUTE
        value: "30"
      - key: PYTHON_VERSION
        value: "3.13.0"
```

## Step 3: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/multi-agent-genai.git
git push -u origin main
```

## Step 4: Deploy on Render

1. Go to https://dashboard.render.com
2. Click **New +** → **Web Service**
3. Connect your GitHub repo
4. Render will detect `render.yaml` and configure automatically
5. Add environment variable:
   - Key: `GROQ_API_KEY`
   - Value: `your-groq-key-here`
6. Click **Create Web Service**

## Step 5: Wait for Deploy

Render will:
1. Install Python dependencies
2. Start your FastAPI app
3. Give you a public URL like:
   ```
   https://multi-agent-genai.onrender.com
   ```

## Step 6: Test

```bash
curl https://multi-agent-genai.onrender.com/health
curl -X POST https://multi-agent-genai.onrender.com/api/query \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","query":"Hello","top_k":2}'
```

## Free Tier Limits

| Resource | Free Tier | Your Usage |
|----------|-----------|------------|
| Web Service | 750 hrs/month | ~720 hrs (24/7) ✅ |
| Bandwidth | 100GB/month | Small API ✅ |
| Build minutes | 500 min/month | ~5 min/build ✅ |
| PostgreSQL | 1GB storage | Small data ✅ |

## Notes
- Free web services **spin down after 15 min idle** (cold start ~30s)
- For always-on, need paid plan ($7/month)
- Database expires after 30 days on free tier (backup data!)

## Alternative: Railway (also easy)
- https://railway.app
- $5 free credit, then pay-as-you-go
- Better for always-on services
