# Deploy Multi-Agent GenAI to Google Cloud Run (Console Method)

Since Cloud Build API is blocked by organization policy, use this manual method via Google Cloud Console.

## Step 1: Prepare Source Code

1. Zip your project folder (excluding .git, __pycache__, .venv)
2. Or use the prepared source: Upload to Cloud Storage

## Step 2: Upload Source to Cloud Storage

1. Go to https://console.cloud.google.com/storage/browser?project=gen-lang-client-0113521438
2. Create bucket: `gen-lang-client-0113521438-source`
3. Upload your project zip file

## Step 3: Use Cloud Run Console Deploy

1. Go to https://console.cloud.google.com/run?project=gen-lang-client-0113521438
2. Click **CREATE SERVICE**
3. Select **Deploy one revision from an existing container image**
4. Since we can't build, use a pre-built Python image and override command

### Alternative: Use Cloud Shell Editor (has all tools)

1. Go to https://console.cloud.google.com/cloudshell/editor?project=gen-lang-client-0113521438
2. In the editor, open terminal (Ctrl+`)
3. Clone your repo or upload files
4. Run:

```bash
# Set project
gcloud config set project gen-lang-client-0113521438

# Enable APIs (if not already enabled)
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# Build using local Docker (Cloud Shell has Docker)
gcloud builds submit --tag gcr.io/gen-lang-client-0113521438/multi-agent-genai:latest

# Or build with local Docker
docker build -t gcr.io/gen-lang-client-0113521438/multi-agent-genai:latest -f docker/Dockerfile.cloudrun .
docker push gcr.io/gen-lang-client-0113521438/multi-agent-genai:latest

# Deploy
gcloud run deploy multi-agent-genai \
  --image gcr.io/gen-lang-client-0113521438/multi-agent-genai:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="LLM_PROVIDER=groq,GROQ_API_KEY=your-groq-key-here,GROQ_MODEL=llama-3.1-8b-instant" \
  --memory 2Gi \
  --cpu 1
```

## Step 4: Verify Deployment

```bash
# Get URL
SERVICE_URL=$(gcloud run services describe multi-agent-genai --region us-central1 --format 'value(status.url)')
echo $SERVICE_URL

# Test
curl $SERVICE_URL/health
curl -X POST $SERVICE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","query":"Hello","top_k":2}'
```

## Troubleshooting Organization Policy

If Cloud Build is blocked, try:

1. **Use Cloud Shell** (has all tools pre-installed)
2. **Use Artifact Registry** instead of Container Registry
3. **Request policy exception** from organization admin

## Alternative: Deploy Pre-built Image

If you have a pre-built image, deploy directly:

```bash
gcloud run deploy multi-agent-genai \
  --image gcr.io/gen-lang-client-0113521438/multi-agent-genai:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated
```
