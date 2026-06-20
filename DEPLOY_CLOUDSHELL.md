# Deploy using Google Cloud Shell (No local Docker needed)

If local Docker/gcloud is not available, use Google Cloud Shell which has all tools pre-installed.

## Step 1: Open Cloud Shell

1. Go to: https://console.cloud.google.com/?project=gen-lang-client-0113521438
2. Click the **Cloud Shell** icon (terminal icon) in the top-right corner
3. Wait for Cloud Shell to start

## Step 2: Upload your project

In Cloud Shell, run:

```bash
# Clone from GitHub (recommended)
git clone https://github.com/YOUR_USERNAME/multi-agent-genai.git
cd multi-agent-genai

# OR upload via Cloud Shell Editor
# Click "Open Editor" → Upload your project files
```

## Step 3: Set environment variables

```bash
export GROQ_API_KEY='your-groq-key-here'
export LLM_PROVIDER='groq'
export GROQ_MODEL='llama-3.1-8b-instant'
```

## Step 4: Build and deploy

```bash
# Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# Build and deploy in one command
gcloud run deploy multi-agent-genai \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="LLM_PROVIDER=groq,GROQ_API_KEY=${GROQ_API_KEY},GROQ_MODEL=llama-3.1-8b-instant,API_BEARER_TOKEN=,RATE_LIMIT_PER_MINUTE=30" \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 3 \
  --timeout 300
```

## Step 5: Get URL

After deployment, run:

```bash
gcloud run services describe multi-agent-genai --region us-central1 --format 'value(status.url)'
```

This will output your public URL like:
```
https://multi-agent-genai-xxx.a.run.app
```

## Step 6: Test

```bash
SERVICE_URL=$(gcloud run services describe multi-agent-genai --region us-central1 --format 'value(status.url)')
curl ${SERVICE_URL}/health
curl -X POST ${SERVICE_URL}/api/query \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","query":"Hello","top_k":2}'
```

## Alternative: Build image separately then deploy

```bash
# Build image
gcloud builds submit --tag gcr.io/gen-lang-client-0113521438/multi-agent-genai:latest --file docker/Dockerfile.cloudrun .

# Deploy image
gcloud run deploy multi-agent-genai \
  --image gcr.io/gen-lang-client-0113521438/multi-agent-genai:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="LLM_PROVIDER=groq,GROQ_API_KEY=${GROQ_API_KEY},GROQ_MODEL=llama-3.1-8b-instant" \
  --memory 2Gi \
  --cpu 1
```
