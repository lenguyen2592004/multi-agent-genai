# Deploy to Cloud Run using gcloud buildpacks (No Docker needed)
# Run this in Google Cloud Shell terminal

# Step 1: Set project
gcloud config set project gen-lang-client-0113521438

# Step 2: Enable APIs
gcloud services enable run.googleapis.com

# Step 3: Deploy directly using gcloud buildpacks (auto-detects Python)
gcloud run deploy multi-agent-genai \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="LLM_PROVIDER=groq,GROQ_API_KEY=your-groq-key-here,GROQ_MODEL=llama-3.1-8b-instant,API_BEARER_TOKEN=,RATE_LIMIT_PER_MINUTE=30" \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 3 \
  --timeout 300

# Step 4: Get URL
SERVICE_URL=$(gcloud run services describe multi-agent-genai --region us-central1 --format 'value(status.url)')
echo "Deployed to: $SERVICE_URL"
echo "Health: $SERVICE_URL/health"
echo "UI: $SERVICE_URL/ui"
echo "Docs: $SERVICE_URL/docs"
