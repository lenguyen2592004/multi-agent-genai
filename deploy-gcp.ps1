# Deploy Multi-Agent GenAI Platform to Google Cloud Run
# Using PowerShell + gcloud CLI (run in PowerShell, not bash)
#
# Prerequisites:
# 1. GCP Service Account Key: gen-lang-client-0113521438-2984d5ff2d7a.json
# 2. Gemini API Key (free from https://aistudio.google.com/app/apikey)
# 3. gcloud CLI installed

# Step 1: Set environment variables
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\Users\PHAMN55\OneDrive - Heineken International\Documents\multi-agent-genai\gen-lang-client-0113521438-2984d5ff2d7a.json"
$env:GCP_PROJECT_ID = "project-445fe1d3-0888-4fad-97e"
$env:GCP_REGION = "us-central1"
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # <-- REPLACE THIS

# Step 2: Authenticate with GCP
gcloud auth activate-service-account --key-file="$env:GOOGLE_APPLICATION_CREDENTIALS"
gcloud config set project $env:GCP_PROJECT_ID

# Step 3: Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Step 4: Build and deploy using Cloud Build
gcloud builds submit --tag "gcr.io/$env:GCP_PROJECT_ID/multi-agent-genai:latest" --file docker/Dockerfile.cloudrun .

# Step 5: Deploy to Cloud Run
gcloud run deploy multi-agent-genai `
  --image "gcr.io/$env:GCP_PROJECT_ID/multi-agent-genai:latest" `
  --region $env:GCP_REGION `
  --platform managed `
  --allow-unauthenticated `
  --set-env-vars="LLM_PROVIDER=groq,GROQ_API_KEY=$env:GROQ_API_KEY,GROQ_MODEL=llama-3.1-8b-instant,API_BEARER_TOKEN=,RATE_LIMIT_PER_MINUTE=30" `
  --memory 2Gi `
  --cpu 1 `
  --max-instances 3 `
  --timeout 300

# Step 6: Get the deployed URL
$serviceUrl = gcloud run services describe multi-agent-genai --region $env:GCP_REGION --format 'value(status.url)'
Write-Host "Deployed to: $serviceUrl"
Write-Host "Health check: $serviceUrl/health"
Write-Host "UI: $serviceUrl/ui"
Write-Host "API Docs: $serviceUrl/docs"
