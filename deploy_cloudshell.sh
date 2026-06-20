#!/bin/bash
set -euo pipefail

# Quick Deploy Script for Google Cloud Shell
#
# INSTRUCTIONS:
# 1. Open https://console.cloud.google.com/cloudshell/editor?project=gen-lang-client-0113521438
# 2. Upload this project folder (or git clone)
# 3. Export GROQ_API_KEY in Cloud Shell
# 4. Run: bash deploy_cloudshell.sh

PROJECT_ID="gen-lang-client-0113521438"
REGION="us-central1"
SERVICE_NAME="multi-agent-genai"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  echo "ERROR: GROQ_API_KEY is not set."
  echo "Run: export GROQ_API_KEY='your-groq-key-here'"
  exit 1
fi

echo "=========================================="
echo "Deploy Multi-Agent GenAI to Cloud Run"
echo "=========================================="

# Set project
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "Enabling APIs..."
gcloud services enable run.googleapis.com || true
gcloud services enable cloudbuild.googleapis.com || true
gcloud services enable containerregistry.googleapis.com || true

# Build image in Cloud Build so Cloud Shell does not depend on local docker auth.
echo "Building Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} --file docker/Dockerfile.cloudrun .

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --region ${REGION} \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="LLM_PROVIDER=groq,GROQ_API_KEY=${GROQ_API_KEY},GROQ_MODEL=llama-3.1-8b-instant,API_BEARER_TOKEN=,RATE_LIMIT_PER_MINUTE=30" \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 3 \
  --timeout 300

# Get URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')

echo ""
echo "=========================================="
echo "DEPLOYMENT SUCCESSFUL!"
echo "=========================================="
echo "URL: ${SERVICE_URL}"
echo "Health: ${SERVICE_URL}/health"
echo "UI: ${SERVICE_URL}/ui"
echo "Docs: ${SERVICE_URL}/docs"
echo ""
echo "Test with:"
echo "curl ${SERVICE_URL}/health"
