# Quick Deploy Script for Google Cloud Shell
# 
# INSTRUCTIONS:
# 1. Open https://console.cloud.google.com/cloudshell/editor?project=gen-lang-client-0113521438
# 2. Upload this project folder (or git clone)
# 3. In Cloud Shell terminal, run: bash deploy_cloudshell.sh

#!/bin/bash
set -e

PROJECT_ID="gen-lang-client-0113521438"
REGION="us-central1"
SERVICE_NAME="multi-agent-genai"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "=========================================="
echo "Deploy Multi-Agent GenAI to Cloud Run"
echo "=========================================="

# Set project
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "Enabling APIs..."
gcloud services enable run.googleapis.com || true
gcloud services enable artifactregistry.googleapis.com || true

# Create Artifact Registry repository if not exists
echo "Setting up Artifact Registry..."
gcloud artifacts repositories create cloud-run-source-deploy \
  --repository-format=docker \
  --location=${REGION} \
  --description="Docker repository for Cloud Run" || true

# Build image using local Docker (available in Cloud Shell)
echo "Building Docker image..."
docker build -t ${IMAGE_NAME} -f docker/Dockerfile.cloudrun .

# Push image
echo "Pushing image to Container Registry..."
docker push ${IMAGE_NAME}

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --region ${REGION} \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="LLM_PROVIDER=groq,GROQ_API_KEY=your-groq-key-here,GROQ_MODEL=llama-3.1-8b-instant,API_BEARER_TOKEN=,RATE_LIMIT_PER_MINUTE=30" \
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
