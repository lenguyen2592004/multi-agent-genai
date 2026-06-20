#!/usr/bin/env python3
"""Deploy Multi-Agent GenAI Platform to Google Cloud Run using REST API."""

import os
import sys
import json
import time
import base64
import subprocess
import http.client
import urllib.parse
from pathlib import Path

# Configuration
PROJECT_ID = "gen-lang-client-0113521438"
REGION = "us-central1"
SERVICE_NAME = "multi-agent-genai"
IMAGE_NAME = f"gcr.io/{PROJECT_ID}/{SERVICE_NAME}:latest"

# Paths
BASE_DIR = Path(__file__).resolve().parent
KEY_FILE = BASE_DIR / "gen-lang-client-0113521438-2984d5ff2d7a.json"
DOCKERFILE = BASE_DIR / "docker" / "Dockerfile.cloudrun"

# Environment variables for Cloud Run
ENV_VARS = {
    "LLM_PROVIDER": "groq",
    "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
    "GROQ_MODEL": "llama-3.1-8b-instant",
    "API_BEARER_TOKEN": "",
    "RATE_LIMIT_PER_MINUTE": "30",
}


def get_access_token():
    """Get access token from service account key."""
    if not KEY_FILE.exists():
        print(f"ERROR: Service account key not found: {KEY_FILE}")
        sys.exit(1)
    
    # Use gcloud or direct OAuth
    try:
        # Try using gcloud auth first
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    
    # Fallback: use Python google-auth
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
        
        credentials = service_account.Credentials.from_service_account_file(
            str(KEY_FILE),
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
        return credentials.token
    except ImportError:
        print("ERROR: google-auth library not installed.")
        print("Run: pip install google-auth google-auth-oauthlib")
        sys.exit(1)


def enable_apis(token):
    """Enable required APIs."""
    apis = ["run.googleapis.com", "cloudbuild.googleapis.com"]
    
    for api in apis:
        print(f"Enabling API: {api}...")
        conn = http.client.HTTPSConnection("serviceusage.googleapis.com")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        conn.request(
            "POST",
            f"/v1/projects/{PROJECT_ID}/services/{api}:enable",
            body="{}",
            headers=headers,
        )
        response = conn.getresponse()
        data = response.read().decode()
        conn.close()
        
        if response.status in [200, 202]:
            print(f"  [OK] {api} enabled")
        else:
            print("  [WARN] {api} - Status: {response.status}, may already be enabled")


def build_image(token):
    """Build Docker image using Cloud Build API."""
    print(f"Building image: {IMAGE_NAME}...")
    
    # Create build config
    build_config = {
        "source": {
            "storageSource": {
                "bucket": f"{PROJECT_ID}_cloudbuild",
                "object": "source.tar.gz",
            }
        },
        "steps": [
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": [
                    "build",
                    "-t", IMAGE_NAME,
                    "-f", str(DOCKERFILE.relative_to(BASE_DIR)),
                    "."
                ]
            },
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": ["push", IMAGE_NAME]
            }
        ],
        "images": [IMAGE_NAME],
    }
    
    # Alternative: Use gcloud builds submit if available
    try:
        result = subprocess.run(
            ["gcloud", "builds", "submit", "--tag", IMAGE_NAME, 
             "--file", str(DOCKERFILE), str(BASE_DIR)],
            capture_output=True, text=True, timeout=600, cwd=str(BASE_DIR)
        )
        if result.returncode == 0:
            print("  [OK] Build completed")
            return True
        else:
            print(f"  [ERROR] Build failed: {result.stderr}")
    except FileNotFoundError:
        print("  [WARN] gcloud not available, trying alternative...")
    
    return False


def deploy_cloud_run(token):
    """Deploy to Cloud Run."""
    print(f"Deploying to Cloud Run: {SERVICE_NAME}...")
    
    # Format environment variables
    env_list = [{"name": k, "value": v} for k, v in ENV_VARS.items()]
    
    service_config = {
        "apiVersion": "serving.knative.dev/v1",
        "kind": "Service",
        "metadata": {
            "name": SERVICE_NAME,
            "annotations": {
                "run.googleapis.com/ingress": "all",
            }
        },
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "autoscaling.knative.dev/maxScale": "3",
                        "run.googleapis.com/timeout": "300",
                    }
                },
                "spec": {
                    "containerConcurrency": 80,
                    "containers": [
                        {
                            "image": IMAGE_NAME,
                            "resources": {
                                "limits": {
                                    "cpu": "1",
                                    "memory": "2Gi",
                                }
                            },
                            "env": env_list,
                            "ports": [
                                {
                                    "containerPort": 8080,
                                }
                            ],
                        }
                    ],
                }
            },
            "traffic": [
                {
                    "percent": 100,
                    "latestRevision": True,
                }
            ]
        }
    }
    
    conn = http.client.HTTPSConnection(f"{REGION}-run.googleapis.com")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    # Try to update existing service first (PATCH)
    conn.request(
        "GET",
        f"/apis/serving.knative.dev/v1/namespaces/{PROJECT_ID}/services/{SERVICE_NAME}",
        headers=headers,
    )
    
    check_response = conn.getresponse()
    check_data = check_response.read().decode()
    check_response.close()
    
    if check_response.status == 200:
        # Service exists, use PATCH to update
        print(f"  Service exists, updating...")
        conn = http.client.HTTPSConnection(f"{REGION}-run.googleapis.com")
        
        # For update, we need to use the replace (PUT) or patch
        conn.request(
            "PUT",
            f"/apis/serving.knative.dev/v1/namespaces/{PROJECT_ID}/services/{SERVICE_NAME}",
            body=json.dumps(service_config),
            headers=headers,
        )
    else:
        # Create new service
        conn = http.client.HTTPSConnection(f"{REGION}-run.googleapis.com")
        conn.request(
            "POST",
            f"/apis/serving.knative.dev/v1/namespaces/{PROJECT_ID}/services",
            body=json.dumps(service_config),
            headers=headers,
        )
    
    response = conn.getresponse()
    data = response.read().decode()
    conn.close()
    
    if response.status in [200, 201, 202]:
        print("  [OK] Deployment initiated")
        result = json.loads(data)
        if "status" in result and "url" in result["status"]:
            url = result["status"]["url"]
            print(f"\n[OK] DEPLOYED SUCCESSFULLY!")
            print(f"URL: {url}")
            print(f"Health: {url}/health")
            print(f"UI: {url}/ui")
            print(f"Docs: {url}/docs")
            return url
        return True
    else:
        print(f"  [ERROR] Deployment failed: {response.status}")
        print(f"Response: {data}")
        return False


def main():
    """Main deployment flow."""
    print("=" * 60)
    print("Deploy Multi-Agent GenAI to Google Cloud Run")
    print("=" * 60)
    
    # Check prerequisites
    if not KEY_FILE.exists():
        print(f"\nERROR: Service account key not found:")
        print(f"  {KEY_FILE}")
        print("\nPlease ensure the key file exists.")
        sys.exit(1)
    
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
        print("\nWARNING: GEMINI_API_KEY not set!")
        print("Get free key at: https://aistudio.google.com/app/apikey")
        print("\nSet it with:")
        print("  $env:GEMINI_API_KEY='your-key'  (PowerShell)")
        print("  export GEMINI_API_KEY='your-key'  (bash)")
        
        # Try to read from .env file
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY="):
                        groq_key = line.split("=", 1)[1].strip('"\'')
                        ENV_VARS["GROQ_API_KEY"] = groq_key
                        print("\n[OK] Found GROQ_API_KEY in .env file")
                        break
                    elif line.startswith("GEMINI_API_KEY="):
                        gemini_key = line.split("=", 1)[1].strip('"\'')
                        ENV_VARS["GEMINI_API_KEY"] = gemini_key
                        print("\n[OK] Found GEMINI_API_KEY in .env file")
                        break
                    elif line.startswith("OPENAI_API_KEY="):
                        openai_key = line.split("=", 1)[1].strip('"\'')
                        ENV_VARS["OPENAI_API_KEY"] = openai_key
                        print("\n[OK] Found OPENAI_API_KEY in .env file")
                        break
        
        if not ENV_VARS.get("GROQ_API_KEY") and not ENV_VARS.get("GEMINI_API_KEY") and not ENV_VARS.get("OPENAI_API_KEY"):
            print("\nPlease set LLM API key in .env and try again.")
            sys.exit(1)
    else:
        ENV_VARS["GEMINI_API_KEY"] = gemini_key
    
    print(f"\nProject ID: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"Service: {SERVICE_NAME}")
    print(f"Image: {IMAGE_NAME}")
    print(f"Key file: {KEY_FILE}")
    print()
    
    # Get access token
    print("Authenticating...")
    token = get_access_token()
    print("  [OK] Authenticated")
    
    # Enable APIs
    enable_apis(token)
    
    # Build image
    build_success = build_image(token)
    if not build_success:
        print("\n[WARN] Build may have failed, but continuing to deploy...")
        print("  (Image might already exist from previous build)")
    
    # Deploy
    deploy_cloud_run(token)
    
    print("\n" + "=" * 60)
    print("Deployment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
