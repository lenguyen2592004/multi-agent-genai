#!/usr/bin/env python3
"""Build Docker image and push to GCR using Cloud Build REST API."""

import os
import sys
import json
import tarfile
import io
import http.client
from pathlib import Path
from google.oauth2 import service_account
from google.auth.transport.requests import Request

PROJECT_ID = "gen-lang-client-0113521438"
IMAGE_NAME = f"gcr.io/{PROJECT_ID}/multi-agent-genai:latest"
BASE_DIR = Path(__file__).resolve().parent


def get_access_token():
    """Get access token from service account key."""
    key_file = BASE_DIR / "gen-lang-client-0113521438-2984d5ff2d7a.json"
    credentials = service_account.Credentials.from_service_account_file(
        str(key_file),
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials.token


def create_source_tarball():
    """Create a tarball of the source code."""
    print("Creating source tarball...")
    
    # Files to exclude
    exclude = {
        '.git', '__pycache__', '.venv', 'venv', 'env',
        'node_modules', '*.pyc', '.env', '.gitignore',
        'data/app.db', 'data/vector_store.json', 'logs/',
    }
    
    tarball = io.BytesIO()
    with tarfile.open(fileobj=tarball, mode='w:gz') as tar:
        for item in BASE_DIR.iterdir():
            if item.name in exclude or item.name.startswith('.'):
                continue
            tar.add(item, arcname=item.name)
    
    tarball.seek(0)
    return tarball


def upload_to_gcs(data, bucket_name, object_name, token):
    """Upload data to Google Cloud Storage."""
    print(f"Uploading to gs://{bucket_name}/{object_name}...")
    
    # Check if bucket exists, create if not
    conn = http.client.HTTPSConnection("storage.googleapis.com")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    conn.request("GET", f"/storage/v1/b/{bucket_name}", headers=headers)
    response = conn.getresponse()
    response.read()
    
    if response.status == 404:
        # Create bucket
        print(f"  Creating bucket {bucket_name}...")
        conn = http.client.HTTPSConnection("storage.googleapis.com")
        payload = json.dumps({
            "name": bucket_name,
            "location": "US",
            "storageClass": "STANDARD",
        })
        conn.request("POST", "/storage/v1/b?project=" + PROJECT_ID, 
                    body=payload, headers=headers)
        response = conn.getresponse()
        response.read()
        if response.status not in [200, 201]:
            print(f"  [ERROR] Failed to create bucket: {response.status}")
            return False
    
    # Upload object
    conn = http.client.HTTPSConnection("storage.googleapis.com")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-gzip",
    }
    
    conn.request("POST", f"/upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={object_name}",
                body=data, headers=headers)
    response = conn.getresponse()
    response.read()
    
    if response.status in [200, 201]:
        print(f"  [OK] Uploaded")
        return True
    else:
        print(f"  [ERROR] Upload failed: {response.status}")
        return False


def trigger_cloud_build(token, bucket_name, object_name):
    """Trigger Cloud Build using REST API."""
    print("Triggering Cloud Build...")
    
    build_config = {
        "source": {
            "storageSource": {
                "bucket": bucket_name,
                "object": object_name,
            }
        },
        "steps": [
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": [
                    "build",
                    "-t", IMAGE_NAME,
                    "-f", "docker/Dockerfile.cloudrun",
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
    
    conn = http.client.HTTPSConnection("cloudbuild.googleapis.com")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    conn.request(
        "POST",
        f"/v1/projects/{PROJECT_ID}/builds",
        body=json.dumps(build_config),
        headers=headers,
    )
    
    response = conn.getresponse()
    data = response.read().decode()
    conn.close()
    
    if response.status in [200, 201]:
        result = json.loads(data)
        build_id = result.get("id", "unknown")
        print(f"  [OK] Build triggered: {build_id}")
        return build_id
    else:
        print(f"  [ERROR] Build trigger failed: {response.status}")
        print(f"Response: {data}")
        return None


def wait_for_build(build_id, token, timeout=600):
    """Wait for Cloud Build to complete."""
    print(f"Waiting for build {build_id} to complete...")
    
    import time
    start = time.time()
    
    while time.time() - start < timeout:
        conn = http.client.HTTPSConnection("cloudbuild.googleapis.com")
        headers = {"Authorization": f"Bearer {token}"}
        conn.request("GET", f"/v1/projects/{PROJECT_ID}/builds/{build_id}", headers=headers)
        response = conn.getresponse()
        data = json.loads(response.read().decode())
        conn.close()
        
        status = data.get("status", "UNKNOWN")
        print(f"  Build status: {status}")
        
        if status == "SUCCESS":
            print("  [OK] Build completed successfully!")
            return True
        elif status in ["FAILURE", "CANCELLED", "EXPIRED"]:
            print(f"  [ERROR] Build failed: {status}")
            print(f"  Logs: {data.get('logUrl', 'N/A')}")
            return False
        
        time.sleep(10)
    
    print("  [WARN] Timeout waiting for build")
    return False


def main():
    print("=" * 60)
    print("Build Docker Image for Cloud Run")
    print("=" * 60)
    
    token = get_access_token()
    print("[OK] Authenticated")
    
    # Create source tarball
    tarball = create_source_tarball()
    
    # Upload to GCS
    bucket_name = f"{PROJECT_ID}_cloudbuild"
    object_name = "source.tar.gz"
    
    if not upload_to_gcs(tarball.read(), bucket_name, object_name, token):
        print("\n[ERROR] Failed to upload source")
        sys.exit(1)
    
    # Trigger build
    build_id = trigger_cloud_build(token, bucket_name, object_name)
    if not build_id:
        print("\n[ERROR] Failed to trigger build")
        sys.exit(1)
    
    # Wait for build
    success = wait_for_build(build_id, token)
    if not success:
        print("\n[ERROR] Build failed")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("[OK] Image built successfully!")
    print(f"Image: {IMAGE_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()
