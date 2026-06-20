# Deploy Multi-Agent GenAI Platform lên Google Cloud Run

## Tại sao chọn GCP Cloud Run?

- ✅ **Free forever**: 2M requests/tháng, 180K vCPU-seconds, 360K GB-seconds
- ✅ **Container-native**: Chạy Docker container trực tiếp
- ✅ **Auto-scaling**: 0 → n instances tự động
- ✅ **FastAPI hoàn hảo**: Hỗ trợ ASGI frameworks
- ✅ **CV Value cao**: Google Cloud AI/ML rất hot ở VN
- ✅ **Dễ nhất trong 3 cloud lớn**

---

## Architecture trên GCP

```
┌─────────────────────────────────────────┐
│           Cloud Run (FastAPI)           │
│         Container: Python 3.13          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ FastAPI │  │LangGraph│  │  RAG    │ │
│  │  App    │  │ Agents  │  │Pipeline │ │
│  └────┬────┘  └────┬────┘  └────┬────┘ │
│       │            │            │       │
│  ┌────┴────────────┴────────────┘       │
│  │      Cloud Run Service (auto-scale)  │
│  └─────────────────────────────────────┘
└─────────────────────────────────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌─────────┐   ┌─────────────┐
│Firestore│   │Cloud Storage│
│(NoSQL)  │   │ (PDF files) │
└─────────┘   └─────────────┘
            │
            ▼
    ┌───────────────┐
    │  Vertex AI /    │
    │  OpenAI API     │
    │  (External LLM) │
    └───────────────┘
```

---

## Bước 1: Chuẩn bị

### 1.1 Tạo Google Cloud Account

1. Vào [https://cloud.google.com/free](https://cloud.google.com/free)
2. Click "Get started for free"
3. Đăng nhập bằng Gmail
4. Nhập thông tin + credit card (verify, **không charge** cho free tier)
5. Nhận **$300 credit** cho 90 ngày (bonus!)

### 1.2 Cài đặt Google Cloud CLI

```bash
# Windows (PowerShell - Admin)
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& $env:Temp\GoogleCloudSDKInstaller.exe

# Sau khi cài xong, restart terminal
# Login
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 1.3 Enable APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable firestore.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

---

## Bước 2: Thay đổi code cho Cloud Run

### 2.1 Tạo `cloudbuild.yaml`

```yaml
steps:
  # Build container image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/multi-agent-genai:$COMMIT_SHA'
      - '-f'
      - 'docker/Dockerfile'
      - '.'
  
  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - 'gcr.io/$PROJECT_ID/multi-agent-genai:$COMMIT_SHA'
  
  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'multi-agent-genai'
      - '--image'
      - 'gcr.io/$PROJECT_ID/multi-agent-genai:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--set-env-vars'
      - 'LLM_PROVIDER=openai,OPENAI_API_KEY=${_OPENAI_API_KEY},FIRESTORE_PROJECT_ID=$PROJECT_ID'
      - '--memory'
      - '1Gi'
      - '--cpu'
      - '1'
      - '--max-instances'
      - '5'
      - '--concurrency'
      - '80'

images:
  - 'gcr.io/$PROJECT_ID/multi-agent-genai:$COMMIT_SHA'

substitutions:
  _OPENAI_API_KEY: ""  # Set when triggering build
```

### 2.2 Tạo `Dockerfile.cloudrun` (optimized)

```dockerfile
# Multi-stage build for smaller image
FROM python:3.13-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.13-slim

WORKDIR /app

# Copy only necessary files
COPY --from=builder /root/.local /root/.local
COPY . .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Cloud Run sets PORT environment variable
ENV PORT=8080
EXPOSE 8080

# Use uvicorn with optimized settings for Cloud Run
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --loop uvloop
```

### 2.3 Tạo `api/cloud_config.py` - Cloud adapter

```python
"""Cloud-specific configuration adapters."""

import os
from typing import Optional


def get_database_config():
    """Return database config based on environment."""
    cloud_provider = os.getenv("CLOUD_PROVIDER", "local")
    
    if cloud_provider == "gcp":
        return {
            "type": "firestore",
            "project_id": os.getenv("FIRESTORE_PROJECT_ID"),
            "collection": "documents",
        }
    elif cloud_provider == "aws":
        return {
            "type": "dynamodb",
            "region": os.getenv("AWS_REGION", "us-east-1"),
            "table_name": os.getenv("DYNAMODB_TABLE", "documents"),
        }
    else:
        return {
            "type": "sqlite",
            "path": "data/app.db",
        }


def get_vector_store_config():
    """Return vector store config."""
    cloud_provider = os.getenv("CLOUD_PROVIDER", "local")
    
    if cloud_provider == "gcp":
        return {
            "type": "firestore",
            "project_id": os.getenv("FIRESTORE_PROJECT_ID"),
        }
    else:
        return {
            "type": "json",
            "path": "data/vector_store.json",
        }


def get_storage_config():
    """Return file storage config."""
    cloud_provider = os.getenv("CLOUD_PROVIDER", "local")
    
    if cloud_provider == "gcp":
        return {
            "type": "gcs",
            "bucket": os.getenv("GCS_BUCKET_NAME"),
        }
    elif cloud_provider == "aws":
        return {
            "type": "s3",
            "bucket": os.getenv("S3_BUCKET_NAME"),
        }
    else:
        return {
            "type": "local",
            "path": "data/uploads",
        }
```

### 2.4 Tạo `rag/firestore_vector_store.py`

```python
"""Firestore-based vector store for GCP deployment."""

import os
from typing import Dict, List, Any, Optional

try:
    from google.cloud import firestore
    from google.cloud.firestore_v1.vector import Vector
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False

from rag.embeddings import EmbeddingModel


class FirestoreVectorStore:
    """Vector store using Firestore (free tier: 1GB storage, 50K reads/day)."""
    
    def __init__(self, collection_name: str = "embeddings"):
        if not FIRESTORE_AVAILABLE:
            raise ImportError("google-cloud-firestore required. Install: pip install google-cloud-firestore")
        
        self.db = firestore.Client(project=os.getenv("FIRESTORE_PROJECT_ID"))
        self.collection = self.db.collection(collection_name)
        self.embedder = EmbeddingModel()
    
    def add(self, document_id: str, chunk_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """Add a document chunk with embedding."""
        embedding = self.embedder.embed(text)
        
        doc_ref = self.collection.document(f"{document_id}_{chunk_id}")
        doc_ref.set({
            "document_id": document_id,
            "chunk_id": chunk_id,
            "text": text,
            "embedding": embedding,  # Firestore supports array fields
            "metadata": metadata,
        })
    
    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Search using cosine similarity (brute-force for small collections)."""
        query_embedding = self.embedder.embed(query)
        
        # For small collections, fetch all and compute similarity
        # For production, use Firestore vector search or Vertex AI Matching Engine
        docs = self.collection.stream()
        
        results = []
        for doc in docs:
            data = doc.to_dict()
            doc_embedding = data.get("embedding", [])
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            results.append({
                "document_id": data["document_id"],
                "chunk_id": data["chunk_id"],
                "text": data["text"],
                "metadata": data.get("metadata", {}),
                "similarity": similarity,
            })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
```

### 2.5 Cập nhật `requirements.txt` cho cloud

```txt
fastapi>=0.116,<0.117
uvicorn[standard]>=0.35,<0.36
pydantic>=2.11,<2.12
langgraph>=0.4,<0.5
requests>=2.32,<2.33
pytest>=8.3,<8.4
httpx>=0.27,<0.28
pypdf
python-multipart

# Cloud dependencies (install as needed)
google-cloud-firestore>=2.16,<2.17
google-cloud-storage>=2.18,<2.19
openai>=1.0,<2.0
redis>=5.0,<6.0
```

---

## Bước 3: Deploy

### 3.1 Cách 1: Cloud Build (Recommended)

```bash
# Submit build và deploy
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_OPENAI_API_KEY="sk-your-key-here"
```

### 3.2 Cách 2: Local Docker + Push

```bash
# Build locally
docker build -t gcr.io/PROJECT_ID/multi-agent-genai -f docker/Dockerfile.cloudrun .

# Push to Google Container Registry
docker push gcr.io/PROJECT_ID/multi-agent-genai

# Deploy to Cloud Run
gcloud run deploy multi-agent-genai \
  --image gcr.io/PROJECT_ID/multi-agent-genai \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="LLM_PROVIDER=openai,OPENAI_API_KEY=sk-xxx" \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 5
```

### 3.3 Cách 3: GitHub Actions (CI/CD)

Tạo `.github/workflows/deploy-gcp.yml`:

```yaml
name: Deploy to Google Cloud Run

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Google Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
        with:
          project_id: ${{ secrets.GCP_PROJECT_ID }}
          service_account_key: ${{ secrets.GCP_SA_KEY }}
      
      - name: Configure Docker
        run: gcloud auth configure-docker
      
      - name: Build and Push
        run: |
          docker build -t gcr.io/${{ secrets.GCP_PROJECT_ID }}/multi-agent-genai:${{ github.sha }} -f docker/Dockerfile.cloudrun .
          docker push gcr.io/${{ secrets.GCP_PROJECT_ID }}/multi-agent-genai:${{ github.sha }}
      
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy multi-agent-genai \
            --image gcr.io/${{ secrets.GCP_PROJECT_ID }}/multi-agent-genai:${{ github.sha }} \
            --region us-central1 \
            --platform managed \
            --allow-unauthenticated \
            --set-env-vars="LLM_PROVIDER=openai,OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}" \
            --memory 1Gi \
            --cpu 1
```

---

## Bước 4: Verify

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe multi-agent-genai --region us-central1 --format 'value(status.url)')

# Test health
curl $SERVICE_URL/health

# Test query
curl -X POST $SERVICE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","query":"Hello","top_k":2}'

# Open UI
start $SERVICE_URL/ui
```

---

## Bước 5: Monitoring & Billing

```bash
# Xem logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=multi-agent-genai" --limit=50

# Xem metrics
gcloud monitoring metrics list --filter="metric.type:run.googleapis.com"

# Set billing alert (QUAN TRỌNG!)
# Vào Cloud Console → Billing → Budgets & alerts → Create budget
# Set: $0.01 alert (sẽ email khi có charge)
```

---

## Free Tier Limits (Cloud Run)

| Resource | Free Tier | Project của bạn dùng bao nhiêu? |
|----------|-----------|--------------------------------|
| Requests | 2 triệu/tháng | ~1000 requests/ngày = 30K/tháng ✅ |
| vCPU-seconds | 180,000/tháng | 1 vCPU × 24h × 30d = 2.5M ❌ (need limit) |
| GB-seconds | 360,000/tháng | 1GB × 24h × 30d = 2.5M ❌ (need limit) |
| Egress (US) | 1 GB/tháng | Nhỏ với API JSON ✅ |

**⚠️ QUAN TRỌNG**: Cloud Run free tier là "per month" nhưng tính theo **usage**. Nếu bạn để chạy 24/7 sẽ vượt quá free tier!

**Giải pháp:**
- Set `--max-instances 1` hoặc `--no-traffic` khi không dùng
- Dùng `--cpu-boost` chỉ khi cần
- Hoặc: Deploy lên Oracle Cloud ARM VM cho 24/7 free

---

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| Cold start chậm | Thêm `--min-instances 1` (nhưng tốn tiền) |
| Memory exceeded | Tăng `--memory 2Gi` hoặc optimize code |
| Timeout | Cloud Run max 3600s, nhưng HTTP request nên < 30s |
| Firestore quota exceeded | Đợi 24h reset hoặc upgrade |
| Container fails | Check logs: `gcloud logging read` |

---

## Next Steps

1. ✅ Deploy thành công
2. ✅ Test endpoints
3. ✅ Ghi URL vào CV
4. ➡️ Consider: Thêm custom domain (Cloud Run hỗ trợ miễn phí)
5. ➡️ Consider: Thêm Cloud CDN cho static assets
6. ➡️ Consider: Thêm Cloud Monitoring dashboards

---

## Resources

- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Cloud Run Quickstart](https://cloud.google.com/run/docs/quickstarts/build-and-deploy)
- [Firestore Free Tier](https://cloud.google.com/firestore/pricing)
- [GCP Free Tier](https://cloud.google.com/free)
