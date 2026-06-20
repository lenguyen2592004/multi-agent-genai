# Cloud Deployment Guide - Multi-Agent GenAI Platform

## Tóm tắt: Nền tảng nào miễn phí + CV value cao nhất?

| STT | Nền tảng | Chi phí | Độ khó | CV Value | Khuyến nghị |
|-----|----------|---------|--------|----------|-------------|
| 1 | **Google Cloud Run** | ✅ Free forever | ⭐⭐ Dễ | ⭐⭐⭐⭐⭐ Rất cao | **#1 Khuyến nghị** |
| 2 | **Oracle Cloud (OCI)** | ✅ Free forever | ⭐⭐⭐⭐ Khó | ⭐⭐⭐⭐ Cao | Mạnh nhất, khó hơn |
| 3 | **AWS Lambda + API Gateway** | ✅ Free forever | ⭐⭐⭐⭐ Khó | ⭐⭐⭐⭐⭐ Rất cao | CV đỉnh, phức tạp |
| 4 | **Render** | ✅ Free | ⭐⭐ Dễ | ⭐⭐⭐ Trung bình | Deploy nhanh nhất |
| 5 | **Railway** | ❌ $5/tháng | ⭐⭐ Dễ | ⭐⭐⭐ Trung bình | Không free |

---

## 🏆 TOP 3 Lựa chọn tốt nhất cho bạn

### 1. Google Cloud Run (KHUYẾN NGHỊ #1)

**Tại sao chọn GCP Cloud Run:**
- ✅ **Free forever**: 2 triệu requests/tháng, 180K vCPU-seconds, 360K GB-seconds
- ✅ **Container-native**: Hỗ trợ Docker, auto-scale
- ✅ **FastAPI hoàn hảo**: Chạy container Python, cold start nhanh
- ✅ **CV Value cao**: Google Cloud AI/ML rất hot ở VN
- ✅ **Dễ deploy**: `gcloud run deploy` 1 lệnh là xong
- ✅ **Domain miễn phí**: `https://your-app.a.run.app`

**Chi phí thực tế:**
- Dưới 2M requests/tháng = **$0**
- Bandwidth egress 1GB/tháng miễn phí (US regions)

**Nhược điểm:**
- Cold start ~2-5 giây (chấp nhận được)
- Phải dùng US regions cho free tier
- Database phải dùng Firestore (NoSQL) hoặc Cloud SQL (trả phí)

---

### 2. Oracle Cloud Infrastructure (OCI) - Always Free

**Tại sao chọn OCI:**
- ✅ **Mạnh nhất free tier**: 2 ARM VMs (4 OCPU + 24GB RAM total)
- ✅ **Always Free thật sự**: Không hết hạn
- ✅ **2 Database miễn phí**: Oracle Autonomous DB
- ✅ **10TB bandwidth/tháng**
- ✅ **CV Value**: Oracle là enterprise cloud lớn, đặc biệt ở VN (ngân hàng, chính phủ)

**Chi phí thực tế:**
- 100% miễn phí nếu stay trong limits
- Cần credit card verify (không charge nếu đúng free tier)

**Nhược điểm:**
- UI phức tạp, enterprise-style
- ARM instances hay hết capacity ở popular regions
- Oracle DB không phổ biến bằng PostgreSQL
- Khó hơn GCP/AWS để setup

---

### 3. AWS Lambda + API Gateway (Serverless)

**Tại sao chọn AWS:**
- ✅ **#1 Cloud Provider toàn cầu**
- ✅ **Free tier hào phóng**: 1M Lambda requests/tháng forever
- ✅ **DynamoDB miễn phí**: 25GB storage forever
- ✅ **CV Value đỉnh nhất**: AWS cert là "vàng" trên CV
- ✅ **Serverless**: Không cần quản lý server

**Chi phí thực tế:**
- Lambda: 1M requests free/tháng
- API Gateway: 1M requests free (12 tháng), sau đó ~$3.5/1M requests
- DynamoDB: 25GB free forever

**Nhược điểm:**
- **Phức tạp nhất**: IAM, VPC, Security Groups, CloudFormation
- Dễ "accidentally" tốn tiền (NAT Gateway $32/tháng, Elastic IP $3.6/tháng)
- Lambda timeout 15 phút (không phù hợp long-running tasks)
- Cold start có thể chậm với Python

---

## 📊 So sánh chi tiết cho project của bạn

Project: FastAPI + LangGraph + RAG + SQLite + Ollama

| Yêu cầu | Cloud Run | OCI | AWS Lambda |
|---------|-----------|-----|------------|
| **FastAPI container** | ✅ Native | ✅ Docker trên VM | ⚠️ Via Lambda container |
| **Database** | Firestore (NoSQL) | Oracle Autonomous DB | DynamoDB |
| **Vector Store** | Firestore hoặc in-memory | JSON file hoặc Oracle DB | DynamoDB |
| **LLM (Ollama)** | ❌ Không chạy được | ✅ Chạy trên ARM VM | ❌ Không chạy được |
| **File Upload (PDF)** | Cloud Storage (trả phí) | Block Storage (free) | S3 (trả phí) |
| **Custom Domain** | ✅ Miễn phí | ✅ Miễn phí | CloudFront (trả phí) |
| **SSL/HTTPS** | ✅ Auto | ✅ Auto | ✅ Auto |

---

## 🔥 Quyết định cuối cùng

### Scenario A: Zero cost, CV cao, dễ deploy
→ **Google Cloud Run** + Firestore
- Deploy FastAPI container
- Thay SQLite bằng Firestore
- Thay Ollama bằng OpenAI API (free tier $5) hoặc Gemini API (free tier)

### Scenario B: Zero cost, mạnh nhất, chấp nhận khó hơn
→ **Oracle Cloud ARM VM** + Docker Compose
- Chạy toàn bộ stack (FastAPI + Ollama + PostgreSQL) trên 1 VM
- 4 OCPU + 24GB RAM đủ cho everything
- Giữ nguyên architecture local-first

### Scenario C: CV đỉnh nhất, chấp nhận phức tạp
→ **AWS Lambda + API Gateway + DynamoDB**
- Refactor FastAPI thành Lambda functions
- Thay SQLite bằng DynamoDB
- Thay Ollama bằng Bedrock/Anthropic API

---

## 📝 Script deploy nhanh

### Cloud Run (1 lệnh deploy)

```bash
# 1. Build Docker image
gcloud builds submit --tag gcr.io/PROJECT_ID/multi-agent-genai

# 2. Deploy
gcloud run deploy multi-agent-genai \
  --image gcr.io/PROJECT_ID/multi-agent-genai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="LLM_PROVIDER=openai,OPENAI_API_KEY=sk-xxx"
```

### Oracle Cloud (Terraform)

```bash
# 1. Tạo VM
oci compute instance launch \
  --availability-domain "AD-1" \
  --shape "VM.Standard.A1.Flex" \
  --shape-config '{"ocpus":4, "memory_in_gbs":24}' \
  --image-id "ocid1.image..." \
  --subnet-id "ocid1.subnet..."

# 2. SSH vào, cài Docker, chạy docker-compose
ssh opc@IP_ADDRESS
docker-compose up -d
```

---

## 💡 Tips cho phỏng vấn

Khi được hỏi về cloud deployment, bạn có thể nói:

> "Tôi đã deploy multi-agent GenAI platform trên **Google Cloud Run** với containerized FastAPI, sử dụng **Firestore** cho NoSQL data và **Cloud Storage** cho file uploads. Tôi chọn Cloud Run vì nó serverless, auto-scaling, và phù hợp với microservices architecture. Tôi cũng có kinh nghiệm với **Oracle Cloud** ARM instances cho workloads cần compute mạnh mẽ hơn."

**Keywords để nhắc trong phỏng vấn:**
- Containerization (Docker)
- Serverless computing
- Cloud-native architecture
- Auto-scaling
- Managed database
- CI/CD pipeline
- Infrastructure as Code (Terraform)
- Cost optimization (free tier utilization)

---

## 📚 Next Steps

1. Chọn 1 nền tảng (khuyến nghị: **Google Cloud Run**)
2. Tạo account (cần credit card verify nhưng không charge)
3. Install CLI tools (`gcloud`, `oci`, `aws`)
4. Deploy theo hướng dẫn chi tiết trong thư mục `infrastructure/`
5. Test endpoints
6. Ghi lại URL và đưa vào CV/Portfolio

---

## ⚠️ Cảnh báo quan trọng

**Đừng bao giờ:**
- Để resources chạy qua giới hạn free tier
- Quên monitor billing dashboard
- Sử dụng Elastic IP không attach vào instance (AWS charge $3.6/tháng)
- Tạo NAT Gateway không cần thiết (AWS charge $32+/tháng)
- Để Oracle Autonomous DB idle quá 90 ngày

**Luôn luôn:**
- Set billing alerts (100% free tier usage)
- Dùng Terraform/CloudFormation để dễ dàng destroy
- Tắt resources khi không dùng (trừ Always Free)
- Kiểm tra billing dashboard hàng tuần
