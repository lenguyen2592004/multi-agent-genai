# Deploy Multi-Agent GenAI Platform lên AWS Lambda + API Gateway

## Tại sao chọn AWS?

- ✅ **#1 Cloud Provider toàn cầu**: Được nhắc nhiều nhất trong JD
- ✅ **Free tier hào phóng**: 1M Lambda requests/tháng **forever**
- ✅ **DynamoDB miễn phí**: 25GB storage **forever**
- ✅ **CV Value đỉnh nhất**: AWS cert là "vàng" trên CV
- ✅ **Serverless**: Không quản lý server, auto-scale tự động

---

## Architecture trên AWS (Serverless)

```
┌─────────────────────────────────────────────────┐
│                    AWS Cloud                       │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │         Amazon API Gateway (HTTP API)      │   │
│  │              (Free: 1M requests)           │   │
│  └───────────────────┬──────────────────────┘   │
│                      │                           │
│  ┌───────────────────▼──────────────────────┐   │
│  │         AWS Lambda (Python 3.13)          │   │
│  │         (Free: 1M requests/month)        │   │
│  │                                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
│  │  │ FastAPI │  │LangGraph│  │  RAG    │  │   │
│  │  │ (Mangum)│  │ Agents  │  │Pipeline │  │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  │   │
│  │       │            │            │       │   │
│  │  ┌────┴────────────┴────────────┘       │   │
│  │  │    Lambda Function (container)       │   │
│  │  └─────────────────────────────────────┘   │
│  └───────────────────┬──────────────────────┘   │
│                      │                           │
│  ┌───────────────────▼──────────────────────┐   │
│  │         Amazon DynamoDB                   │   │
│  │    (Free: 25GB + 200M read/write)       │   │
│  │                                          │   │
│  │  ┌─────────────┐  ┌──────────────────┐  │   │
│  │  │  Documents  │  │  Request Logs    │  │   │
│  │  │  (RAG data) │  │  (Metrics)       │  │   │
│  │  └─────────────┘  └──────────────────┘  │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │         Amazon S3 (Optional)              │   │
│  │    (Free: 5GB + 20K GET requests)         │   │
│  │    PDF uploads, static assets             │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │         Amazon Bedrock (LLM)              │   │
│  │    OR OpenAI API (external)               │   │
│  │    (Pay-per-use, không có free tier)      │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## ⚠️ QUAN TRỌNG: AWS Free Tier Gotchas

**DỄ TỐN TIỀN NHẤT:**

| Service | Chi phí ẩn | Cách tránh |
|---------|-----------|------------|
| **NAT Gateway** | $32.40/tháng | Dùng public subnet, không cần NAT |
| **Elastic IP** | $3.60/tháng/IP | Không attach → bị charge. Release khi không dùng |
| **Data Transfer** | $0.09/GB | Giữ trong cùng region |
| **CloudWatch Logs** | $0.50/GB ingested | Giảm log retention |
| **API Gateway REST** | $3.50/1M requests | Dùng **HTTP API** ($1/1M) |
| **Route 53** | $0.50/zone/tháng | Dùng free DNS khác |

**Rule #1**: Không tạo VPC với NAT Gateway cho project này!
**Rule #2**: Luôn dùng HTTP API thay vì REST API!
**Rule #3**: Set billing alert ngay sau khi tạo account!

---

## Bước 1: Tạo AWS Account

1. Vào [https://aws.amazon.com/free/](https://aws.amazon.com/free/)
2. Click "Create a Free Account"
3. Đăng ký với email + credit card (verify)
4. Chọn **Basic Support** (free)
5. **QUAN TRỌNG**: Vào Billing → Budgets → Create budget
   - Budget type: Zero spend budget
   - Alert: 100% of budget (email khi có charge)

---

## Bước 2: Cài AWS CLI + SAM

```bash
# Windows (PowerShell)
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi

# Verify
aws --version

# Configure
aws configure
# AWS Access Key ID: [tạo trong IAM]
# AWS Secret Access Key: [tạo trong IAM]
# Default region: ap-southeast-1 (Singapore, gần VN)
# Default output: json

# Cài SAM CLI (Serverless Application Model)
# https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html
winget install Amazon.SAM-CLI

# Verify
sam --version
```

---

## Bước 3: Refactor Code cho Lambda

### 3.1 Thêm `mangum` adapter (FastAPI → Lambda)

```txt
# requirements.txt (thêm)
mangum>=0.17,<0.18
```

### 3.2 Tạo `api/lambda_handler.py`

```python
"""AWS Lambda handler for FastAPI application."""

from mangum import Mangum
from api.main import app

# Mangum adapter converts API Gateway events to ASGI
handler = Mangum(app, lifespan="off")
```

### 3.3 Tạo `template.yaml` (SAM)

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Multi-Agent GenAI Platform

Globals:
  Function:
    Timeout: 30
    MemorySize: 1024
    Runtime: python3.12
    Architectures:
      - x86_64
    Environment:
      Variables:
        PYTHONPATH: /var/task
        DYNAMODB_TABLE: !Ref DocumentsTable
        LOG_LEVEL: INFO

Parameters:
  OpenAIApiKey:
    Type: String
    NoEcho: true
    Description: OpenAI API Key (for LLM)

Resources:
  # DynamoDB Table for RAG documents
  DocumentsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: genai-documents
      BillingMode: PAY_PER_REQUEST  # Free tier: 25GB + 2.5M read/write
      AttributeDefinitions:
        - AttributeName: pk
          AttributeType: S
        - AttributeName: sk
          AttributeType: S
      KeySchema:
        - AttributeName: pk
          KeyType: HASH
        - AttributeName: sk
          KeyType: RANGE
      StreamSpecification:
        StreamViewType: NEW_AND_OLD_IMAGES

  # DynamoDB Table for request logs
  LogsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: genai-logs
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: trace_id
          AttributeType: S
      KeySchema:
        - AttributeName: trace_id
          KeyType: HASH
      TimeToLiveSpecification:
        AttributeName: ttl
        Enabled: true

  # Lambda Function
  GenAIFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: multi-agent-genai
      CodeUri: .
      Handler: api.lambda_handler.handler
      Description: Multi-Agent GenAI Platform
      Environment:
        Variables:
          OPENAI_API_KEY: !Ref OpenAIApiKey
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref DocumentsTable
        - DynamoDBCrudPolicy:
            TableName: !Ref LogsTable
      Events:
        # HTTP API (cheaper than REST API)
        ApiEvent:
          Type: HttpApi
          Properties:
            Path: /{proxy+}
            Method: ANY
            PayloadFormatVersion: "2.0"

  # S3 Bucket for PDF uploads (optional)
  UploadsBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "genai-uploads-${AWS::AccountId}"
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      LifecycleConfiguration:
        Rules:
          - Id: DeleteOldUploads
            Status: Enabled
            ExpirationInDays: 7

Outputs:
  ApiUrl:
    Description: API Gateway endpoint URL
    Value: !Sub "https://${ServerlessHttpApi}.execute-api.${AWS::Region}.amazonaws.com/"
  
  DocumentsTable:
    Description: DynamoDB table for documents
    Value: !Ref DocumentsTable
  
  UploadsBucket:
    Description: S3 bucket for uploads
    Value: !Ref UploadsBucket
```

### 3.4 Tạo DynamoDB adapter cho RAG

```python
# rag/dynamo_vector_store.py
"""DynamoDB-based vector store for AWS deployment."""

import os
import json
import time
from typing import Dict, List, Any, Optional
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

from rag.embeddings import EmbeddingModel


class DynamoDBVectorStore:
    """Vector store using DynamoDB (free tier: 25GB storage)."""
    
    def __init__(self, table_name: Optional[str] = None):
        self.table_name = table_name or os.getenv("DYNAMODB_TABLE", "genai-documents")
        self.dynamodb = boto3.resource("dynamodb")
        self.table = self.dynamodb.Table(self.table_name)
        self.embedder = EmbeddingModel()
    
    def add(self, document_id: str, chunk_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """Add a document chunk with embedding."""
        embedding = self.embedder.embed(text)
        
        # Convert embedding to Decimal for DynamoDB
        decimal_embedding = [Decimal(str(x)) for x in embedding]
        
        item = {
            "pk": f"DOC#{document_id}",
            "sk": f"CHUNK#{chunk_id}",
            "text": text,
            "embedding": decimal_embedding,
            "metadata": metadata,
            "created_at": Decimal(str(time.time())),
        }
        
        self.table.put_item(Item=item)
    
    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Search using cosine similarity."""
        query_embedding = self.embedder.embed(query)
        
        # For small collections, scan all (DynamoDB free tier allows 25GB)
        # For production, use OpenSearch or ElastiCache
        response = self.table.scan()
        items = response.get("Items", [])
        
        # Handle pagination
        while "LastEvaluatedKey" in response:
            response = self.table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response.get("Items", []))
        
        results = []
        for item in items:
            doc_embedding = [float(x) for x in item.get("embedding", [])]
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            results.append({
                "document_id": item["pk"].replace("DOC#", ""),
                "chunk_id": item["sk"].replace("CHUNK#", ""),
                "text": item["text"],
                "metadata": item.get("metadata", {}),
                "similarity": similarity,
            })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        import math
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
```

### 3.5 Tạo `api/dynamo_deps.py` - DynamoDB dependencies

```python
"""DynamoDB-based dependencies for AWS Lambda."""

import os
import time
from decimal import Decimal
from typing import Optional

import boto3
from fastapi import HTTPException, Header


class DynamoDBRateLimiter:
    """Rate limiter using DynamoDB (distributed across Lambda instances)."""
    
    def __init__(self, table_name: Optional[str] = None, max_requests: int = 30):
        self.table_name = table_name or os.getenv("DYNAMODB_TABLE", "genai-logs")
        self.max_requests = max_requests
        self.dynamodb = boto3.resource("dynamodb")
        self.table = self.dynamodb.Table(self.table_name)
    
    def check(self, user_id: str) -> None:
        window_start = int(time.time()) - 60
        
        # Query requests in last 60 seconds
        response = self.table.query(
            KeyConditionExpression=Key("pk").eq(f"RATE#{user_id}") & 
                                  Key("sk").gt(f"TS#{window_start}")
        )
        
        request_count = response["Count"]
        
        if request_count >= self.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Record this request
        self.table.put_item(Item={
            "pk": f"RATE#{user_id}",
            "sk": f"TS#{int(time.time())}",
            "ttl": int(time.time()) + 120,  # Auto-delete after 2 minutes
        })
```

---

## Bước 4: Build & Deploy

### 4.1 Cách 1: SAM CLI (Recommended)

```bash
# Build
sam build --use-container

# Deploy (first time - creates resources)
sam deploy --guided \
  --stack-name multi-agent-genai \
  --region ap-southeast-1 \
  --parameter-overrides OpenAIApiKey=sk-your-key-here \
  --capabilities CAPABILITY_IAM

# Deploy (subsequent times)
sam deploy --no-confirm-changeset
```

### 4.2 Cách 2: AWS CDK (TypeScript/Python)

```python
# infrastructure/aws/cdk_stack.py
from aws_cdk import (
    Stack,
    aws_lambda as _lambda,
    aws_apigatewayv2 as apigwv2,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
    Duration,
)
from constructs import Construct

class GenAIStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # DynamoDB table
        documents_table = dynamodb.Table(
            self, "DocumentsTable",
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
        )
        
        # Lambda function
        genai_function = _lambda.Function(
            self, "GenAIFunction",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="api.lambda_handler.handler",
            code=_lambda.Code.from_asset("."),
            timeout=Duration.seconds(30),
            memory_size=1024,
            environment={
                "DYNAMODB_TABLE": documents_table.table_name,
            },
        )
        
        documents_table.grant_read_write_data(genai_function)
        
        # HTTP API
        api = apigwv2.HttpApi(self, "GenAIApi")
        api.add_routes(
            path="/{proxy+}",
            methods=[apigwv2.HttpMethod.ANY],
            integration=apigwv2_integrations.HttpLambdaIntegration("LambdaIntegration", genai_function),
        )
```

### 4.3 Cách 3: Terraform

```hcl
# infrastructure/aws/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-southeast-1"
}

# DynamoDB table
resource "aws_dynamodb_table" "documents" {
  name         = "genai-documents"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  range_key    = "sk"

  attribute {
    name = "pk"
    type = "S"
  }

  attribute {
    name = "sk"
    type = "S"
  }
}

# Lambda function
resource "aws_lambda_function" "genai" {
  function_name = "multi-agent-genai"
  role          = aws_iam_role.lambda_role.arn
  handler       = "api.lambda_handler.handler"
  runtime       = "python3.12"
  timeout       = 30
  memory_size   = 1024
  filename      = "function.zip"
  source_code_hash = filebase64sha256("function.zip")

  environment {
    variables = {
      DYNAMODB_TABLE = aws_dynamodb_table.documents.name
    }
  }
}

# HTTP API
resource "aws_apigatewayv2_api" "genai" {
  name          = "genai-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id           = aws_apigatewayv2_api.genai.id
  integration_type = "AWS_PROXY"
  integration_uri  = aws_lambda_function.genai.invoke_arn
}

resource "aws_apigatewayv2_route" "proxy" {
  api_id    = aws_apigatewayv2_api.genai.id
  route_key = "ANY /{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}
```

---

## Bước 5: Verify

```bash
# Get API URL
API_URL=$(aws cloudformation describe-stacks \
  --stack-name multi-agent-genai \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
  --output text)

# Test health
curl $API_URL/health

# Test query
curl -X POST $API_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","query":"Hello","top_k":2}'
```

---

## Bước 6: Monitoring & Cost Control

### 6.1 CloudWatch Alarms (Free tier)

```bash
# Tạo billing alarm
aws cloudwatch put-metric-alarm \
  --alarm-name BillingAlarm \
  --alarm-description "Alert when estimated charges > $0" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 0.01 \
  --comparison-operator GreaterThanThreshold \
  --alarm-actions arn:aws:sns:ap-southeast-1:YOUR_ACCOUNT:YOUR_TOPIC
```

### 6.2 AWS Budgets

```bash
aws budgets create-budget \
  --account-id YOUR_ACCOUNT_ID \
  --budget file://budget.json
```

`budget.json`:
```json
{
  "BudgetName": "ZeroSpendBudget",
  "BudgetLimit": {
    "Amount": "0.01",
    "Unit": "USD"
  },
  "BudgetType": "COST",
  "TimeUnit": "MONTHLY",
  "Notification": {
    "NotificationType": "ACTUAL",
    "ComparisonOperator": "GREATER_THAN",
    "Threshold": 100
  }
}
```

---

## Free Tier Limits (AWS)

| Resource | Free Tier | Project của bạn |
|----------|-----------|-----------------|
| Lambda requests | 1M/tháng | ✅ ~1000/ngày = 30K/tháng |
| Lambda duration | 400K GB-seconds | ✅ 1GB × 30s × 1000 = 30K |
| DynamoDB storage | 25GB | ✅ < 1GB cho demo |
| DynamoDB read | 25M read units | ✅ Quá đủ |
| DynamoDB write | 25M write units | ✅ Quá đủ |
| S3 storage | 5GB | ✅ PDF uploads nhỏ |
| S3 GET requests | 20K/tháng | ✅ Quá đủ |
| HTTP API requests | 1M/tháng (không free tier) | ⚠️ $1/1M requests |
| Data transfer | 100GB out/tháng (12 tháng) | ✅ Quá đủ |

---

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| Lambda timeout | Tăng timeout (max 15 phút) hoặc optimize code |
| Cold start chậm | Dùng Provisioned Concurrency (tốn tiền) hoặc accept delay |
| DynamoDB throttling | Chuyển sang On-Demand (PAY_PER_REQUEST) |
| Package quá lớn | Dùng Lambda Layers hoặc container image |
| CORS errors | Thêm CORS middleware trong FastAPI |
| IAM permission denied | Kiểm tra Lambda execution role |

---

## Pros & Cons

### ✅ Pros
- **CV Value đỉnh nhất**: AWS là #1 cloud, cert là "vàng"
- **Serverless**: Không quản lý server
- **Auto-scale**: Tự động từ 0 → ∞
- **Free tier generous**: 1M Lambda + 25GB DynamoDB forever
- **Ecosystem lớn nhất**: Dịch vụ đa dạng

### ❌ Cons
- **Phức tạp nhất**: IAM, VPC, security groups
- **Dễ tốn tiền**: NAT Gateway, Elastic IP, data transfer
- **Cold start**: Lambda chậm khi idle
- **Timeout 15 phút**: Không phù hợp long-running
- **Không chạy Ollama**: Phải dùng external LLM API
- **No persistent disk**: SQLite không work

---

## Next Steps

1. ✅ Tạo AWS account + set billing alert
2. ✅ Cài AWS CLI + SAM
3. ✅ Refactor code (Mangum + DynamoDB)
4. ✅ Deploy với SAM
5. ✅ Test endpoints
6. ✅ Setup CloudWatch alarms
7. ✅ Ghi URL vào CV
8. ➡️ Consider: AWS Certified Cloud Practitioner (free tier + cert = CV đỉnh)

---

## Resources

- [AWS Free Tier](https://aws.amazon.com/free/)
- [SAM Documentation](https://docs.aws.amazon.com/serverless-application-model/)
- [Lambda Python](https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html)
- [DynamoDB Free Tier](https://aws.amazon.com/dynamodb/pricing/on-demand/)
- [HTTP API vs REST API](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-vs-rest.html)
- [AWS Pricing Calculator](https://calculator.aws/)
