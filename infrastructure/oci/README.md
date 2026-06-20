# Deploy Multi-Agent GenAI Platform lên Oracle Cloud Infrastructure (OCI)

## Tại sao chọn OCI?

- ✅ **Always Free thật sự**: Không hết hạn, không charge nếu đúng limits
- ✅ **Mạnh nhất**: 4 OCPU + 24GB RAM (ARM), 2 VMs x86
- ✅ **Database miễn phí**: 2 Autonomous Databases
- ✅ **10TB bandwidth/tháng**
- ✅ **CV Value**: Oracle là enterprise cloud lớn, đặc biệt ở VN (ngân hàng, chính phủ)

---

## Architecture trên OCI

```
┌─────────────────────────────────────────────────┐
│              Oracle Cloud (OCI)                  │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │     VM.Standard.A1.Flex (ARM)            │   │
│  │     4 OCPU + 24GB RAM (Always Free)       │   │
│  │                                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
│  │  │ FastAPI │  │ Ollama  │  │PostgreSQL│  │   │
│  │  │  App    │  │  LLM    │  │  (opt)  │  │   │
│  │  │ (Docker)│  │(Docker) │  │(Docker) │  │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  │   │
│  │       │            │            │      │   │
│  │  ┌────┴────────────┴────────────┘      │   │
│  │  │        Docker Compose               │   │
│  │  │    (All-in-one container)           │   │
│  │  └─────────────────────────────────────┘   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  Autonomous Database (Optional)          │   │
│  │  Oracle DB / JSON / APEX (Always Free)   │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## Bước 1: Tạo OCI Account

1. Vào [https://www.oracle.com/cloud/free/](https://www.oracle.com/cloud/free/)
2. Click "Start for free"
3. Đăng ký với email + credit card (verify, **không charge** cho Always Free)
4. Chọn Home Region: **Sydney** (ap-sydney-1) hoặc **Tokyo** (ap-tokyo-1) - gần VN hơn
5. Hoàn thành verification

---

## Bước 2: Tạo VM (Always Free)

### 2.1 Via OCI Console (Web UI)

1. Vào OCI Console → Compute → Instances → Create Instance
2. Chọn:
   - **Name**: `multi-agent-genai`
   - **Shape**: `VM.Standard.A1.Flex` (ARM) - **Always Free eligible**
   - **OCPUs**: 4 (max for Always Free)
   - **Memory**: 24GB (max for Always Free)
   - **Image**: `Canonical Ubuntu 22.04` (ARM64)
   - **Boot Volume**: 200GB (max for Always Free)
   - **VNIC**: Tạo new VCN + public subnet
   - **SSH Keys**: Generate new pair hoặc upload public key
   - **Add Cloud-Init**: (optional, see below)

3. Click "Create"

### 2.2 Via OCI CLI

```bash
# Cài OCI CLI
curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh | bash

# Configure
oci setup config
# Enter user OCID, tenancy OCID, region, key file

# Tạo VM
oci compute instance launch \
  --compartment-id $COMPARTMENT_ID \
  --availability-domain $AD \
  --display-name "multi-agent-genai" \
  --shape "VM.Standard.A1.Flex" \
  --shape-config '{"ocpus":4, "memory_in_gbs":24}' \
  --source-boot-volume-id $BOOT_VOLUME_ID \
  --image-id $UBUNTU_ARM_IMAGE_ID \
  --subnet-id $SUBNET_ID \
  --ssh-authorized-keys-file ~/.ssh/id_rsa.pub \
  --wait-for-state RUNNING
```

### 2.3 Cloud-Init Script (Auto-setup)

```yaml
#cloud-config
package_update: true
packages:
  - docker.io
  - docker-compose
  - git
  - curl
  - htop

runcmd:
  # Start Docker
  - systemctl start docker
  - systemctl enable docker
  - usermod -aG docker ubuntu
  
  # Install Docker Compose
  - curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  - chmod +x /usr/local/bin/docker-compose
  
  # Clone repo (replace with your repo)
  - cd /home/ubuntu
  - git clone https://github.com/YOUR_USERNAME/multi-agent-genai.git
  - chown -R ubuntu:ubuntu /home/ubuntu/multi-agent-genai
  
  # Setup complete - user needs to run docker-compose manually
  - echo "Setup complete! SSH in and run: cd /home/ubuntu/multi-agent-genai && docker-compose up -d"
```

---

## Bước 3: SSH vào VM và Deploy

### 3.1 SSH vào VM

```bash
# Lấy public IP từ OCI Console
chmod 600 ~/.ssh/id_rsa
ssh -i ~/.ssh/id_rsa ubuntu@YOUR_VM_PUBLIC_IP
```

### 3.2 Clone repo và setup

```bash
# Clone repo (nếu chưa có cloud-init)
cd ~
git clone https://github.com/YOUR_USERNAME/multi-agent-genai.git
cd multi-agent-genai

# Verify Docker
sudo docker --version
sudo docker-compose --version
```

### 3.3 Tạo `docker-compose.prod.yml` cho OCI

```yaml
version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - OLLAMA_MODEL=qwen2.5:3b
      - API_BEARER_TOKEN=${API_BEARER_TOKEN:-}
      - RATE_LIMIT_PER_MINUTE=30
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
    restart: unless-stopped
    # Resource limits (VM có 4 OCPU + 24GB RAM)
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G
        reservations:
          cpus: '1'
          memory: 4G

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 16G
        reservations:
          cpus: '1'
          memory: 8G
    # ARM64 image
    platform: linux/arm64

  # Optional: PostgreSQL thay SQLite
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=genai
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-changeme}
      - POSTGRES_DB=genai_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G

  # Optional: Nginx reverse proxy + SSL
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: unless-stopped

volumes:
  ollama_data:
  postgres_data:
```

### 3.4 Deploy

```bash
# Pull models trước (tốn thời gian)
docker-compose -f docker-compose.prod.yml pull

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Pull Ollama model
docker-compose -f docker-compose.prod.yml exec ollama ollama pull qwen2.5:3b

# Verify
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs -f api
```

---

## Bước 4: Security & Networking

### 4.1 OCI Security List (Firewall)

Vào OCI Console → Networking → Virtual Cloud Networks → Your VCN → Security Lists

Thêm Ingress Rules:

| Source | Protocol | Port | Description |
|--------|----------|------|-------------|
| 0.0.0.0/0 | TCP | 22 | SSH |
| 0.0.0.0/0 | TCP | 80 | HTTP |
| 0.0.0.0/0 | TCP | 443 | HTTPS |
| 0.0.0.0/0 | TCP | 8000 | FastAPI (temporary) |
| YOUR_IP/32 | TCP | 11434 | Ollama (restrict!) |

### 4.2 UFW (Ubuntu Firewall)

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8000/tcp
sudo ufw enable
```

### 4.3 SSL với Let's Encrypt + Nginx

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

`nginx/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

---

## Bước 5: Monitoring & Maintenance

### 5.1 Basic Monitoring

```bash
# System resources
htop

# Docker stats
docker stats

# Disk usage
df -h
docker system df

# Logs
docker-compose -f docker-compose.prod.yml logs -f --tail=100
```

### 5.2 Setup Auto-restart

```bash
# Tạo systemd service
sudo tee /etc/systemd/system/multi-agent-genai.service > /dev/null <<EOF
[Unit]
Description=Multi-Agent GenAI Platform
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/multi-agent-genai
ExecStart=/usr/local/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable multi-agent-genai
sudo systemctl start multi-agent-genai
```

### 5.3 Backup Data

```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/ubuntu/backups"
mkdir -p $BACKUP_DIR

# Backup data directory
tar czf $BACKUP_DIR/data_$DATE.tar.gz /home/ubuntu/multi-agent-genai/data

# Backup Ollama models (large!)
docker run --rm -v multi-agent-genai_ollama_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/ollama_$DATE.tar.gz -C /data .

# Upload to OCI Object Storage (optional)
oci os object put -bn your-backup-bucket --file $BACKUP_DIR/data_$DATE.tar.gz

# Keep only last 7 backups
ls -t $BACKUP_DIR/*.tar.gz | tail -n +8 | xargs -r rm
```

---

## Bước 6: Domain & DNS

### 6.1 Free Domain Options

| Provider | Free? | Notes |
|----------|-------|-------|
| Freenom | ✅ | .tk, .ml, .ga, .cf, .gq |
| DuckDNS | ✅ | Subdomain, dynamic DNS |
| No-IP | ✅ | Subdomain, need confirm monthly |
| Cloudflare Pages | ✅ | Great DNS + CDN |

### 6.2 Setup với DuckDNS

```bash
# Đăng ký tại https://www.duckdns.org/
# Tạo subdomain: yourname.duckdns.org

# Update IP script
curl "https://www.duckdns.org/update?domains=yourname&token=YOUR_TOKEN&ip="

# Add to crontab
echo "*/5 * * * * curl -s https://www.duckdns.org/update?domains=yourname&token=YOUR_TOKEN&ip= > /dev/null" | crontab -
```

---

## Free Tier Limits (OCI Always Free)

| Resource | Limit | Project của bạn |
|----------|-------|-----------------|
| ARM VMs | 4 OCPU + 24GB RAM | ✅ Dùng hết = 1 VM mạnh |
| x86 VMs | 2 VMs (1/8 OCPU + 1GB RAM each) | ✅ Dùng cho dev/test |
| Block Storage | 200GB | ✅ Boot volume |
| Object Storage | 10GB | ✅ Backups |
| Autonomous DB | 2 instances (1 OCPU + 20GB) | ✅ Optional |
| Bandwidth | 10TB/tháng | ✅ Quá đủ |

---

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| ARM capacity unavailable | Thử region khác (Sydney, Tokyo, Seoul) |
| Docker permission denied | `sudo usermod -aG docker $USER` + re-login |
| Ollama model download chậm | Dùng mirror hoặc download trước rồi copy |
| Out of memory | Giảm Ollama model size hoặc thêm swap |
| SSL certificate fail | Kiểm tra domain DNS trỏ đúng IP |
| VM không start | Check boot volume size, shape eligibility |

---

## Pros & Cons

### ✅ Pros
- **Mạnh nhất free tier**: 4 OCPU + 24GB RAM
- **Giữ nguyên Ollama**: Chạy local LLM, không cần API key
- **Full control**: Root access, cài đặt tùy ý
- **Always Free**: Không lo hết hạn
- **Enterprise recognition**: Oracle là big name

### ❌ Cons
- **Phức tạp**: Nhiều steps hơn Cloud Run
- **Capacity issues**: ARM instances hay hết ở popular regions
- **Oracle DB**: Không phổ biến bằng PostgreSQL
- **Self-managed**: Phải tự lo security, updates, backups
- **Cold start**: VM reboot mất vài phút

---

## Next Steps

1. ✅ Tạo OCI account
2. ✅ Launch ARM VM
3. ✅ SSH vào, cài Docker
4. ✅ Clone repo, chạy docker-compose
5. ✅ Pull Ollama model
6. ✅ Test endpoints
7. ✅ Setup domain + SSL
8. ✅ Ghi URL vào CV

---

## Resources

- [OCI Always Free](https://www.oracle.com/cloud/free/)
- [OCI Documentation](https://docs.oracle.com/en-us/iaas/Content/home.htm)
- [OCI CLI](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/cliconcepts.htm)
- [Docker on ARM](https://docs.docker.com/desktop/install/linux-install/)
