# Robi Face Verification System - Deployment Guide

## 🚀 Production Deployment Guide

This guide provides step-by-step instructions for deploying the Robi Face Verification System in production environments with GPU acceleration and Qdrant vector database.

## 📋 System Requirements

### Hardware Requirements

- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or equivalent
- **CPU**: 8+ cores, 3.0GHz+ recommended
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ SSD storage
- **Network**: Gigabit Ethernet for high-throughput scenarios

### Software Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or CentOS 8+
- **Python**: 3.12+
- **CUDA**: 12.1+ with cuDNN 8.9+
- **Docker**: 24.0+ with Docker Compose
- **Git**: For repository management

## 🛠️ Pre-Deployment Setup

### 1. GPU Driver Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot

# Verify installation
nvidia-smi
```

### 2. CUDA Toolkit Installation

```bash
# Download and install CUDA 12.1+
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Docker Installation

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
sudo systemctl enable docker
sudo systemctl start docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## 📦 Application Deployment

### 1. Repository Setup

```bash
# Clone repository
git clone <repository-url>
cd robi-face-stage

# Create data directories
mkdir -p data/qdrant data/faiss logs models
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Key Production Settings:**

```bash
# Application
DEBUG=false
LOG_LEVEL=INFO
MAX_WORKERS=4

# GPU Configuration
USE_GPU=true
GPU_DEVICE_ID=0
GPU_MEMORY_FRACTION=0.90
BATCH_SIZE=16

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=face_embeddings
EMBEDDING_DIMENSION=512

# Security
CORS_ORIGINS=["https://yourdomain.com"]
MAX_FILE_SIZE=5242880  # 5MB
RATE_LIMIT_PER_MINUTE=60

# Performance
FAISS_ENABLED=true
FAISS_CACHE_SIZE=100000
CONCURRENT_REQUESTS=8
```

### 3. Python Environment Setup

```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate
uv sync

# Or using pip
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Qdrant Service Deployment

```bash
# Start Qdrant with Docker Compose
docker-compose up -d qdrant

# Verify service
curl http://localhost:6333/
docker logs qdrant-container
```

### 5. Database Initialization

```bash
# Create Qdrant collection
python create_collection.py

# Verify collection
python -c "
import requests
r = requests.get('http://localhost:6333/collections')
print('Collections:', r.json())
"
```

## 🚀 Application Startup

### Production Startup Script

```bash
#!/bin/bash
# production_start.sh

set -e

echo "Starting Robi Face Verification System..."

# Check GPU availability
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Check Qdrant service
curl -f http://localhost:6333/ || (echo "Qdrant not available" && exit 1)

# Start application with GPU optimization
python run_gpu.py

echo "Application started successfully"
```

### Systemd Service (Linux)

```ini
# /etc/systemd/system/robi-face.service
[Unit]
Description=Robi Face Verification System
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=robi
Group=robi
WorkingDirectory=/opt/robi-face-stage
Environment=PATH=/opt/robi-face-stage/.venv/bin
ExecStart=/opt/robi-face-stage/.venv/bin/python run_gpu.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable robi-face
sudo systemctl start robi-face
sudo systemctl status robi-face
```

## 🔒 Security Configuration

### 1. Firewall Setup

```bash
# Ubuntu/Debian with ufw
sudo ufw allow 8000/tcp  # API port
sudo ufw allow 8501/tcp  # Streamlit port (if needed)
sudo ufw allow 6333/tcp  # Qdrant port (internal only)
sudo ufw enable
```

### 2. SSL/TLS Configuration

```nginx
# /etc/nginx/sites-available/robi-face
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # File upload limits
        client_max_body_size 10M;
        proxy_read_timeout 30s;
    }
}
```

### 3. Authentication Setup

```python
# Add to .env for API key authentication
API_KEY_ENABLED=true
API_KEY=your-secure-api-key-here
```

## 📊 Monitoring and Logging

### 1. Log Configuration

```bash
# Create log rotation
sudo tee /etc/logrotate.d/robi-face << EOF
/opt/robi-face-stage/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 robi robi
}
EOF
```

### 2. Health Monitoring

```bash
#!/bin/bash
# health_check.sh

# Check API health
curl -f http://localhost:8000/health || exit 1

# Check GPU memory
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{if($1>20000) exit 1}'

# Check Qdrant
curl -f http://localhost:6333/ || exit 1

echo "All services healthy"
```

### 3. Performance Monitoring

```bash
# Add to crontab for regular monitoring
*/5 * * * * /opt/robi-face-stage/health_check.sh >> /var/log/robi-face-health.log 2>&1
```

## 🔧 Performance Tuning

### 1. GPU Optimization

```python
# Optimal settings for RTX 4090
GPU_MEMORY_FRACTION=0.90
BATCH_SIZE=16
MAX_IMAGE_SIZE=1024
ENABLE_CUDA_GRAPH=false  # Conservative for stability
USE_TF32=false
```

### 2. Qdrant Optimization

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=8
      - QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=4
      - QDRANT__STORAGE__OPTIMIZERS__MEMMAP_THRESHOLD_KB=65536
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### 3. Application Tuning

```python
# Uvicorn production settings
uvicorn src.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --access-log \
  --log-level info \
  --limit-concurrency 8 \
  --timeout-keep-alive 30
```

## 🚨 Troubleshooting

### Common Issues

#### 1. GPU Memory Issues

```bash
# Check GPU memory usage
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size in .env
BATCH_SIZE=8
GPU_MEMORY_FRACTION=0.80
```

#### 2. Qdrant Connection Issues

```bash
# Check Qdrant logs
docker logs qdrant-container

# Restart Qdrant
docker-compose restart qdrant

# Verify network connectivity
netstat -tlnp | grep 6333
```

#### 3. Performance Issues

```bash
# Monitor system resources
htop
iotop
nvidia-smi -l 1

# Check API performance
curl http://localhost:8000/stats
```

## 📈 Scaling Considerations

### Horizontal Scaling

```yaml
# docker-compose.yml for multiple instances
version: '3.8'
services:
  robi-face-1:
    build: .
    ports:
      - "8001:8000"
    environment:
      - GPU_DEVICE_ID=0
  
  robi-face-2:
    build: .
    ports:
      - "8002:8000"
    environment:
      - GPU_DEVICE_ID=1

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Load Balancer Configuration

```nginx
# nginx.conf
upstream robi_face {
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://robi_face;
    }
}
```

## 🔄 Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup_qdrant.sh

BACKUP_DIR="/backup/qdrant/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Stop Qdrant temporarily
docker-compose stop qdrant

# Copy data
cp -r data/qdrant/* "$BACKUP_DIR/"

# Restart Qdrant
docker-compose start qdrant

echo "Backup completed: $BACKUP_DIR"
```

### Application Backup

```bash
# Backup configuration and models
tar -czf robi-face-backup-$(date +%Y%m%d).tar.gz \
  .env \
  data/ \
  models/ \
  logs/ \
  --exclude='logs/*.log'
```

## 📝 Maintenance

### Regular Maintenance Tasks

```bash
#!/bin/bash
# maintenance.sh

# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# Clean temporary files
find /tmp -name "robi-*" -mtime +1 -delete

# Update system packages
sudo apt update && sudo apt upgrade -y

# Restart services monthly
sudo systemctl restart robi-face
```

### Performance Monitoring

```bash
# Weekly performance report
curl http://localhost:8000/stats | jq '.' > "performance-$(date +%Y%m%d).json"
```

## ✅ Deployment Checklist

- [ ] GPU drivers and CUDA installed
- [ ] Docker and Docker Compose installed
- [ ] Repository cloned and configured
- [ ] Environment variables set
- [ ] Qdrant service running
- [ ] Collection created and verified
- [ ] Application starts without errors
- [ ] Health endpoints responding
- [ ] SSL/TLS configured (production)
- [ ] Firewall rules configured
- [ ] Monitoring and logging setup
- [ ] Backup procedures implemented
- [ ] Performance tuning applied

---

**Status**: ✅ **PRODUCTION READY** - All components tested and operational
