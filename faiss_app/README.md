# FAISS GPU Microservice

A high-performance vector similarity search microservice using FAISS with GPU acceleration, optimized for RTX 4090 (24GB VRAM).

## Features

- **GPU-Accelerated FAISS**: Leverages NVIDIA GPU for fast vector operations
- **RESTful API**: FastAPI-based web service with automatic documentation
- **Docker Support**: Containerized deployment with GPU support
- **Data Migration**: Automatic migration from legacy FAISS indexes
- **Health Monitoring**: Built-in health checks and statistics
- **Batch Operations**: Support for batch vector operations

## Prerequisites

- **Docker Desktop** with GPU support enabled
- **NVIDIA GPU** with CUDA 12.1+ support
- **NVIDIA Container Toolkit** installed
- **Windows 10/11** or **Linux** with Docker

### GPU Requirements

- NVIDIA GPU with compute capability 6.0+
- Minimum 8GB VRAM (optimized for RTX 4090 24GB)
- CUDA 12.1 or later

## Quick Start

### Option 1: Using Build Scripts (Recommended)

**Windows:**
```cmd
cd faiss_app
build_and_run.bat
```

**Linux/Mac:**
```bash
cd faiss_app
chmod +x build_and_run.sh
./build_and_run.sh
```

### Option 2: Manual Docker Commands

1. **Build the containers:**
```bash
cd faiss_app
docker-compose -f docker-compose.faiss.yml build --no-cache
```

2. **Start the services:**
```bash
docker-compose -f docker-compose.faiss.yml up -d
```

3. **Check status:**
```bash
docker-compose -f docker-compose.faiss.yml ps
```

## API Endpoints

### FAISS GPU Service (Port 8001)

- **Health Check**: `GET /health`
- **Add Vector**: `POST /vectors/add`
- **Batch Add**: `POST /vectors/batch-add`
- **Search Vectors**: `POST /vectors/search`
- **Remove Vector**: `DELETE /vectors/{user_id}`
- **Statistics**: `GET /stats`
- **Reset Index**: `POST /admin/reset`
- **API Docs**: `http://localhost:8001/docs`

### Example Usage

**Add a vector:**
```bash
curl -X POST "http://localhost:8001/vectors/add" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "embedding": [0.1, 0.2, 0.3, ...],
    "metadata": {"name": "John Doe"}
  }'
```

**Search for similar vectors:**
```bash
curl -X POST "http://localhost:8001/vectors/search" \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.1, 0.2, 0.3, ...],
    "k": 5,
    "threshold": 0.8
  }'
```

## Configuration

Environment variables can be set in `docker-compose.faiss.yml`:

- `EMBEDDING_DIM`: Vector dimension (default: 512)
- `GPU_DEVICE_ID`: GPU device ID (default: 0)
- `GPU_MEMORY_GB`: GPU memory allocation (default: 16GB)
- `USE_FP16`: Enable FP16 precision (default: true)
- `INDEX_TYPE`: FAISS index type (default: IndexFlatIP)
- `HOST`: Service host (default: 0.0.0.0)
- `PORT`: Service port (default: 8001)
- `LOG_LEVEL`: Logging level (default: INFO)

## Data Persistence

- **FAISS Index**: Stored in `/app/data/faiss_index.bin`
- **Metadata**: Stored in `/app/data/metadata.json`
- **Docker Volumes**: Persistent storage across container restarts

## Monitoring and Logs

**View logs:**
```bash
docker-compose -f docker-compose.faiss.yml logs -f
```

**Check service health:**
```bash
curl http://localhost:8001/health
```

**Get statistics:**
```bash
curl http://localhost:8001/stats
```

## Troubleshooting

### GPU Not Detected

1. Ensure NVIDIA drivers are installed
2. Install NVIDIA Container Toolkit
3. Restart Docker Desktop
4. Verify GPU access: `docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi`

### Build Failures

1. Clear Docker cache: `docker system prune -a`
2. Rebuild without cache: `docker-compose build --no-cache`
3. Check Docker Desktop memory allocation (minimum 8GB recommended)

### Service Not Starting

1. Check port availability: `netstat -an | findstr :8001`
2. Review logs: `docker-compose logs faiss-gpu-service`
3. Verify GPU resources: `nvidia-smi`

### Performance Issues

1. Increase GPU memory allocation in docker-compose.yml
2. Enable FP16 precision for memory efficiency
3. Adjust batch size based on available VRAM
4. Monitor GPU utilization: `nvidia-smi -l 1`

## Development

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run locally:**
```bash
python run_faiss_service.py
```

### Testing

```bash
# Test health endpoint
curl http://localhost:8001/health

# Test vector operations
python -c "
import requests
import numpy as np

# Add a test vector
response = requests.post('http://localhost:8001/vectors/add', json={
    'user_id': 'test_user',
    'embedding': np.random.rand(512).tolist(),
    'metadata': {'test': True}
})
print('Add response:', response.json())

# Search for similar vectors
response = requests.post('http://localhost:8001/vectors/search', json={
    'embedding': np.random.rand(512).tolist(),
    'k': 5
})
print('Search response:', response.json())
"
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   Client App    │───▶│  FAISS Service   │
│                 │    │   (Port 8001)    │
└─────────────────┘    └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   FAISS GPU      │
                       │   Index + GPU    │
                       └──────────────────┘
```

## Performance Benchmarks

**RTX 4090 (24GB VRAM):**
- Vector Addition: ~10,000 vectors/second
- Search (k=5): ~1ms per query
- Index Size: 1M vectors @ 512D = ~2GB GPU memory
- Batch Processing: Up to 32 vectors per batch

## License

This project is part of the face verification system and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker and service logs
3. Verify GPU compatibility and drivers
4. Ensure sufficient system resources
