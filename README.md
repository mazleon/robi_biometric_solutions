# Robi Biometric Solution - Face Verification System

## 🚀 GPU-Accelerated Face Verification API with Qdrant Vector Database

A high-performance face verification system optimized for NVIDIA RTX 4090 with real-time enrollment, verification, and detection capabilities using InsightFace and Qdrant vector database.

## ✨ Features

- **GPU-Accelerated Processing**: Optimized for NVIDIA RTX 4090 with CUDA 12.8+
- **Real-time Face Verification**: Sub-second response times for face matching
- **Hybrid Vector Store**: Qdrant + FAISS integration for optimal performance
- **Multi-face Detection**: Process up to 5 faces per image
- **RESTful API**: Complete FastAPI implementation with OpenAPI documentation
- **Web Interface**: Modern Streamlit-based UI for testing and management
- **Performance Monitoring**: Real-time metrics and system statistics

## 🏗️ Architecture

### Core Components
- **Backend**: FastAPI with async/await optimization
- **Face Processing**: InsightFace buffalo_l model with ONNX Runtime
- **Vector Database**: Qdrant (primary) + FAISS (cache/fallback)
- **GPU Acceleration**: CUDA with conservative optimization settings
- **Frontend**: Streamlit web application

### Technology Stack
- **Python**: 3.12+
- **AI/ML**: InsightFace, ONNX Runtime GPU, PyTorch
- **Database**: Qdrant vector database, SQLite (legacy)
- **Web**: FastAPI, Streamlit, Uvicorn
- **GPU**: CUDA 12.1+, cuDNN

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 4090 (or compatible with 16GB+ VRAM)
- **RAM**: 32GB+ recommended
- **Storage**: 10GB+ free space

### Software Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.12+
- **CUDA**: 12.1+ with cuDNN
- **Docker**: For Qdrant service

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd robi-face-stage
```

### 2. Setup Python Environment
```bash
# Using uv (recommended)
uv venv
uv sync

# Or using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start Qdrant Service
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/data/qdrant:/qdrant/storage \
    qdrant/qdrant:latest
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your specific settings
```

### 5. Initialize System
```bash
# Create Qdrant collection
python create_collection.py

# Test GPU setup
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### Start GPU-Optimized Server
```bash
# Recommended: GPU-optimized startup
uv run run_gpu.py

# Alternative: Standard startup
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Start Web Interface
```bash
streamlit run streamlit_app.py
```

### Access Services
- **API Documentation**: http://localhost:8000/docs
- **Web Interface**: http://localhost:8501
- **Health Check**: http://localhost:8000/health
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📡 API Endpoints

### Core Endpoints
- `POST /enroll` - Enroll user with face image
- `POST /verify` - Verify face against enrolled users
- `POST /detect` - Detect faces in image
- `GET /users` - List all enrolled users
- `DELETE /users/{user_id}` - Delete user
- `GET /stats` - System performance statistics
- `GET /health` - Health check

### Example Usage
```bash
# Enroll a user
curl -X POST "http://localhost:8000/enroll" \
  -F "user_id=john_doe" \
  -F "name=John Doe" \
  -F "image=@face_image.jpg"

# Verify a face
curl -X POST "http://localhost:8000/verify" \
  -F "image=@test_image.jpg"
```

## ⚙️ Configuration

### GPU Settings (RTX 4090 Optimized)
```python
# Key settings in .env
USE_GPU=true
GPU_DEVICE_ID=0
GPU_MEMORY_FRACTION=0.95
BATCH_SIZE=16
MAX_IMAGE_SIZE=1024
```

### Qdrant Configuration
```python
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=face_embeddings
EMBEDDING_DIMENSION=512
```

### Performance Tuning
```python
# Conservative ONNX Runtime settings for stability
ENABLE_CUDA_GRAPH=false
USE_TF32=false
TUNABLE_OP_ENABLE=false
GPU_MEM_LIMIT=17179869184  # 16GB
```

## 🧪 Testing

### Run System Tests
```bash
# Test API endpoints
python test_api_endpoints.py

# Test Qdrant connection
python test_qdrant_connection.py

# Test startup sequence
python test_startup.py
```

### Performance Benchmarks
- **Enrollment**: <1000ms per face
- **Verification**: <500ms per query
- **Detection**: <300ms per image
- **Throughput**: 8 concurrent requests

## 🔧 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

#### Qdrant Connection Failed
```bash
# Check Qdrant service
curl http://localhost:6333/
docker ps | grep qdrant
```

#### Memory Issues
```bash
# Monitor GPU memory
nvidia-smi
# Reduce batch size or image resolution in .env
```

#### Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Performance Optimization
1. **GPU Memory**: Adjust `GPU_MEMORY_FRACTION` based on available VRAM
2. **Batch Size**: Increase `BATCH_SIZE` for better throughput
3. **Image Size**: Reduce `MAX_IMAGE_SIZE` for faster processing
4. **Workers**: Use single worker for GPU workloads

## 📊 Monitoring

### System Metrics
- GPU utilization and memory usage
- API response times and throughput
- Vector database performance
- Error rates and system health

### Logging
```bash
# View application logs
tail -f logs/app.log

# View performance metrics
curl http://localhost:8000/stats
```

## 🔒 Security

### Best Practices
- Use environment variables for sensitive configuration
- Implement proper authentication for production
- Validate all input data and file uploads
- Monitor for unusual access patterns
- Regular security updates

## 📚 Development

### Project Structure
```
robi-face-stage/
├── src/                    # Main source code
│   ├── api/               # FastAPI endpoints and schemas
│   ├── core/              # Face processing and vector store
│   ├── config/            # Configuration management
│   ├── models/            # Data models
│   └── utils/             # Utility modules
├── data/                  # Data storage
├── logs/                  # Application logs
├── models/                # ML model storage
└── tests/                 # Test files
```

### Adding New Features
1. Follow existing code patterns and structure
2. Add comprehensive error handling
3. Include performance monitoring
4. Update documentation and tests
5. Maintain GPU optimization compatibility

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Update documentation
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Review logs for error details
3. Test with provided examples
4. Create detailed issue reports

---

**Status**: ✅ **FULLY OPERATIONAL** - All critical issues resolved, GPU acceleration enabled, Qdrant integration working