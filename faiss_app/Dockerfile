# Use the proven NVIDIA CUDA runtime image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install numpy with specific version first
RUN pip install numpy==1.23.5

# Install FAISS with GPU support using force-reinstall
RUN pip install --force-reinstall --no-cache-dir faiss-gpu

# Install other Python dependencies
RUN pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    requests==2.31.0

# Set up the working directory
WORKDIR /app

# Copy the application code into the container
COPY faiss_app/ /app/

# Set environment variables for the application
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Define the command for runing FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]
