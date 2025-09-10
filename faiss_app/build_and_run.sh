#!/bin/bash

# Build and Run FAISS GPU Application
# This script builds the Docker containers and runs the application with GPU support

set -e

echo "🚀 Building and Running FAISS GPU Application"
echo "=============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if nvidia-docker is available
if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "⚠️  GPU support may not be available. Continuing anyway..."
fi

# Navigate to the correct directory
cd "$(dirname "$0")"

echo "📁 Current directory: $(pwd)"

# Build the containers
echo "🔨 Building Docker containers..."
docker-compose -f docker-compose.faiss.yml build --no-cache

# Create necessary directories
echo "📂 Creating data directories..."
mkdir -p data logs

# Start the services
echo "🚀 Starting services..."
docker-compose -f docker-compose.faiss.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker-compose -f docker-compose.faiss.yml ps

# Show logs
echo "📋 Recent logs:"
docker-compose -f docker-compose.faiss.yml logs --tail=20

echo ""
echo "✅ Application should now be running!"
echo "🌐 FAISS GPU Service: http://localhost:8001"
echo "🌐 Face Verification Service: http://localhost:8000"
echo ""
echo "📖 API Documentation:"
echo "   - FAISS Service: http://localhost:8001/docs"
echo "   - Face Verification: http://localhost:8000/docs"
echo ""
echo "🔧 To view logs: docker-compose -f docker-compose.faiss.yml logs -f"
echo "🛑 To stop: docker-compose -f docker-compose.faiss.yml down"
