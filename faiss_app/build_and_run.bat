@echo off
REM Build and Run FAISS GPU Application for Windows
REM This script builds the Docker containers and runs the application with GPU support

echo 🚀 Building and Running FAISS GPU Application
echo ==============================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Navigate to the script directory
cd /d "%~dp0"

echo 📁 Current directory: %cd%

REM Build the containers
echo 🔨 Building Docker containers...
docker-compose -f docker-compose.faiss.yml build --no-cache

REM Create necessary directories
echo 📂 Creating data directories...
if not exist "data" mkdir data
if not exist "logs" mkdir logs

REM Start the services
echo 🚀 Starting services...
docker-compose -f docker-compose.faiss.yml up -d

REM Wait for services to be ready
echo ⏳ Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Check service status
echo 🔍 Checking service status...
docker-compose -f docker-compose.faiss.yml ps

REM Show logs
echo 📋 Recent logs:
docker-compose -f docker-compose.faiss.yml logs --tail=20

echo.
echo ✅ Application should now be running!
echo 🌐 FAISS GPU Service: http://localhost:8001
echo 🌐 Face Verification Service: http://localhost:8000
echo.
echo 📖 API Documentation:
echo    - FAISS Service: http://localhost:8001/docs
echo    - Face Verification: http://localhost:8000/docs
echo.
echo 🔧 To view logs: docker-compose -f docker-compose.faiss.yml logs -f
echo 🛑 To stop: docker-compose -f docker-compose.faiss.yml down
echo.
pause
