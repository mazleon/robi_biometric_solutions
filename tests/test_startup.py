#!/usr/bin/env python3
"""
Test script to verify application startup fixes.
"""

import sys
import os
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all critical imports work."""
    print("Testing imports...")
    
    try:
        from src.config.settings import settings
        print("✓ Settings import successful")
        
        # Test settings loading
        print(f"  - Qdrant URL: {settings.qdrant_url}")
        print(f"  - GPU enabled: {settings.use_gpu}")
        print(f"  - Model name: {settings.model_name}")
        print(f"  - Detection size: {settings.detection_size}")
        
    except Exception as e:
        print(f"✗ Settings import failed: {e}")
        return False
    
    try:
        from src.utils.gpu_config import get_gpu_info, apply_gpu_config
        print("✓ GPU config import successful")
        
        # Test GPU info
        gpu_info = get_gpu_info()
        print(f"  - GPU available: {gpu_info['gpu_available']}")
        if gpu_info['gpu_available']:
            print(f"  - GPU name: {gpu_info['gpu_name']}")
            print(f"  - GPU memory: {gpu_info['gpu_memory_gb']:.1f}GB")
        
        # Apply GPU config
        apply_gpu_config()
        print("✓ GPU configuration applied")
        
    except Exception as e:
        print(f"✗ GPU config failed: {e}")
        return False
    
    try:
        # Test Qdrant connection
        import requests
        response = requests.get("http://localhost:6333/health", timeout=5)
        if response.status_code == 200:
            print("✓ Qdrant service is responding")
        else:
            print(f"✗ Qdrant service returned status {response.status_code}")
            
    except Exception as e:
        print(f"✗ Qdrant connection failed: {e}")
        return False
    
    return True

def test_face_processor():
    """Test face processor initialization."""
    print("\nTesting face processor...")
    
    try:
        from src.core.face_processor import FaceProcessor
        print("✓ Face processor import successful")
        
        # This will test the ONNX Runtime configuration
        processor = FaceProcessor()
        print("✓ Face processor initialization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Face processor failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_store():
    """Test hybrid vector store initialization."""
    print("\nTesting hybrid vector store...")
    
    try:
        from src.core.hybrid_vector_store import HybridVectorStore
        print("✓ Hybrid store import successful")
        
        # Initialize with test settings
        store = HybridVectorStore()
        print("✓ Hybrid store initialization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Hybrid store failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Application Startup Test")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    success = True
    
    # Test imports and basic functionality
    if not test_imports():
        success = False
    
    # Test face processor (the main issue)
    if not test_face_processor():
        success = False
    
    # Test hybrid store
    if not test_hybrid_store():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Application should start successfully!")
    else:
        print("❌ SOME TESTS FAILED - Check the errors above")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
