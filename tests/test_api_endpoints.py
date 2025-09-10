#!/usr/bin/env python3
"""
Test script to verify all API endpoints are working after fixes.
"""

import requests
import json
import time
import base64
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_health_endpoint():
    """Test the health endpoint."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        print(f"Health endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"✓ Health check passed: {response.json()}")
            return True
        else:
            print(f"✗ Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Health endpoint error: {e}")
        return False

def test_stats_endpoint():
    """Test the stats endpoint."""
    try:
        response = requests.get(f"{API_BASE}/stats", timeout=10)
        print(f"Stats endpoint: {response.status_code}")
        if response.status_code == 200:
            stats = response.json()
            print(f"✓ Stats retrieved: {len(stats)} metrics")
            return True
        else:
            print(f"✗ Stats failed: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Stats endpoint error: {e}")
        return False

def test_users_endpoint():
    """Test the users list endpoint."""
    try:
        response = requests.get(f"{API_BASE}/users", timeout=10)
        print(f"Users endpoint: {response.status_code}")
        if response.status_code == 200:
            users = response.json()
            print(f"✓ Users list retrieved: {users.get('total_users', 0)} users")
            return True
        else:
            print(f"✗ Users list failed: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Users endpoint error: {e}")
        return False

def create_test_image():
    """Create a simple test image for enrollment."""
    try:
        from PIL import Image
        import io
        
        # Create a simple 200x200 RGB image
        img = Image.new('RGB', (200, 200), color='white')
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    except Exception as e:
        print(f"Could not create test image: {e}")
        return None

def test_enrollment_endpoint():
    """Test the enrollment endpoint with a test image."""
    try:
        # Create test image
        test_image = create_test_image()
        if not test_image:
            print("✗ Could not create test image for enrollment")
            return False
        
        # Prepare enrollment request
        files = {
            'image': ('test.jpg', test_image, 'image/jpeg')
        }
        data = {
            'user_id': 'test_user_001'
        }
        
        response = requests.post(f"{API_BASE}/enroll", files=files, data=data, timeout=30)
        print(f"Enrollment endpoint: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Enrollment successful: {result.get('message', 'No message')}")
            return True
        else:
            print(f"✗ Enrollment failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Enrollment endpoint error: {e}")
        return False

def main():
    """Run all API tests."""
    print("=" * 60)
    print("API Endpoints Test Suite")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Stats Endpoint", test_stats_endpoint),
        ("Users List", test_users_endpoint),
        ("Enrollment", test_enrollment_endpoint)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All API endpoints are working correctly!")
        return 0
    else:
        print("⚠️  Some endpoints need attention.")
        return 1

if __name__ == "__main__":
    exit(main())
