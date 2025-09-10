#!/usr/bin/env python3
"""Test script to verify FAISS GPU service deployment and functionality."""

import requests
import numpy as np
import time
import json
import sys
from typing import Dict, Any

class FaissServiceTester:
    """Test the FAISS GPU service deployment."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details
        })
    
    def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                gpu_available = data.get('gpu_available', False)
                self.log_test(
                    "Health Check", 
                    True, 
                    f"Status: {data.get('status')}, GPU: {gpu_available}"
                )
                return True
            else:
                self.log_test("Health Check", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Error: {str(e)}")
            return False
    
    def test_add_vector(self) -> bool:
        """Test adding a vector."""
        try:
            test_vector = np.random.rand(512).tolist()
            payload = {
                "user_id": "test_user_001",
                "embedding": test_vector,
                "metadata": {"name": "Test User", "test": True}
            }
            
            response = self.session.post(
                f"{self.base_url}/vectors/add",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                faiss_id = data.get('faiss_id')
                self.log_test(
                    "Add Vector", 
                    True, 
                    f"FAISS ID: {faiss_id}"
                )
                return True
            else:
                self.log_test("Add Vector", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Add Vector", False, f"Error: {str(e)}")
            return False
    
    def test_search_vectors(self) -> bool:
        """Test searching for vectors."""
        try:
            query_vector = np.random.rand(512).tolist()
            payload = {
                "embedding": query_vector,
                "k": 5,
                "threshold": 0.0
            }
            
            response = self.session.post(
                f"{self.base_url}/vectors/search",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                query_time = data.get('query_time_ms', 0)
                self.log_test(
                    "Search Vectors", 
                    True, 
                    f"Found {len(results)} results in {query_time:.2f}ms"
                )
                return True
            else:
                self.log_test("Search Vectors", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Search Vectors", False, f"Error: {str(e)}")
            return False
    
    def test_get_stats(self) -> bool:
        """Test getting index statistics."""
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                total_vectors = data.get('total_vectors', 0)
                gpu_enabled = data.get('gpu_enabled', False)
                self.log_test(
                    "Get Statistics", 
                    True, 
                    f"Vectors: {total_vectors}, GPU: {gpu_enabled}"
                )
                return True
            else:
                self.log_test("Get Statistics", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Get Statistics", False, f"Error: {str(e)}")
            return False
    
    def test_batch_add(self) -> bool:
        """Test batch adding vectors."""
        try:
            vectors = []
            for i in range(3):
                vectors.append({
                    "user_id": f"batch_user_{i:03d}",
                    "embedding": np.random.rand(512).tolist(),
                    "metadata": {"batch": True, "index": i}
                })
            
            payload = {"vectors": vectors}
            
            response = self.session.post(
                f"{self.base_url}/vectors/batch-add",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                successful = data.get('successful', 0)
                failed = data.get('failed', 0)
                self.log_test(
                    "Batch Add Vectors", 
                    True, 
                    f"Success: {successful}, Failed: {failed}"
                )
                return True
            else:
                self.log_test("Batch Add Vectors", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Batch Add Vectors", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        print("🧪 Starting FAISS GPU Service Tests")
        print("=" * 50)
        
        # Wait for service to be ready
        print("⏳ Waiting for service to be ready...")
        time.sleep(5)
        
        # Run tests
        tests = [
            self.test_health_check,
            self.test_add_vector,
            self.test_search_vectors,
            self.test_get_stats,
            self.test_batch_add
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
            time.sleep(1)  # Brief pause between tests
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Service is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the logs above.")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': passed / total,
            'all_passed': passed == total,
            'results': self.test_results
        }

def main():
    """Main function to run tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FAISS GPU Service')
    parser.add_argument('--url', default='http://localhost:8001', 
                       help='Base URL of the FAISS service')
    parser.add_argument('--output', help='Output file for test results (JSON)')
    
    args = parser.parse_args()
    
    tester = FaissServiceTester(args.url)
    results = tester.run_all_tests()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if results['all_passed'] else 1)

if __name__ == "__main__":
    main()
