#!/usr/bin/env python3
"""
Test script to verify Qdrant connection and create collection.
"""

import asyncio
import sys
import os
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_qdrant_connection():
    """Test Qdrant connection and create collection."""
    print("Testing Qdrant connection...")
    
    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        # Create client
        client = AsyncQdrantClient(
            url="http://localhost:6333",
            timeout=30,
            prefer_grpc=True,
            grpc_port=6334
        )
        
        # Test connection
        collections = await client.get_collections()
        print(f"✓ Connected to Qdrant. Found {len(collections.collections)} collections:")
        for col in collections.collections:
            print(f"  - {col.name}")
        
        # Check if face_embeddings exists
        collection_name = "face_embeddings"
        collection_exists = any(col.name == collection_name for col in collections.collections)
        
        if not collection_exists:
            print(f"Creating collection: {collection_name}")
            
            # Create collection with simple configuration
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=512,  # Face embedding dimension
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Collection {collection_name} created successfully")
        else:
            print(f"✓ Collection {collection_name} already exists")
        
        # Verify collection
        collection_info = await client.get_collection(collection_name)
        print(f"✓ Collection info: {collection_info.config.params.vectors}")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"✗ Qdrant connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the test."""
    print("=" * 60)
    print("Qdrant Connection Test")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    success = await test_qdrant_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Qdrant connection test PASSED!")
    else:
        print("❌ Qdrant connection test FAILED!")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
