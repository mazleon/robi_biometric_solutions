#!/usr/bin/env python3
"""
Simple script to create the face_embeddings collection in Qdrant.
"""

import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

async def create_collection():
    """Create the face_embeddings collection."""
    try:
        client = AsyncQdrantClient(
            url="http://localhost:6333",
            timeout=30
        )
        
        # Check if collection exists
        collections = await client.get_collections()
        collection_exists = any(col.name == "face_embeddings" for col in collections.collections)
        
        if collection_exists:
            print("Collection 'face_embeddings' already exists")
        else:
            # Create collection
            await client.create_collection(
                collection_name="face_embeddings",
                vectors_config=VectorParams(
                    size=512,
                    distance=Distance.COSINE
                )
            )
            print("Collection 'face_embeddings' created successfully")
        
        # Verify collection
        collection_info = await client.get_collection("face_embeddings")
        print(f"Collection verified: {collection_info.config.params.vectors}")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(create_collection())
    exit(0 if success else 1)
