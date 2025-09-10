"""Alternative Vector Database Service supporting ChromaDB and Qdrant."""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
from abc import ABC, abstractmethod

# Import vector database clients
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from config import config
from models import VectorSearchResult

logger = logging.getLogger(__name__)


class VectorDatabaseInterface(ABC):
    """Abstract interface for vector database implementations."""
    
    @abstractmethod
    def add_vector(self, user_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a vector to the database."""
        pass
    
    @abstractmethod
    def search_vectors(self, query_embedding: List[float], k: int = 5, threshold: Optional[float] = None) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def remove_vector(self, user_id: str) -> bool:
        """Remove a vector from the database."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass
    
    @abstractmethod
    def reset_index(self) -> bool:
        """Reset the entire database."""
        pass


class ChromaDBService(VectorDatabaseInterface):
    """ChromaDB implementation of vector database service."""
    
    def __init__(self):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
        
        self.client = None
        self.collection = None
        self.collection_name = "face_embeddings"
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize persistent ChromaDB client
            persist_directory = os.path.dirname(config.INDEX_PATH)
            os.makedirs(persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_vector(self, user_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a vector to ChromaDB."""
        try:
            # Normalize embedding for cosine similarity
            embedding_array = np.array(embedding, dtype=np.float32)
            embedding_normalized = (embedding_array / np.linalg.norm(embedding_array)).tolist()
            
            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata['user_id'] = user_id
            doc_metadata['created_at'] = time.time()
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding_normalized],
                documents=[f"Face embedding for user {user_id}"],
                metadatas=[doc_metadata],
                ids=[user_id]
            )
            
            logger.info(f"Added vector for user {user_id} to ChromaDB")
            return user_id
            
        except Exception as e:
            logger.error(f"Failed to add vector to ChromaDB: {e}")
            raise
    
    def search_vectors(self, query_embedding: List[float], k: int = 5, threshold: Optional[float] = None) -> List[VectorSearchResult]:
        """Search for similar vectors in ChromaDB."""
        try:
            start_time = time.time()
            
            # Normalize query embedding
            query_array = np.array(query_embedding, dtype=np.float32)
            query_normalized = (query_array / np.linalg.norm(query_array)).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_normalized],
                n_results=k,
                include=['metadatas', 'distances']
            )
            
            # Process results
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, (user_id, distance, metadata) in enumerate(zip(
                    results['ids'][0],
                    results['distances'][0],
                    results['metadatas'][0]
                )):
                    # Convert distance to similarity (ChromaDB returns distances)
                    similarity = 1.0 - distance
                    
                    # Apply threshold if specified
                    if threshold is not None and similarity < threshold:
                        continue
                    
                    search_results.append(VectorSearchResult(
                        user_id=user_id,
                        similarity=similarity,
                        faiss_id=i,  # Use index as faiss_id equivalent
                        metadata=metadata
                    ))
            
            query_time = (time.time() - start_time) * 1000
            logger.info(f"ChromaDB search completed in {query_time:.2f}ms, found {len(search_results)} results")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in ChromaDB: {e}")
            raise
    
    def remove_vector(self, user_id: str) -> bool:
        """Remove a vector from ChromaDB."""
        try:
            self.collection.delete(ids=[user_id])
            logger.info(f"Removed vector for user {user_id} from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove vector from ChromaDB: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics."""
        try:
            count = self.collection.count()
            return {
                'total_vectors': count,
                'dimension': config.EMBEDDING_DIM,
                'database_type': 'ChromaDB',
                'collection_name': self.collection_name,
                'gpu_enabled': False  # ChromaDB doesn't use GPU acceleration
            }
            
        except Exception as e:
            logger.error(f"Failed to get ChromaDB stats: {e}")
            return {}
    
    def reset_index(self) -> bool:
        """Reset ChromaDB collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB collection reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB collection: {e}")
            return False


class QdrantService(VectorDatabaseInterface):
    """Qdrant implementation of vector database service."""
    
    def __init__(self):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client is not installed. Install with: pip install qdrant-client")
        
        self.client = None
        self.collection_name = "face_embeddings"
        self._initialize_qdrant()
    
    def _initialize_qdrant(self):
        """Initialize Qdrant client and collection."""
        try:
            # Initialize Qdrant client (local mode)
            qdrant_path = os.path.join(os.path.dirname(config.INDEX_PATH), "qdrant_db")
            os.makedirs(qdrant_path, exist_ok=True)
            
            self.client = QdrantClient(path=qdrant_path)
            
            # Create collection if it doesn't exist
            try:
                self.client.get_collection(self.collection_name)
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
            except:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=config.EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created new Qdrant collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    def add_vector(self, user_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a vector to Qdrant."""
        try:
            # Normalize embedding for cosine similarity
            embedding_array = np.array(embedding, dtype=np.float32)
            embedding_normalized = (embedding_array / np.linalg.norm(embedding_array)).tolist()
            
            # Prepare metadata
            payload = metadata or {}
            payload['user_id'] = user_id
            payload['created_at'] = time.time()
            
            # Create point
            point = PointStruct(
                id=hash(user_id) % (2**63),  # Convert user_id to integer ID
                vector=embedding_normalized,
                payload=payload
            )
            
            # Add to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Added vector for user {user_id} to Qdrant")
            return user_id
            
        except Exception as e:
            logger.error(f"Failed to add vector to Qdrant: {e}")
            raise
    
    def search_vectors(self, query_embedding: List[float], k: int = 5, threshold: Optional[float] = None) -> List[VectorSearchResult]:
        """Search for similar vectors in Qdrant."""
        try:
            start_time = time.time()
            
            # Normalize query embedding
            query_array = np.array(query_embedding, dtype=np.float32)
            query_normalized = (query_array / np.linalg.norm(query_array)).tolist()
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_normalized,
                limit=k,
                score_threshold=threshold,
                with_payload=True
            )
            
            # Process results
            results = []
            for i, result in enumerate(search_results):
                user_id = result.payload.get('user_id')
                if user_id:
                    results.append(VectorSearchResult(
                        user_id=user_id,
                        similarity=result.score,
                        faiss_id=result.id,
                        metadata=result.payload
                    ))
            
            query_time = (time.time() - start_time) * 1000
            logger.info(f"Qdrant search completed in {query_time:.2f}ms, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in Qdrant: {e}")
            raise
    
    def remove_vector(self, user_id: str) -> bool:
        """Remove a vector from Qdrant."""
        try:
            # Search for the point by user_id
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=[0.0] * config.EMBEDDING_DIM,  # Dummy vector
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match={"value": user_id}
                        )
                    ]
                ),
                limit=1
            )
            
            if search_results:
                point_id = search_results[0].id
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=[point_id]
                )
                logger.info(f"Removed vector for user {user_id} from Qdrant")
                return True
            else:
                logger.warning(f"No vector found for user {user_id} in Qdrant")
                return False
            
        except Exception as e:
            logger.error(f"Failed to remove vector from Qdrant: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Qdrant statistics."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'total_vectors': collection_info.points_count,
                'dimension': config.EMBEDDING_DIM,
                'database_type': 'Qdrant',
                'collection_name': self.collection_name,
                'gpu_enabled': False  # Local Qdrant doesn't use GPU
            }
            
        except Exception as e:
            logger.error(f"Failed to get Qdrant stats: {e}")
            return {}
    
    def reset_index(self) -> bool:
        """Reset Qdrant collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            logger.info("Qdrant collection reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset Qdrant collection: {e}")
            return False


class VectorDatabaseFactory:
    """Factory class for creating vector database instances."""
    
    @staticmethod
    def create_service(db_type: str = "faiss") -> VectorDatabaseInterface:
        """Create a vector database service instance."""
        db_type = db_type.lower()
        
        if db_type == "chromadb":
            return ChromaDBService()
        elif db_type == "qdrant":
            return QdrantService()
        else:
            # Default to optimized FAISS service
            from optimized_faiss_service import OptimizedFaissGPUService
            return OptimizedFaissGPUService()


# Configuration-based service selection
def get_vector_service() -> VectorDatabaseInterface:
    """Get the configured vector database service."""
    db_type = os.getenv("VECTOR_DB_TYPE", "faiss").lower()
    return VectorDatabaseFactory.create_service(db_type)
