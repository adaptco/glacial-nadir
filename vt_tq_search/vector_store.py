import hashlib
import logging
from typing import List, Dict, Any

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not available. Using in-memory mock.")

class ToolVectorStore:
    """
    Qdrant-backed tool vector storage.
    Separate collection from document embeddings.
    """
    
    def __init__(self, qdrant_url="http://localhost:6333", force_mock=False):
        self.collection_name = "tool_registry_v1"
        self.force_mock = force_mock
        if QDRANT_AVAILABLE and not force_mock:
            self.client = QdrantClient(url=qdrant_url)
            self._ensure_collection()
        else:
            self.mock_storage = {} # id -> point
        
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not QDRANT_AVAILABLE or self.force_mock: return
        
        collections = self.client.get_collections().collections
        if self.collection_name not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=Distance.COSINE
                )
            )
    
    def upsert_tool(self, tool_id: str, embedding: list, metadata: dict):
        """Store or update tool embedding."""
        point_id = self._hash_to_id(tool_id)
        
        if QDRANT_AVAILABLE and not self.force_mock:
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "tool_id": tool_id,
                    "content_hash": metadata["content_hash"],
                    "model_version": metadata["model_version"],
                    "timestamp": metadata["timestamp"]
                }
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
        else:
            self.mock_storage[point_id] = {
                "vector": embedding,
                "payload": {
                    "tool_id": tool_id,
                    "content_hash": metadata["content_hash"],
                    "model_version": metadata["model_version"],
                    "timestamp": metadata["timestamp"]
                }
            }
    
    def search_tools(self, query_vector: list, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for tools by semantic similarity.
        Returns: List of {tool_id, score, metadata}
        """
        if QDRANT_AVAILABLE and not self.force_mock:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            return [
                {
                    "tool_id": hit.payload["tool_id"],
                    "score": hit.score,
                    "content_hash": hit.payload["content_hash"],
                    "model_version": hit.payload["model_version"]
                }
                for hit in results
            ]
        else:
            # Simple mock search (dot product)
            results = []
            for pid, data in self.mock_storage.items():
                # Simplified dot product
                score = sum(a*b for a,b in zip(query_vector, data["vector"]))
                results.append({
                    "tool_id": data["payload"]["tool_id"],
                    "score": score,
                    "content_hash": data["payload"]["content_hash"],
                    "model_version": data["payload"]["model_version"]
                })
            # Sort by score desc
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
    
    @staticmethod
    def _hash_to_id(tool_id: str) -> int:
        """Convert tool_id to Qdrant point ID (uint64)."""
        return int(hashlib.sha256(tool_id.encode()).hexdigest()[:16], 16)
