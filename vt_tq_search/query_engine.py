import hashlib
from datetime import datetime
from typing import Dict, Any, List

class SemanticToolQuery:
    """
    Natural language tool discovery engine.
    """
    
    def __init__(self, embedder, vector_store, fossil_logger=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.fossil_logger = fossil_logger
        
    def find_tool(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Find best matching tool for user query.
        """
        # Generate query embedding
        # In a real scenario, embedder.embed_tool uses construct_semantic_text
        # We need a separate method for query embedding usually, or reuse the tokenizer part
        # For this implementation, we'll assuming embedder has a method or we adapt
        
        if hasattr(self.embedder, '_embed_query'):
            query_embedding = self.embedder._embed_query(user_query)
        elif hasattr(self.embedder, 'tokenizer'):
             # Direct access if available (transformers installed)
             import torch
             with torch.no_grad():
                inputs = self.embedder.tokenizer(
                    user_query,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.embedder.device)
                outputs = self.embedder.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                query_embedding = embedding.cpu().tolist()
        else:
             # Mock
             query_embedding = [0.1] * 384
             
        
        # Search vector store
        candidates = self.vector_store.search_tools(
            query_vector=query_embedding,
            top_k=top_k
        )
        
        # Log selection to fossilization ledger
        selection_event = {
            "event_type": "tool_discovery",
            "query": user_query,
            "query_hash": hashlib.sha256(user_query.encode()).hexdigest(),
            "selected_tool": candidates[0]["tool_id"] if candidates else None,
            "confidence": candidates[0]["score"] if candidates else 0.0,
            "candidates": candidates,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.fossil_logger:
            self.fossil_logger.log_tool_discovery(selection_event)
        
        return {
            "selected_tool": candidates[0]["tool_id"] if candidates else None,
            "confidence": candidates[0]["score"] if candidates else 0.0,
            "alternatives": candidates[1:] if len(candidates) > 1 else [],
            "query_hash": selection_event["query_hash"]
        }
