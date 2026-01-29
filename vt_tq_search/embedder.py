import hashlib
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import sys

# Try imports, handle missing deps for testability in lightweight envs
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers/Torch not available. Embedding will be mocked.")

try:
    import jcs  # JSON Canonicalization Scheme
    JCS_AVAILABLE = True
except ImportError:
    from jcs_utils import jcs_serialize # Fallback to local utils if installed
    # Or define simple fallback
    JCS_AVAILABLE = False

class ToolEmbedder:
    """
    Deterministic tool embedding generator.
    Uses frozen model weights for reproducible embeddings.
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = "cpu"
        
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()  # Freeze for determinism
        
    def embed_tool(self, tool_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate embedding for a tool document.
        Returns: {embedding: List[float], content_hash: str}
        """
        # Canonicalize tool document for deterministic hashing
        if 'jcs' in sys.modules:
             # This part might be tricky if jcs isn't installed, will use simple json dump for now if missing
             canonical_json = json.dumps(tool_doc, sort_keys=True).encode('utf-8')
        else:
             canonical_json = json.dumps(tool_doc, sort_keys=True).encode('utf-8')
             
        content_hash = hashlib.sha256(canonical_json).hexdigest()
        
        # Construct semantic text from key fields
        semantic_text = self._construct_semantic_text(tool_doc)
        
        # Generate embedding
        if TRANSFORMERS_AVAILABLE:
            with torch.no_grad():
                inputs = self.tokenizer(
                    semantic_text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # Mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                # L2 normalization for cosine similarity
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                embedding_list = embedding.cpu().tolist()
        else:
            # Mock embedding for environments without torch
            padding = [0.0] * 383
            # Simple deterministic mock based on hash
            seed = int(content_hash[:8], 16)
            embedding_list = [float(seed)/0xFFFFFFFF] + padding
            
        return {
            "tool_id": tool_doc["tool_id"],
            "embedding": embedding_list,
            "content_hash": content_hash,
            "model_version": self.model_name,
            "timestamp": datetime.now().isoformat()
        }
    
    def _construct_semantic_text(self, tool_doc: Dict[str, Any]) -> str:
        """Construct searchable text from tool metadata."""
        parts = [
            tool_doc.get("name", ""),
            tool_doc.get("description", ""),
            " ".join(tool_doc.get("capabilities", [])),
            " ".join(tool_doc.get("use_cases", []))
        ]
        return " | ".join(parts)
