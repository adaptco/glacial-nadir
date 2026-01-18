"""
Vector Store Phase Map - Auditable RAG Phase Management

This module implements a deterministic, phase-based vector store lifecycle manager
for the Agent Q RAG Desk (toy.rag_desk.v1). It provides:
- Strict phase transitions (INIT → INDEXING → READY → QUERYING → UPDATING)
- Canonical hashing for each phase state
- Audit trail for all operations
- Deterministic replay capability
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import hashlib
import jcs_utils

# --- Phase Definitions ---

class VectorPhase(Enum):
    """Discrete phases in the vector store lifecycle."""
    INIT = "INIT"
    INDEXING = "INDEXING"
    READY = "READY"
    QUERYING = "QUERYING"
    UPDATING = "UPDATING"
    ERROR = "ERROR"
    SEALED = "SEALED"  # Immutable state, no further changes allowed

# Valid phase transitions
PHASE_TRANSITIONS = {
    VectorPhase.INIT: [VectorPhase.INDEXING, VectorPhase.ERROR],
    VectorPhase.INDEXING: [VectorPhase.READY, VectorPhase.ERROR],
    VectorPhase.READY: [VectorPhase.QUERYING, VectorPhase.UPDATING, VectorPhase.SEALED, VectorPhase.ERROR],
    VectorPhase.QUERYING: [VectorPhase.READY, VectorPhase.ERROR],
    VectorPhase.UPDATING: [VectorPhase.INDEXING, VectorPhase.ERROR],
    VectorPhase.ERROR: [VectorPhase.INIT],  # Allow recovery
    VectorPhase.SEALED: []  # Terminal state
}

# --- Core Data Structures ---

@dataclass
class VectorDocument:
    """A single document in the vector store."""
    doc_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    indexed_at: Optional[str] = None
    
    def hash(self) -> str:
        """Canonical hash of the document."""
        doc_dict = asdict(self)
        canonical_bytes = jcs_utils.jcs_serialize(doc_dict)
        return hashlib.sha256(canonical_bytes).hexdigest()

@dataclass
class QueryResult:
    """Result of a vector similarity query."""
    query_id: str
    query_embedding: List[float]
    top_k: int
    results: List[Tuple[str, float]]  # (doc_id, similarity_score)
    timestamp: str
    phase_at_query: str
    
    def hash(self) -> str:
        """Canonical hash of the query result."""
        result_dict = asdict(self)
        canonical_bytes = jcs_utils.jcs_serialize(result_dict)
        return hashlib.sha256(canonical_bytes).hexdigest()

@dataclass
class PhaseTransitionEvent:
    """Audit log entry for a phase transition."""
    event_id: str
    tick: int
    timestamp: str
    from_phase: str
    to_phase: str
    trigger: str  # What caused the transition
    state_hash_before: str
    state_hash_after: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def hash(self) -> str:
        """Canonical hash of the transition event."""
        event_dict = asdict(self)
        canonical_bytes = jcs_utils.jcs_serialize(event_dict)
        return hashlib.sha256(canonical_bytes).hexdigest()

# --- Vector Store Phase Map ---

@dataclass
class VectorStorePhaseMap:
    """
    Manages the lifecycle and phase transitions of a vector store.
    
    Design Philosophy:
    - Each phase has strict entry/exit conditions
    - All transitions are logged and auditable
    - State is deterministically hashable at any point
    - Errors move to ERROR phase, not crash
    """
    
    # Identity
    store_id: str
    created_at: str
    
    # Current State
    current_phase: VectorPhase = VectorPhase.INIT
    tick: int = 0
    
    # Storage
    documents: Dict[str, VectorDocument] = field(default_factory=dict)
    query_history: List[QueryResult] = field(default_factory=list)
    
    # Audit Trail
    phase_transitions: List[PhaseTransitionEvent] = field(default_factory=list)
    
    # Configuration
    embedding_dim: int = 384  # Default dimension
    max_documents: int = 10000
    similarity_threshold: float = 0.7
    
    # Error State
    last_error: Optional[str] = None
    
    def __post_init__(self):
        """Initialize with INIT phase logged."""
        self._log_transition(
            from_phase=VectorPhase.INIT,
            to_phase=VectorPhase.INIT,
            trigger="INITIALIZATION"
        )
    
    # --- State Hashing ---
    
    def compute_state_hash(self) -> str:
        """
        Compute canonical hash of current store state.
        This is the 'eigenstate' of the vector store at this tick.
        """
        state_dict = {
            "store_id": self.store_id,
            "tick": self.tick,
            "current_phase": self.current_phase.value,
            "document_hashes": {doc_id: doc.hash() for doc_id, doc in self.documents.items()},
            "query_count": len(self.query_history),
            "embedding_dim": self.embedding_dim
        }
        canonical_bytes = jcs_utils.jcs_serialize(state_dict)
        return hashlib.sha256(canonical_bytes).hexdigest()
    
    # --- Phase Transition Logic ---
    
    def _log_transition(self, from_phase: VectorPhase, to_phase: VectorPhase, trigger: str, metadata: Optional[Dict] = None):
        """Internal: Log a phase transition."""
        self.tick += 1
        
        event = PhaseTransitionEvent(
            event_id=f"{self.store_id}_t{self.tick}",
            tick=self.tick,
            timestamp=datetime.utcnow().isoformat(),
            from_phase=from_phase.value,
            to_phase=to_phase.value,
            trigger=trigger,
            state_hash_before=self.compute_state_hash() if from_phase != VectorPhase.INIT else "GENESIS",
            state_hash_after="",  # Will be filled after transition
            metadata=metadata or {}
        )
        
        self.phase_transitions.append(event)
    
    def _validate_transition(self, to_phase: VectorPhase) -> Tuple[bool, str]:
        """Check if transition is valid."""
        if self.current_phase == VectorPhase.SEALED:
            return False, "Cannot transition from SEALED state"
        
        if to_phase not in PHASE_TRANSITIONS[self.current_phase]:
            return False, f"Invalid transition: {self.current_phase.value} → {to_phase.value}"
        
        return True, "OK"
    
    def transition_to(self, to_phase: VectorPhase, trigger: str, metadata: Optional[Dict] = None) -> bool:
        """
        Attempt to transition to a new phase.
        Returns True if successful, False if invalid.
        """
        valid, reason = self._validate_transition(to_phase)
        
        if not valid:
            self.last_error = reason
            self._force_error_state(reason)
            return False
        
        from_phase = self.current_phase
        self._log_transition(from_phase, to_phase, trigger, metadata)
        self.current_phase = to_phase
        
        # Update the state_hash_after for the event we just created
        self.phase_transitions[-1].state_hash_after = self.compute_state_hash()
        
        return True
    
    def _force_error_state(self, error_msg: str):
        """Force transition to ERROR state."""
        self.last_error = error_msg
        self._log_transition(
            from_phase=self.current_phase,
            to_phase=VectorPhase.ERROR,
            trigger="FORCED_ERROR",
            metadata={"error": error_msg}
        )
        self.current_phase = VectorPhase.ERROR
        self.phase_transitions[-1].state_hash_after = self.compute_state_hash()
    
    # --- Vector Store Operations ---
    
    def start_indexing(self) -> bool:
        """Begin indexing phase."""
        return self.transition_to(
            VectorPhase.INDEXING,
            trigger="START_INDEXING"
        )
    
    def add_document(self, doc: VectorDocument) -> bool:
        """Add a document during INDEXING phase."""
        if self.current_phase != VectorPhase.INDEXING:
            self.last_error = f"Cannot add documents in {self.current_phase.value} phase"
            return False
        
        if len(self.documents) >= self.max_documents:
            self._force_error_state("MAX_DOCUMENTS_EXCEEDED")
            return False
        
        if doc.embedding and len(doc.embedding) != self.embedding_dim:
            self._force_error_state(f"EMBEDDING_DIM_MISMATCH: expected {self.embedding_dim}, got {len(doc.embedding)}")
            return False
        
        doc.indexed_at = datetime.utcnow().isoformat()
        self.documents[doc.doc_id] = doc
        return True
    
    def finalize_indexing(self) -> bool:
        """Complete indexing and transition to READY."""
        if self.current_phase != VectorPhase.INDEXING:
            return False
        
        return self.transition_to(
            VectorPhase.READY,
            trigger="INDEXING_COMPLETE",
            metadata={"document_count": len(self.documents)}
        )
    
    def start_querying(self) -> bool:
        """Enter QUERYING phase."""
        return self.transition_to(
            VectorPhase.QUERYING,
            trigger="START_QUERY"
        )
    
    def execute_query(self, query_id: str, query_embedding: List[float], top_k: int = 5) -> Optional[QueryResult]:
        """Execute a similarity query."""
        if self.current_phase != VectorPhase.QUERYING:
            self.last_error = f"Cannot query in {self.current_phase.value} phase"
            return None
        
        # Simple cosine similarity (in real implementation, use FAISS or similar)
        results = []
        for doc_id, doc in self.documents.items():
            if doc.embedding:
                similarity = self._cosine_similarity(query_embedding, doc.embedding)
                if similarity >= self.similarity_threshold:
                    results.append((doc_id, similarity))
        
        # Sort by similarity and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]
        
        query_result = QueryResult(
            query_id=query_id,
            query_embedding=query_embedding,
            top_k=top_k,
            results=results,
            timestamp=datetime.utcnow().isoformat(),
            phase_at_query=self.current_phase.value
        )
        
        self.query_history.append(query_result)
        return query_result
    
    def end_querying(self) -> bool:
        """Exit QUERYING phase back to READY."""
        return self.transition_to(
            VectorPhase.READY,
            trigger="END_QUERY"
        )
    
    def start_updating(self) -> bool:
        """Enter UPDATING phase to modify the store."""
        return self.transition_to(
            VectorPhase.UPDATING,
            trigger="START_UPDATE"
        )
    
    def seal_store(self) -> bool:
        """Seal the store, making it immutable."""
        if self.current_phase != VectorPhase.READY:
            return False
        
        return self.transition_to(
            VectorPhase.SEALED,
            trigger="SEAL_REQUESTED",
            metadata={"final_state_hash": self.compute_state_hash()}
        )
    
    # --- Utility Methods ---
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    # --- Audit & Replay ---
    
    def get_audit_trail(self) -> List[PhaseTransitionEvent]:
        """Return the complete audit trail."""
        return self.phase_transitions
    
    def get_current_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "store_id": self.store_id,
            "tick": self.tick,
            "current_phase": self.current_phase.value,
            "document_count": len(self.documents),
            "query_count": len(self.query_history),
            "last_error": self.last_error,
            "state_hash": self.compute_state_hash(),
            "is_sealed": self.current_phase == VectorPhase.SEALED
        }
    
    def replay_from_genesis(self) -> List[Dict[str, Any]]:
        """
        Replay all phase transitions from genesis.
        Returns a timeline of state snapshots.
        """
        timeline = []
        for event in self.phase_transitions:
            timeline.append({
                "tick": event.tick,
                "transition": f"{event.from_phase} → {event.to_phase}",
                "trigger": event.trigger,
                "state_hash": event.state_hash_after,
                "metadata": event.metadata
            })
        return timeline
