# Vector Store Phase Map - Reference

## Overview

The `VectorStorePhaseMap` is a deterministic, auditable vector store lifecycle manager designed for the **Agent Q RAG Desk** (`toy.rag_desk.v1`). It enforces strict phase transitions, canonical state hashing, and complete audit trails.

## Key Features

### 1. **Strict Phase Transitions**

```
INIT → INDEXING → READY → QUERYING → READY
                    ↓
                 SEALED (terminal, immutable)
```

**All phases can transition to ERROR on validation failures**

### 2. **Canonical State Hashing**

Every state change is hashed using JCS (JSON Canonicalization Scheme) + SHA-256:

- Deterministic across runs
- Enables verification and replay
- Creates audit trail

### 3. **Error Handling**

- Invalid transitions → ERROR phase
- Embedding dimension mismatches → ERROR
- Max document limits → ERROR
- Recovery path: ERROR → INIT

### 4. **Auditability**

Every phase transition creates a `PhaseTransitionEvent`:

```python
@dataclass
class PhaseTransitionEvent:
    event_id: str
    tick: int
    timestamp: str
    from_phase: str
    to_phase: str
    trigger: str
    state_hash_before: str
    state_hash_after: str
    metadata: Dict[str, Any]
```

## Usage Examples

### Basic RAG Flow

```python
from vector_phase_map import VectorStorePhaseMap, VectorDocument

# Initialize
store = VectorStorePhaseMap(
    store_id="my_rag_store",
    created_at=datetime.utcnow().isoformat(),
    embedding_dim=384
)

# Index documents
store.start_indexing()
doc = VectorDocument(
    doc_id="doc_001",
    content="Machine learning is awesome",
    embedding=[0.1, 0.2, ...],  # 384-dim vector
    metadata={"source": "ai"}
)
store.add_document(doc)
store.finalize_indexing()

# Query
store.start_querying()
result = store.execute_query(
    query_id="q1",
    query_embedding=[0.15, 0.25, ...],
    top_k=5
)
store.end_querying()

# Seal for immutability
store.seal_store()
```

### Audit Trail

```python
# Get complete history
timeline = store.replay_from_genesis()
for entry in timeline:
    print(f"T{entry['tick']}: {entry['transition']}")

# Get current state
summary = store.get_current_state_summary()
print(f"Phase: {summary['current_phase']}")
print(f"State Hash: {summary['state_hash']}")
```

## Phase Transition Rules

| From Phase | To Phase | Condition |
|------------|----------|-----------|
| INIT | INDEXING | Manual trigger |
| INDEXING | READY | All documents indexed |
| READY | QUERYING | Manual trigger |
| QUERYING | READY | Query complete |
| READY | UPDATING | Manual trigger |
| UPDATING | INDEXING | Update complete |
| READY | SEALED | Manual seal |
| ANY | ERROR | Validation failure |
| ERROR | INIT | Recovery |

## Validation Rules

1. **Embedding Dimension**: All embeddings must match `embedding_dim`
2. **Max Documents**: Cannot exceed `max_documents` (default: 10,000)
3. **Phase Constraints**:
   - Can only add documents in INDEXING phase
   - Can only query in QUERYING phase
   - Cannot modify SEALED stores

## Integration with Agent Q

This component fits into the Agent Q framework as:

- **Toy Model**: `toy.rag_desk.v1` (Memory & Retrieval Desk)
- **Focus**: Retrieval-Augmented Generation (RAG)
- **Key Concepts**:
  - Embedding distance
  - Hallucination prevention (via similarity threshold)
  - Grounding checks (via audit trail)

## Architecture Alignment

Follows the workspace patterns:

- ✅ Uses `@dataclass` (like `schemas.py`)
- ✅ Canonical hashing with `jcs_utils` (like `audit_fossil.py`)
- ✅ Deterministic state transitions
- ✅ Append-only audit log (Merkle-ready)
- ✅ Blueprint aesthetic: clean, engineered, auditable

## Files

| File | Purpose |
|------|---------|
| `vector_phase_map.py` | Core implementation |
| `demo_vector_phase_map.py` | 4 demo scenarios |

## Demo Scenarios

Run `python demo_vector_phase_map.py` to see:

1. **Demo 1**: Happy path (INIT → INDEXING → READY → QUERYING)
2. **Demo 2**: Invalid transition detection (ERROR recovery)
3. **Demo 3**: Embedding dimension mismatch
4. **Demo 4**: Store sealing (immutability guarantee)

## Future Enhancements

- [ ] Integration with actual embedding models (OpenAI, Sentence Transformers)
- [ ] FAISS/Annoy backend for real similarity search
- [ ] Persistent storage (IPFS integration)
- [ ] Merkle root calculation for query history
- [ ] Multi-index support (multiple embedding spaces)
- [ ] Incremental updates without full re-indexing

---

**Status**: ✅ Core implementation complete  
**Test Coverage**: 4 demo scenarios passing  
**Ready for**: Integration into Agent Q curriculum
