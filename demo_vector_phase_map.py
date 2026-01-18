"""
Vector Phase Map Demo - Showcasing the RAG Desk Toy Model

This demo illustrates:
1. Proper phase transitions (INIT → INDEXING → READY → QUERYING)
2. Error handling (invalid transitions, dimension mismatches)
3. Audit trail generation
4. Deterministic replay
"""

from vector_phase_map import (
    VectorStorePhaseMap,
    VectorDocument,
    VectorPhase
)
from datetime import datetime
import random

def print_header(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def print_state(store: VectorStorePhaseMap):
    """Print current store state."""
    summary = store.get_current_state_summary()
    print(f"┌─ Store State (Tick {summary['tick']}) ─────────────────")
    print(f"│ Phase: {summary['current_phase']}")
    print(f"│ Documents: {summary['document_count']}")
    print(f"│ Queries: {summary['query_count']}")
    print(f"│ State Hash: {summary['state_hash'][:16]}...")
    if summary['last_error']:
        print(f"│ ⚠️  Last Error: {summary['last_error']}")
    print(f"└────────────────────────────────────────────────────")

def generate_dummy_embedding(dim: int = 384) -> list:
    """Generate a random embedding vector."""
    return [random.random() for _ in range(dim)]

# =============================================================================
# DEMO 1: Happy Path - Normal RAG Flow
# =============================================================================

def demo_happy_path():
    print_header("Demo 1: Happy Path - Normal RAG Flow")
    
    # Initialize store
    store = VectorStorePhaseMap(
        store_id="rag_desk_demo_001",
        created_at=datetime.utcnow().isoformat(),
        embedding_dim=384
    )
    
    print("✓ Store initialized")
    print_state(store)
    
    # Start indexing
    print("\n→ Starting indexing phase...")
    success = store.start_indexing()
    print(f"  {'✓' if success else '✗'} Transitioned to INDEXING")
    print_state(store)
    
    # Add documents
    print("\n→ Adding documents...")
    documents = [
        VectorDocument(
            doc_id="doc_001",
            content="The sky is blue and the grass is green",
            embedding=generate_dummy_embedding(),
            metadata={"source": "nature"}
        ),
        VectorDocument(
            doc_id="doc_002",
            content="Machine learning requires large datasets",
            embedding=generate_dummy_embedding(),
            metadata={"source": "ai"}
        ),
        VectorDocument(
            doc_id="doc_003",
            content="The weather today is sunny with clouds",
            embedding=generate_dummy_embedding(),
            metadata={"source": "weather"}
        )
    ]
    
    for doc in documents:
        success = store.add_document(doc)
        print(f"  {'✓' if success else '✗'} Added {doc.doc_id}")
    
    print_state(store)
    
    # Finalize indexing
    print("\n→ Finalizing indexing...")
    success = store.finalize_indexing()
    print(f"  {'✓' if success else '✗'} Transitioned to READY")
    print_state(store)
    
    # Query the store
    print("\n→ Starting query phase...")
    success = store.start_querying()
    print(f"  {'✓' if success else '✗'} Transitioned to QUERYING")
    
    query_embedding = generate_dummy_embedding()
    result = store.execute_query("query_001", query_embedding, top_k=2)
    
    if result:
        print(f"\n  Query Results (top {result.top_k}):")
        for doc_id, score in result.results:
            print(f"    - {doc_id}: {score:.4f}")
        print(f"  Result Hash: {result.hash()[:16]}...")
    
    # End querying
    print("\n→ Ending query phase...")
    success = store.end_querying()
    print(f"  {'✓' if success else '✗'} Returned to READY")
    print_state(store)
    
    # Show audit trail
    print("\n→ Audit Trail:")
    timeline = store.replay_from_genesis()
    for entry in timeline:
        print(f"  T{entry['tick']:02d}: {entry['transition']} ({entry['trigger']})")
    
    return store

# =============================================================================
# DEMO 2: Error Handling - Invalid Transitions
# =============================================================================

def demo_error_handling():
    print_header("Demo 2: Error Handling - Invalid Transitions")
    
    store = VectorStorePhaseMap(
        store_id="rag_desk_demo_002",
        created_at=datetime.utcnow().isoformat()
    )
    
    print("✓ Store initialized in INIT phase")
    print_state(store)
    
    # Try to query before indexing (INVALID)
    print("\n→ Attempting to query in INIT phase (should fail)...")
    success = store.start_querying()
    print(f"  {'✓' if success else '✗ EXPECTED'} Transition blocked")
    print_state(store)
    
    # The store should now be in ERROR state
    print(f"\n  Current Phase: {store.current_phase.value}")
    print(f"  Error Message: {store.last_error}")
    
    # Recover by going back to INIT
    print("\n→ Recovering from ERROR...")
    success = store.transition_to(VectorPhase.INIT, "RECOVERY")
    print(f"  {'✓' if success else '✗'} Recovered to INIT")
    print_state(store)
    
    return store

# =============================================================================
# DEMO 3: Dimension Mismatch Error
# =============================================================================

def demo_dimension_mismatch():
    print_header("Demo 3: Embedding Dimension Mismatch")
    
    store = VectorStorePhaseMap(
        store_id="rag_desk_demo_003",
        created_at=datetime.utcnow().isoformat(),
        embedding_dim=384  # Expected dimension
    )
    
    print("✓ Store configured for 384-dim embeddings")
    store.start_indexing()
    print_state(store)
    
    # Try to add document with wrong dimension
    print("\n→ Adding document with 512-dim embedding (should fail)...")
    bad_doc = VectorDocument(
        doc_id="bad_doc",
        content="This has the wrong embedding size",
        embedding=generate_dummy_embedding(512),  # WRONG!
        metadata={"source": "test"}
    )
    
    success = store.add_document(bad_doc)
    print(f"  {'✓' if success else '✗ EXPECTED'} Document rejected")
    print_state(store)
    
    print(f"\n  Error: {store.last_error}")
    
    return store

# =============================================================================
# DEMO 4: Sealing the Store
# =============================================================================

def demo_sealing():
    print_header("Demo 4: Sealing the Store (Immutability)")
    
    store = VectorStorePhaseMap(
        store_id="rag_desk_demo_004",
        created_at=datetime.utcnow().isoformat()
    )
    
    # Quick setup
    store.start_indexing()
    doc = VectorDocument(
        doc_id="final_doc",
        content="Final document before sealing",
        embedding=generate_dummy_embedding()
    )
    store.add_document(doc)
    store.finalize_indexing()
    
    print("✓ Store prepared with 1 document")
    print_state(store)
    
    # Seal the store
    print("\n→ Sealing the store...")
    success = store.seal_store()
    print(f"  {'✓' if success else '✗'} Store sealed")
    print_state(store)
    
    # Try to modify (should fail)
    print("\n→ Attempting to enter INDEXING phase (should fail)...")
    success = store.start_indexing()
    print(f"  {'✓' if success else '✗ EXPECTED'} Modification blocked")
    
    print(f"\n  Store is now immutable.")
    print(f"  Final State Hash: {store.compute_state_hash()}")
    
    return store

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       Vector Store Phase Map - Interactive Demo           ║")
    print("║       Agent Q: RAG Desk Toy Model (toy.rag_desk.v1)       ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Run all demos
    store1 = demo_happy_path()
    store2 = demo_error_handling()
    store3 = demo_dimension_mismatch()
    store4 = demo_sealing()
    
    # Final summary
    print_header("Demo Complete - Summary")
    print("✓ Demo 1: Happy path with indexing → querying")
    print("✓ Demo 2: Invalid transition detection")
    print("✓ Demo 3: Embedding dimension validation")
    print("✓ Demo 4: Store sealing for immutability")
    print("\n" + "="*60)
    print("All phase transitions are:")
    print("  - Deterministic")
    print("  - Auditable (logged with hashes)")
    print("  - Replayable (from genesis)")
    print("="*60 + "\n")
