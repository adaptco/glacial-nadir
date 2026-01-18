from audit_fossil import FossilReceipt
import json
import time

def verify_kansei_drift():
    print("--- Kansei Drift Node Startup & Verification ---\n")
    
    # 1. BAEK-CHEON-2026 Manifest Parameters
    genesis_root = "8d9f2a1b5c3d4e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f"
    node_id = "KANSEI-DRIFT-001"
    epoch_id = "BAEK-CHEON-2026"
    # Simulated Merkle root of the Qube system state
    world_state_hash = "sha256:7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8" 
    hamiltonian_seed = 44  # LH=44 Deterministic Seed
    
    print(f"Target Epoch: {epoch_id}")
    print(f"Node Strategy: {node_id}")
    print(f"Genesis Root (512-bit): {genesis_root}")
    print(f"Han Root Alignment (Seed): {hamiltonian_seed}\n")
    
    # 2. Bind the 512-bit Bead (Create the Receipt)
    print(">>> Engaging Genesis Lock...")
    start_time = time.perf_counter()
    
    receipt = FossilReceipt.create(
        epoch_id=epoch_id,
        node_id=node_id,
        world_state_hash=world_state_hash,
        seed=hamiltonian_seed,
        genesis_root=genesis_root
    )
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    print(f"✅ Genesis Bound in {latency_ms:.4f}ms.")
    
    # 3. Validation Logic
    print("\n>>> Verifying Han Root Alignment...")
    canonical_json = receipt.to_json()
    parsed = json.loads(canonical_json)
    
    # Check if the root is actually in the payload
    c_manifest = parsed.get("canonical_manifest", {})
    bound_root = c_manifest.get("genesis_root")
    
    if bound_root == genesis_root:
        print("✅ PASS: Genesis Root correctly anchored in Canonical Manifest.")
    else:
        print(f"❌ FAIL: Genesis Root mismatch! Found: {bound_root}")
        return

    # Verify Cryptographic Seal
    is_seal_valid = receipt.verify_seal()
    if is_seal_valid:
        print("✅ PASS: Audit Seal validates against Manifest content.")
        print(f"    Seal: {receipt.audit_seal}")
        print(f"    CID:  {receipt.provenance.cid_keccak_256}")
    else:
        print("❌ FAIL: Cryptographic Seal Invalid!")
        return

    print("\n--- Status: GENESIS-LOCKED ---")
    print("Ready for High-Speed Telemetry Cycle.")

if __name__ == "__main__":
    verify_kansei_drift()
