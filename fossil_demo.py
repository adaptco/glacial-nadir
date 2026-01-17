from audit_fossil import FossilReceipt
import json
import time

def demo_fossilization():
    print("--- Starting N3 Audit Fossilization Demo ---\n")

    # 1. Simulate a World State (The Manifest Content)
    # In a real run, this comes from the QubeState Merkle Root
    simulated_world_state_hash = "sha256:3EB838E5D4D256A45DE1584C1379B72C412729799CF2E3CFA745CF7CEE2B68B1"
    hamiltonian_seed = 1052244197
    current_epoch = "168h_burn_001"
    node_id = "OPAL-R32-N3"

    print(f"Targeting Epoch: {current_epoch}")
    print(f"Node: {node_id}")
    print(f"Input State Hash: {simulated_world_state_hash}")
    print(f"Hamiltonian Seed: {hamiltonian_seed}\n")

    # 2. Create the Seal
    print(">>> Engaging Inductive Sealing Sequence...")
    receipt = FossilReceipt.create(
        epoch_id=current_epoch,
        node_id=node_id,
        world_state_hash=simulated_world_state_hash,
        seed=hamiltonian_seed
    )

    # 3. Output the Canonical Receipt
    canonical_json = receipt.to_json()
    print("\n✅ SEALED. Generated Fossil Receipt:")
    
    # Pretty printing for the user (The actual artifact is compact)
    parsed = json.loads(canonical_json)
    print(json.dumps(parsed, indent=2))

    # 4. Verify the Seal
    print("\n>>> Verifying Seal Integrity...")
    is_valid = receipt.verify_seal()
    if is_valid:
        print("✅ PASS: Audit Seal validates against Canonical Manifest.")
    else:
        print("❌ FAIL: Cryptographic Mismatch Detected!")

    print("\n--- Sealing Complete. Ready for IPFS Pinning. ---")

if __name__ == "__main__":
    demo_fossilization()
