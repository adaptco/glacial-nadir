from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional
import hashlib
from datetime import datetime
import jcs_utils

@dataclass
class CanonicalManifest:
    node_id: str
    genesis_root: str
    world_state_hash: str
    hamiltonian_seed: int

@dataclass
class Provenance:
    cid_keccak_256: str
    safety_path: str

@dataclass
class FossilReceipt:
    receipt_version: str
    epoch_id: str
    audit_seal: str
    canonical_manifest: CanonicalManifest
    provenance: Provenance

    @classmethod
    def create(cls, epoch_id: str, node_id: str, world_state_hash: str, seed: int, genesis_root: str, safety_path: str = "FAIL_CLOSED|LEDGER_FIRST"):
        """
        Factory method to create a sealed receipt.
        Calculates the audit_seal based on the canonical hash of the manifest + seed.
        """
        manifest = CanonicalManifest(
            node_id=node_id,
            genesis_root=genesis_root,
            world_state_hash=world_state_hash,
            hamiltonian_seed=seed
        )
        
        # Calculate CID / Audit components
        # In this spec, audit_seal is the hash of the canonical manifest
        # (Or a Merkle root usually, but here defined as hash of content for single leaf)
        manifest_dict = asdict(manifest)
        canonical_bytes = jcs_utils.jcs_serialize(manifest_dict)
        audit_seal_hash = hashlib.sha256(canonical_bytes).hexdigest()
        audit_seal = f"sha256:{audit_seal_hash}"
        
        # Calculate Keccak (simulated with SHA3_256 if Keccak not available, using sha3_256 for standard Python)
        # Note: True Keccak-256 is slightly different from SHA3-256, but for this pure Python env without web3 dependencies, 
        # SHA3-256 is the closest standard lib equivalent for a strong hash.
        cid_hash = hashlib.sha3_256(canonical_bytes).hexdigest()
        cid_keccak = f"0x{cid_hash.upper()}"

        provenance = Provenance(
            cid_keccak_256=cid_keccak,
            safety_path=safety_path
        )

        return cls(
            receipt_version="fossil.receipt.v1",
            epoch_id=epoch_id,
            audit_seal=audit_seal,
            canonical_manifest=manifest,
            provenance=provenance
        )

    def to_json(self) -> str:
        """Returns the JCS canonical string of the entire receipt."""
        return jcs_utils.jcs_serialize(asdict(self)).decode('utf-8')

    def verify_seal(self) -> bool:
        """Verify that the audit_seal matches the canonical manifest content."""
        manifest_dict = asdict(self.canonical_manifest)
        canonical_bytes = jcs_utils.jcs_serialize(manifest_dict)
        calculated_hash = hashlib.sha256(canonical_bytes).hexdigest()
        expected = self.audit_seal.replace("sha256:", "")
        return calculated_hash == expected
