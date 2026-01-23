import hashlib
import json
from typing import Any, Dict, Optional

# --- CIE-V1: Domain Swap Protocol (Inquiry Intake) ---
# Purpose: Canonicalization and Deterministic Fossilization of Customer Inquiries.
# Invariants: SHA-256 Digest Law // Deterministic Fallback (Delta 1)


def sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def execute_fossilization(input_data: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Field Extraction (The New Schema)
    customer_id = input_data.get("customer_id")
    inquiry_type = input_data.get("inquiry_type")
    message = input_data.get("message")
    channel = input_data.get("channel")
    timestamp = input_data.get("timestamp")

    # Header Override Rule (Patch 1)
    header_event_id = input_data.get("hdr_event_id")

    # 2. Build Identity Preimage
    preimage = {
        "channel": channel,
        "customer_id": customer_id,
        "inquiry_type": inquiry_type,
        "message": message,
        "timestamp": timestamp,
    }

    # 3. Canonicalization (Digest Law)
    # Locked Spec: sort_keys=True, separators=(',', ':')
    canonical_json = json.dumps(preimage, sort_keys=True, separators=(",", ":"))
    leaf_digest = sha256_hex(canonical_json)

    # 4. Deterministic Identity Derivation (Delta 1)
    # If no header override exists, derive ID from leaf_digest
    if header_event_id:
        final_event_id = header_event_id
        adjudication_type = "HEADER_OVERRIDE"
    else:
        # Fallback formula: sha256("event:" + leaf_digest)
        final_event_id = sha256_hex(f"event:{leaf_digest}")
        adjudication_type = "DETERMINISTIC_FALLBACK"

    # 5. Output Construction
    return {
        "event_id": final_event_id,
        "leaf_digest": f"sha256:{leaf_digest}",
        "raw_payload": canonical_json,
        "merkle_root": leaf_digest,  # Single-leaf topology
        "adjudication_verdict": "PASS",
        "adjudication_type": adjudication_type,
        "corridor_state": "State-3: Active Lineage",
    }


def zapier_entrypoint(input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Zapier-compatible entry point that delegates to the fossilization routine."""
    return execute_fossilization(input_data or {})


if "input_data" in globals():
    output = execute_fossilization(globals()["input_data"])
