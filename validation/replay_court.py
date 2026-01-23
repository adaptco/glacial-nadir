import hashlib
import yaml
import jcs_utils


def canonicalize_jcs(data: dict) -> str:
    """Implements JSON Canonicalization Scheme (RFC 8785) for YAML objects."""
    return jcs_utils.jcs_serialize(data).decode("utf-8")


def verify_fossil(path_to_yaml: str) -> bool:
    """Verifies the integrity of a fossil artifact against its digest."""
    with open(path_to_yaml, "r", encoding="utf-8") as file_handle:
        artifact = yaml.safe_load(file_handle)

    provided_digest = artifact.pop("digest")
    computed_digest = hashlib.sha256(jcs_utils.jcs_serialize(artifact)).hexdigest()

    return provided_digest == computed_digest


# Kernel usage:
# if verify_fossil("validation/fossils/AgentProposalReceipt.v1.yaml"):
#     engage_replay_court()
