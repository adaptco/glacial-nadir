package validation.sovereign

import future.keywords.if

default allow = false

# S1: Must include a validated artifact
violation[msg] if {
    not input.drift_receipt
    not input.agent_checkpoint
    msg := "S1 Violation: Missing required DriftReceipt or AgentCheckpoint."
}

# P1: Air-Gap enforcement for file paths
violation[msg] if {
    input.agent_role == "INFRA"
    not startswith(input.target_path, "charts/")
    msg := "P1 Violation: INFRA agent attempted to access restricted path."
}

violation[msg] if {
    input.agent_role == "POLICY"
    not startswith(input.target_path, "policy/")
    msg := "P1 Violation: POLICY agent attempted to access restricted path."
}

# H1: Human approval for production impact
violation[msg] if {
    input.requires_human_approval == true
    not input.golden_snapshot
    msg := "H1 Violation: Production-impacting change missing GoldenSnapshot artifact."
}

allow if {
    count(violation) == 0
}
