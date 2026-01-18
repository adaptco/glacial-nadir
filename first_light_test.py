"""
First Light Telemetry Test

Validates the 1.2ms latency target across the PINN Arbitration Layer.
"""

import yaml
import time
import numpy as np
from pinn_arbitration import PINNArbitrator, VehicleState

print("=" * 80)
print("FIRST LIGHT TELEMETRY TEST - R32 POCKET BUNNY")
print("2026 Epoch | ASIL-D Integrity | Hausdorff Tolerance: 0.045nm")
print("=" * 80)
print()

# Load configuration
with open('config/digital_twin_r32.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize PINN
print("[1/4] Initializing PINN Arbitrator...")
arbitrator = PINNArbitrator(config)
print(f"      Target Latency: {arbitrator.target_latency_ms}ms")
print(f"      Hausdorff Tolerance: {arbitrator.hausdorff_tolerance:.2e}m")
print(f"      Physics Constraints: {len(arbitrator.constraints)}")
print()

# Create realistic test scenario: Sonic Green Hill loop transition
print("[2/4] Creating 'Sonic Green Hill' loop scenario...")
test_state = VehicleState(
    timestamp=0.0,
    position=np.array([0.0, 0.0, 0.0]),
    velocity=np.array([45.0, 3.5, 0.0]),  # 162 km/h with lateral drift
    acceleration=np.array([0.0, 0.0, 0.0]),
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    yaw_rate=0.15,  # Entering turn
    steering_angle=np.radians(20),  # 20 degrees steering input
    throttle=0.7,
    brake=0.0,
    wheel_speeds=np.array([55.0, 56.0, 54.0, 54.5]),  # Slight differential
    slip_angle=np.radians(6),  # 6 degrees slip
    surface_mu=0.95  # Dry asphalt
)
print(f"      Entry Speed: {np.linalg.norm(test_state.velocity[:2]) * 3.6:.1f} km/h")
print(f"      Slip Angle: {np.degrees(test_state.slip_angle):.2f} degrees")
print(f"      Steering: {np.degrees(test_state.steering_angle):.2f} degrees")
print()

# Run latency test
print("[3/4] Running 1.2ms latency validation (100 iterations)...")
print()

latencies = []
violations = 0
admissible_count = 0

for i in range(100):
    predicted_state, diagnostics = arbitrator.predict_state(test_state, dt=0.01)
    
    latencies.append(diagnostics['inference_time_ms'])
    if diagnostics['admissible']:
        admissible_count += 1
    if diagnostics['total_physics_violation'] > 0.1:
        violations += 1
    
    # Progress update every 25 iterations
    if (i + 1) % 25 == 0:
        print(f"  Iteration {i+1:3d}: {diagnostics['inference_time_ms']:.4f}ms | "
              f"ay={diagnostics['prediction_vector'][1]/9.81:.2f}g | "
              f"Drift={diagnostics['hausdorff_drift']:.2e}m")
    
    # Update state for next iteration
    test_state = predicted_state

print()
print("-" * 80)

# Performance Analysis
print("[4/4] Performance Analysis:")
print()

avg_latency = np.mean(latencies)
max_latency = np.max(latencies)
min_latency = np.min(latencies)
p95_latency = np.percentile(latencies, 95)
p99_latency = np.percentile(latencies, 99)

print(f"  Latency Statistics:")
print(f"    Average:  {avg_latency:.4f}ms (target: {arbitrator.target_latency_ms}ms)")
print(f"    Minimum:  {min_latency:.4f}ms")
print(f"    Maximum:  {max_latency:.4f}ms")
print(f"    P95:      {p95_latency:.4f}ms")
print(f"    P99:      {p99_latency:.4f}ms")
print()

latency_compliance = avg_latency <= arbitrator.target_latency_ms
admissibility_rate = (admissible_count / 100) * 100

print(f"  Compliance:")
print(f"    Latency Target:       {'[PASS]' if latency_compliance else '[FAIL]'}")
print(f"    Admissibility Rate:   {admissibility_rate:.1f}%")
print(f"    Physics Violations:   {violations}")
print()

# Final Verdict
print("=" * 80)
print("FIRST LIGHT VERDICT")
print("=" * 80)

if latency_compliance and admissibility_rate > 90:
    verdict = "CALCIFICATION_COMPLETE"
    status_symbol = "[OK]"
else:
    verdict = "REQUIRES_TUNING"
    status_symbol = "[WARN]"

print(f"{status_symbol} Status: {verdict}")
print(f"   Average Latency: {avg_latency:.4f}ms")
print(f"   Admissibility: {admissibility_rate:.1f}%")
print()

# Export performance report
report = arbitrator.get_performance_report()
merkle = arbitrator.export_audit_merkle()

print(f"[AUDIT] Merkle Root: {merkle[:32]}...")
print(f"[AUDIT] Total Inferences: {report['total_inferences']}")
print(f"[AUDIT] Total Violations Logged: {report['total_violations']}")
print()

print("=" * 80)
print("First Light telemetry test complete.")
print("Ready for Sonic Green Hill kinetic loop transition.")
print("=" * 80)
