from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

# --- Core Ontology: The Qube ---

@dataclass
class Voxel:
    """Discretized 3D volume element."""
    position: tuple  # (x, y, z)
    occupancy: float  # 0.0 to 1.0
    velocity_field: tuple  # (vx, vy, vz)
    grip_coefficient: float
    local_entropy: float

@dataclass
class QubeState:
    """The root eigenstate vector (q_t)."""
    tick: int
    track_state: Dict[str, Any]  # Surface conditions, weather
    kart_states: Dict[str, 'KartState']
    driver_states: Dict[str, 'DriverState']
    env_state: Dict[str, Any]  # Global environment (wind, temp)
    race_control_state: Dict[str, Any]  # Flags, SC status
    
    # The Voxel Grid (Pixel -> Voxel expansion)
    voxels: Dict[tuple, Voxel] = field(default_factory=dict)

    @property
    def scalar_volume(self) -> float:
        """Total 'mass' of state: energy + entropy + uncertainty."""
        # This would be a complex calculation in the real implementation
        return 0.0

# --- Telemetry & Merkle Structures ---

@dataclass
class KartState:
    kart_id: str
    position: tuple
    velocity: tuple
    yaw: float
    slip_angle: float
    inputs: Dict[str, float]  # throttle, brake, steering
    energy: float
    health_metrics: Dict[str, float]

@dataclass
class DriverState:
    driver_id: str
    aggression: float
    focus: float
    current_intent: str

@dataclass
class RaceEvent:
    """Leaf node for the Merkle tree."""
    event_id: str
    tick: int
    kart_id: Optional[str]
    event_type: str  # overtake, collision, pit_stop, etc.
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    agent_decisions: Dict[str, str]  # Which expert voted for what
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str, sort_keys=True)

    def hash(self) -> str:
        """SHA-256 hash of the canonical JSON representation."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()

class MerkleTree:
    """Merkle tree implementation for race events."""
    def __init__(self, events: List[RaceEvent]):
        self.leaves = [e.hash() for e in events]
        self.root = self.build_tree(self.leaves)

    def build_tree(self, nodes: List[str]) -> str:
        if not nodes:
            return ""
        if len(nodes) == 1:
            return nodes[0]
        
        new_level = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i+1] if i + 1 < len(nodes) else left
            combined = hashlib.sha256((left + right).encode()).hexdigest()
            new_level.append(combined)
        
        return self.build_tree(new_level)

# --- Routing ---

@dataclass
class ExpertRouting:
    expert_name: str
    gating_weight: float
    is_active: bool
