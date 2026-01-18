"""
State definitions for The Steering Problem (toy.steering.v1).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

class SteeringAction(Enum):
    STEER_LEFT = "STEER_LEFT"
    STEER_RIGHT = "STEER_RIGHT"
    HOLD_LINE = "HOLD_LINE"
    BRAKE = "BRAKE"
    ACCEL = "ACCEL"

@dataclass
class SteeringState:
    """
    Represents the state of the kart in the 1D steering simulation.
    """
    # Dynamic State
    heading_deg: float = 0.0
    speed: float = 0.6
    lap_progress: float = 0.0
    damage: float = 0.0
    
    # Environment State
    track_curvature: float = 0.0
    grip: float = 0.8
    offtrack_risk: float = 0.0
    
    # Meta
    tick: int = 0
    terminated: bool = False
    last_action: Optional[SteeringAction] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "heading_deg": round(self.heading_deg, 2),
            "speed": round(self.speed, 2),
            "lap_progress": round(self.lap_progress, 2),
            "damage": round(self.damage, 2),
            "track_curvature": round(self.track_curvature, 2),
            "grip": round(self.grip, 2),
            "offtrack_risk": round(self.offtrack_risk, 3),
            "terminated": self.terminated,
            "last_action": self.last_action.value if self.last_action else None
        }

@dataclass
class SteeringParams:
    """
    Simulation parameters.
    """
    dt: float = 1.0
    max_steps: int = 200
    steer_delta_deg: float = 6.0
    brake_delta: float = 0.08
    accel_delta: float = 0.06
    damage_on_offtrack: float = 0.03
    offtrack_threshold: float = 0.2
