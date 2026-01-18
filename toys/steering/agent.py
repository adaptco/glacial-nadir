"""
Agent definitions for The Steering Problem.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import random

from .state import SteeringState, SteeringAction

class BaseAgent(ABC):
    @abstractmethod
    def act(self, state: SteeringState) -> SteeringAction:
        """Decide on an action based on the current state."""
        pass
    
    @abstractmethod
    def update(self, state: SteeringState, reward: float):
        """Optional: Update internal policy based on reward."""
        pass

class RuleBasedAgent(BaseAgent):
    """
    A simple P-controller agent that tries to align heading with curvature.
    """
    def __init__(self, aggressiveness: float = 1.0):
        self.aggressiveness = aggressiveness
        
    def act(self, state: SteeringState) -> SteeringAction:
        # 1. Determine Target Heading
        # (Assuming the physics convention: Curvature * 30 is target)
        target_heading = state.track_curvature * 30.0
        
        # 2. Calculate Error
        error = target_heading - state.heading_deg
        
        # 3. Decide Action
        # Threshold for steering (deadzone to prevent oscillation)
        deadzone = 2.0
        
        if error > deadzone:
            return SteeringAction.STEER_LEFT  # Need to increase heading
        elif error < -deadzone:
            return SteeringAction.STEER_RIGHT # Need to decrease heading
        
        # 4. Speed Control (Simple Logic)
        # If error is high, brake. If low, accel.
        if abs(error) > 10.0:
            return SteeringAction.BRAKE
        elif state.speed < 0.8:
             return SteeringAction.ACCEL
        
        return SteeringAction.HOLD_LINE

    def update(self, state: SteeringState, reward: float):
        # Rule-based agent doesn't learn, but we could log something here.
        pass

class RandomAgent(BaseAgent):
    """
    Agent that acts randomly. Good for baseline failure comparisons.
    """
    def act(self, state: SteeringState) -> SteeringAction:
        return random.choice(list(SteeringAction))

    def update(self, state: SteeringState, reward: float):
        pass
