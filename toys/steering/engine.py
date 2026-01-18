"""
Simulation Engine for The Steering Problem.
Runs the loop, handles Hooks, and Metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import random

from .state import SteeringState, SteeringAction, SteeringParams
from .physics import SteeringPhysics
from .agent import BaseAgent

@dataclass
class SimulationResult:
    final_state: SteeringState
    trace: List[Dict[str, Any]]
    total_reward: float
    success: bool
    reason: str

class SteeringEngine:
    def __init__(self, physics: SteeringPhysics = None):
        self.physics = physics or SteeringPhysics()
        self.hooks: Dict[str, List[Callable]] = {
            "pre_step": [],
            "post_step": [],
            "on_finish": []
        }

    def register_hook(self, event: str, callback: Callable):
        if event in self.hooks:
            self.hooks[event].append(callback)

    def _generate_curvature_profile(self, max_tick: int) -> Callable[[int], float]:
        """
        Generates a deterministic track profile (curvature vs tick).
        Simple S-curve.
        """
        def curvature_fn(tick: int) -> float:
            # 0-50: Straight
            if tick < 50: return 0.0
            # 50-100: Left Turn
            if tick < 100: return 1.0
            # 100-150: Straight
            if tick < 150: return 0.0
            # 150-200: Right Turn
            if tick < 200: return -1.0
            return 0.0
        return curvature_fn

    def calculate_reward(self, state: SteeringState, action: SteeringAction, params: SteeringParams) -> float:
        """
        Reward Function:
        R = Speed Bonus + Smoothness - Penalty
        """
        # 1. Speed Bonus
        r_speed = state.speed * 1.0
        
        # 2. Smoothness (Penalize erratic steering)
        r_smooth = 0.0
        if action in [SteeringAction.STEER_LEFT, SteeringAction.STEER_RIGHT]:
             r_smooth = -0.1
        
        # 3. Offtrack Penalty
        r_penalty = 0.0
        if state.offtrack_risk > params.offtrack_threshold:
            r_penalty = -5.0 * state.offtrack_risk
            
        return r_speed + r_smooth + r_penalty

    def run(self, agent: BaseAgent, seed: int = 42) -> SimulationResult:
        random.seed(seed)
        
        # Initialize
        state = SteeringState()
        params = self.physics.params
        trace = []
        total_reward = 0.0
        
        # Track Gen
        curvature_fn = self._generate_curvature_profile(params.max_steps)
        
        # Loop
        while not state.terminated:
            # 1. Update Environment Context (Curvature)
            state.track_curvature = curvature_fn(state.tick)
            
            # 2. Agent Decision
            action = agent.act(state)
            
            # 3. Physics Step
            # Note: The physics step modifies state in-place currently, which is efficient but 
            # we should capture 'before' state if we want strict transitions.
            # For simplicity, we capture the state *after* standard update.
            
            self.physics.step(state, action)
            
            # 4. Reward
            reward = self.calculate_reward(state, action, params)
            total_reward += reward
            
            # 5. Agent Update (Learning)
            agent.update(state, reward)
            
            # 6. Logging
            snapshot = state.to_dict()
            snapshot['action'] = action.value
            snapshot['reward'] = round(reward, 3)
            trace.append(snapshot)
            
            if state.terminated:
                break
                
        # Result
        success = state.damage < 1.0 and state.tick >= params.max_steps
        reason = "Completed Track" if success else "Crashed (Damage > 1.0)"
        if state.tick >= params.max_steps and not success:
             reason = "Time Limit Reached" # Should be success contextually, but check strictness
             if state.damage < 1.0: success = True # Completing frames is success if alive

        return SimulationResult(
            final_state=state,
            trace=trace,
            total_reward=total_reward,
            success=success,
            reason=reason
        )
