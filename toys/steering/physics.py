"""
Deterministic physics engine for The Steering Problem (toy.steering.v1).
"""

from .state import SteeringState, SteeringAction, SteeringParams

class SteeringPhysics:
    def __init__(self, params: SteeringParams = SteeringParams()):
        self.params = params

    def get_desired_heading(self, curvature: float) -> float:
        """
        Model: Ideally you want to face 'into' the turn slightly to counteract drift,
        or just align with the tangent. For this toy, let's say desired heading
        is proportional to curvature (e.g., following the road tangent).
        
        Simple model: Desired heading is exactly the curvature angle.
        """
        # In this 1D abstraction, 'curvature' can be treated as the 'target heading'
        # relative to the straight line.
        return curvature * 100.0  # Scale curvature to degrees, just a simple mapping

    def step(self, state: SteeringState, action: SteeringAction) -> SteeringState:
        """
        Advance the state by one tick based on the action.
        Pure function: does not modify input state in-place (returns new copy or updates fields if mutable).
        Here we update in place for performance, or return new? 
        Let's update in place for this simple toy model simplicity, or copy if we want strict history.
        """
        # 1. Update Heading
        if action == SteeringAction.STEER_LEFT:
            state.heading_deg += self.params.steer_delta_deg
        elif action == SteeringAction.STEER_RIGHT:
            state.heading_deg -= self.params.steer_delta_deg
        # HOLD_LINE, BRAKE, ACCEL do not change heading directly in this simplified model
        # (Though biologically, braking usually transfers weight -> more grip -> steering effect, ignoring here)

        # 2. Update Speed
        if action == SteeringAction.ACCEL:
            state.speed += self.params.accel_delta
        elif action == SteeringAction.BRAKE:
            state.speed -= self.params.brake_delta
        
        # Clamp speed
        state.speed = max(0.0, min(1.0, state.speed))

        # 3. Calculate Risks
        # Desired heading depends on track curvature (which changes over time/distance)
        # For this step, we assume curvature was set by the Environment before step() or stays constant.
        # Let's assume the Environment sets curvature.
        
        # Simple Logic: "offtrack_risk = abs(heading - desired) * (1 - grip)"
        # Note: We need a 'reference' heading for the track.
        # Let's assume Curvature IS the target heading for local segment relative to world frame 0.
        desired_heading = self.params.steer_delta_deg * state.track_curvature * 5 # arbitrary scaling
        
        # Actually, let's stick to the starter_pack logic:
        # "offtrack_risk = abs(heading_deg - desired_heading(track_curvature)) * (1 - grip)"
        # We need to define desired_heading func.
        # Let's say track_curvature is -1.0 (Hard Right) to 1.0 (Hard Left).
        # Max heading needed is let's say 30 degrees.
        desired_heading = state.track_curvature * 30.0
        
        state.offtrack_risk = abs(state.heading_deg - desired_heading) / 30.0 * (1.0 - state.grip)
        
        # 4. Apply Damage
        if state.offtrack_risk > self.params.offtrack_threshold:
            state.damage += self.params.damage_on_offtrack
        
        # 5. Lap Progress
        # "progress_gain = clamp(speed, 0, 1) * progress_gain(grip)"
        # If grip is low, maybe you slip and gain less progress?
        # Logic: effective_speed = speed * grip
        state.lap_progress += state.speed * state.grip * 0.01 # Base progress
        
        # 6. Tick
        state.tick += 1
        state.last_action = action
        
        # Terminate?
        if state.damage >= 1.0 or state.tick >= self.params.max_steps:
            state.terminated = True
            
        return state
