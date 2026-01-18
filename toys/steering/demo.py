"""
Demo runner for The Steering Problem (toy.steering.v1).
"""
import sys
import os
import argparse

# Add repo root to path if running directly
sys.path.append(os.getcwd())

from toys.steering.engine import SteeringEngine, SteeringPhysics
from toys.steering.agent import RuleBasedAgent, RandomAgent

def print_banner(text: str):
    print(f"\n{'='*50}")
    print(f"  {text}")
    print(f"{'='*50}\n")

def run_demo(agent_type: str = "rule", seed: int = 42, verbose: bool = False):
    print_banner(f"Running Demo: {agent_type.upper()} AGENT (Seed={seed})")
    
    # 1. Setup
    physics = SteeringPhysics()
    engine = SteeringEngine(physics)
    
    if agent_type == "random":
        agent = RandomAgent()
    else:
        agent = RuleBasedAgent(aggressiveness=1.0)
        
    # 2. Run
    result = engine.run(agent, seed=seed)
    
    # 3. Report
    print(f"Outcome: {result.reason}")
    print(f"Success: {'✅' if result.success else '❌'}")
    print(f"Total Reward: {result.total_reward:.2f}")
    print(f"Final Damage: {result.final_state.damage:.2f}")
    print(f"Ticks Survived: {result.final_state.tick} / {physics.params.max_steps}")
    
    # 4. Trace Visualization (Simple ASCII)
    if verbose:
        print("\nTrace Preview (Last 10 steps or critical events):")
        print(f"{'Tick':<6} | {'Curv':<6} | {'Head':<6} | {'Act':<12} | {'Risk':<6} | {'Dmg':<6}")
        print("-" * 60)
        
        # Show start, some middle, and end
        interesting_indices = list(range(0, 5)) + \
                              list(range(45, 55)) + \
                              list(range(len(result.trace)-5, len(result.trace)))
        
        interesting_indices = sorted(list(set(interesting_indices))) # Dedup
        
        last_idx = -1
        for idx in interesting_indices:
            if idx >= len(result.trace): continue
            
            if idx > last_idx + 1 and last_idx != -1:
                print("...")
            
            step = result.trace[idx]
            print(f"{step['tick']:<6} | {step['track_curvature']:<6.2f} | {step['heading_deg']:<6.2f} | {step['action']:<12} | {step['offtrack_risk']:<6.3f} | {step['damage']:<6.2f}")
            last_idx = idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["rule", "random"], default="rule")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    run_demo(args.agent, args.seed, args.verbose)
