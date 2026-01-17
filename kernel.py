import socket
import json
import yaml
import time
import threading
from typing import Dict, Any
from schemas import QubeState, RaceEvent, MerkleTree, ExpertRouting
from datetime import datetime

# Configuration
HOST = '0.0.0.0'
PORT = 65432
CHARTER_PATH = 'agent_charter.yaml'

class AgenticKernel:
    def __init__(self, charter_path: str):
        self.charter = self._load_charter(charter_path)
        self.qube_state = self._init_qube()
        self.event_log = []
        self.merkle_tree = None
        self.tick_counter = 0
        
        # Load Experts (Mock implementation based on Charter)
        self.experts = [exp['name'] for exp in self.charter['mixture_of_agents']['experts']]
        print(f"[Kernel] Initialized with Experts: {self.experts}")

    def _load_charter(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[Kernel] Failed to load charter: {e}")
            # Fallback for dev/testing if file not found immediately
            return {"mixture_of_agents": {"experts": [{"name": "FallbackExpert"}]}}

    def _init_qube(self) -> QubeState:
        # Initialize a blank Qube state
        return QubeState(
            tick=0,
            track_state={},
            kart_states={},
            driver_states={},
            env_state={"status": "nominal"},
            race_control_state={}
        )

    def route_and_execute(self, input_token: Dict[str, Any]) -> Dict[str, Any]:
        """
        The Main Loop:
        1. Embed state (Mock)
        2. Route to Experts
        3. Execute
        4. Update Qube
        5. Log to Merkle
        """
        self.tick_counter += 1
        current_tick = self.tick_counter
        
        # 1. Routing (Logic from Charter: Gated Sparse)
        active_experts = []
        decisions = {}
        
        # Simple rule-based routing for demonstration
        command = input_token.get("command", "tick")
        
        if command == "incident":
            active_experts.append("StewardAgent")
            active_experts.append("RaceControllerAgent")
        elif command == "weather_change":
            active_experts.append("TrackDynamicsAgent")
            active_experts.append("StrategyAgent")
        else:
            # Default: Physics is always on
            active_experts.append("KartPhysicsAgent")
            if self.tick_counter % 10 == 0:
                 active_experts.append("StrategyAgent")

        # 2. Execution (Mock)
        for expert in active_experts:
            # In a real MoE, this would call the expert's forward() method
            decisions[expert] = f"Processed {command} at tick {current_tick}"

        # 3. Update State (Mock)
        self.qube_state.tick = current_tick
        
        # 4. Merkle Logging
        event = RaceEvent(
            event_id=f"evt_{current_tick}_{int(time.time())}",
            tick=current_tick,
            kart_id=input_token.get("kart_id", "global"),
            event_type=command,
            state_before={"tick": current_tick - 1},
            state_after={"tick": current_tick},
            agent_decisions=decisions
        )
        self.event_log.append(event)
        
        # Recompute Merkle Root
        # Optimization: In real systems, use an accumulator or append-only tree
        self.merkle_tree = MerkleTree(self.event_log)
        
        return {
            "tick": current_tick,
            "merkle_root": self.merkle_tree.root,
            "active_experts": active_experts,
            "decisions": decisions,
            "qube_status": "stable"
        }

def start_socket_server(kernel: AgenticKernel):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[Kernel] Socket Server listening on {HOST}:{PORT}")
        
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"[Kernel] Connected by {addr}")
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    
                    try:
                        # Expecting JSON tokens
                        input_token = json.loads(data.decode('utf-8'))
                        print(f"[Kernel] Received Token: {input_token}")
                        
                        response = kernel.route_and_execute(input_token)
                        
                        conn.sendall(json.dumps(response).encode('utf-8'))
                    except json.JSONDecodeError:
                        conn.sendall(b"Error: Invalid JSON Token")
                    except Exception as e:
                        conn.sendall(f"Error: {str(e)}".encode('utf-8'))

if __name__ == "__main__":
    print("[Kernel] Booting Agentic Kernel...")
    kernel = AgenticKernel(CHARTER_PATH)
    start_socket_server(kernel)
