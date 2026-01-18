import socket
import json
import yaml
import time
import threading
from typing import Dict, Any
from schemas import QubeState, RaceEvent, MerkleTree, ExpertRouting
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Configuration
SOCKET_HOST = '0.0.0.0'
SOCKET_PORT = 65432
HTTP_HOST = '0.0.0.0'
HTTP_PORT = 8080
CHARTER_PATH = 'agent_charter.yaml'

class AgenticKernel:
    def __init__(self, charter_path: str):
        self.charter = self._load_charter(charter_path)
        self.qube_state = self._init_qube()
        self.event_log = []
        self.merkle_tree = None
        self.tick_counter = 0
        self.current_status = "IDLE"
        self.active_experts = []
        self.last_decision = {}
        
        # Load Experts (Mock implementation based on Charter)
        self.experts = [exp['name'] for exp in self.charter['mixture_of_agents']['experts']]
        print(f"[Kernel] Initialized with Experts: {self.experts}")

    def _load_charter(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[Kernel] Failed to load charter: {e}")
            return {"mixture_of_agents": {"experts": [{"name": "FallbackExpert"}]}}

    def _init_qube(self) -> QubeState:
        return QubeState(
            tick=0,
            track_state={},
            kart_states={},
            driver_states={},
            env_state={"status": "nominal"},
            race_control_state={}
        )

    def route_and_execute(self, input_token: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution loop"""
        self.tick_counter += 1
        current_tick = self.tick_counter
        
        # Routing (Logic from Charter: Gated Sparse)
        active_experts = []
        decisions = {}
        
        command = input_token.get("command", "tick")
        
        if command == "incident":
            active_experts.append("StewardAgent")
            active_experts.append("RaceControllerAgent")
            self.current_status = "INCIDENT"
        elif command == "weather_change":
            active_experts.append("TrackDynamicsAgent")
            active_experts.append("StrategyAgent")
            self.current_status = "WEATHER_EVENT"
        else:
            active_experts.append("KartPhysicsAgent")
            if self.tick_counter % 10 == 0:
                 active_experts.append("StrategyAgent")
            self.current_status = "RUNNING"

        # Execution (Mock)
        for expert in active_experts:
            decisions[expert] = f"Processed {command} at tick {current_tick}"

        # Update State
        self.qube_state.tick = current_tick
        self.active_experts = active_experts
        self.last_decision = decisions
        
        # Merkle Logging
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
        
        self.merkle_tree = MerkleTree(self.event_log)
        
        return {
            "tick": current_tick,
            "merkle_root": self.merkle_tree.root,
            "active_experts": active_experts,
            "decisions": decisions,
            "qube_status": "stable"
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current kernel status for HTTP API"""
        return {
            "tick": self.tick_counter,
            "status": self.current_status,
            "merkle_root": self.merkle_tree.root if self.merkle_tree else "none",
            "active_experts": self.active_experts,
            "total_experts": len(self.experts),
            "event_count": len(self.event_log),
            "qube_state": {
                "env_status": self.qube_state.env_state.get("status", "unknown")
            }
        }

class HTTPHandler(BaseHTTPRequestHandler):
    kernel = None  # Will be set by main()
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            status = self.kernel.get_status()
            self.wfile.write(json.dumps(status).encode())
        elif parsed.path == '/experts':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"experts": self.kernel.experts}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP logs

def start_socket_server(kernel: AgenticKernel):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((SOCKET_HOST, SOCKET_PORT))
        s.listen()
        print(f"[Kernel] Socket Server listening on {SOCKET_HOST}:{SOCKET_PORT}")
        
        while True:
            conn, addr = s.accept()
            with conn:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    
                    try:
                        input_token = json.loads(data.decode('utf-8'))
                        response = kernel.route_and_execute(input_token)
                        conn.sendall(json.dumps(response).encode('utf-8'))
                    except json.JSONDecodeError:
                        conn.sendall(b"Error: Invalid JSON Token")
                    except Exception as e:
                        conn.sendall(f"Error: {str(e)}".encode('utf-8'))

def start_http_server(kernel: AgenticKernel):
    HTTPHandler.kernel = kernel
    server = HTTPServer((HTTP_HOST, HTTP_PORT), HTTPHandler)
    print(f"[Kernel] HTTP API listening on {HTTP_HOST}:{HTTP_PORT}")
    server.serve_forever()

if __name__ == "__main__":
    print("[Kernel] Booting Agentic Kernel...")
    kernel = AgenticKernel(CHARTER_PATH)
    
    # Start HTTP server in a thread
    http_thread = threading.Thread(target=start_http_server, args=(kernel,), daemon=True)
    http_thread.start()
    
    # Start socket server in main thread
    start_socket_server(kernel)
