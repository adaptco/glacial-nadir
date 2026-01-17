import socket
import json
import time

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # The port used by the server

def send_token(command, kart_id="kart_1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            
            token = {
                "command": command,
                "kart_id": kart_id,
                "timestamp": time.time()
            }
            
            print(f"Sending: {token}")
            s.sendall(json.dumps(token).encode('utf-8'))
            
            data = s.recv(4096)
            print(f"Received: {data.decode('utf-8')}")
        except ConnectionRefusedError:
            print("Could not connect to Kernel. Is it running?")

if __name__ == "__main__":
    # Simulate a race sequence
    print("--- Simulating Race Sequence ---")
    send_token("tick")
    time.sleep(0.5)
    send_token("tick")
    time.sleep(0.5)
    send_token("weather_change")
    time.sleep(0.5)
    send_token("incident", kart_id="kart_4")
