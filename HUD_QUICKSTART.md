# Master Operator HUD - Quick Start Guide

## What You Have

A **live telemetry dashboard** for the Agentic Kernel that displays:
- Real-time Qube state (tick counter, status, Merkle root)
- Expert network activation (which agents are firing)
- Telemetry graph (active expert count over time)
- Event log (recent tick history)

## How to Launch

### Step 1: Start the Enhanced Kernel
```powershell
python kernel_websocket.py
```

This starts:
- Socket server on port **65432** (for token commands)
- HTTP API on port **8080** (for the dashboard)

### Step 2: Open the Dashboard
Open this file in your browser:
```
ui/dashboard.html
```

Or use a local HTTP server (recommended):
```powershell
cd ui
python -m http.server 3000
```
Then navigate to: `http://localhost:3000/dashboard.html`

### Step 3: Send Commands to the Kernel
Use the test client:
```powershell
python client_test.py
```

Or send manual commands:
```python
import socket, json
s = socket.socket()
s.connect(('localhost', 65432))
s.sendall(json.dumps({"command": "weather_change"}).encode())
print(s.recv(4096).decode())
s.close()
```

## Visual Features

- **British Racing Green Theme**: Matches your Aston Martin aesthetic
- **Blueprint Grid**: Technical, high-precision UI
- **Live Expert Activation**: Cards glow green when experts fire
- **Merkle Root Display**: Cryptographic audit trail
- **Telemetry Graph**: Real-time visualization of system load

## API Endpoints

The kernel exposes:
- `GET /status` - Current state (tick, merkle root, active experts)
- `GET /experts` - List of all available experts

## Master Operator Bootstrap

For a single-command launch, use:
```powershell
./bootstrap.ps1
```

This will:
1. Clear port 65432
2. Start the kernel
3. (You can extend it to open the dashboard automatically)
