import time
import random
import json
import os
from datetime import datetime

# This is our 'Agent' - a software robot that does a specific job repeatedly.
# Its job right now is to simulate checking the 'health' of a server.

LOG_FILE = "logs/activity_log.json"

def perform_task():
    """Simulates the agent doing work."""
    # 1. Gather data (Input)
    cpu_usage = random.randint(10, 90)
    memory_usage = random.randint(20, 80)
    timestamp = datetime.now().isoformat()

    # 2. Make a decision (The 'Model' logic - very simple here)
    status = "HEALTHY"
    if cpu_usage > 80 or memory_usage > 75:
        status = "WARNING"
    
    # 3. Create a record
    record = {
        "timestamp": timestamp,
        "cpu": cpu_usage,
        "memory": memory_usage,
        "status": status,
        "message": f"System checked at {timestamp}"
    }
    
    return record

def save_record(record):
    """Writes the work to a permanent record (The 'Ledger')."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # Read existing logs or start new
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    
    logs.append(record)
    
    # Keep only last 10 records to keep file small for this demo
    logs = logs[-10:]
    
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"[{record['timestamp']}] Agent Status: {record['status']} (Saved to log)")

def main():
    print("Agent started. Press Ctrl+C to stop.")
    try:
        while True:
            # Run the task
            data = perform_task()
            save_record(data)
            
            # Wait 3 seconds before next run (Repeatable)
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nAgent stopped.")

if __name__ == "__main__":
    main()
