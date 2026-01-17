import json
import time
import os

# This represents the "AI Monitor" or "Observer".
# It watches the work the Agent does by reading the logs.

LOG_FILE = "logs/activity_log.json"

def analyze_logs():
    if not os.path.exists(LOG_FILE):
        print("Waiting for Agent to generate logs...")
        return

    try:
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)
    except Exception:
        return

    if not logs:
        return

    # Simple analysis logic
    total = len(logs)
    warnings = sum(1 for log in logs if log['status'] == "WARNING")
    last_entry = logs[-1]

    os.system('cls' if os.name == 'nt' else 'clear')
    print("=== LIVE SYSTEM MONITOR ===")
    print(f"Total Records Analyzed: {total}")
    print(f"Warnings Detected:      {warnings}")
    print("-" * 30)
    print(f"Latest Status: {last_entry['status']}")
    print(f"Latest CPU:    {last_entry['cpu']}%")
    print(f"Latest Mem:    {last_entry['memory']}%")
    print("===========================")

def main():
    print("Monitor started...")
    try:
        while True:
            analyze_logs()
            time.sleep(3)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
