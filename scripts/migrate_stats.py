import json
import os
import time
from collections import deque

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STATS_FILE = os.path.join(DATA_DIR, 'traffic_stats.json')

def migrate():
    print("Starting migration...")
    if not os.path.exists(STATS_FILE):
        print("No stats file found.")
        return

    with open(STATS_FILE, 'r') as f:
        old_data = json.load(f)

    # Check if already migrated
    if "sources" in old_data and "global_total" in old_data:
        print("Data already appears to be in new format.")
        return

    print("Detected legacy format. Converting...")

    new_data = {
        "sources": {},
        "global_total": {
            "accumulated_count": 0,
            "cars": 0,
            "motorcycles": 0
        },
        "window_stats": {},
        "last_update": time.time()
    }

    # Iterate over old data (assuming keys are UUIDs)
    for key, value in old_data.items():
        if not isinstance(value, dict):
            continue
            
        # Copy source data
        new_data["sources"][key] = value
        
        # Aggregate totals
        new_data["global_total"]["accumulated_count"] += value.get("accumulated_count", 0)
        
        class_counts = value.get("accumulated_class_counts", {})
        # Assuming '0' is car, '1' is motorcycle based on config
        new_data["global_total"]["cars"] += class_counts.get("0", 0)
        new_data["global_total"]["motorcycles"] += class_counts.get("1", 0)

    # Save new file
    backup_file = STATS_FILE + ".legacy_backup"
    with open(backup_file, 'w') as f:
        json.dump(old_data, f, indent=4)
    print(f"Backed up legacy data to {backup_file}")

    with open(STATS_FILE, 'w') as f:
        json.dump(new_data, f, indent=4)
    
    print("Migration complete. New structure saved.")

if __name__ == "__main__":
    migrate()
