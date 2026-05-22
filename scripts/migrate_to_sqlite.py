import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_db, insert_history_batch
from app.config import STATS_FILE

def migrate():
    print("Initializing Database...")
    init_db()
    
    if not os.path.exists(STATS_FILE):
        print("No stats file found.")
        return

    print(f"Loading JSON data from {STATS_FILE}...")
    try:
        with open(STATS_FILE, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return

    print("Migrating data...")
    # data IS the sources dict (mostly)
    total_records = 0
    
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        if "history" not in val:
            continue
            
        cam_id = key
        stats = val
        name = stats.get("name", "Unknown")
        history = stats.get("history", [])
        print(f"Processing {name} ({cam_id})... {len(history)} records")
        
        batch = []
        for item in history:
            # item: {'ts': float, 'count': int, 'cars': int, 'motors': int}
            batch.append((
                cam_id,
                item.get("ts"),
                item.get("count", 0),
                item.get("cars", 0),
                item.get("motors", 0),
                item.get("new_count", 0),
                item.get("new_cars", 0),
                item.get("new_motors", 0)
            ))
            
            if len(batch) >= 1000:
                insert_history_batch(batch)
                total_records += len(batch)
                batch = []
        
        if batch:
            insert_history_batch(batch)
            total_records += len(batch)
            
    print(f"Migration complete. Inserted {total_records} records.")

if __name__ == "__main__":
    migrate()
