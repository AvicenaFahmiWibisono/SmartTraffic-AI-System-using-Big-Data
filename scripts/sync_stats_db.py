import sqlite3
import json
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "traffic_data.db")
STATS_PATH = os.path.join(DATA_DIR, "traffic_stats.json")
CONFIG_PATH = os.path.join(DATA_DIR, "cctv_config.json")

def sync_db_to_json():
    print("Starting sync from DB to JSON...")
    
    if not os.path.exists(DB_PATH):
        print("Database not found!")
        return
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Load current stats
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
    else:
        stats = {"sources": {}, "global_total": {}}
        
    # Load config to ensure we have all cameras
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
    else:
        config = []
        
    # Get totals from DB
    print("Querying database totals...")
    c.execute("""
        SELECT 
            camera_id, 
            SUM(new_count) as total, 
            SUM(new_cars) as cars, 
            SUM(new_motors) as motors 
        FROM traffic_history 
        GROUP BY camera_id
    """)
    rows = c.fetchall()
    
    db_totals = {}
    grand_total = 0
    grand_cars = 0
    grand_motors = 0
    
    for row in rows:
        cam_id, total, cars, motors = row
        db_totals[cam_id] = {
            "total": total or 0,
            "cars": cars or 0,
            "motors": motors or 0
        }
        grand_total += (total or 0)
        grand_cars += (cars or 0)
        grand_motors += (motors or 0)
        
    print(f"Found {len(db_totals)} cameras in DB with total {grand_total} vehicles.")
    
    # Update stats
    if "sources" not in stats:
        stats["sources"] = {}
        
    for cam_id, db_data in db_totals.items():
        if cam_id not in stats["sources"]:
            # Find name from config
            name = "Unknown Camera"
            for cam in config:
                if cam["id"] == cam_id:
                    name = cam["name"]
                    break
            
            stats["sources"][cam_id] = {
                "name": name,
                "current_count": 0,
                "current_class_counts": {"0": 0, "1": 0},
                "accumulated_count": 0,
                "accumulated_class_counts": {"0": 0, "1": 0},
                "history": []
            }
            
        # Update accumulated counts
        # We replace the accumulated count with the DB total because DB is the source of truth
        stats["sources"][cam_id]["accumulated_count"] = db_data["total"]
        if "accumulated_class_counts" not in stats["sources"][cam_id]:
            stats["sources"][cam_id]["accumulated_class_counts"] = {"0": 0, "1": 0}
        stats["sources"][cam_id]["accumulated_class_counts"]["0"] = db_data["cars"]
        stats["sources"][cam_id]["accumulated_class_counts"]["1"] = db_data["motors"]
        
    # Update global total
    stats["global_total"] = {
        "accumulated_count": grand_total,
        "cars": grand_cars,
        "motorcycles": grand_motors,
        "current_count": 0,
        "current_cars": 0,
        "current_motorcycles": 0
    }
    stats["last_update"] = time.time()
    
    # Save back to JSON
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Successfully synced stats. New Global Total: {grand_total}")
    conn.close()

if __name__ == "__main__":
    sync_db_to_json()
