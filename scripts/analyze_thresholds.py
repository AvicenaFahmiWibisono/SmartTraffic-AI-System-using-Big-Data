import sqlite3
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "traffic_data.db")

def analyze_traffic_distribution():
    if not os.path.exists(DB_PATH):
        print("Database not found!")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    print("Analyzing traffic distribution per camera to set realistic thresholds...")

    # 1. Get unique cameras
    c.execute("SELECT DISTINCT camera_id FROM traffic_history")
    cameras = [row['camera_id'] for row in c.fetchall()]

    thresholds = {}

    for cam_id in cameras:
        # Get hourly sums for this camera
        # Group by Date + Hour
        query = """
            SELECT 
                strftime('%Y-%m-%d %H', datetime(timestamp, 'unixepoch', 'localtime')) as hour_str,
                SUM(new_count) as hourly_count
            FROM traffic_history
            WHERE camera_id = ?
            GROUP BY hour_str
        """
        c.execute(query, (cam_id,))
        rows = c.fetchall()
        
        counts = [row['hourly_count'] for row in rows]
        
        if not counts:
            continue
            
        # Calculate percentiles
        p50 = np.percentile(counts, 50) # Median
        p75 = np.percentile(counts, 75) # Padat
        p90 = np.percentile(counts, 90) # Macet
        max_val = np.max(counts)
        
        thresholds[cam_id] = {
            "p50": int(p50),
            "p75": int(p75),
            "p90": int(p90),
            "max": int(max_val)
        }
        
        print(f"Cam {cam_id[:8]}... : Median={int(p50)}, Padat(75%)={int(p75)}, Macet(90%)={int(p90)}, Max={int(max_val)}")

    # Save to JSON
    json_path = os.path.join(DATA_DIR, "camera_thresholds.json")
    import json
    with open(json_path, 'w') as f:
        json.dump(thresholds, f, indent=4)
    print(f"Thresholds saved to {json_path}")

    conn.close()
    return thresholds

if __name__ == "__main__":
    analyze_traffic_distribution()
