import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "traffic_data.db")

def get_total():
    try:
        if not os.path.exists(DB_PATH):
            print(f"Database not found at {DB_PATH}")
            return
            
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get total count
        c.execute("SELECT SUM(new_count) FROM traffic_history")
        result = c.fetchone()
        total = result[0] if result and result[0] is not None else 0
        
        print(f"Total in DB: {total}")
        
        # Get count per camera
        c.execute("SELECT camera_id, SUM(new_count) FROM traffic_history GROUP BY camera_id")
        rows = c.fetchall()
        print("\nPer Camera:")
        for row in rows:
            print(f"{row[0]}: {row[1]}")
            
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_total()
