import sqlite3
import csv
import os
import json
import datetime

# Konfigurasi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data/traffic_data.db')
CONFIG_PATH = os.path.join(BASE_DIR, 'data/cctv_config.json')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data/exports/traffic_data_limited.csv')

def load_camera_names():
    """Load camera configuration and return a dictionary mapping ID to Name."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            cameras = json.load(f)
            # Create a dictionary: { "uuid": "Location Name" }
            return {cam["id"]: cam["name"] for cam in cameras}
    except Exception as e:
        print(f"Warning: Could not load camera config: {e}")
        return {}

def export_limited_data(limit=100000):
    """
    Export N data terakhir dari database SQLite ke CSV, 
    mengganti Camera ID dengan Nama Lokasi.
    """
    print(f"Loading camera names from: {CONFIG_PATH}")
    camera_map = load_camera_names()
    
    print(f"Connecting to database: {DB_PATH}")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Query untuk mengambil data terakhir dengan LIMIT
        query = f"""
            SELECT 
                datetime(timestamp, 'unixepoch', 'localtime') as time_str,
                camera_id,
                total_count,
                car_count,
                motorcycle_count
            FROM traffic_history
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        print(f"Executing query with LIMIT {limit}...")
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        
        if not rows:
            print("No data found in database.")
            return

        # Tulis ke CSV
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header baru dengan Location Name
            writer.writerow(['Timestamp', 'Location Name', 'Total Count', 'Car Count', 'Motorcycle Count'])
            
            count = 0
            for row in rows:
                ts, cam_id, total, cars, motors = row
                # Ganti ID dengan Nama, atau gunakan ID jika nama tidak ditemukan
                location_name = camera_map.get(cam_id, cam_id)
                writer.writerow([ts, location_name, total, cars, motors])
                count += 1
            
        print(f"✅ Berhasil export {count} baris data ke '{OUTPUT_FILE}'")
        print(f"   (Camera ID telah diganti dengan Nama Lokasi)")
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Menggunakan limit 100,000 sesuai permintaan
    export_limited_data(100000)
