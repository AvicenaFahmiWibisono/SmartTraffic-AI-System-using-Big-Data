import csv
import datetime
import json
import os
import time
import uuid
import shutil
import random
import math
from collections import deque
from app.config import CONFIG_FILE, STATS_FILE, HISTORY_MAX_LEN
import app.globals as g

from app.database import insert_history_batch, clear_all_history

def get_camera_profile(name):
    """
    Determines the traffic profile based on the camera location name.
    Returns: 'EXTREME', 'HEAVY', 'ARTERIAL', 'RESIDENTIAL', or 'DEFAULT'
    """
    name = name.lower()
    if any(k in name for k in ['gedebage', 'soekarno hatta', 'kiaracondong', 'samsat', 'binong']):
        return 'EXTREME'
    elif any(k in name for k in ['dago', 'dipatiukur', 'gasibu', 'cihampelas', 'braga', 'asia afrika', 'merdeka', 'surapati']):
        return 'HEAVY'
    elif any(k in name for k in ['fly over', 'flyover', 'pasupati', 'pasteur', 'sudirman', 'peta', 'laswi', 'pelajar pejuang']):
        return 'ARTERIAL'
    elif any(k in name for k in ['waas', 'batununggal', 'sukahaji', 'cijerah', 'sariningsih', 'komplek']):
        return 'RESIDENTIAL'
    return 'DEFAULT'

def generate_varied_history(hours=24):
    """
    Generates varied synthetic history for all cameras based on their location profile.
    """
    # Ensure all configured cameras exist in global stats
    if not g.CCTV_SOURCES:
        g.CCTV_SOURCES = load_config()

    # Clear existing history to avoid duplicates and ensure clean slate
    try:
        clear_all_history()
    except Exception as e:
        print(f"Error clearing history: {e}")

    current_source_ids = {s["id"] for s in g.CCTV_SOURCES}
    for s in g.CCTV_SOURCES:
        if s["id"] not in g.global_stats:
             g.global_stats[s["id"]] = {
                "name": s["name"],
                "current_count": 0,
                "current_class_counts": {"0": 0, "1": 0},
                "accumulated_count": 0,
                "accumulated_class_counts": {"0": 0, "1": 0},
                "history": deque(maxlen=HISTORY_MAX_LEN)
            }

    now = time.time()
    start_ts = now - (hours * 3600)
    
    # 60s step for 7-day history (Manageable size: ~10k points)
    step = 60
    
    timestamps = []
    t = start_ts
    while t <= now:
        timestamps.append(t)
        t += step
        
    for source_id, stats in g.global_stats.items():
        if source_id not in current_source_ids:
            continue

        name = stats.get("name", "")
        profile = get_camera_profile(name)
        
        # Profile Configuration
        if profile == 'EXTREME':
            base_density = random.randint(40, 60)
            peak_boost = random.randint(80, 120)
            morning_peak_hour = 7.0  # Early rush
            evening_peak_hour = 17.5 # Late rush
            peak_width = 2.5         # Wide peak (long jams)
        elif profile == 'HEAVY':
            base_density = random.randint(20, 35)
            peak_boost = random.randint(40, 60)
            morning_peak_hour = 7.5
            evening_peak_hour = 17.0
            peak_width = 2.0
        elif profile == 'ARTERIAL':
            base_density = random.randint(30, 50) # Steady flow
            peak_boost = random.randint(30, 50)   # Less dramatic spikes
            morning_peak_hour = 8.0
            evening_peak_hour = 18.0
            peak_width = 3.0
        elif profile == 'RESIDENTIAL':
            base_density = random.randint(2, 8)   # Quiet usually
            peak_boost = random.randint(20, 40)   # Sharp school/work runs
            morning_peak_hour = 6.5
            evening_peak_hour = 18.0
            peak_width = 1.0         # Sharp peaks
        else: # DEFAULT
            base_density = random.randint(10, 20)
            peak_boost = random.randint(20, 30)
            morning_peak_hour = 7.5
            evening_peak_hour = 17.0
            peak_width = 1.5
            
        # Add slight randomness to hours so not everyone is identical
        morning_peak_hour += random.uniform(-0.3, 0.3)
        evening_peak_hour += random.uniform(-0.3, 0.3)
        
        # Reset stats
        stats["history"] = deque(maxlen=HISTORY_MAX_LEN)
        stats["accumulated_count"] = 0
        stats["accumulated_class_counts"] = {"0": 0, "1": 0}
        
        history_batch = []
        
        for ts in timestamps:
            dt = datetime.datetime.fromtimestamp(ts)
            hour_float = dt.hour + dt.minute / 60.0
            
            # Traffic Curve
            m_peak = peak_boost * math.exp(-((hour_float - morning_peak_hour)**2) / peak_width)
            e_peak = (peak_boost * 1.2) * math.exp(-((hour_float - evening_peak_hour)**2) / peak_width) # Evening usually worse
            
            flow = base_density + m_peak + e_peak
            
            # Noise
            actual_density = int(flow * (1.0 + random.uniform(-0.15, 0.15)))
            if actual_density < 0: actual_density = 0
            
            # Class Ratio
            motor_ratio = 0.65 if profile in ['RESIDENTIAL', 'EXTREME'] else 0.55
            motor_ratio += random.uniform(-0.05, 0.05)
            
            motors = int(actual_density * motor_ratio)
            cars = actual_density - motors
            
            # New Count simulation (Flow Rate)
            # Higher density usually means higher flow.
            # Flow factor: % of cars moving out of frame per step (15s)
            flow_factor = 0.15 # 15% move
            if profile == 'EXTREME': flow_factor = 0.25
            elif profile == 'HEAVY': flow_factor = 0.20
            elif profile == 'RESIDENTIAL': flow_factor = 0.05
            
            new_count = int(actual_density * flow_factor)
            
            # Add randomness and minimum movement
            new_count = int(new_count * random.uniform(0.8, 1.2))
            if actual_density > 5 and new_count == 0: new_count = 1
            
            new_motors = int(new_count * motor_ratio)
            new_cars = new_count - new_motors
            
            item = {
                "ts": ts,
                "count": actual_density,
                "cars": cars,
                "motors": motors,
                "new_count": new_count,
                "new_cars": new_cars,
                "new_motors": new_motors
            }
            history_batch.append(item)
            
            stats["accumulated_count"] += new_count
            stats["accumulated_class_counts"]["0"] += new_cars
            stats["accumulated_class_counts"]["1"] += new_motors
            
        stats["history"].extend(history_batch)
        
        if history_batch:
            # Sync to SQLite for Prediction API
            db_records = []
            for item in history_batch:
                db_records.append((
                    source_id,
                    item["ts"],
                    item["count"],
                    item["cars"],
                    item["motors"],
                    item["new_count"],
                    item["new_cars"],
                    item["new_motors"]
                ))
            try:
                insert_history_batch(db_records)
            except Exception as e:
                print(f"[ERROR] Failed to insert history batch for {source_id}: {e}")

            last = history_batch[-1]
            stats["current_count"] = last["count"]
            stats["current_class_counts"] = {"0": last["cars"], "1": last["motors"]}
            
    save_stats()
    return {"status": "success", "message": f"Generated location-aware history for {len(g.global_stats)} cameras"}

def backfill_camera_history(new_id, template_id, hours=24, generate_datalake=False, start_date=None):
    now = time.time()
    if start_date:
        try:
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            start_ts = start_dt.timestamp()
        except ValueError:
            return {"status": "error", "message": "Invalid date format. Use YYYY-MM-DD"}
    else:
        start_ts = now - float(hours) * 3600.0

    if template_id not in g.global_stats:
        return {"status": "error", "message": "Template source not found"}

    template_stats = g.global_stats[template_id]
    template_history = list(template_stats.get("history", []))
    if not template_history:
        return {"status": "error", "message": "Template has no history data"}

    items_to_add = []
    if start_date:
        last_ts = template_history[-1]["ts"]
        pattern_start = last_ts - 86400
        pattern_items = [h for h in template_history if h["ts"] > pattern_start]
        if not pattern_items:
            pattern_items = template_history

        daily_pattern = []
        for item in pattern_items:
            dt = datetime.datetime.fromtimestamp(item["ts"])
            secs = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
            daily_pattern.append((secs, item))
        daily_pattern.sort(key=lambda x: x[0])

        loop_date = datetime.datetime.fromtimestamp(start_ts).date()
        end_date = datetime.datetime.fromtimestamp(now).date()
        while loop_date <= end_date:
            day_start = datetime.datetime.combine(loop_date, datetime.time.min).timestamp()
            for secs, item in daily_pattern:
                new_ts = day_start + secs
                if new_ts < start_ts:
                    continue
                if new_ts > now:
                    break
                new_item = item.copy()
                new_item["ts"] = new_ts
                items_to_add.append(new_item)
            loop_date += datetime.timedelta(days=1)
    else:
        items_to_add = [h for h in template_history if h.get("ts", 0) >= start_ts]

    if new_id not in g.global_stats:
        name = next((s["name"] for s in g.CCTV_SOURCES if s["id"] == new_id), "Unknown")
        g.global_stats[new_id] = {
            "name": name,
            "current_count": 0,
            "current_class_counts": {"0": 0, "1": 0},
            "accumulated_count": 0,
            "accumulated_class_counts": {"0": 0, "1": 0},
            "history": deque(maxlen=HISTORY_MAX_LEN)
        }

    dst = g.global_stats[new_id]
    dst["history"] = deque(maxlen=HISTORY_MAX_LEN)
    dst["accumulated_count"] = 0
    dst["accumulated_class_counts"] = {"0": 0, "1": 0}
    for item in items_to_add:
        dst["history"].append(item)
        dst["accumulated_count"] += item.get("new_count", 0)
        dst["accumulated_class_counts"]["0"] += item.get("new_cars", 0)
        dst["accumulated_class_counts"]["1"] += item.get("new_motors", 0)

    if items_to_add:
        # Sync to SQLite
        db_records = []
        for item in items_to_add:
            db_records.append((
                new_id,
                item["ts"],
                item.get("count", 0),
                item.get("cars", 0),
                item.get("motors", 0),
                item.get("new_count", 0),
                item.get("new_cars", 0),
                item.get("new_motors", 0)
            ))
        try:
            insert_history_batch(db_records)
        except Exception as e:
            print(f"[ERROR] Failed to insert backfill batch to DB: {e}")

        last = items_to_add[-1]
        dst["current_count"] = last.get("count", 0)
        dst["current_class_counts"] = {
            "0": last.get("new_cars", 0),
            "1": last.get("new_motors", 0)
        }

    save_stats()

    if generate_datalake and items_to_add:
        from collections import defaultdict
        items_by_date = defaultdict(list)
        for item in items_to_add:
            ts = item.get("ts", now)
            dt = datetime.datetime.fromtimestamp(ts)
            date_key = (dt.year, dt.month, dt.day)
            items_by_date[date_key].append(item)

        name = dst.get("name", new_id)
        for (year, month, day), day_items in items_by_date.items():
            base = os.path.join("/var/www/vehicle-counter/data_lake/raw", str(year), f"{month:02d}", f"{day:02d}")
            os.makedirs(base, exist_ok=True)
            fp = os.path.join(base, f"traffic_log_{new_id}.csv")
            file_exists = os.path.isfile(fp)
            with open(fp, "a", newline="") as f:
                w = csv.writer(f)
                if not file_exists:
                    w.writerow(["timestamp", "source_id", "source_name", "class_id", "confidence", "bbox"])
                for item in day_items:
                    ts = item.get("ts")
                    for _ in range(item.get("new_cars", 0)):
                        w.writerow([ts, new_id, name, "car", "0.50", "[0,0,0,0]"])
                    for _ in range(item.get("new_motors", 0)):
                        w.writerow([ts, new_id, name, "motorcycle", "0.50", "[0,0,0,0]"])

    return {"status": "success", "message": "Backfill completed"}

def get_datalake_stats(date_str=None):
    """
    Read aggregated stats from Data Lake for a specific date (YYYY-MM-DD).
    If date_str is None, defaults to today.
    """
    if date_str is None:
        now = datetime.datetime.now()
    else:
        try:
            now = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}
            
    # Path to Data Lake partition
    partition_path = os.path.join(
        "/var/www/vehicle-counter/data_lake/raw", 
        str(now.year), 
        f"{now.month:02d}", 
        f"{now.day:02d}"
    )
    
    if not os.path.exists(partition_path):
        return {"total_vehicles": 0, "by_camera": {}, "date": now.strftime("%Y-%m-%d"), "message": "No data found for this date"}
        
    stats = {
        "date": now.strftime("%Y-%m-%d"),
        "total_vehicles": 0,
        "by_camera": {}
    }
    
    try:
        # Iterate over all CSV files in the partition
        for filename in os.listdir(partition_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(partition_path, filename)
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        stats["total_vehicles"] += 1
                        
                        src_name = row.get("source_name", "Unknown")
                        cls_id = row.get("class_id", "unknown")
                        
                        if src_name not in stats["by_camera"]:
                            stats["by_camera"][src_name] = {"total": 0, "car": 0, "motorcycle": 0}
                            
                        stats["by_camera"][src_name]["total"] += 1
                        if cls_id == "car":
                            stats["by_camera"][src_name]["car"] += 1
                        elif cls_id == "motorcycle":
                            stats["by_camera"][src_name]["motorcycle"] += 1
                            
        return stats
    except Exception as e:
        print(f"[ERROR] Failed to read Data Lake: {e}")
        return {"error": str(e)}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        example_path = os.path.join(os.path.dirname(CONFIG_FILE), "cctv_config.example.json")
        if os.path.exists(example_path):
            try:
                with open(example_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return []

def save_config(sources):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(sources, f, indent=4)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save config: {e}")
        return False

def load_stats():
    # Try loading from main file
    files_to_try = [STATS_FILE, STATS_FILE + ".bak"]
    
    for file_path in files_to_try:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Check if it's nested structure (v2)
                stats = {}
                if "sources" in data:
                    stats = data["sources"]
                else:
                    # Legacy flat structure or unexpected format
                    # If it has UUID keys, it's legacy sources
                    # Filter out non-dict or special keys
                    stats = {k: v for k, v in data.items() if isinstance(v, dict) and "id" not in v} 
                    # Actually legacy was just {uuid: data}
                    if not stats and data: # fallback
                         stats = data

                # Convert history lists back to deque
                for src_id, src_data in stats.items():
                    if isinstance(src_data, dict) and "history" in src_data:
                        src_data["history"] = deque(src_data["history"], maxlen=HISTORY_MAX_LEN)
                
                print(f"[INFO] Successfully loaded stats from {file_path}")
                return stats
        except Exception as e:
            print(f"[ERROR] Failed to load stats from {file_path}: {e}")
            # Continue to try backup if this was main file
            
    example_path = os.path.join(os.path.dirname(STATS_FILE), "traffic_stats.example.json")
    if os.path.exists(example_path):
        try:
            with open(example_path, 'r') as f:
                data = json.load(f)
            stats = data.get("sources", {}) if isinstance(data, dict) else {}
            for src_id, src_data in stats.items():
                if isinstance(src_data, dict) and "history" in src_data:
                    src_data["history"] = deque(src_data["history"], maxlen=HISTORY_MAX_LEN)
            return stats
        except Exception:
            return {}
    return {}

def save_stats():
    try:
        # Create a copy for saving, converting deque to list
        sources_data = {}
        global_accumulated = 0
        global_cars = 0
        global_motors = 0
        
        global_current = 0
        global_current_cars = 0
        global_current_motors = 0
        
        all_history = []

        for k, v in g.global_stats.items():
            sources_data[k] = v.copy()
            if "history" in v and isinstance(v["history"], deque):
                # Convert to list for JSON serialization
                hist_list = list(v["history"])
                sources_data[k]["history"] = hist_list
                all_history.extend(hist_list)
            
            # Aggregate globals
            global_accumulated += v.get("accumulated_count", 0)
            global_cars += v.get("accumulated_class_counts", {}).get("0", 0)
            global_motors += v.get("accumulated_class_counts", {}).get("1", 0)
            
            # Aggregate current
            global_current += v.get("current_count", 0)
            global_current_cars += v.get("current_class_counts", {}).get("0", 0)
            global_current_motors += v.get("current_class_counts", {}).get("1", 0)
        
        # Calculate Global Window Stats
        window_stats = calculate_window_stats(all_history)

        # Construct final structure
        final_data = {
            "sources": sources_data,
            "global_total": {
                "accumulated_count": global_accumulated,
                "cars": global_cars,
                "motorcycles": global_motors,
                "current_count": global_current,
                "current_cars": global_current_cars,
                "current_motorcycles": global_current_motors
            },
            "window_stats": window_stats,
            "last_update": time.time()
        }

        try:
            if os.path.exists(STATS_FILE):
                with open(STATS_FILE, 'r') as f:
                    existing_text = f.read()
                existing_data, _ = json.JSONDecoder().raw_decode(existing_text)
                if isinstance(existing_data, dict):
                    existing_global = existing_data.get("global_total")
                    if isinstance(existing_global, dict):
                        existing_acc = existing_global.get("accumulated_count")
                        if isinstance(existing_acc, (int, float)) and existing_acc > 0 and global_accumulated < existing_acc:
                            return
        except Exception:
            pass
        
        # Atomic Write: Write to temp -> Move to final
        temp_file = f"{STATS_FILE}.{os.getpid()}.tmp"
        backup_file = STATS_FILE + ".bak"
        
        # Write to temp file first
        with open(temp_file, 'w') as f:
            json.dump(final_data, f, indent=4)
            
        # If write successful, backup old file then replace
        if os.path.exists(STATS_FILE):
            try:
                shutil.copy2(STATS_FILE, backup_file)
            except Exception as e:
                print(f"[WARN] Failed to create backup: {e}")
                
        shutil.move(temp_file, STATS_FILE)
        
    except Exception as e:
        print(f"[ERROR] Failed to save stats: {e}")

def sync_stats_with_config():
    valid_ids = {src["id"] for src in g.CCTV_SOURCES}
    to_remove = [k for k in g.global_stats.keys() if k not in valid_ids]
    
    if to_remove:
        print(f"[INFO] Cleaning up {len(to_remove)} zombie stats entries.")
        for k in to_remove:
            del g.global_stats[k]
        save_stats()

def calculate_window_stats(history):
    now = time.time()
    windows = {
        "10s": 10,
        "30m": 1800,
        "1h": 3600,
        "5h": 18000,
        "24h": 86400
    }
    
    results = {}
    
    hist_list = list(history)
    
    for label, seconds in windows.items():
        # Filter items within window
        cutoff = now - seconds
        relevant_items = [item for item in hist_list if item["ts"] >= cutoff]
        
        count = len(relevant_items)
        if count > 0:
            # Calculate Total Volume (Flux) - Sum of new vehicles
            # Use .get() for backward compatibility with old history data
            total_volume = sum(item.get("new_count", 0) for item in relevant_items)
            total_cars = sum(item.get("new_cars", 0) for item in relevant_items)
            total_motors = sum(item.get("new_motors", 0) for item in relevant_items)
            
            # Also calculate Average Density for reference (optional, but keeping logic)
            avg_density = round(sum(item["count"] for item in relevant_items) / count)
        else:
            total_volume = 0
            total_cars = 0
            total_motors = 0
            avg_density = 0
            
        results[label] = {
            "total_volume": total_volume,
            "cars": total_cars,
            "motors": total_motors,
            "avg_density": avg_density
        }
        
    return results

def get_history_series(history, period="30m", start_ts=None):
    now = time.time()
    
    # Custom 24h view for a specific day
    if period == "custom" and start_ts:
        try:
            # Align start_ts to 00:00 of that day
            ts_float = float(start_ts)
            t = time.localtime(ts_float)
            start_time = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
            
            duration = 86400 # 24 hours
            bucket_size = 3600 # 1 hour buckets
            time_format = "%H:%M"
            
            # Create buckets
            buckets = []
            num_buckets = 24
            for i in range(num_buckets):
                bucket_start = start_time + (i * bucket_size)
                buckets.append({
                    "ts": bucket_start,
                    "label": time.strftime(time_format, time.localtime(bucket_start)),
                    "count": 0,
                    "cars": 0,
                    "motors": 0
                })
                
            # Fill buckets
            end_time = start_time + duration
            hist_list = list(history)
            for item in hist_list:
                ts = item["ts"]
                if ts < start_time or ts >= end_time:
                    continue
                    
                idx = int((ts - start_time) / bucket_size)
                if 0 <= idx < num_buckets:
                    buckets[idx]["count"] += item.get("new_count", 0)
                    buckets[idx]["cars"] += item.get("new_cars", 0)
                    buckets[idx]["motors"] += item.get("new_motors", 0)
            
            return buckets
            
        except ValueError:
            pass # Fallback to standard logic if invalid start_ts

    # Define period duration and bucket size
    if period == "30d":
        duration = 2592000 # 30 days
        bucket_size = 86400 # 24 hour buckets (1 point per day)
        time_format = "%a, %d %b" # e.g. Mon, 25 Jan
    elif period == "7d":
        duration = 604800 # 7 days
        bucket_size = 86400 # 24 hour buckets (1 point per day)
        time_format = "%A, %d %b" # e.g. Monday, 25 Jan
    elif period == "24h":
        duration = 86400
        bucket_size = 3600 # 1 hour buckets (24 points)
        time_format = "%H:%M"
    elif period == "12h":
        duration = 43200
        bucket_size = 1800 # 30 min buckets (24 points)
        time_format = "%H:%M"
    elif period == "6h":
        duration = 21600
        bucket_size = 900 # 15 min buckets (24 points)
        time_format = "%H:%M"
    elif period == "1h":
        duration = 3600
        bucket_size = 120 # 2 min buckets (30 points)
        time_format = "%H:%M"
    elif period == "30m":
        duration = 1800
        bucket_size = 60 # 1 minute buckets (30 points)
        time_format = "%H:%M"
    else:
        # Default fallback (30m)
        duration = 1800
        bucket_size = 60
        time_format = "%H:%M"
    
    start_time = now - duration
    
    # Special handling for 24h: Align to today 00:00 - 24:00
    if period == "24h":
        t = time.localtime(now)
        start_time = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
        # Ensure we cover full 24h from start of day
        duration = 86400 
    
    # Initialize buckets
    # Align start time to the nearest bucket boundary for cleaner charts
    # e.g. if bucket is 1 hour, start at XX:00:00
    # But for sliding window, we might just want "last 24h"
    
    num_buckets = int(duration / bucket_size)
    buckets = []
    for i in range(num_buckets):
        bucket_start = start_time + (i * bucket_size)
        buckets.append({
            "ts": bucket_start,
            "label": time.strftime(time_format, time.localtime(bucket_start)),
            "count": 0,
            "cars": 0,
            "motors": 0
        })
    
    # Fill buckets
    hist_list = list(history)
    for item in hist_list:
        ts = item["ts"]
        if ts < start_time:
            continue
            
        # Find bucket index
        idx = int((ts - start_time) / bucket_size)
        if 0 <= idx < num_buckets:
            buckets[idx]["count"] += item.get("new_count", 0)
            buckets[idx]["cars"] += item.get("new_cars", 0)
            buckets[idx]["motors"] += item.get("new_motors", 0)
            
    return buckets
