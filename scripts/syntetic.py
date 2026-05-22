import json
import math
import os
import random
import sys
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app.database import clear_all_history, init_db, insert_history_batch

CONFIG_PATH = os.path.join(BASE_DIR, "data", "cctv_config.json")

def _traffic_factor(dt: datetime) -> float:
    hour = dt.hour + (dt.minute / 60.0)
    factor = 1.0
    factor += 2.8 * math.exp(-((hour - 7.5) ** 2) / 1.4)
    factor += 3.8 * math.exp(-((hour - 17.5) ** 2) / 2.2)
    if hour >= 22.0 or hour <= 5.0:
        factor *= 0.35

    dow = dt.weekday()
    if dow >= 5:
        factor *= 0.75

    factor *= random.uniform(0.85, 1.15)
    return max(0.1, factor)

def _sample_nonneg_int(lam: float, rng: random.Random) -> int:
    lam = max(0.0, lam)
    sigma = math.sqrt(lam) + 1.0
    return max(0, int(rng.gauss(lam, sigma)))

def rebuild_history_from_april(step_minutes: int = 5, seed: int = 20260401):
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cameras = json.load(f)

    now = datetime.now()
    start_dt = datetime(now.year, 4, 1, 0, 0, 0)
    end_dt = now
    step = timedelta(minutes=step_minutes)

    init_db()
    clear_all_history()

    global_rng = random.Random(seed)

    camera_profiles = {}
    for cam in cameras:
        cam_id = cam.get("id")
        if not cam_id:
            continue
        cam_rng = random.Random(f"{seed}:{cam_id}")
        camera_profiles[cam_id] = {
            "scale": cam_rng.uniform(0.7, 1.8),
            "car_ratio": cam_rng.uniform(0.5, 0.7),
            "rng": cam_rng,
        }

    total_expected = int(((end_dt - start_dt).total_seconds() // step.total_seconds()) + 1) * max(1, len(camera_profiles))
    print(f"Rebuilding DB history from {start_dt} to {end_dt} every {step_minutes} minutes...")
    print(f"Cameras: {len(camera_profiles)} | Target rows: ~{total_expected}")

    batch = []
    batch_size = 5000
    written = 0

    t = start_dt
    while t <= end_dt:
        tf = _traffic_factor(t)
        for cam_id, prof in camera_profiles.items():
            rng = prof["rng"]
            scale = prof["scale"]
            base = global_rng.uniform(18.0, 40.0)
            lam = base * scale * tf
            new_count = _sample_nonneg_int(lam, rng)

            occ_mult = 1.6 + (tf * 0.9) + rng.uniform(-0.2, 0.6)
            total_count = max(0, int(new_count * occ_mult))

            cr = min(0.85, max(0.15, prof["car_ratio"] + rng.uniform(-0.08, 0.08)))
            new_cars = int(new_count * cr)
            new_motors = max(0, new_count - new_cars)

            car_count = int(total_count * cr)
            motor_count = max(0, total_count - car_count)

            batch.append((
                cam_id,
                t.timestamp(),
                total_count,
                car_count,
                motor_count,
                new_count,
                new_cars,
                new_motors,
            ))

            if len(batch) >= batch_size:
                insert_history_batch(batch)
                written += len(batch)
                batch = []
                if written % 100000 == 0:
                    print(f"Inserted {written} rows...")
        t += step

    if batch:
        insert_history_batch(batch)
        written += len(batch)

    print(f"Done. Inserted {written} rows into traffic_history.")

if __name__ == "__main__":
    rebuild_history_from_april(step_minutes=5)
