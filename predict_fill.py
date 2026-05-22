"""
Fill total_count, car_count, motorcycle_count with predicted values
for rows where total_count = 0 (no real detection data).

Uses DOW x Hour average from historical data per camera.
"""
import psycopg2
import psycopg2.extras
import os
import time
import datetime

PG_HOST = "localhost"
PG_PORT = 5432
PG_NAME = "smarttraffic"
PG_USER = "postgres"
PG_PASS = os.environ.get("DB_PASSWORD", "")
TZ = "Asia/Jakarta"


def main():
    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_NAME, user=PG_USER, password=PG_PASS)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # 1. Count how many rows need filling
    cur.execute("SELECT COUNT(*) AS cnt FROM traffic_history WHERE total_count = 0")
    zero_count = cur.fetchone()["cnt"]
    print(f"[INFO] Rows with total_count = 0: {zero_count:,}")

    if zero_count == 0:
        print("[DONE] Nothing to fill.")
        conn.close()
        return

    # 2. Build DOW x Hour averages per camera from rows that DO have data
    print("[1/3] Building DOW x Hour averages per camera...")
    cur.execute(f"""
        SELECT 
            camera_id,
            EXTRACT(DOW FROM to_timestamp(timestamp) AT TIME ZONE '{TZ}')::int AS dow,
            EXTRACT(HOUR FROM to_timestamp(timestamp) AT TIME ZONE '{TZ}')::int AS hour,
            AVG(total_count)::int AS avg_total,
            AVG(car_count)::int AS avg_cars,
            AVG(motorcycle_count)::int AS avg_motors
        FROM traffic_history
        WHERE total_count > 0
        GROUP BY camera_id, dow, hour
    """)
    averages = {}
    for row in cur.fetchall():
        key = (row["camera_id"], row["dow"], row["hour"])
        averages[key] = {
            "total": max(1, row["avg_total"]),
            "cars": max(0, row["avg_cars"]),
            "motors": max(0, row["avg_motors"]),
        }
    print(f"       Built {len(averages):,} (camera, dow, hour) averages")

    # Global fallback if a specific camera/dow/hour combo has no history
    cur.execute(f"""
        SELECT 
            EXTRACT(DOW FROM to_timestamp(timestamp) AT TIME ZONE '{TZ}')::int AS dow,
            EXTRACT(HOUR FROM to_timestamp(timestamp) AT TIME ZONE '{TZ}')::int AS hour,
            AVG(total_count)::int AS avg_total,
            AVG(car_count)::int AS avg_cars,
            AVG(motorcycle_count)::int AS avg_motors
        FROM traffic_history
        WHERE total_count > 0
        GROUP BY dow, hour
    """)
    global_avg = {}
    for row in cur.fetchall():
        global_avg[(row["dow"], row["hour"])] = {
            "total": max(1, row["avg_total"]),
            "cars": max(0, row["avg_cars"]),
            "motors": max(0, row["avg_motors"]),
        }

    # 3. Fetch all zero rows
    print("[2/3] Fetching rows to fill...")
    cur.execute(f"""
        SELECT id, camera_id, timestamp,
               EXTRACT(DOW FROM to_timestamp(timestamp) AT TIME ZONE '{TZ}')::int AS dow,
               EXTRACT(HOUR FROM to_timestamp(timestamp) AT TIME ZONE '{TZ}')::int AS hour
        FROM traffic_history
        WHERE total_count = 0
        ORDER BY id
    """)
    rows = cur.fetchall()
    print(f"       {len(rows):,} rows to update")

    # 4. Batch update
    print("[3/3] Updating with predicted values...")
    update_sql = "UPDATE traffic_history SET total_count = %s, car_count = %s, motorcycle_count = %s WHERE id = %s"
    batch = []
    updated = 0
    start = time.time()

    for row in rows:
        key = (row["camera_id"], row["dow"], row["hour"])
        avg = averages.get(key) or global_avg.get((row["dow"], row["hour"])) or {"total": 8, "cars": 5, "motors": 3}
        batch.append((avg["total"], avg["cars"], avg["motors"], row["id"]))

        if len(batch) >= 5000:
            psycopg2.extras.execute_batch(cur, update_sql, batch, page_size=1000)
            conn.commit()
            updated += len(batch)
            batch = []
            pct = updated / len(rows) * 100
            print(f"  [{pct:5.1f}%] {updated:,} / {len(rows):,} rows", end="\r")

    if batch:
        psycopg2.extras.execute_batch(cur, update_sql, batch, page_size=1000)
        conn.commit()
        updated += len(batch)

    elapsed = time.time() - start
    print(f"\n[DONE] Updated {updated:,} rows in {elapsed:.1f}s with predicted values.")
    print("       Prediction method: DOW x Hour average per camera (fallback: global average)")

    conn.close()


if __name__ == "__main__":
    main()
