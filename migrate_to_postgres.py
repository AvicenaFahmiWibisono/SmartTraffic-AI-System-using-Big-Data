"""
Migrate all data from SQLite (data/traffic_data.db) to PostgreSQL.
Run this ONCE before switching database.py to PostgreSQL.

Usage:
    python migrate_to_postgres.py
"""
import sqlite3
import os
import sys
import time

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

# ─── Config ───────────────────────────────────────────────────────────────────
SQLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "traffic_data.db")
PG_HOST = os.environ.get("DB_HOST", "localhost")
PG_PORT = int(os.environ.get("DB_PORT", "5432"))
PG_NAME = os.environ.get("DB_NAME", "smarttraffic")
PG_USER = os.environ.get("DB_USER", "postgres")
PG_PASS = os.environ.get("DB_PASSWORD", "")

BATCH_SIZE = 5000  # rows per INSERT batch


def pg_connect():
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_NAME, user=PG_USER, password=PG_PASS)


def create_tables(pg):
    cur = pg.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS traffic_history (
            id BIGSERIAL PRIMARY KEY,
            camera_id TEXT NOT NULL,
            timestamp DOUBLE PRECISION NOT NULL,
            total_count INTEGER DEFAULT 0,
            car_count INTEGER DEFAULT 0,
            motorcycle_count INTEGER DEFAULT 0,
            new_count INTEGER DEFAULT 0,
            new_cars INTEGER DEFAULT 0,
            new_motors INTEGER DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_camera_timestamp
        ON traffic_history (camera_id, timestamp)
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_profile (
            session_id TEXT PRIMARY KEY,
            updated_ts DOUBLE PRECISION NOT NULL,
            last_intent TEXT,
            last_camera_id TEXT,
            last_camera_name TEXT,
            last_destination TEXT,
            prefs_json TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            ts DOUBLE PRECISION NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            page TEXT,
            meta_json TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_ts ON chat_messages (session_id, ts)")
    pg.commit()
    print("[OK] PostgreSQL tables created.")


def count_rows(sqlite_conn, table):
    c = sqlite_conn.cursor()
    c.execute(f"SELECT COUNT(*) FROM {table}")
    return c.fetchone()[0]


def migrate_traffic_history(sqlite_conn, pg):
    total = count_rows(sqlite_conn, "traffic_history")
    print(f"\n[INFO] Migrating traffic_history: {total:,} rows...")

    c = sqlite_conn.cursor()
    c.execute("""
        SELECT camera_id, timestamp, total_count, car_count, motorcycle_count, new_count, new_cars, new_motors
        FROM traffic_history
        ORDER BY id ASC
    """)

    pg_cur = pg.cursor()
    insert_sql = """
        INSERT INTO traffic_history (camera_id, timestamp, total_count, car_count, motorcycle_count, new_count, new_cars, new_motors)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    migrated = 0
    batch = []
    start = time.time()

    while True:
        rows = c.fetchmany(BATCH_SIZE)
        if not rows:
            break
        for row in rows:
            batch.append((row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))

        psycopg2.extras.execute_batch(pg_cur, insert_sql, batch, page_size=1000)
        pg.commit()
        migrated += len(batch)
        batch = []

        elapsed = time.time() - start
        rate = migrated / max(elapsed, 0.001)
        pct = (migrated / total * 100) if total > 0 else 100
        print(f"  [{pct:5.1f}%] {migrated:>10,} / {total:,} rows  ({rate:,.0f} rows/sec)", end="\r")

    print(f"\n[OK] traffic_history: {migrated:,} rows migrated in {time.time() - start:.1f}s")


def migrate_chat_profile(sqlite_conn, pg):
    total = count_rows(sqlite_conn, "chat_profile")
    if total == 0:
        print("[SKIP] chat_profile: 0 rows")
        return

    print(f"[INFO] Migrating chat_profile: {total} rows...")
    c = sqlite_conn.cursor()
    c.execute("SELECT session_id, updated_ts, last_intent, last_camera_id, last_camera_name, last_destination, prefs_json FROM chat_profile")
    rows = c.fetchall()

    pg_cur = pg.cursor()
    insert_sql = """
        INSERT INTO chat_profile (session_id, updated_ts, last_intent, last_camera_id, last_camera_name, last_destination, prefs_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (session_id) DO NOTHING
    """
    batch = [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows]
    psycopg2.extras.execute_batch(pg_cur, insert_sql, batch, page_size=500)
    pg.commit()
    print(f"[OK] chat_profile: {len(batch)} rows migrated")


def migrate_chat_messages(sqlite_conn, pg):
    total = count_rows(sqlite_conn, "chat_messages")
    if total == 0:
        print("[SKIP] chat_messages: 0 rows")
        return

    print(f"[INFO] Migrating chat_messages: {total} rows...")
    c = sqlite_conn.cursor()
    c.execute("SELECT session_id, ts, role, content, page, meta_json FROM chat_messages ORDER BY id ASC")
    rows = c.fetchall()

    pg_cur = pg.cursor()
    insert_sql = """
        INSERT INTO chat_messages (session_id, ts, role, content, page, meta_json)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    batch = [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]
    psycopg2.extras.execute_batch(pg_cur, insert_sql, batch, page_size=500)
    pg.commit()
    print(f"[OK] chat_messages: {len(batch)} rows migrated")


def verify(sqlite_conn, pg):
    """Quick sanity check: compare row counts."""
    print("\n[VERIFY] Comparing row counts...")
    pg_cur = pg.cursor()

    for table in ["traffic_history", "chat_profile", "chat_messages"]:
        sqlite_count = count_rows(sqlite_conn, table)
        pg_cur.execute(f"SELECT COUNT(*) FROM {table}")
        pg_count = pg_cur.fetchone()[0]
        status = "OK" if pg_count >= sqlite_count else "MISMATCH"
        print(f"  {table}: SQLite={sqlite_count:,}  PostgreSQL={pg_count:,}  [{status}]")


def main():
    if not os.path.exists(SQLITE_PATH):
        print(f"ERROR: SQLite file not found: {SQLITE_PATH}")
        sys.exit(1)

    print(f"SQLite: {SQLITE_PATH}")
    print(f"PostgreSQL: {PG_USER}@{PG_HOST}:{PG_PORT}/{PG_NAME}")
    print("=" * 60)

    # Connect SQLite
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = None  # tuple mode for speed

    # Connect PostgreSQL
    try:
        pg = pg_connect()
    except Exception as e:
        print(f"ERROR: Cannot connect to PostgreSQL: {e}")
        sys.exit(1)

    # Create tables
    create_tables(pg)

    # Check if already migrated
    pg_cur = pg.cursor()
    pg_cur.execute("SELECT COUNT(*) FROM traffic_history")
    existing = pg_cur.fetchone()[0]
    if existing > 0:
        print(f"\n[WARN] PostgreSQL traffic_history already has {existing:,} rows.")
        answer = input("  Truncate and re-migrate? (y/N): ").strip().lower()
        if answer == 'y':
            pg_cur.execute("TRUNCATE traffic_history RESTART IDENTITY")
            pg_cur.execute("TRUNCATE chat_profile")
            pg_cur.execute("TRUNCATE chat_messages RESTART IDENTITY")
            pg.commit()
            print("  Truncated.")
        else:
            print("  Skipping migration. Existing data preserved.")
            verify(sqlite_conn, pg)
            sqlite_conn.close()
            pg.close()
            return

    # Migrate
    migrate_traffic_history(sqlite_conn, pg)
    migrate_chat_profile(sqlite_conn, pg)
    migrate_chat_messages(sqlite_conn, pg)

    # Verify
    verify(sqlite_conn, pg)

    sqlite_conn.close()
    pg.close()
    print("\n[DONE] Migration complete. You can now switch database.py to PostgreSQL.")


if __name__ == "__main__":
    main()
