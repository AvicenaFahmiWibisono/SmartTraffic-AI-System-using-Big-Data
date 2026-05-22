import os
import time
import datetime
import json
import threading

import psycopg2
import psycopg2.extras

from app.config import DATA_DIR

# ─── PostgreSQL connection settings ───────────────────────────────────────────
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "smarttraffic")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

# ─── PyTorch (optional, for Transformer forecaster) ───────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

_transformer_models = {}
_transformer_training = set()
_transformer_training_lock = threading.Lock()


# ─── Connection helper ────────────────────────────────────────────────────────
def get_db_connection(timeout_s=30, **kwargs):
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
        connect_timeout=int(timeout_s or 30),
    )
    conn.autocommit = False
    return conn


def _cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


# ─── Timezone helpers ─────────────────────────────────────────────────────────
def _local_tzinfo():
    return datetime.datetime.now().astimezone().tzinfo


def _hour_bucket_ts(ts, tzinfo):
    try:
        dt = datetime.datetime.fromtimestamp(float(ts), tz=tzinfo)
    except Exception:
        return None
    dt0 = dt.replace(minute=0, second=0, microsecond=0)
    return int(dt0.timestamp())


def _build_hourly_series(camera_id, days=60):
    tzinfo = _local_tzinfo()
    cutoff = time.time() - (float(days or 60) * 24 * 3600)
    conn = get_db_connection()
    cur = _cursor(conn)
    try:
        cur.execute(
            "SELECT timestamp, new_count FROM traffic_history WHERE camera_id = %s AND timestamp >= %s ORDER BY timestamp ASC",
            (camera_id, cutoff),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    buckets = {}
    last_bucket = None
    for r in rows:
        ts = r["timestamp"]
        b = _hour_bucket_ts(ts, tzinfo)
        if b is None:
            continue
        last_bucket = b if last_bucket is None else max(last_bucket, b)
        v = int(r["new_count"] or 0)
        buckets[b] = int(buckets.get(b, 0) + v)

    if not buckets:
        return {"series": [], "tzinfo": tzinfo, "max_bucket_ts": None}

    first = min(buckets.keys())
    last = max(buckets.keys())
    out = []
    cur_ts = first
    while cur_ts <= last:
        out.append((cur_ts, int(buckets.get(cur_ts, 0))))
        cur_ts += 3600
    return {"series": out, "tzinfo": tzinfo, "max_bucket_ts": last}


def _time_features_from_bucket_ts(bucket_ts, tzinfo):
    dt = datetime.datetime.fromtimestamp(int(bucket_ts), tz=tzinfo)
    dow = int(dt.strftime("%w"))
    hour = int(dt.strftime("%H"))
    return dow, hour


# ─── Tiny Transformer Forecaster ──────────────────────────────────────────────
if torch is not None and nn is not None:
    class _TinyTransformerForecaster(nn.Module):
        def __init__(self, d_model=32, nhead=4, num_layers=2, dropout=0.1, max_len=256):
            super().__init__()
            self.d_model = int(d_model)
            self.value_proj = nn.Linear(1, self.d_model)
            self.hour_emb = nn.Embedding(24, self.d_model)
            self.dow_emb = nn.Embedding(7, self.d_model)
            self.pos_emb = nn.Embedding(int(max_len), self.d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=int(nhead), dropout=float(dropout), batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
            self.head = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 1))

        def forward(self, x_val, x_hour, x_dow):
            b, t, _ = x_val.shape
            pos = torch.arange(t, device=x_val.device).unsqueeze(0).expand(b, t)
            h = self.value_proj(x_val) + self.hour_emb(x_hour) + self.dow_emb(x_dow) + self.pos_emb(pos)
            z = self.encoder(h)
            y = self.head(z[:, -1, :])
            return y
else:
    _TinyTransformerForecaster = None


def _get_or_train_transformer(camera_id, context_len=48, max_days=60):
    if torch is None or nn is None or optim is None or _TinyTransformerForecaster is None:
        return None
    cam_id = str(camera_id or "").strip()
    if not cam_id:
        return None
    info = _build_hourly_series(cam_id, days=max_days)
    series = info.get("series") or []
    tzinfo = info.get("tzinfo")
    max_bucket_ts = info.get("max_bucket_ts")
    if len(series) < max(context_len + 24, 96):
        return None
    cache = _transformer_models.get(cam_id)
    if cache and cache.get("max_bucket_ts") == max_bucket_ts:
        return cache
    values = [float(v) for _, v in series]
    mean = sum(values) / float(len(values) or 1)
    var = sum((v - mean) ** 2 for v in values) / float(max(1, len(values) - 1))
    std = (var ** 0.5) if var > 1e-8 else 1.0
    xs_val, xs_hour, xs_dow, ys = [], [], [], []
    for i in range(context_len, len(series)):
        window = series[i - context_len: i]
        target = series[i][1]
        xw, xh, xd = [], [], []
        for bucket_ts, v in window:
            dow, hour = _time_features_from_bucket_ts(bucket_ts, tzinfo)
            xw.append([(float(v) - mean) / std])
            xh.append(hour)
            xd.append(dow)
        xs_val.append(xw)
        xs_hour.append(xh)
        xs_dow.append(xd)
        ys.append([(float(target) - mean) / std])
    device = torch.device("cpu")
    x_val = torch.tensor(xs_val, dtype=torch.float32, device=device)
    x_hour = torch.tensor(xs_hour, dtype=torch.long, device=device)
    x_dow = torch.tensor(xs_dow, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.float32, device=device)
    model = _TinyTransformerForecaster(d_model=32, nhead=4, num_layers=2, dropout=0.1, max_len=max(256, int(context_len) + 8)).to(device)
    model.train()
    opt = optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()
    n = x_val.shape[0]
    batch = 64 if n >= 256 else 32
    epochs = 6
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    for _ in range(epochs):
        idx = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            j = idx[s: s + batch]
            pred = model(x_val[j], x_hour[j], x_dow[j])
            loss = loss_fn(pred, y[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    cache = {
        "model": model, "context_len": int(context_len),
        "mean": float(mean), "std": float(std), "tzinfo": tzinfo,
        "series": series, "max_bucket_ts": max_bucket_ts, "trained_at": time.time(),
    }
    _transformer_models[cam_id] = cache
    return cache


def _get_transformer_cache(camera_id):
    return _transformer_models.get(str(camera_id or "").strip()) or None


def _predict_with_transformer(camera_id, target_dt_local, context_len=48):
    if torch is None or not isinstance(target_dt_local, datetime.datetime):
        return None
    cache = _get_transformer_cache(camera_id)
    if not cache:
        return None
    model = cache.get("model")
    tzinfo = cache.get("tzinfo")
    mean = float(cache.get("mean") or 0.0)
    std = float(cache.get("std") or 1.0)
    series = cache.get("series") or []
    if not model or not series:
        return None
    target_dt_local = target_dt_local.astimezone(tzinfo)
    target_bucket = int(target_dt_local.replace(minute=0, second=0, microsecond=0).timestamp())
    last_bucket = int(series[-1][0])

    def _infer_window(window):
        device = torch.device("cpu")
        xw, xh, xd = [], [], []
        for bucket_ts, v in window:
            dow, hour = _time_features_from_bucket_ts(bucket_ts, tzinfo)
            xw.append([(float(v) - mean) / std])
            xh.append(hour)
            xd.append(dow)
        x_val = torch.tensor([xw], dtype=torch.float32, device=device)
        x_hour = torch.tensor([xh], dtype=torch.long, device=device)
        x_dow = torch.tensor([xd], dtype=torch.long, device=device)
        with torch.no_grad():
            y_out = model(x_val, x_hour, x_dow).cpu().numpy().reshape(-1)[0]
        return max(0, int(round(float(y_out) * std + mean)))

    if target_bucket <= last_bucket:
        i = None
        for k, (b, _) in enumerate(series):
            if int(b) == int(target_bucket):
                i = k
                break
        if i is None or i < context_len:
            return None
        return _infer_window(series[i - context_len: i])

    steps = min(48, int((target_bucket - last_bucket) // 3600))
    if steps <= 0:
        return None
    window = list(series[-context_len:])
    cur_last = last_bucket
    pred_val = None
    for _ in range(steps):
        pred_val = _infer_window(window)
        cur_last += 3600
        window.append((cur_last, pred_val))
        if len(window) > context_len:
            window = window[-context_len:]
    return pred_val


# ─── Schema initialization ────────────────────────────────────────────────────
def init_db():
    conn = get_db_connection(timeout_s=10)
    cur = conn.cursor()
    try:
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
        cur.execute("CREATE INDEX IF NOT EXISTS idx_camera_timestamp ON traffic_history (camera_id, timestamp)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_profile (
                session_id TEXT PRIMARY KEY,
                updated_ts DOUBLE PRECISION NOT NULL,
                last_intent TEXT, last_camera_id TEXT, last_camera_name TEXT, last_destination TEXT, prefs_json TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                ts DOUBLE PRECISION NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                page TEXT, meta_json TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_ts ON chat_messages (session_id, ts)")
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
    print(f"Database initialized (PostgreSQL {DB_HOST}:{DB_PORT}/{DB_NAME})")


# ─── Chat ─────────────────────────────────────────────────────────────────────
def get_chat_profile(session_id):
    sid = str(session_id or "").strip()
    if not sid:
        return {}
    conn = get_db_connection(timeout_s=2)
    cur = _cursor(conn)
    try:
        cur.execute("SELECT * FROM chat_profile WHERE session_id = %s", (sid,))
        row = cur.fetchone()
        if not row:
            return {}
        out = dict(row)
        try:
            prefs = json.loads(out.get("prefs_json") or "{}")
        except Exception:
            prefs = {}
        out["prefs"] = prefs if isinstance(prefs, dict) else {}
        return out
    finally:
        conn.close()


def upsert_chat_profile(session_id, fields):
    sid = str(session_id or "").strip()
    if not sid:
        return False
    f = fields or {}
    now = time.time()
    prefs_json = None
    if "prefs" in f:
        try:
            prefs_json = json.dumps(f.get("prefs") or {}, ensure_ascii=False)
        except Exception:
            prefs_json = "{}"
    conn = get_db_connection(timeout_s=5)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_profile (session_id, updated_ts, last_intent, last_camera_id, last_camera_name, last_destination, prefs_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
                updated_ts = EXCLUDED.updated_ts,
                last_intent = COALESCE(EXCLUDED.last_intent, chat_profile.last_intent),
                last_camera_id = COALESCE(EXCLUDED.last_camera_id, chat_profile.last_camera_id),
                last_camera_name = COALESCE(EXCLUDED.last_camera_name, chat_profile.last_camera_name),
                last_destination = COALESCE(EXCLUDED.last_destination, chat_profile.last_destination),
                prefs_json = COALESCE(EXCLUDED.prefs_json, chat_profile.prefs_json)
        """, (sid, now, f.get("last_intent"), f.get("last_camera_id"), f.get("last_camera_name"), f.get("last_destination"), prefs_json))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


def add_chat_message(session_id, role, content, page=None, meta=None):
    sid = str(session_id or "").strip()
    if not sid:
        return False
    r = str(role or "").strip()
    txt = str(content or "").strip()
    if not r or not txt:
        return False
    meta_json = json.dumps(meta or {}, ensure_ascii=False) if meta is not None else None
    conn = get_db_connection(timeout_s=5)
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO chat_messages (session_id, ts, role, content, page, meta_json) VALUES (%s, %s, %s, %s, %s, %s)",
                    (sid, time.time(), r, txt, str(page or "") or None, meta_json))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


def get_recent_chat_messages(session_id, limit=12):
    sid = str(session_id or "").strip()
    if not sid:
        return []
    lim = min(50, max(1, int(limit or 12)))
    conn = get_db_connection(timeout_s=3)
    cur = _cursor(conn)
    try:
        cur.execute("SELECT ts, role, content FROM chat_messages WHERE session_id = %s ORDER BY ts DESC LIMIT %s", (sid, lim))
        rows = cur.fetchall() or []
        return [{"ts": row["ts"], "role": row["role"], "content": row["content"]} for row in reversed(rows)]
    finally:
        conn.close()


# ─── Traffic history CRUD ─────────────────────────────────────────────────────
# ─── Write pause control ──────────────────────────────────────────────────────
_write_paused = False
_write_pause_lock = threading.Lock()


def is_write_paused():
    return _write_paused


def set_write_paused(paused):
    global _write_paused
    with _write_pause_lock:
        _write_paused = bool(paused)


def get_live_status_from_db():
    """Read current_count per camera from PostgreSQL live_status table."""
    conn = get_db_connection(timeout_s=2)
    cur = _cursor(conn)
    try:
        cur.execute("SELECT camera_id, current_count, current_cars, current_motors, status FROM live_status")
        rows = cur.fetchall()
        return {row["camera_id"]: row for row in rows}
    except Exception:
        return {}
    finally:
        conn.close()


def update_live_status(camera_id, current_count, current_cars, current_motors, status="online"):
    """Write current_count to PostgreSQL live_status table (called by camera agents)."""
    if _write_paused:
        return
    conn = get_db_connection(timeout_s=2)
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE live_status
            SET current_count = %s, current_cars = %s, current_motors = %s, status = %s, updated_at = NOW()
            WHERE camera_id = %s
        """, (current_count, current_cars, current_motors, status, camera_id))
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


# Cache camera names for insert_history_batch
_camera_name_cache = {}
_camera_name_cache_ts = 0.0


def _get_camera_name(camera_id):
    """Get camera name from cached config. Refreshes every 60s."""
    global _camera_name_cache, _camera_name_cache_ts
    now = time.time()
    if not _camera_name_cache or (now - _camera_name_cache_ts) > 60:
        try:
            from app.utils import load_config
            config = load_config() or []
            _camera_name_cache = {c.get("id"): c.get("name", c.get("id", "")) for c in config if c.get("id")}
            _camera_name_cache_ts = now
        except Exception:
            pass
    return _camera_name_cache.get(camera_id, camera_id)


def insert_history_batch(records):
    if not records:
        return
    if _write_paused:
        return  # Skip writing when paused
    sql = """INSERT INTO traffic_history
             (camera_id, timestamp, total_count, car_count, motorcycle_count, new_count, new_cars, new_motors, recorded_at, camera_name)
             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, to_timestamp(%s) AT TIME ZONE 'Asia/Jakarta', %s)"""
    enriched = [
        (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[1], _get_camera_name(r[0]))
        for r in records
    ]
    conn = get_db_connection(timeout_s=30)
    cur = conn.cursor()
    try:
        psycopg2.extras.execute_batch(cur, sql, enriched, page_size=200)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error inserting batch: {e}")
    finally:
        conn.close()


def clear_all_history():
    conn = get_db_connection(timeout_s=10)
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM traffic_history")
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error clearing history: {e}")
    finally:
        conn.close()


def get_camera_history(camera_id, start_ts=None, end_ts=None):
    conn = get_db_connection(timeout_s=2)
    cur = _cursor(conn)
    try:
        q = "SELECT timestamp, total_count, car_count, motorcycle_count, new_count, new_cars, new_motors FROM traffic_history WHERE camera_id = %s"
        p = [camera_id]
        if start_ts:
            q += " AND timestamp >= %s"; p.append(start_ts)
        if end_ts:
            q += " AND timestamp <= %s"; p.append(end_ts)
        q += " ORDER BY timestamp ASC"
        cur.execute(q, p)
        return [{"ts": r["timestamp"], "count": r["total_count"], "cars": r["car_count"], "motors": r["motorcycle_count"], "new_count": r["new_count"], "new_cars": r["new_cars"], "new_motors": r["new_motors"]} for r in cur.fetchall()]
    finally:
        conn.close()


def predict_future_traffic(camera_id, day_of_week, hour_of_day, target_dt_local=None):
    if target_dt_local is not None and torch is not None:
        try:
            pred = _predict_with_transformer(camera_id, target_dt_local, context_len=48)
            if pred is not None:
                return float(pred)
        except Exception:
            pass
        cam_id = str(camera_id or "").strip()
        if cam_id:
            should_start = False
            with _transformer_training_lock:
                if cam_id not in _transformer_models and cam_id not in _transformer_training:
                    _transformer_training.add(cam_id)
                    should_start = True
            if should_start:
                def _train_bg():
                    try:
                        _get_or_train_transformer(cam_id, context_len=48, max_days=60)
                    finally:
                        with _transformer_training_lock:
                            _transformer_training.discard(cam_id)
                threading.Thread(target=_train_bg, daemon=True).start()

    conn = get_db_connection(timeout_s=2)
    cur = _cursor(conn)
    try:
        # EXTRACT(DOW) in PostgreSQL: 0=Sunday, same as SQLite strftime('%w')
        cur.execute("""
            WITH hourly_sums AS (
                SELECT DATE(to_timestamp(timestamp) AT TIME ZONE 'Asia/Jakarta') AS date_str,
                       SUM(new_count) AS hourly_total
                FROM traffic_history
                WHERE camera_id = %s
                  AND EXTRACT(DOW FROM to_timestamp(timestamp) AT TIME ZONE 'Asia/Jakarta') = %s
                  AND EXTRACT(HOUR FROM to_timestamp(timestamp) AT TIME ZONE 'Asia/Jakarta') = %s
                GROUP BY date_str
            )
            SELECT AVG(hourly_total) AS avg_hourly_traffic FROM hourly_sums
        """, (camera_id, day_of_week, hour_of_day))
        result = cur.fetchone()
        return float(result["avg_hourly_traffic"]) if result and result["avg_hourly_traffic"] is not None else 0
    except Exception as e:
        print(f"Prediction Error: {e}")
        return 0
    finally:
        conn.close()


def get_total_lifetime():
    conn = get_db_connection(timeout_s=2)
    cur = _cursor(conn)
    try:
        cur.execute("SELECT COALESCE(SUM(new_cars),0) AS cars, COALESCE(SUM(new_motors),0) AS motors FROM traffic_history")
        row = cur.fetchone()
        if not row:
            return {"accumulated_count": 0, "cars": 0, "motorcycles": 0}
        total = int((row["cars"] or 0) + (row["motors"] or 0))
        return {"accumulated_count": total, "cars": int(row["cars"] or 0), "motorcycles": int(row["motors"] or 0)}
    except Exception:
        return {"accumulated_count": 0, "cars": 0, "motorcycles": 0}
    finally:
        conn.close()


def get_totals_by_camera(camera_ids=None, start_ts=None, end_ts=None):
    conn = get_db_connection(timeout_s=2)
    cur = _cursor(conn)
    try:
        params = []
        conditions = []
        if camera_ids:
            placeholders = ",".join(["%s"] * len(camera_ids))
            conditions.append(f"camera_id IN ({placeholders})")
            params.extend(list(camera_ids))
        if start_ts:
            conditions.append("timestamp >= %s"); params.append(start_ts)
        if end_ts:
            conditions.append("timestamp <= %s"); params.append(end_ts)
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        cur.execute(f"SELECT camera_id, COALESCE(SUM(new_cars),0) AS cars, COALESCE(SUM(new_motors),0) AS motors FROM traffic_history {where_clause} GROUP BY camera_id", params)
        out = {}
        for row in cur.fetchall():
            total = int((row["cars"] or 0) + (row["motors"] or 0))
            out[row["camera_id"]] = {"accumulated_count": total, "cars": int(row["cars"] or 0), "motorcycles": int(row["motors"] or 0)}
        return out
    except Exception:
        return {}
    finally:
        conn.close()


def get_aggregated_stats(days=30):
    conn = get_db_connection(timeout_s=2)
    cur = _cursor(conn)
    try:
        cutoff = time.time() - (days * 24 * 3600)
        cur.execute("SELECT COALESCE(SUM(new_cars),0) AS cars, COALESCE(SUM(new_motors),0) AS motors FROM traffic_history WHERE timestamp >= %s", (cutoff,))
        row = cur.fetchone()
        if not row:
            return {"accumulated_count": 0, "cars": 0, "motorcycles": 0}
        total = int((row["cars"] or 0) + (row["motors"] or 0))
        return {"accumulated_count": total, "cars": int(row["cars"] or 0), "motorcycles": int(row["motors"] or 0)}
    except Exception as e:
        print(f"Error getting aggregated stats: {e}")
        return {"accumulated_count": 0, "cars": 0, "motorcycles": 0}
    finally:
        conn.close()


def get_history_range(camera_id=None, start_ts=None, end_ts=None):
    conn = get_db_connection(timeout_s=2)
    cur = _cursor(conn)
    try:
        conditions, params = [], []
        if camera_id:
            conditions.append("camera_id = %s"); params.append(camera_id)
        if start_ts:
            conditions.append("timestamp >= %s"); params.append(start_ts)
        if end_ts:
            conditions.append("timestamp <= %s"); params.append(end_ts)
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        cur.execute(f"SELECT camera_id, timestamp, total_count, car_count, motorcycle_count, new_count, new_cars, new_motors FROM traffic_history {where_clause} ORDER BY camera_id, timestamp ASC", params)
        return [{"camera_id": r["camera_id"], "ts": r["timestamp"], "count": r["total_count"], "cars": r["car_count"], "motors": r["motorcycle_count"], "new_count": r["new_count"], "new_cars": r["new_cars"], "new_motors": r["new_motors"]} for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def get_last_history_row(camera_id):
    conn = get_db_connection(timeout_s=2)
    cur = _cursor(conn)
    try:
        cur.execute("SELECT timestamp, total_count, car_count, motorcycle_count, new_count, new_cars, new_motors FROM traffic_history WHERE camera_id = %s ORDER BY timestamp DESC LIMIT 1", (camera_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {"ts": row["timestamp"], "count": row["total_count"], "cars": row["car_count"], "motors": row["motorcycle_count"], "new_count": row["new_count"], "new_cars": row["new_cars"], "new_motors": row["new_motors"]}
    finally:
        conn.close()


def get_recent_history_averages(camera_id, start_ts, end_ts):
    conn = get_db_connection(timeout_s=2)
    cur = _cursor(conn)
    try:
        cur.execute("""
            SELECT AVG(total_count) AS avg_total, AVG(car_count) AS avg_cars, AVG(motorcycle_count) AS avg_motors,
                   AVG(new_count) AS avg_new, AVG(new_cars) AS avg_new_cars, AVG(new_motors) AS avg_new_motors, COUNT(*) AS n
            FROM traffic_history WHERE camera_id = %s AND timestamp >= %s AND timestamp <= %s
        """, (camera_id, start_ts, end_ts))
        row = cur.fetchone()
        if not row or row["n"] == 0:
            return None
        return {"avg_total": float(row["avg_total"] or 0), "avg_cars": float(row["avg_cars"] or 0), "avg_motors": float(row["avg_motors"] or 0), "avg_new": float(row["avg_new"] or 0), "avg_new_cars": float(row["avg_new_cars"] or 0), "avg_new_motors": float(row["avg_new_motors"] or 0), "n": int(row["n"] or 0)}
    finally:
        conn.close()
