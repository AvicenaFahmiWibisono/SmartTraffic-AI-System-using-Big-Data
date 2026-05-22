from flask import Flask
from app.utils import load_config, load_stats, sync_stats_with_config, warm_history_from_db, recover_downtime_gaps
from app.services.camera import start_camera_agents
from app.database import init_db
import app.globals as g
import threading

def create_app():
    # Initialize Globals
    g.CCTV_SOURCES = load_config()
    if g.CCTV_SOURCES:
        active = next((s for s in (g.CCTV_SOURCES or []) if (s or {}).get("active") is True and (s or {}).get("url")), None)
        g.VIDEO_SOURCE = (active or g.CCTV_SOURCES[0])["url"]
    
    g.global_stats = load_stats()
    
    # Initialize Database
    init_db()
    
    # Sync stats with config (Remove zombie entries)
    sync_stats_with_config()

    # Recover gaps if app was down (fills history up to "now" using recent pattern)
    try:
        warm_history_from_db(hours=24)
        threading.Thread(target=recover_downtime_gaps, daemon=True).start()
    except Exception as e:
        print(f"[WARN] Downtime recovery failed: {e}")
    
    app = Flask(__name__)
    
    # Register Blueprints
    from app.routes import bp
    app.register_blueprint(bp)
    
    return app
