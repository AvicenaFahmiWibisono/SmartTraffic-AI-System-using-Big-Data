from flask import Flask
from app.utils import load_config, load_stats, sync_stats_with_config
from app.services.camera import start_camera_agents
from app.database import init_db
import app.globals as g

def create_app():
    # Initialize Globals
    g.CCTV_SOURCES = load_config()
    if g.CCTV_SOURCES:
        g.VIDEO_SOURCE = g.CCTV_SOURCES[0]["url"]
    
    g.global_stats = load_stats()
    
    # Initialize Database
    init_db()
    
    # Sync stats with config (Remove zombie entries)
    sync_stats_with_config()
    
    app = Flask(__name__)
    
    # Register Blueprints
    from app.routes import bp
    app.register_blueprint(bp)
    
    start_camera_agents()
    
    return app
