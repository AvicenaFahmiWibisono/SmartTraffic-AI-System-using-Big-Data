import os
import sys

sys.dont_write_bytecode = True

os.environ.setdefault("OLLAMA_MODEL", "mistral:7b")

deps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deps")
if os.path.isdir(deps_dir) and deps_dir not in sys.path:
    sys.path.insert(0, deps_dir)

from app import create_app
from app.services.camera import start_camera_agents
from app.config import HOST_IP, HOST_PORT

# Create Flask Application
app = create_app()

if __name__ == "__main__":
    print(f"[INFO] Starting Vehicle Counter System...")
    
    # Start Camera Agents (Background Threads)
    start_camera_agents()
    
    print(f"[INFO] Server running on http://{HOST_IP}:{HOST_PORT}")
    
    # Run Flask Server
    # use_reloader=False is important when using background threads to avoid duplicates
    app.run(host=HOST_IP, port=HOST_PORT, debug=False, use_reloader=False, threaded=True)
