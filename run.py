
from app import create_app
from app.config import HOST_IP, HOST_PORT

# Create Flask Application
app = create_app()

if __name__ == "__main__":
    print(f"[INFO] Starting Vehicle Counter System...")
    print(f"[INFO] Server running on http://{HOST_IP}:{HOST_PORT}")
    
    # Run Flask Server
    # use_reloader=False is important when using background threads to avoid duplicates
    app.run(host=HOST_IP, port=HOST_PORT, debug=False, use_reloader=False)
