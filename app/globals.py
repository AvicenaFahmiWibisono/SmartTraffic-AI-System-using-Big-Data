import threading

# Shared Global State
global_stats = {}
CCTV_SOURCES = []
camera_agents = {}

# Video Feed State
VIDEO_SOURCE = ""
ACTIVE_CAMERA_ID = None
outputFrame = None

# Locks
lock = threading.Lock()
model_lock = threading.Lock()

# YOLO Instance (Lazy loaded)
yolo_model_instance = None
yolo_model_path = None
yolo_model_loaded_ts = None
