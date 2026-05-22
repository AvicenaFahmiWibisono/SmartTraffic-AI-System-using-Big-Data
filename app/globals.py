import threading

# Shared Global State
global_stats = {}
CCTV_SOURCES = []
camera_agents = {}

# Video Feed State
VIDEO_SOURCE = ""
outputFrame = None

# Locks
lock = threading.Lock()
model_lock = threading.Lock()
stats_lock = threading.Lock()
chat_lock = threading.Lock()

# YOLO Instance (Lazy loaded)
yolo_model_instance = None
