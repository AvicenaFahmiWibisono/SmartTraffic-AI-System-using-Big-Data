import os

# Base Directories
# Moved inside app/, so go up two levels to reach root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Files
CONFIG_FILE = os.path.join(DATA_DIR, "cctv_config.json")
STATS_FILE = os.path.join(DATA_DIR, "traffic_stats.json")
_CUSTOM_YOLO_PATH = os.path.join(MODELS_DIR, "yolov8_mobil_motor.pt")
_DEFAULT_YOLO_PATH = os.path.join(MODELS_DIR, "yolov8l.pt")
def get_yolo_model_path():
    env_path = os.getenv("YOLO_MODEL_PATH")
    if env_path:
        return env_path
    return _CUSTOM_YOLO_PATH if os.path.exists(_CUSTOM_YOLO_PATH) else _DEFAULT_YOLO_PATH

YOLO_MODEL_PATH = get_yolo_model_path()

# Server
HOST_IP = "0.0.0.0"
HOST_PORT = 5000

# YOLO & Detection Config
# Tuning for better precision on CCTV scenes
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
PROCESS_INTERVAL = 2
# Increase history length to support up to ~24h in memory (Hot Data)
# 24h * 60m * 30 (2s intervals) = ~43,200 points
HISTORY_MAX_LEN = 50000

# Vehicle Classes
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_CAR = 0
CLASS_MOTORCYCLE = 1
CLASS_MAPPING = {
    2: CLASS_CAR,        # Car -> Car
    3: CLASS_MOTORCYCLE, # Motorcycle -> Motorcycle
    5: CLASS_CAR,        # Bus -> Car
    7: CLASS_CAR         # Truck -> Car
}
