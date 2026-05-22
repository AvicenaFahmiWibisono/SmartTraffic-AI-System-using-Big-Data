import os

# Base Directories
# Moved inside app/, so go up two levels to reach root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Files
CONFIG_FILE = os.path.join(DATA_DIR, "cctv_config.json")
STATS_FILE = os.path.join(DATA_DIR, "traffic_stats.json")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8l.pt")
YOLO_ONNX_PATH = os.path.join(MODELS_DIR, "yolov5n.onnx")
YOLO_ONNX_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
DATA_LAKE_PATH = os.environ.get("DATA_LAKE_PATH") or os.path.join(BASE_DIR, "data_lake", "raw")

# Server
HOST_IP = "0.0.0.0"
HOST_PORT = 5000

# YOLO & Detection Config
# Tuning for higher recall in crowded scenes
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.50
PROCESS_INTERVAL = 2
# Increase history length to support up to ~24h in memory (Hot Data)
# 24h * 60m * 30 (2s intervals) = ~43,200 points
HISTORY_MAX_LEN = 50000

# Vehicle Classes
VEHICLE_CLASSES = [1, 2, 3, 5, 7]
CLASS_CAR = 0
CLASS_MOTORCYCLE = 1
CLASS_MAPPING = {
    1: CLASS_MOTORCYCLE, # Bicycle -> Motorcycle
    2: CLASS_CAR,        # Car -> Car
    3: CLASS_MOTORCYCLE, # Motorcycle -> Motorcycle
    5: CLASS_CAR,        # Bus -> Car
    7: CLASS_CAR         # Truck -> Car
}
