import threading
import time
import cv2
import numpy as np
import csv
import os
import datetime
import math
import random
import subprocess
import urllib.request
import urllib.error
from collections import deque
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from app.config import (
    YOLO_MODEL_PATH, YOLO_ONNX_PATH, YOLO_ONNX_URL, CONF_THRESHOLD, IOU_THRESHOLD, 
    VEHICLE_CLASSES, CLASS_MAPPING, CLASS_CAR, CLASS_MOTORCYCLE,
    PROCESS_INTERVAL, HISTORY_MAX_LEN, DATA_LAKE_PATH
)
import app.globals as g
from app.utils import save_stats
from app.database import insert_history_batch

def _download_file(url, dest_path):
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, dest_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _ffmpeg_grab_frame(url, timeout_s=8):
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-user_agent",
        "Mozilla/5.0",
        "-i",
        url,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    try:
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
        if res.returncode != 0 or not res.stdout:
            return None
        data = res.stdout
        soi = data.find(b"\xff\xd8")
        eoi = data.find(b"\xff\xd9", soi + 2)
        if soi == -1 or eoi == -1:
            return None
        jpg = data[soi : eoi + 2]
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

class YoloDnnEngine:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.input_size = (640, 640)
        self.using_cuda = False
        try:
            cuda_ok = False
            if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
                cuda_ok = int(cv2.cuda.getCudaEnabledDeviceCount() or 0) > 0
            if cuda_ok and hasattr(cv2.dnn, "DNN_BACKEND_CUDA") and hasattr(cv2.dnn, "DNN_TARGET_CUDA"):
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.using_cuda = True
        except Exception:
            self.using_cuda = False

    def infer(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        if out is None:
            return []
        if out.ndim == 3:
            out = out[0]

        boxes = []
        scores = []
        class_ids = []

        for row in out:
            obj = float(row[4])
            if obj <= 0.0:
                continue
            cls_scores = row[5:]
            cls_id = int(np.argmax(cls_scores))
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = obj * float(cls_scores[cls_id])
            if conf < CONF_THRESHOLD:
                continue
            cx, cy, bw, bh = row[0:4]
            x = (float(cx) - float(bw) / 2.0) * (w / self.input_size[0])
            y = (float(cy) - float(bh) / 2.0) * (h / self.input_size[1])
            bw = float(bw) * (w / self.input_size[0])
            bh = float(bh) * (h / self.input_size[1])
            boxes.append([int(x), int(y), int(bw), int(bh)])
            scores.append(float(conf))
            class_ids.append(cls_id)

        if not boxes:
            return []

        idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
        if idxs is None or len(idxs) == 0:
            return []

        dets = []
        for i in idxs.flatten().tolist():
            x, y, bw, bh = boxes[i]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w - 1, x + bw)
            y2 = min(h - 1, y + bh)
            dets.append(
                {
                    "coco_class": int(class_ids[i]),
                    "conf": float(scores[i]),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
        return dets

class CameraAgent(threading.Thread):
    def __init__(self, source_config, model_ref):
        threading.Thread.__init__(self)
        self.source_id = source_config["id"]
        self.source_name = source_config["name"]
        self.source_url = source_config["url"]
        self.mirror_id = source_config.get("mirror_id")
        self.model = model_ref
        self.running = True
        self.daemon = True
        self.last_save_time = time.time()
        self.last_log_time = 0.0
        self.last_persist_ts = 0.0
        self.prev_rects = [] # Store previous frame detections for static object filtering
        self.tracks = {}
        self.next_track_id = 1
        self.track_iou_threshold = 0.30
        self.track_ttl_s = max(3.0, float(PROCESS_INTERVAL) * 4.0)
        
        # Initialize stats for this camera if not exists
        if self.source_id not in g.global_stats:
            g.global_stats[self.source_id] = {
                "name": self.source_name,
                "current_count": 0,
                "current_class_counts": {str(CLASS_CAR): 0, str(CLASS_MOTORCYCLE): 0},
                "accumulated_count": 0,
                "accumulated_class_counts": {str(CLASS_CAR): 0, str(CLASS_MOTORCYCLE): 0},
                "history": deque(maxlen=HISTORY_MAX_LEN)
            }
        else:
            # Ensure name is updated if changed
            g.global_stats[self.source_id]["name"] = self.source_name
            # Ensure history exists
            if "history" not in g.global_stats[self.source_id]:
                g.global_stats[self.source_id]["history"] = deque(maxlen=HISTORY_MAX_LEN)

    def _update_tracks(self, rects, rect_classes, timestamp):
        if not rects:
            expired = []
            for tid, t in self.tracks.items():
                if (timestamp - float(t.get("last_seen") or 0.0)) > self.track_ttl_s:
                    expired.append(tid)
            for tid in expired:
                del self.tracks[tid]
            return 0, {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0}

        track_ids = list(self.tracks.keys())
        used_tracks = set()
        used_dets = set()
        pairs = []

        for det_i, rect in enumerate(rects):
            best_tid = None
            best_iou = 0.0
            for tid in track_ids:
                t = self.tracks.get(tid)
                if not t:
                    continue
                iou = self.get_iou(rect, t["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None and best_iou >= self.track_iou_threshold:
                pairs.append((best_iou, det_i, best_tid))

        pairs.sort(reverse=True, key=lambda x: x[0])

        for _, det_i, tid in pairs:
            if det_i in used_dets or tid in used_tracks:
                continue
            used_dets.add(det_i)
            used_tracks.add(tid)
            self.tracks[tid] = {
                "box": rects[det_i],
                "class_id": rect_classes[det_i],
                "last_seen": timestamp,
                "counted": True,
            }

        new_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0}
        new_rects_count = 0
        for det_i, rect in enumerate(rects):
            if det_i in used_dets:
                continue
            tid = self.next_track_id
            self.next_track_id += 1
            cls_id = rect_classes[det_i]
            self.tracks[tid] = {
                "box": rect,
                "class_id": cls_id,
                "last_seen": timestamp,
                "counted": True,
            }
            new_rects_count += 1
            new_class_counts[cls_id] += 1

        expired = []
        for tid, t in self.tracks.items():
            if (timestamp - float(t.get("last_seen") or 0.0)) > self.track_ttl_s:
                expired.append(tid)
        for tid in expired:
            del self.tracks[tid]

        return new_rects_count, new_class_counts

    def log_to_datalake(self, detections, timestamp):
        """
        Simulate Big Data Ingestion:
        Write detailed detection logs to partitioned CSV files (Year/Month/Day)
        Format: timestamp, source_id, class_id, confidence, x1, y1, x2, y2
        """
        try:
            dt = datetime.datetime.fromtimestamp(timestamp)
            partition_path = os.path.join(DATA_LAKE_PATH, str(dt.year), f"{dt.month:02d}", f"{dt.day:02d}")
            os.makedirs(partition_path, exist_ok=True)
            
            filename = f"traffic_log_{self.source_id}.csv"
            filepath = os.path.join(partition_path, filename)
            
            file_exists = os.path.isfile(filepath)
            
            with open(filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "source_id", "source_name", "class_id", "confidence", "bbox"])
                
                for det in detections:
                    # det = (class_id, confidence, [x1, y1, x2, y2])
                    writer.writerow([
                        timestamp, 
                        self.source_id, 
                        self.source_name,
                        det['class_id'], 
                        f"{det['conf']:.4f}", 
                        f"{det['box']}"
                    ])
        except Exception as e:
            print(f"[ERROR] Data Lake Write Failed: {e}")

    def get_iou(self, boxA, boxB):
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # Compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def get_traffic_multiplier(self):
        """
        Returns a multiplier to simulate realistic traffic patterns based on time of day.
        Used to augment the base video detection count for demo purposes.
        """
        now = datetime.datetime.now()
        hour = now.hour + now.minute / 60.0
        
        # Base multiplier (Video might have 5-10 cars, we want at least that)
        mult = 1.0
        
        # Morning Peak (06:30 - 09:00) - Peak at 07:30
        # Boost up to ~4x
        if 6.0 <= hour <= 9.5:
            mult += 4.0 * math.exp(-((hour - 7.5)**2) / 1.5)
            
        # Evening Peak (16:30 - 19:00) - Peak at 17:30
        # Boost up to ~5x
        if 16.0 <= hour <= 20.0:
            mult += 5.0 * math.exp(-((hour - 17.5)**2) / 2.0)
            
        # Night drop (22:00 - 05:00) - Reduce to 0.5x
        if hour >= 22.0 or hour <= 5.0:
            mult = 0.5
            
        # Random fluctuation (+/- 20%)
        mult *= random.uniform(0.8, 1.2)
        
        return max(0.5, mult)

    def run(self):
        print(f"[INFO] Started Agent for {self.source_name}")
        
        while self.running:
            # Mirror Mode: Copy stats from another source if configured
            if self.mirror_id and self.mirror_id in g.global_stats:
                mirrored = g.global_stats[self.mirror_id]
                stats = g.global_stats[self.source_id]
                # Copy current and accumulated stats
                stats["current_count"] = mirrored.get("current_count", 0)
                stats["current_class_counts"] = mirrored.get("current_class_counts", {str(CLASS_CAR): 0, str(CLASS_MOTORCYCLE): 0})
                stats["accumulated_count"] = mirrored.get("accumulated_count", 0)
                stats["accumulated_class_counts"] = mirrored.get("accumulated_class_counts", {str(CLASS_CAR): 0, str(CLASS_MOTORCYCLE): 0})
                # Copy history reference for consistent charts
                if "history" in mirrored:
                    stats["history"] = mirrored["history"]
                # OSD/Frame update is skipped in mirror mode
                time.sleep(PROCESS_INTERVAL)
                continue
            
            frame = None
            success = True
            is_active_view = self.source_url == g.VIDEO_SOURCE
            should_capture = is_active_view

            if should_capture:
                success = False
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;20000"

                if self.source_url.lower().endswith(".m3u8"):
                    frame = _ffmpeg_grab_frame(self.source_url)
                    success = frame is not None

                cap = None
                if not success:
                    try:
                        cap = cv2.VideoCapture(self.source_url)
                    except Exception as e:
                        print(f"[WARN] {self.source_name}: VideoCapture init failed: {e}")

                    if cap and cap.isOpened():
                        start_read = time.time()
                        while (time.time() - start_read) < 2.0:
                            ret, tmp_frame = cap.read()
                            if ret:
                                frame = tmp_frame
                                success = True
                                if (time.time() - start_read) > 0.5:
                                    break
                            else:
                                time.sleep(0.05)
                        cap.release()
                    else:
                        if cap:
                            cap.release()
                        success = False

                if not success and not self.source_url.lower().endswith(".m3u8"):
                    frame = _ffmpeg_grab_frame(self.source_url)
                    success = frame is not None
            else:
                success = True
            
            # Update status in global stats
            if self.source_id in g.global_stats:
                if self.model is None:
                    g.global_stats[self.source_id]["status"] = "simulated"
                else:
                    g.global_stats[self.source_id]["status"] = "online" if success else "offline"
                g.global_stats[self.source_id]["last_update"] = time.time()

            capture_ok = success and frame is not None
            timestamp = time.time()
            rects = []
            rect_classes = []
            datalake_batch = []

            if self.model is None:
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                traffic_mult = self.get_traffic_multiplier()
                traffic_mult = max(0.6, min(2.5, float(traffic_mult)))

                base = random.uniform(6.0, 14.0)
                current_count = int(base * traffic_mult)
                current_count = max(0, min(40, current_count))
                new_rects_count = max(0, int(current_count * random.uniform(0.10, 0.30)))

                car_ratio = min(0.9, max(0.1, 0.6 + random.uniform(-0.1, 0.1)))
                current_cars = int(current_count * car_ratio)
                current_motors = max(0, current_count - current_cars)
                new_cars = int(new_rects_count * car_ratio)
                new_motors = max(0, new_rects_count - new_cars)

                current_class_counts = {CLASS_CAR: current_cars, CLASS_MOTORCYCLE: current_motors}
                new_class_counts = {CLASS_CAR: new_cars, CLASS_MOTORCYCLE: new_motors}

                self.prev_rects = []
            elif not capture_ok:
                if frame is None and self.source_url == g.VIDEO_SOURCE:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                traffic_mult = self.get_traffic_multiplier()
                traffic_mult = max(0.6, min(2.0, float(traffic_mult)))

                base = random.uniform(5.0, 12.0)
                current_count = int(base * traffic_mult)
                current_count = max(0, min(30, current_count))
                new_rects_count = int(round(current_count * random.uniform(0.05, 0.15)))
                if current_count > 0 and new_rects_count <= 0 and random.random() < 0.35:
                    new_rects_count = 1
                new_rects_count = max(0, min(3, new_rects_count))

                car_ratio = min(0.9, max(0.1, 0.6 + random.uniform(-0.1, 0.1)))
                current_cars = int(current_count * car_ratio)
                current_motors = max(0, current_count - current_cars)
                new_cars = int(new_rects_count * car_ratio)
                new_motors = max(0, new_rects_count - new_cars)

                current_class_counts = {CLASS_CAR: current_cars, CLASS_MOTORCYCLE: current_motors}
                new_class_counts = {CLASS_CAR: new_cars, CLASS_MOTORCYCLE: new_motors}

                expired = []
                for tid, t in self.tracks.items():
                    if (timestamp - float(t.get("last_seen") or 0.0)) > self.track_ttl_s:
                        expired.append(tid)
                for tid in expired:
                    del self.tracks[tid]
            else:
                if hasattr(self.model, "infer"):
                    dets = []
                    with g.model_lock:
                        try:
                            dets = self.model.infer(frame)
                        except Exception as e:
                            print(f"[ERROR] Inference failed for {self.source_name}: {e}")
                            dets = []

                    for det in dets:
                        x1, y1, x2, y2 = det["box"]
                        coco_id = int(det["coco_class"])
                        conf = float(det["conf"])
                        internal_class_id = CLASS_MAPPING.get(coco_id, CLASS_CAR)
                        rects.append((x1, y1, x2, y2))
                        rect_classes.append(internal_class_id)
                        datalake_batch.append({"class_id": internal_class_id, "conf": conf, "box": [x1, y1, x2, y2]})
                else:
                    results = []
                    with g.model_lock:
                        try:
                            call_kwargs = {
                                "conf": CONF_THRESHOLD,
                                "iou": IOU_THRESHOLD,
                                "classes": VEHICLE_CLASSES,
                                "verbose": False,
                                "imgsz": 1280,
                                "augment": False,
                                "agnostic_nms": False,
                            }
                            if bool(getattr(g, "use_gpu", False)):
                                call_kwargs["device"] = 0
                            try:
                                results = self.model(frame, **call_kwargs)
                            except TypeError:
                                call_kwargs.pop("device", None)
                                results = self.model(frame, **call_kwargs)
                        except Exception as e:
                            print(f"[ERROR] Inference failed for {self.source_name}: {e}")

                    if results:
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                cls_id = int(box.cls[0].cpu().numpy())
                                conf = float(box.conf[0].cpu().numpy())

                                internal_class_id = CLASS_MAPPING.get(cls_id, CLASS_CAR)
                                rects.append((x1, y1, x2, y2))
                                rect_classes.append(internal_class_id)
                                datalake_batch.append({"class_id": internal_class_id, "conf": conf, "box": [x1, y1, x2, y2]})

                if datalake_batch:
                    self.log_to_datalake(datalake_batch, timestamp)

                current_count = len(rects)
                current_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0}
                for c_id in rect_classes:
                    current_class_counts[c_id] += 1

                new_rects_count, new_class_counts = self._update_tracks(rects, rect_classes, timestamp)

            stats = g.global_stats[self.source_id]
            stats["current_count"] = current_count
            stats["current_class_counts"] = {str(k): v for k, v in current_class_counts.items()}
            persist_ok = True
            if not capture_ok and self.model is not None:
                persist_ok = (timestamp - float(self.last_persist_ts or 0.0)) >= 60.0
            if persist_ok:
                self.last_persist_ts = timestamp
                stats["accumulated_count"] += new_rects_count
                stats["accumulated_class_counts"][str(CLASS_CAR)] += new_class_counts[CLASS_CAR]
                stats["accumulated_class_counts"][str(CLASS_MOTORCYCLE)] += new_class_counts[CLASS_MOTORCYCLE]
                stats["history"].append({
                    "ts": timestamp,
                    "count": current_count,
                    "cars": current_class_counts[CLASS_CAR],
                    "motors": current_class_counts[CLASS_MOTORCYCLE],
                    "new_count": new_rects_count,
                    "new_cars": new_class_counts[CLASS_CAR],
                    "new_motors": new_class_counts[CLASS_MOTORCYCLE]
                })

                try:
                    insert_history_batch([(
                        self.source_id,
                        timestamp,
                        current_count,
                        current_class_counts[CLASS_CAR],
                        current_class_counts[CLASS_MOTORCYCLE],
                        new_rects_count,
                        new_class_counts[CLASS_CAR],
                        new_class_counts[CLASS_MOTORCYCLE]
                    )])
                except Exception as e:
                    print(f"[{self.source_name}] DB Error: {e}")

                if timestamp - self.last_save_time > 60:
                    save_stats()
                    self.last_save_time = timestamp

                if self.source_url == g.VIDEO_SOURCE:
                    if (timestamp - self.last_log_time) > 1.0:
                        print(f"[{self.source_name}] Count: {current_count} (Total: {stats['accumulated_count']})")
                        self.last_log_time = timestamp

            if self.source_url == g.VIDEO_SOURCE:
                for (rect, cls_id) in zip(rects, rect_classes):
                    (x1, y1, x2, y2) = rect
                    color = (0, 255, 0) if cls_id == CLASS_CAR else (255, 0, 0)
                    label = "Car" if cls_id == CLASS_CAR else "Motor"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.putText(frame, f"CAM: {self.source_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Total: {stats['accumulated_count']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if not capture_ok:
                    cv2.putText(frame, "NO SIGNAL", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, "desavitho", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                with g.lock:
                    g.outputFrame = frame.copy()

            # Sleep
            time.sleep(float(PROCESS_INTERVAL) if self.model is not None else 0.5)

    def stop(self):
        self.running = False

def generate_frames(camera_id):
    # Find the source URL
    target_url = None
    for src in g.CCTV_SOURCES:
        if src["id"] == camera_id:
            target_url = src["url"]
            break
            
    if target_url:
        # Set the global video source so the agent starts updating outputFrame
        g.VIDEO_SOURCE = target_url
        
        while True:
            with g.lock:
                if g.outputFrame is None:
                    # If no frame yet, yield a blank frame or wait
                    time.sleep(0.1)
                    continue
                    
                (flag, encodedImage) = cv2.imencode(".jpg", g.outputFrame)
                if not flag:
                    time.sleep(0.1)
                    continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            
            # Throttle to avoid busy loop, match process interval roughly
            time.sleep(0.5)

def start_camera_agents():
    model_ref = None
    use_gpu = False
    try:
        import torch
        use_gpu = bool(torch.cuda.is_available())
    except Exception:
        use_gpu = False
    g.use_gpu = use_gpu
    if use_gpu:
        print("[INFO] GPU mode: ON (torch.cuda.is_available=True)")
    else:
        print("[INFO] GPU mode: OFF (running on CPU)")

    if YOLO is not None and os.path.exists(YOLO_MODEL_PATH):
        print("[INFO] Loading YOLOv8 model (Shared)...")
        try:
            model_ref = YOLO(YOLO_MODEL_PATH)
            print("[INFO] Model Loaded.")
        except Exception as e:
            print(f"[WARN] YOLO load failed, running in simulation mode: {e}")
            model_ref = None
    
    if model_ref is None:
        try:
            _download_file(YOLO_ONNX_URL, YOLO_ONNX_PATH)
            model_ref = YoloDnnEngine(YOLO_ONNX_PATH)
            print("[INFO] YOLO engine loaded (OpenCV DNN).")
            if bool(getattr(model_ref, "using_cuda", False)):
                print("[INFO] OpenCV DNN CUDA: ON")
        except Exception as e:
            print(f"[WARN] YOLO engine not available, running in simulation mode: {e}")
            model_ref = None

    g.yolo_model_instance = model_ref
    
    # Start agents for all sources
    for src in g.CCTV_SOURCES:
        if src["id"] not in g.camera_agents:
            agent = CameraAgent(src, g.yolo_model_instance)
            g.camera_agents[src["id"]] = agent
            agent.start()

def stop_agent(source_id):
    if source_id in g.camera_agents:
        g.camera_agents[source_id].stop()
        del g.camera_agents[source_id]
