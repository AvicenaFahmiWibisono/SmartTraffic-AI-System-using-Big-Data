import threading
import time
import cv2
import csv
import os
import datetime
import math
import random
from collections import deque
try:
    import numpy as np
except ModuleNotFoundError:
    np = None
try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None

from app.config import (
    YOLO_MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, 
    VEHICLE_CLASSES, CLASS_MAPPING, CLASS_CAR, CLASS_MOTORCYCLE,
    PROCESS_INTERVAL, HISTORY_MAX_LEN, get_yolo_model_path
)
import app.globals as g
from app.utils import save_stats
from app.database import insert_history_batch

# Data Lake Configuration
DATA_LAKE_PATH = "/var/www/vehicle-counter/data_lake/raw"

def _normalize_model_names(names):
    if names is None:
        return None
    if isinstance(names, dict):
        max_k = max(names.keys()) if names else -1
        out = [None] * (max_k + 1)
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names]
    return None

class CameraAgent(threading.Thread):
    def __init__(self, source_config, model_ref):
        threading.Thread.__init__(self)
        self.source_id = source_config["id"]
        self.source_name = source_config["name"]
        self.source_url = source_config["url"]
        self.mirror_id = source_config.get("mirror_id")
        self.model = model_ref
        self.model_names = _normalize_model_names(getattr(model_ref, "names", None)) if model_ref is not None else None
        names_lower = set()
        if self.model_names:
            names_lower = {str(n).strip().lower() for n in self.model_names if n is not None}
        self.is_custom_vehicle_model = ("mobil" in names_lower and "motor" in names_lower and len(names_lower) <= 3)
        self.yolo_classes = None if self.is_custom_vehicle_model else VEHICLE_CLASSES
        self.running = True
        self.daemon = True
        self.last_save_time = time.time()
        self.prev_rects = [] # Store previous frame detections for static object filtering
        self.cap = None
        self.last_infer_time = 0.0
        self.last_overlay = []
        self._infer_running = False
        self._infer_state_lock = threading.Lock()
        self._data_lock = threading.Lock()
        self.tracks = {}
        self.track_candidates = {}
        self.next_track_id = 1
        self.count_line = source_config.get("count_line") if isinstance(source_config, dict) else None
        if not isinstance(self.count_line, dict):
            self.count_line = {"type": "horizontal", "pos": 0.6}
        self.latest_detections = {"ts": 0, "items": [], "line": dict(self.count_line)}
        try:
            if isinstance(source_config, dict) and "det_conf" in source_config:
                self.det_conf = float(source_config.get("det_conf"))
            else:
                self.det_conf = 0.15 if self.is_custom_vehicle_model else float(CONF_THRESHOLD)
        except Exception:
            self.det_conf = 0.15 if self.is_custom_vehicle_model else float(CONF_THRESHOLD)
        try:
            self.det_iou = float(source_config.get("det_iou", IOU_THRESHOLD))
        except Exception:
            self.det_iou = float(IOU_THRESHOLD)
        try:
            self.min_area_ratio = float(source_config.get("min_area_ratio", 0.00015))
        except Exception:
            self.min_area_ratio = 0.00015
        try:
            self.min_wh = int(source_config.get("min_wh", 12))
        except Exception:
            self.min_wh = 12
        try:
            self.track_iou_threshold = float(source_config.get("track_iou", 0.2))
        except Exception:
            self.track_iou_threshold = 0.2
        try:
            self.track_dist_ratio = float(source_config.get("track_dist", 0.10))
        except Exception:
            self.track_dist_ratio = 0.10
        try:
            self.max_track_misses = int(source_config.get("max_misses", 6))
        except Exception:
            self.max_track_misses = 6
        if self.max_track_misses > 3:
            self.max_track_misses = 3
        try:
            self.active_infer_interval = float(source_config.get("active_infer_interval", 0.0))
        except Exception:
            self.active_infer_interval = 0.0
        try:
            self.imgsz = int(source_config.get("imgsz", 640))
        except Exception:
            self.imgsz = 640
        self.preprocess = bool(source_config.get("preprocess", True))
        try:
            self.roi_min_y = float(source_config.get("roi_min_y", 0.0))
        except Exception:
            self.roi_min_y = 0.0
        try:
            self.roi_max_y = float(source_config.get("roi_max_y", 1.0))
        except Exception:
            self.roi_max_y = 1.0
        self.roi_min_y = max(0.0, min(1.0, self.roi_min_y))
        self.roi_max_y = max(self.roi_min_y, min(1.0, self.roi_max_y))
        self.candidate_min_hits = int(source_config.get("candidate_hits", 2) or 2)
        self.candidate_ttl = float(source_config.get("candidate_ttl", 8.0) or 8.0)
        self.pending_new_count = 0
        self.pending_new_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0}
        self.last_persist_time = 0.0
        
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

    def set_model(self, model_ref):
        self.model = model_ref
        self.model_names = _normalize_model_names(getattr(model_ref, "names", None)) if model_ref is not None else None
        names_lower = set()
        if self.model_names:
            names_lower = {str(n).strip().lower() for n in self.model_names if n is not None}
        self.is_custom_vehicle_model = ("mobil" in names_lower and "motor" in names_lower and len(names_lower) <= 3)
        self.yolo_classes = None if self.is_custom_vehicle_model else VEHICLE_CLASSES

    def open_capture(self):
        try:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;20000000|rw_timeout;20000000|reconnect;1|reconnect_streamed;1|reconnect_delay_max;2"
        except Exception:
            pass
        cap = None
        try:
            cap = cv2.VideoCapture(self.source_url)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            except Exception:
                pass
        except Exception:
            cap = None
        return cap

    def close_capture(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    def preprocess_frame(self, frame):
        if not self.preprocess:
            return frame
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            mean_v = float(v.mean())
            std_v = float(v.std())
            if mean_v >= 100 and mean_v <= 170 and std_v >= 45:
                return frame
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return out
        except Exception:
            return frame

    def _infer_worker(self, frame, now):
        try:
            frame_in = self.preprocess_frame(frame)
            results = []
            if self.model is not None:
                with g.model_lock:
                    try:
                        results = self.model(frame_in, conf=self.det_conf, iou=self.det_iou, classes=self.yolo_classes, verbose=False, imgsz=self.imgsz, augment=False, agnostic_nms=False)
                    except Exception as e:
                        print(f"[ERROR] Inference failed for {self.source_name}: {e}")

            rects = []
            rect_classes = []
            rect_confs = []
            datalake_batch = []
            frame_h, frame_w = frame.shape[:2]
            min_area = int(frame_w * frame_h * max(0.0, float(self.min_area_ratio)))

            if results:
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cls_id = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())

                        w = max(0, x2 - x1)
                        h = max(0, y2 - y1)
                        if w < self.min_wh or h < self.min_wh or (w * h) < min_area:
                            continue

                        area_ratio = (w * h) / float(frame_w * frame_h)
                        aspect = h / float(w + 1e-6)
                        cy = (y1 + y2) / 2.0
                        cy_norm = cy / float(frame_h + 1e-6)
                        if cy_norm < self.roi_min_y or cy_norm > self.roi_max_y:
                            continue

                        if area_ratio > 0.08 and conf < 0.60:
                            continue
                        if aspect > 2.8 and conf < 0.60:
                            continue
                        if y2 < int(frame_h * 0.45) and area_ratio > 0.02 and conf < 0.65:
                            continue

                        internal_class_id = CLASS_CAR
                        if self.is_custom_vehicle_model and self.model_names and cls_id < len(self.model_names):
                            n = str(self.model_names[cls_id]).strip().lower()
                            if n == "motor":
                                internal_class_id = CLASS_MOTORCYCLE
                            else:
                                internal_class_id = CLASS_CAR
                        else:
                            internal_class_id = CLASS_MAPPING.get(cls_id, CLASS_CAR)
                        rects.append((x1, y1, x2, y2))
                        rect_classes.append(internal_class_id)
                        rect_confs.append(conf)
                        datalake_batch.append({
                            'class_id': internal_class_id,
                            'conf': conf,
                            'box': [x1, y1, x2, y2]
                        })

            if datalake_batch:
                self.log_to_datalake(datalake_batch, now)

            detections = []
            for rect, cls_id, conf in zip(rects, rect_classes, rect_confs):
                x1, y1, x2, y2 = rect
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections.append({"bbox": rect, "cls": cls_id, "centroid": (cx, cy), "conf": conf})

            current_count = len(detections)
            current_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0}
            for det in detections:
                current_class_counts[det["cls"]] += 1

            with self._data_lock:
                for tid in list(self.tracks.keys()):
                    t = self.tracks[tid]
                    t["misses"] = int(t.get("misses", 0)) + 1

                track_ids = list(self.tracks.keys())
                pairs = []
                for di, det in enumerate(detections):
                    for tid in track_ids:
                        t = self.tracks[tid]
                        t_cent = t.get("centroid")
                        if not t_cent:
                            continue
                        vx = float(t.get("vx", 0.0))
                        vy = float(t.get("vy", 0.0))
                        pred = (float(t_cent[0]) + vx, float(t_cent[1]) + vy)
                        dx = float(det["centroid"][0]) - pred[0]
                        dy = float(det["centroid"][1]) - pred[1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        dist_gate = max(30.0, float(self.track_dist_ratio) * float(min(frame_w, frame_h)))
                        if dist > dist_gate:
                            continue

                        iou = self.get_iou(det["bbox"], t["bbox"])
                        if iou < float(self.track_iou_threshold):
                            continue

                        cls_penalty = 0.12 if int(t.get("cls", det["cls"])) != int(det["cls"]) else 0.0
                        score = float(iou) - (dist / (dist_gate + 1e-6)) * 0.25 - cls_penalty
                        pairs.append((score, tid, di))

                pairs.sort(key=lambda x: x[0], reverse=True)
                used_tracks = set()
                used_dets = set()
                det_to_track = {}
                updated_tracks = set()

                for _, tid, di in pairs:
                    if tid in used_tracks or di in used_dets:
                        continue
                    used_tracks.add(tid)
                    used_dets.add(di)
                    det_to_track[di] = tid

                    t = self.tracks[tid]
                    t["prev_centroid"] = t.get("centroid")
                    t["centroid"] = detections[di]["centroid"]
                    t["bbox"] = detections[di]["bbox"]
                    votes = t.get("cls_votes")
                    if not isinstance(votes, dict):
                        votes = {CLASS_CAR: 0.0, CLASS_MOTORCYCLE: 0.0}
                        t["cls_votes"] = votes
                    prev_c = t.get("prev_centroid")
                    cur_c = t.get("centroid")
                    if prev_c and cur_c:
                        dx = float(cur_c[0]) - float(prev_c[0])
                        dy = float(cur_c[1]) - float(prev_c[1])
                        t["vx"] = float(t.get("vx", 0.0)) * 0.7 + dx * 0.3
                        t["vy"] = float(t.get("vy", 0.0)) * 0.7 + dy * 0.3
                    det_cls = int(detections[di]["cls"])
                    det_conf = float(detections[di].get("conf", 0.0))
                    votes[det_cls] = float(votes.get(det_cls, 0.0)) + max(0.0, det_conf)
                    t["cls"] = CLASS_CAR if votes.get(CLASS_CAR, 0) >= votes.get(CLASS_MOTORCYCLE, 0) else CLASS_MOTORCYCLE
                    t["last_seen"] = now
                    t["misses"] = 0
                    updated_tracks.add(tid)

                for di, det in enumerate(detections):
                    if di in used_dets:
                        continue

                    best_key = None
                    best_iou = 0.0
                    for k, cand in self.track_candidates.items():
                        if (now - float(cand.get("last_seen", 0.0))) > float(self.candidate_ttl):
                            continue
                        iou = self.get_iou(det["bbox"], cand.get("bbox"))
                        if iou > best_iou:
                            best_iou = iou
                            best_key = k

                    if best_key is None or best_iou < 0.2:
                        best_key = f"c{int(det['centroid'][0]//60)}_{int(det['centroid'][1]//60)}_{det['cls']}"

                    cand = self.track_candidates.get(best_key)
                    if not isinstance(cand, dict):
                        cand = {"hits": 0}
                        self.track_candidates[best_key] = cand

                    cand["hits"] = int(cand.get("hits", 0)) + 1
                    cand["bbox"] = det["bbox"]
                    cand["cls"] = det["cls"]
                    cand["conf"] = float(det.get("conf", 0.0))
                    cand["centroid"] = det["centroid"]
                    cand["last_seen"] = now

                    if cand["hits"] < int(self.candidate_min_hits):
                        continue

                    tid = self.next_track_id
                    self.next_track_id += 1
                    det_to_track[di] = tid
                    self.tracks[tid] = {
                        "bbox": det["bbox"],
                        "cls": det["cls"],
                        "cls_votes": {CLASS_CAR: float(det.get("conf", 0.0)) if det["cls"] == CLASS_CAR else 0.0, CLASS_MOTORCYCLE: float(det.get("conf", 0.0)) if det["cls"] == CLASS_MOTORCYCLE else 0.0},
                        "centroid": det["centroid"],
                        "prev_centroid": det["centroid"],
                        "last_seen": now,
                        "counted": False,
                        "vx": 0.0,
                        "vy": 0.0,
                        "misses": 0
                    }
                    updated_tracks.add(tid)
                    try:
                        del self.track_candidates[best_key]
                    except Exception:
                        pass

                stale_cands = [k for k, cand in self.track_candidates.items() if (now - float(cand.get("last_seen", 0.0))) > float(self.candidate_ttl)]
                for k in stale_cands:
                    del self.track_candidates[k]

                stale = [tid for tid, t in self.tracks.items() if int(t.get("misses", 0)) > int(self.max_track_misses)]
                for tid in stale:
                    del self.tracks[tid]

                line_type = str(self.count_line.get("type", "horizontal")).lower()
                pos = self.count_line.get("pos", 0.6)
                try:
                    pos = float(pos)
                except Exception:
                    pos = 0.6
                pos = max(0.05, min(0.95, pos))
                line_value = int(frame_h * pos) if line_type != "vertical" else int(frame_w * pos)

                new_rects_count = 0
                new_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0}

                for tid in list(updated_tracks):
                    t = self.tracks.get(tid)
                    if not t or t.get("counted") is True:
                        continue
                    prev_c = t.get("prev_centroid")
                    cur_c = t.get("centroid")
                    if not prev_c or not cur_c:
                        continue

                    if line_type == "vertical":
                        crossed = (prev_c[0] < line_value <= cur_c[0]) or (prev_c[0] > line_value >= cur_c[0])
                    else:
                        crossed = (prev_c[1] < line_value <= cur_c[1]) or (prev_c[1] > line_value >= cur_c[1])

                    if crossed:
                        t["counted"] = True
                        new_rects_count += 1
                        new_class_counts[t["cls"]] += 1

                self.last_overlay = []
                for di in range(len(detections)):
                    tid = det_to_track.get(di)
                    if tid is None:
                        continue
                    t = self.tracks.get(tid)
                    cls_id = int(t.get("cls", detections[di]["cls"])) if t else int(detections[di]["cls"])
                    self.last_overlay.append((detections[di]["bbox"], cls_id, tid))
                self.latest_detections = {
                    "ts": now,
                    "line": {"type": line_type, "pos": pos},
                    "items": [
                        {
                            "x1": detections[di]["bbox"][0] / frame_w,
                            "y1": detections[di]["bbox"][1] / frame_h,
                            "x2": detections[di]["bbox"][2] / frame_w,
                            "y2": detections[di]["bbox"][3] / frame_h,
                            "cls": int(self.tracks.get(det_to_track.get(di), {}).get("cls", detections[di]["cls"])) if det_to_track.get(di) is not None else int(detections[di]["cls"]),
                            "track_id": det_to_track.get(di),
                        }
                        for di in range(len(detections))
                    ]
                }

            stats = g.global_stats[self.source_id]
            tracked_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0}
            with self._data_lock:
                for t in self.tracks.values():
                    if int(t.get("misses", 0)) == 0:
                        tracked_class_counts[int(t.get("cls", CLASS_CAR))] += 1
            stats["current_count"] = int(tracked_class_counts[CLASS_CAR] + tracked_class_counts[CLASS_MOTORCYCLE])
            stats["current_class_counts"] = {str(CLASS_CAR): int(tracked_class_counts[CLASS_CAR]), str(CLASS_MOTORCYCLE): int(tracked_class_counts[CLASS_MOTORCYCLE])}
            stats["accumulated_count"] += new_rects_count
            stats["accumulated_class_counts"][str(CLASS_CAR)] += new_class_counts[CLASS_CAR]
            stats["accumulated_class_counts"][str(CLASS_MOTORCYCLE)] += new_class_counts[CLASS_MOTORCYCLE]
            self.pending_new_count += int(new_rects_count)
            self.pending_new_class_counts[CLASS_CAR] += int(new_class_counts[CLASS_CAR])
            self.pending_new_class_counts[CLASS_MOTORCYCLE] += int(new_class_counts[CLASS_MOTORCYCLE])

            if (now - float(self.last_persist_time)) >= float(PROCESS_INTERVAL):
                self.last_persist_time = now
                flush_new_count = int(self.pending_new_count)
                flush_new_cars = int(self.pending_new_class_counts[CLASS_CAR])
                flush_new_motors = int(self.pending_new_class_counts[CLASS_MOTORCYCLE])
                self.pending_new_count = 0
                self.pending_new_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0}

                stats["history"].append({
                    "ts": now,
                    "count": current_count,
                    "cars": current_class_counts[CLASS_CAR],
                    "motors": current_class_counts[CLASS_MOTORCYCLE],
                    "new_count": flush_new_count,
                    "new_cars": flush_new_cars,
                    "new_motors": flush_new_motors
                })

                try:
                    insert_history_batch([(
                        self.source_id,
                        now,
                        current_count,
                        current_class_counts[CLASS_CAR],
                        current_class_counts[CLASS_MOTORCYCLE],
                        flush_new_count,
                        flush_new_cars,
                        flush_new_motors
                    )])
                except Exception as e:
                    print(f"[{self.source_name}] DB Error: {e}")

                if now - self.last_save_time > 60:
                    save_stats()
                    self.last_save_time = now
        finally:
            with self._infer_state_lock:
                self._infer_running = False

    def _try_start_inference(self, frame, now):
        if self.model is None:
            return False
        with self._infer_state_lock:
            if self._infer_running:
                return False
            self._infer_running = True
        threading.Thread(target=self._infer_worker, args=(frame, now), daemon=True).start()
        return True

    def _update_predicted_overlay(self, now, frame_w, frame_h):
        with self._data_lock:
            line_type = str(self.count_line.get("type", "horizontal")).lower()
            pos = self.count_line.get("pos", 0.6)
            try:
                pos = float(pos)
            except Exception:
                pos = 0.6
            pos = max(0.05, min(0.95, pos))

            overlay = []
            items = []
            for tid, t in self.tracks.items():
                if int(t.get("misses", 0)) > int(self.max_track_misses):
                    continue
                bbox = t.get("bbox")
                cent = t.get("centroid")
                if not bbox or not cent:
                    continue

                dt = max(0.0, float(now) - float(t.get("last_seen", now)))
                dt = min(dt, 0.5)
                max_disp = max(20.0, float(min(frame_w, frame_h)) * 0.06)
                dx = float(t.get("vx", 0.0)) * dt
                dy = float(t.get("vy", 0.0)) * dt
                dx = max(-max_disp, min(max_disp, dx))
                dy = max(-max_disp, min(max_disp, dy))

                x1 = int(round(bbox[0] + dx))
                y1 = int(round(bbox[1] + dy))
                x2 = int(round(bbox[2] + dx))
                y2 = int(round(bbox[3] + dy))

                x1 = max(0, min(frame_w - 1, x1))
                y1 = max(0, min(frame_h - 1, y1))
                x2 = max(0, min(frame_w - 1, x2))
                y2 = max(0, min(frame_h - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                cls_id = int(t.get("cls", CLASS_CAR))
                overlay.append(((x1, y1, x2, y2), cls_id, tid))
                items.append({
                    "x1": x1 / frame_w,
                    "y1": y1 / frame_h,
                    "x2": x2 / frame_w,
                    "y2": y2 / frame_h,
                    "cls": cls_id,
                    "track_id": tid
                })

            self.last_overlay = overlay
            self.latest_detections = {"ts": now, "line": {"type": line_type, "pos": pos}, "items": items}

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
            
            active_id = getattr(g, "ACTIVE_CAMERA_ID", None)
            is_active = (self.source_id == active_id) if active_id else (self.source_url == g.VIDEO_SOURCE)
            if not is_active and self.cap is not None:
                self.close_capture()
            if is_active:
                if self.cap is None or not self.cap.isOpened():
                    self.close_capture()
                    self.cap = self.open_capture()
                    if self.cap and self.cap.isOpened():
                        for _ in range(5):
                            self.cap.read()
                    else:
                        if self.source_id in g.global_stats:
                            g.global_stats[self.source_id]["status"] = "offline"
                            g.global_stats[self.source_id]["last_update"] = time.time()
                        time.sleep(0.25)
                        continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.close_capture()
                    if self.source_id in g.global_stats:
                        g.global_stats[self.source_id]["status"] = "offline"
                        g.global_stats[self.source_id]["last_update"] = time.time()
                    time.sleep(0.25)
                    continue

                if self.source_id in g.global_stats:
                    g.global_stats[self.source_id]["status"] = "online"
                    g.global_stats[self.source_id]["last_update"] = time.time()

                now = time.time()
                self._update_predicted_overlay(now, frame.shape[1], frame.shape[0])
                if now - self.last_infer_time >= float(self.active_infer_interval):
                    started = self._try_start_inference(frame.copy(), now)
                    if started:
                        self.last_infer_time = now

                stats = g.global_stats.get(self.source_id, {})
                with self._data_lock:
                    overlay = list(self.last_overlay)
                    line_type = str(self.count_line.get("type", "horizontal")).lower()
                    pos = self.count_line.get("pos", 0.6)
                try:
                    pos = float(pos)
                except Exception:
                    pos = 0.6
                pos = max(0.05, min(0.95, pos))
                line_value = int(frame.shape[0] * pos) if line_type != "vertical" else int(frame.shape[1] * pos)

                if line_type == "vertical":
                    cv2.line(frame, (line_value, 0), (line_value, frame.shape[0]), (255, 255, 0), 2)
                else:
                    cv2.line(frame, (0, line_value), (frame.shape[1], line_value), (255, 255, 0), 2)

                for (rect, cls_id, track_id) in overlay:
                    (x1, y1, x2, y2) = rect
                    color = (0, 255, 0) if cls_id == CLASS_CAR else (255, 0, 0)
                    label = "Car" if cls_id == CLASS_CAR else "Motor"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    txt = f"{label} {track_id}" if track_id is not None else label
                    cv2.putText(frame, txt, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.putText(frame, f"CAM: {self.source_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Total: {stats.get('accumulated_count', 0)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "desavitho", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                with g.lock:
                    g.outputFrame = frame.copy()

                time.sleep(0.07)
                continue

            # 1. Connect & Snapshot
            # Set timeout for FFmpeg (20 seconds - increased for slow streams)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;20000000|rw_timeout;20000000|reconnect;1|reconnect_streamed;1|reconnect_delay_max;2"
            
            cap = None
            try:
                cap = cv2.VideoCapture(self.source_url)
            except Exception as e:
                print(f"[WARN] {self.source_name}: VideoCapture init failed: {e}")

            frame = None
            success = False
            
            if cap and cap.isOpened():
                # Burst read to clear buffer and find keyframe
                # Increased to max 2 seconds to handle stream startup artifacts
                start_read = time.time()
                while (time.time() - start_read) < 2.0:
                    ret, tmp_frame = cap.read()
                    if ret:
                        frame = tmp_frame
                        success = True
                        # If we got a good frame, we can break early, 
                        # but reading a few more clears the buffer better.
                        # Let's read at least 3 good frames or until timeout
                        if (time.time() - start_read) > 0.5: 
                            break
                    else:
                        time.sleep(0.05)
                cap.release()
            else:
                if cap: cap.release()
                print(f"[WARN] {self.source_name}: Connection failed or stream closed.")
            
            # Update status in global stats
            if self.source_id in g.global_stats:
                g.global_stats[self.source_id]["status"] = "online" if success else "offline"
                g.global_stats[self.source_id]["last_update"] = time.time()

            if success and frame is not None:
                self._infer_worker(frame, time.time())

            # Sleep
            time.sleep(PROCESS_INTERVAL)

    def stop(self):
        self.running = False
        self.close_capture()

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
        with g.lock:
            g.outputFrame = None

        placeholder = None
        if np is not None:
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank, "Connecting to stream...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            ok, enc = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                placeholder = enc.tobytes()
        
        while True:
            with g.lock:
                frame = None if g.outputFrame is None else g.outputFrame.copy()
            if frame is None:
                if placeholder is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                time.sleep(0.2)
                continue
            
            (flag, encodedImage) = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not flag:
                time.sleep(0.05)
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            
            # Throttle to avoid busy loop, match process interval roughly
            time.sleep(0.1)

def start_camera_agents():
    if g.yolo_model_instance is None and YOLO is not None:
        print("[INFO] Loading YOLOv8 model (Shared)...")
        model_path = get_yolo_model_path()
        g.yolo_model_instance = YOLO(model_path)
        try:
            g.yolo_model_instance.fuse()
        except Exception:
            pass
        g.yolo_model_path = model_path
        g.yolo_model_loaded_ts = time.time()
        print("[INFO] Model Loaded.")
    
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

def reload_yolo_model():
    if YOLO is None:
        return {"ok": False, "message": "Ultralytics YOLO tidak tersedia"}
    model_path = get_yolo_model_path()
    with g.model_lock:
        model = YOLO(model_path)
        try:
            model.fuse()
        except Exception:
            pass
        g.yolo_model_instance = model
        g.yolo_model_path = model_path
        g.yolo_model_loaded_ts = time.time()
        for agent in g.camera_agents.values():
            try:
                agent.set_model(model)
            except Exception:
                agent.model = model
    return {"ok": True, "model_path": model_path}
