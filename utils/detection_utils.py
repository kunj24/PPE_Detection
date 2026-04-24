"""
detection_utils.py – YOLOv8 PPE detection + smart violation logic.

Key change: stats dict now includes 'violations_this_frame' (list of
violation class names newly confirmed this frame) so the alarm system
can speak the right message for each detected violation type.
"""

import os
import time
import cv2
import numpy as np
import torch
import threading
from datetime import datetime
from typing import Dict, List, Tuple
from ultralytics import YOLO

from utils import database_utils, face_utils


class PPEDetector:
    """Stateful detector: YOLO + face recognition + smart violation timer."""

    def __init__(self, model_path: str = "best.pt",
                 threshold_secs: float = 3.0,
                 camera_location: str = "Main Camera"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model       = YOLO(model_path)
        self.class_names = self.model.names
        self.threshold   = threshold_secs
        self.camera      = camera_location

        # {(employee_id, violation_type): first_seen_timestamp}
        self._tracker: Dict[Tuple[str, str], float] = {}
        self._known_faces = database_utils.get_worker_face_encodings()

    def reload_faces(self):
        self._known_faces = database_utils.get_worker_face_encodings()

    # ── duplicate box suppression ───────────────────────────────
    def _suppress_duplicate_boxes(self, results):
        boxes = results.boxes
        if boxes is None or len(boxes) <= 1:
            return results

        xyxy    = boxes.xyxy.cpu().numpy()
        confs   = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        n       = len(cls_ids)
        to_remove: set = set()

        for a in range(n):
            for b in range(a + 1, n):
                if cls_ids[a] != cls_ids[b]:
                    continue
                if a in to_remove or b in to_remove:
                    continue
                x1a, y1a, x2a, y2a = xyxy[a]
                x1b, y1b, x2b, y2b = xyxy[b]
                cxa, cya = (x1a+x2a)/2, (y1a+y2a)/2
                cxb, cyb = (x1b+x2b)/2, (y1b+y2b)/2
                if (x1b<=cxa<=x2b and y1b<=cya<=y2b) or \
                   (x1a<=cxb<=x2a and y1a<=cyb<=y2a):
                    if confs[a] >= confs[b]:
                        to_remove.add(b)
                    else:
                        to_remove.add(a)

        if not to_remove:
            return results
        keep   = [i for i in range(n) if i not in to_remove]
        keep_t = torch.tensor(keep, dtype=torch.long)
        results.boxes = results.boxes[keep_t]
        return results

    # ── main entry point ────────────────────────────────────────
    def process_frame(self, frame: np.ndarray, *,
                      do_faces: bool = True,
                      cached_faces: List[Dict] = None,
                      manual_worker: Dict = None,
                      prev_faces: List[Dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Run YOLO + face ID on one BGR frame.

        Returns (annotated_frame, stats_dict).
        stats_dict keys:
          detections, violations_in_frame, violations_logged,
          workers_identified, pending_tracks, faces,
          violations_this_frame  ← NEW: list of newly-logged class names
        """
        now = time.time()

        # 1) YOLO
        results  = self.model(frame, iou=0.4, conf=0.45, verbose=False)[0]
        results  = self._suppress_duplicate_boxes(results)
        annotated = results.plot()

        # 2) Face ID
        if cached_faces is not None:
            faces = cached_faces
        elif do_faces and self._known_faces:
            faces = face_utils.identify_faces(
                frame, self._known_faces, prev_faces=prev_faces)
        else:
            faces = []

        if faces:
            annotated = face_utils.draw_face_labels(annotated, faces)

        # 3) Violation timing
        violations_in_frame   = 0
        violations_logged     = 0
        violations_this_frame: List[str] = []   # ← NEW
        active_keys: set      = set()

        def _pick_face(x1, y1, x2, y2):
            if not faces:
                return {}
            candidates = [f for f in faces
                          if x1 <= (f["box"][3]+f["box"][1])/2 <= x2
                          and y1 <= (f["box"][0]+f["box"][2])/2 <= y2]
            if not candidates:
                bx, by = (x1+x2)/2, (y1+y2)/2
                candidates = sorted(
                    faces,
                    key=lambda f: ((f["box"][3]+f["box"][1])/2-bx)**2
                               + ((f["box"][0]+f["box"][2])/2-by)**2
                )[:3]
            known = [f for f in candidates if f.get("employee_id") != "Unknown"]
            return max(known, key=lambda f: f.get("confidence", 0)) if known else {}

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = self.class_names[cls_id]
            conf     = float(box.conf[0])

            if "NO" not in cls_name.upper():
                continue
            violations_in_frame += 1

            eid, wname, dept = "Unknown", "Unknown", "Unknown"
            x1, y1, x2, y2  = box.xyxy[0].tolist()
            best_face = _pick_face(x1, y1, x2, y2)
            if best_face and best_face.get("employee_id") != "Unknown":
                eid, wname, dept = (best_face["employee_id"],
                                    best_face["name"],
                                    best_face["department"])
            elif manual_worker and manual_worker.get("employee_id") != "Unknown":
                eid, wname, dept = (manual_worker["employee_id"],
                                    manual_worker["name"],
                                    manual_worker["department"])

            key = (eid, cls_name)
            active_keys.add(key)

            should_log = False
            if self.threshold == 0.0:
                should_log = True
            elif key not in self._tracker:
                self._tracker[key] = now
            elif now - self._tracker[key] >= self.threshold:
                should_log = True

            if should_log:
                ts         = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snap_path  = os.path.join("database", "violations",
                                          f"violation_{eid}_{ts}.jpg")
                frame_copy = frame.copy()

                def _log_bg(eid=eid, wname=wname, dept=dept,
                             cls_name=cls_name, conf=conf,
                             snap_path=snap_path, camera=self.camera,
                             frame_copy=frame_copy):
                    try:
                        os.makedirs(os.path.dirname(snap_path), exist_ok=True)
                        cv2.imwrite(snap_path, frame_copy)
                        severity = database_utils.calc_severity(eid)
                        database_utils.log_violation(
                            employee_id=eid, name=wname, department=dept,
                            violation_type=cls_name, confidence=conf,
                            snapshot_path=snap_path,
                            camera_location=camera,
                            severity=severity,
                        )
                    except Exception as e:
                        print(f"[DB] log error: {e}")

                threading.Thread(target=_log_bg, daemon=True).start()

                violations_logged += 1
                violations_this_frame.append(cls_name)   # ← NEW
                if key in self._tracker:
                    del self._tracker[key]

        # 4) Purge stale trackers
        for k in [k for k in self._tracker if k not in active_keys]:
            del self._tracker[k]

        return annotated, dict(
            detections=len(results.boxes),
            violations_in_frame=violations_in_frame,
            violations_logged=violations_logged,
            workers_identified=len([f for f in faces
                                    if f["employee_id"] != "Unknown"]),
            pending_tracks=len(self._tracker),
            faces=faces,
            violations_this_frame=violations_this_frame,   # ← NEW
        )