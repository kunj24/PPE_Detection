"""
detection_utils.py – YOLOv8 PPE detection + smart 3‑second violation logic.

The PPEDetector class wraps the YOLO model and a per‑(worker, violation_type)
timer so that a violation is only written to the DB when it has been detected
**continuously for N seconds** (default 3).
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

        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.threshold = threshold_secs
        self.camera = camera_location

        # {(employee_id, violation_type): first_seen_timestamp}
        self._tracker: Dict[Tuple[str, str], float] = {}

        # Cache of known face encodings – refreshed on demand
        self._known_faces = database_utils.get_worker_face_encodings()

    # ── public helpers ──────────────────────────────────────────
    def reload_faces(self):
        self._known_faces = database_utils.get_worker_face_encodings()

    # ── duplicate box suppression (all classes) ─────────────────
    def _suppress_duplicate_boxes(self, results):
        """
        NMS misses duplicate boxes when they don't overlap much
        (e.g. one large background Person box + one tight body box,
        or two NO-Safety-Vest boxes at different scales for the same
        torso).  For every pair of boxes with the SAME class we check
        *centre containment*: if the centre of either box falls inside
        the other, they refer to the same object – keep only the one
        with the higher confidence.
        """
        boxes = results.boxes
        if boxes is None or len(boxes) <= 1:
            return results

        xyxy    = boxes.xyxy.cpu().numpy()               # (N,4)
        confs   = boxes.conf.cpu().numpy()               # (N,)
        cls_ids = boxes.cls.cpu().numpy().astype(int)    # (N,)
        n       = len(cls_ids)

        to_remove: set = set()
        for a in range(n):
            for b in range(a + 1, n):
                if cls_ids[a] != cls_ids[b]:
                    continue                              # different class – skip
                if a in to_remove or b in to_remove:
                    continue                              # already eliminated

                x1a, y1a, x2a, y2a = xyxy[a]
                x1b, y1b, x2b, y2b = xyxy[b]

                cxa, cya = (x1a + x2a) / 2, (y1a + y2a) / 2
                cxb, cyb = (x1b + x2b) / 2, (y1b + y2b) / 2

                centre_a_in_b = x1b <= cxa <= x2b and y1b <= cya <= y2b
                centre_b_in_a = x1a <= cxb <= x2a and y1a <= cyb <= y2a

                if centre_a_in_b or centre_b_in_a:
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
        Run detection + face identification on a single BGR frame.

        cached_faces: if provided, skip face detection and use this list
                      (allows caller to run face detection at a lower rate
                      while still drawing labels every frame).
        prev_faces: previous frame's faces for spatial tracking

        Returns (annotated_frame, stats_dict, faces_list).
        """
        now = time.time()

        # 1) YOLO detection
        results = self.model(frame, iou=0.4, conf=0.45, verbose=False)[0]
        results = self._suppress_duplicate_boxes(results)
        annotated = results.plot()

        # 2) Face identification
        if cached_faces is not None:
            # Use caller-supplied cache – still draw labels every frame
            faces = cached_faces
        elif do_faces and self._known_faces:
            faces = face_utils.identify_faces(
                frame, self._known_faces, prev_faces=prev_faces
            )
        else:
            faces = []

        # Always draw cached/fresh face labels on every frame
        if faces:
            annotated = face_utils.draw_face_labels(annotated, faces)

        # 3) Walk detections and apply violation timing rule
        violations_in_frame = 0
        violations_logged = 0
        active_keys: set = set()

        def _pick_face_for_box(x1: float, y1: float, x2: float, y2: float) -> Dict:
            """Pick the best matching face for a given detection box."""
            if not faces:
                return {}
            # Prefer faces whose centre lies inside the detection box
            candidates = []
            for f in faces:
                top, right, bottom, left = f["box"]
                cx = (left + right) / 2.0
                cy = (top + bottom) / 2.0
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    candidates.append(f)
            if not candidates:
                # Fallback: choose closest face centre to box centre
                bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                def _dist2(f: Dict) -> float:
                    top, right, bottom, left = f["box"]
                    cx = (left + right) / 2.0
                    cy = (top + bottom) / 2.0
                    return (cx - bx) ** 2 + (cy - by) ** 2
                candidates = sorted(faces, key=_dist2)[:3]

            known = [f for f in candidates if f.get("employee_id") != "Unknown"]
            if known:
                return max(known, key=lambda f: f.get("confidence", 0.0))
            return {}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id]
            conf = float(box.conf[0])

            if "NO" not in cls_name.upper():
                continue                          # not a violation class
            violations_in_frame += 1

            # Determine worker identity
            eid, wname, dept = "Unknown", "Unknown", "Unknown"

            # Priority 1: face that best corresponds to THIS violation box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best_face = _pick_face_for_box(x1, y1, x2, y2)
            if best_face and best_face.get("employee_id") != "Unknown":
                eid = best_face["employee_id"]
                wname = best_face["name"]
                dept = best_face["department"]
            # Priority 2: manual worker override (fallback)
            elif manual_worker and manual_worker.get("employee_id") != "Unknown":
                eid   = manual_worker["employee_id"]
                wname = manual_worker["name"]
                dept  = manual_worker["department"]

            key = (eid, cls_name)
            active_keys.add(key)

            # Instant logging if threshold is 0
            should_log = False
            if self.threshold == 0.0:
                should_log = True
            elif key not in self._tracker:
                self._tracker[key] = now          # start tracking
            elif now - self._tracker[key] >= self.threshold:
                should_log = True

            if should_log:
                # ── violation confirmed → everything goes to background thread ──
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snap_name = f"violation_{eid}_{ts}.jpg"
                snap_path = os.path.join("database", "violations", snap_name)
                frame_copy = frame.copy()   # copy so the frame buffer isn't reused

                # Disk write + Supabase upload + DB insert all in daemon thread
                # → NEVER blocks the webcam frame loop
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
                        print(f"[DB] background log error: {e}")

                threading.Thread(target=_log_bg, daemon=True).start()

                violations_logged += 1
                if key in self._tracker:
                    del self._tracker[key]        # reset timer

        # 4) Purge stale tracker entries (violation disappeared from frame)
        stale = [k for k in self._tracker if k not in active_keys]
        for k in stale:
            del self._tracker[k]

        stats = dict(
            detections=len(results.boxes),
            violations_in_frame=violations_in_frame,
            violations_logged=violations_logged,
            workers_identified=len([f for f in faces if f['employee_id'] != 'Unknown']),
            pending_tracks=len(self._tracker),
            faces=faces,          # return so caller can cache
        )
        return annotated, stats
