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

    # ── main entry point ────────────────────────────────────────
    def process_frame(self, frame: np.ndarray, *,
                      do_faces: bool = True,
                      manual_worker: Dict = None) -> Tuple[np.ndarray, dict]:
        """
        Run detection + face identification on a single BGR frame.

        Face identification is automatic via OpenCV:
        1. Detect faces in the frame
        2. Compare against registered worker encodings
        3. If match found, use that worker's info for violation logs

        manual_worker: optional override dict (employee_id, name, department)

        Returns (annotated_frame, stats_dict).
        """
        now = time.time()

        # 1) YOLO detection
        results = self.model(frame)[0]
        annotated = results.plot()

        # 2) Face identification (automatic)
        faces: List[Dict] = []
        if do_faces and self._known_faces:
            faces = face_utils.identify_faces(frame, self._known_faces)
            annotated = face_utils.draw_face_labels(annotated, faces)

        # 3) Walk detections and apply violation timing rule
        violations_in_frame = 0
        violations_logged = 0
        active_keys: set = set()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id]
            conf = float(box.conf[0])

            if "NO" not in cls_name.upper():
                continue                          # not a violation class
            violations_in_frame += 1

            # Determine worker identity
            eid, wname, dept = "Unknown", "Unknown", "Unknown"

            # Priority 1: auto face recognition match (best known face)
            known_faces = [f for f in faces if f["employee_id"] != "Unknown"]
            if known_faces:
                # Use the face with highest confidence
                best_face = max(known_faces, key=lambda f: f.get("confidence", 0))
                eid   = best_face["employee_id"]
                wname = best_face["name"]
                dept  = best_face["department"]
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
                # ── violation confirmed → save snapshot + DB row ──
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snap_name = f"violation_{eid}_{ts}.jpg"
                snap_path = os.path.join("database", "violations", snap_name)
                cv2.imwrite(snap_path, frame)

                severity = database_utils.calc_severity(eid)
                database_utils.log_violation(
                    employee_id=eid, name=wname, department=dept,
                    violation_type=cls_name, confidence=conf,
                    snapshot_path=snap_path,
                    camera_location=self.camera,
                    severity=severity,
                )
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
            workers_identified=len(faces),
            pending_tracks=len(self._tracker),
        )
        return annotated, stats
