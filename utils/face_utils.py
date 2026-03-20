"""
face_utils.py - Face detection and matching using OpenCV (no extra libraries needed).

Uses OpenCV's Haar cascade for face detection and histogram comparison
for face matching against registered worker photos.

No dlib or face_recognition needed - works with just opencv-python.
"""

import cv2
import numpy as np
import pickle
import os
from typing import Optional, Tuple, List, Dict

# Face detection is ALWAYS available via OpenCV
FACE_REC_AVAILABLE = True

_HAAR_PATH = os.path.join(
    os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"
)
_face_cascade = cv2.CascadeClassifier(_HAAR_PATH)

# Standard size for face comparison
_FACE_SIZE = (100, 100)


def _detect_faces_in_image(bgr_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces and return list of (x, y, w, h) rects."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    # CLAHE gives much better local contrast than plain equalizeHist,
    # especially under uneven / harsh lighting on a construction site.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # minNeighbors=7 balances detection rate vs false positives;
    # minSize=(80,80) catches faces a bit further from the camera.
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=7, minSize=(80, 80)
    )
    return list(faces) if len(faces) > 0 else []


def _crop_face(bgr_image: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop and resize a face region to standard size."""
    x, y, w, h = rect
    # Add padding around face
    pad = int(0.2 * max(w, h))
    y1 = max(0, y - pad)
    y2 = min(bgr_image.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(bgr_image.shape[1], x + w + pad)
    face_crop = bgr_image[y1:y2, x1:x2]
    return cv2.resize(face_crop, _FACE_SIZE)


def _compute_face_encoding(face_crop: np.ndarray) -> np.ndarray:
    """
    Compute a face 'encoding' using color histogram + structure features.
    Returns a normalized feature vector.
    """
    # Color histogram (in HSV space for better color matching)
    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

    # LBP-like texture from grayscale
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    # Divide face into 4x4 grid and get histogram per region
    h, w = gray.shape
    grid_hists = []
    for gy in range(4):
        for gx in range(4):
            cell = gray[gy*h//4:(gy+1)*h//4, gx*w//4:(gx+1)*w//4]
            cell_hist = cv2.calcHist([cell], [0], None, [16], [0, 256])
            grid_hists.append(cell_hist)

    # Concatenate all features into one vector
    encoding = np.concatenate(
        [hist_h, hist_s, hist_v] + grid_hists
    ).flatten().astype(np.float32)

    # Normalize
    norm = np.linalg.norm(encoding)
    if norm > 0:
        encoding = encoding / norm
    return encoding


def _compare_encodings(enc1: np.ndarray, enc2: np.ndarray) -> float:
    """
    Compare two face encodings. Returns similarity score 0.0 - 1.0.
    Higher = more similar.
    """
    # Cosine similarity
    dot = np.dot(enc1, enc2)
    return float(max(0.0, min(1.0, dot)))


# ---- Public API ----

def generate_encoding_from_file(image_path: str) -> Tuple[Optional[bytes], str]:
    """Load image, detect face, return (pickled_encoding, message)."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, "Cannot read image file."

        faces = _detect_faces_in_image(img)
        if len(faces) == 0:
            # If no face detected, use the whole image as a fallback
            face_crop = cv2.resize(img, _FACE_SIZE)
        else:
            if len(faces) > 1:
                # Use the largest face
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            face_crop = _crop_face(img, faces[0])

        encoding = _compute_face_encoding(face_crop)
        return pickle.dumps(encoding), "Face encoding generated successfully."
    except Exception as e:
        return None, f"Error generating encoding: {e}"


def generate_encoding_from_array(bgr_array: np.ndarray) -> Tuple[Optional[bytes], str]:
    """Same as above but from a BGR numpy array (camera capture)."""
    try:
        faces = _detect_faces_in_image(bgr_array)
        if len(faces) == 0:
            face_crop = cv2.resize(bgr_array, _FACE_SIZE)
        else:
            if len(faces) > 1:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            face_crop = _crop_face(bgr_array, faces[0])

        encoding = _compute_face_encoding(face_crop)
        return pickle.dumps(encoding), "Face encoding generated."
    except Exception as e:
        return None, f"Error: {e}"


def _suppress_duplicate_faces(rects: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
    """
    Remove duplicate face rects where one rect's centre lies inside another.
    Keeps the larger rect (bigger area = better detection).
    """
    if len(rects) <= 1:
        return rects
    rects_sorted = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)
    kept = []
    for r in rects_sorted:
        x, y, w, h = r
        cx, cy = x + w // 2, y + h // 2
        dominated = False
        for kx, ky, kw, kh in kept:
            if kx <= cx <= kx + kw and ky <= cy <= ky + kh:
                dominated = True
                break
        if not dominated:
            kept.append(r)
    return kept


def _calculate_iou(box1: Tuple[int, int, int, int],
                   box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate IoU (Intersection over Union) between two boxes.
    Boxes are in (top, right, bottom, left) format.
    Returns a value between 0.0 (no overlap) and 1.0 (perfect match).
    """
    top1, right1, bottom1, left1 = box1
    top2, right2, bottom2, left2 = box2

    # Calculate intersection area
    x_left = max(left1, left2)
    y_top = max(top1, top2)
    x_right = min(right1, right2)
    y_bottom = min(bottom1, bottom2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def identify_faces(frame_bgr: np.ndarray,
                   known: Dict[str, tuple],
                   threshold: float = 0.62,
                   min_margin: float = 0.10,
                   enforce_unique: bool = True,
                   prev_faces: List[Dict] = None) -> List[Dict]:
    """
    Detect faces in frame, compare against known workers.

    Parameters
    ----------
    known : {employee_id: (name, department, numpy_encoding)}
    threshold : minimum similarity to consider a match (0.0 - 1.0)
    min_margin : best_score - second_best_score must be >= this value
                 to accept the match. Helps reduce false matches.
    enforce_unique : if True, prevents the same employee_id being assigned to
                     multiple faces in the same frame (keeps highest score).
    prev_faces : previous frame's face detections for spatial tracking

    Returns list of dicts with employee_id, name, department, box.
    """
    if not known:
        return []

    try:
        face_rects = _detect_faces_in_image(frame_bgr)
        if not face_rects:
            return []

        # Remove duplicate rects for the same face
        face_rects = _suppress_duplicate_faces(face_rects)

        results: List[Dict] = []

        # Track which previous faces have been matched
        prev_matched = set() if prev_faces else set()

        for rect in face_rects:
            x, y, w, h = rect
            # Convert (x,y,w,h) to (top,right,bottom,left) format
            top, right, bottom, left = y, x + w, y + h, x
            current_box = (top, right, bottom, left)

            best_id, best_name, best_dept = "Unknown", "Unknown", "Unknown"
            best_score = 0.0
            tracked = False

            # SPATIAL TRACKING: Try to match this face to a previous face by position
            if prev_faces:
                best_iou = 0.0
                best_prev_idx = -1

                for prev_idx, prev_face in enumerate(prev_faces):
                    if prev_idx in prev_matched:
                        continue  # already matched to another face

                    prev_box = prev_face["box"]
                    iou = _calculate_iou(current_box, prev_box)

                    # IoU > 0.3 means significant overlap (same person, likely)
                    if iou > 0.3 and iou > best_iou:
                        best_iou = iou
                        best_prev_idx = prev_idx

                # If we found a good spatial match, inherit the identity
                if best_prev_idx >= 0 and prev_faces[best_prev_idx]["employee_id"] != "Unknown":
                    best_id = prev_faces[best_prev_idx]["employee_id"]
                    best_name = prev_faces[best_prev_idx]["name"]
                    best_dept = prev_faces[best_prev_idx]["department"]
                    best_score = prev_faces[best_prev_idx].get("confidence", 0.0)
                    prev_matched.add(best_prev_idx)
                    tracked = True

            # FACE RECOGNITION: Only run if not tracked spatially
            if not tracked:
                face_crop = _crop_face(frame_bgr, rect)
                face_enc = _compute_face_encoding(face_crop)

                second_best = 0.0

                for eid, (name, dept, known_enc) in known.items():
                    score = _compare_encodings(face_enc, known_enc)
                    if score > best_score:
                        second_best = best_score
                        best_score = score
                        best_id, best_name, best_dept = eid, name, dept
                    elif score > second_best:
                        second_best = score

                # Accept match only if it's strong AND unambiguous
                if not (best_score >= threshold and (best_score - second_best) >= min_margin):
                    best_id, best_name, best_dept = "Unknown", "Unknown", "Unknown"
                    best_score = 0.0

            results.append(dict(
                employee_id=best_id, name=best_name,
                department=best_dept, box=current_box,
                confidence=best_score,
            ))

        # ENFORCE UNIQUENESS: Prevent the same employee_id on multiple faces
        if enforce_unique and len(results) > 1:
            # Group faces by employee_id
            eid_groups: Dict[str, List[int]] = {}
            for idx, r in enumerate(results):
                eid = r.get("employee_id", "Unknown")
                if eid == "Unknown":
                    continue
                if eid not in eid_groups:
                    eid_groups[eid] = []
                eid_groups[eid].append(idx)

            # For each duplicate employee_id, keep only the highest confidence
            for eid, indices in eid_groups.items():
                if len(indices) > 1:
                    # Find the best match by confidence
                    best_idx = max(indices, key=lambda i: results[i].get("confidence", 0.0))

                    # Mark all others as Unknown
                    for idx in indices:
                        if idx != best_idx:
                            results[idx]["employee_id"] = "Unknown"
                            results[idx]["name"] = "Unknown"
                            results[idx]["department"] = "Unknown"
                            results[idx]["confidence"] = 0.0

        return results
    except Exception as e:
        print(f"[face_utils] identify error: {e}")
        return []


def draw_face_labels(frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
    """Draw clean coloured boxes + name / ID labels on the frame."""
    for f in faces:
        top, right, bottom, left = f["box"]
        known   = f["employee_id"] != "Unknown"
        colour  = (0, 210, 0) if known else (0, 60, 220)   # green / red-ish
        conf    = f.get("confidence", 0.0)

        # Two-line label: name on top, ID + confidence below
        if known:
            line1 = f'{f["name"]}'
            line2 = f'{f["employee_id"]}  {conf*100:.0f}%'
        else:
            line1 = "Unknown"
            line2 = ""

        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness  = 1

        # Bounding box (thicker for visibility)
        cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)

        # Measure text to size the label background
        (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(line2, font, font_scale - 0.1, thickness)
        lbl_h  = h1 + (h2 + 6 if line2 else 0) + 12
        lbl_w  = max(w1, w2) + 10

        # Draw filled label background above the face box
        lbl_top = max(0, top - lbl_h - 4)
        overlay = frame.copy()
        cv2.rectangle(overlay, (left, lbl_top), (left + lbl_w, top), colour, cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)  # semi-transparent

        # Draw text
        cv2.putText(frame, line1,
                    (left + 5, lbl_top + h1 + 4),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        if line2:
            cv2.putText(frame, line2,
                        (left + 5, lbl_top + h1 + h2 + 10),
                        font, font_scale - 0.1, (220, 220, 220), thickness, cv2.LINE_AA)
    return frame
