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
    gray = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
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


def identify_faces(frame_bgr: np.ndarray,
                   known: Dict[str, tuple],
                   threshold: float = 0.55) -> List[Dict]:
    """
    Detect faces in frame, compare against known workers.

    Parameters
    ----------
    known : {employee_id: (name, department, numpy_encoding)}
    threshold : minimum similarity to consider a match (0.0 - 1.0)

    Returns list of dicts with employee_id, name, department, box.
    """
    if not known:
        return []

    try:
        face_rects = _detect_faces_in_image(frame_bgr)
        if not face_rects:
            return []

        results: List[Dict] = []
        for rect in face_rects:
            x, y, w, h = rect
            face_crop = _crop_face(frame_bgr, rect)
            face_enc = _compute_face_encoding(face_crop)

            best_id, best_name, best_dept = "Unknown", "Unknown", "Unknown"
            best_score = threshold

            for eid, (name, dept, known_enc) in known.items():
                score = _compare_encodings(face_enc, known_enc)
                if score > best_score:
                    best_score = score
                    best_id, best_name, best_dept = eid, name, dept

            # Convert (x,y,w,h) to (top,right,bottom,left) format
            top, right, bottom, left = y, x + w, y + h, x

            results.append(dict(
                employee_id=best_id, name=best_name,
                department=best_dept, box=(top, right, bottom, left),
                confidence=best_score,
            ))
        return results
    except Exception as e:
        print(f"[face_utils] identify error: {e}")
        return []


def draw_face_labels(frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
    """Draw coloured boxes + name labels on the frame."""
    for f in faces:
        top, right, bottom, left = f["box"]
        known = f["employee_id"] != "Unknown"
        colour = (0, 200, 0) if known else (0, 0, 255)
        label = f'{f["name"]} ({f["employee_id"]})' if known else "Unknown"

        cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
        cv2.rectangle(frame, (left, bottom - 28), (right, bottom), colour, cv2.FILLED)
        cv2.putText(frame, label, (left + 4, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame
