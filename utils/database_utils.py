"""
database_utils.py – Supabase (PostgreSQL) backend for PPE detection system.
Drop-in replacement for the original SQLite version — all function signatures
are identical so no other file in the project needs to change.

Setup:
  1. Create a .env file with SUPABASE_URL and SUPABASE_KEY
  2. Run the SQL in supabase_schema.sql inside the Supabase SQL Editor
  3. Create a storage bucket called 'violations' (public read)
"""

import os
import pickle
import base64
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY", "")
STORAGE_BUCKET = "violations"

_client: Optional[Client] = None


# ── connection ──────────────────────────────────────────────────
def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set in your .env file.\n"
                "See .env.example for reference."
            )
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def init_database():
    """Verify Supabase connection on startup. Tables are created via SQL Editor."""
    try:
        get_client()
        print("[DB] Supabase connected ✅")
    except Exception as e:
        print(f"[DB] Supabase connection error: {e}")


# ── internal helpers ────────────────────────────────────────────
def _enc_to_b64(face_enc_bytes: bytes) -> str:
    """Pickle bytes → base64 string (safe for Supabase text column)."""
    return base64.b64encode(face_enc_bytes).decode("utf-8")


def _b64_to_enc(b64_str: str) -> bytes:
    """base64 string → pickle bytes."""
    return base64.b64decode(b64_str)


def _upload_snapshot(local_path: str) -> str:
    """
    Upload a local snapshot JPEG to Supabase Storage.
    Returns the public URL; falls back to local path on error.
    """
    try:
        client   = get_client()
        filename = os.path.basename(local_path)
        with open(local_path, "rb") as f:
            client.storage.from_(STORAGE_BUCKET).upload(
                path=filename,
                file=f,
                file_options={"content-type": "image/jpeg"},
            )
        return client.storage.from_(STORAGE_BUCKET).get_public_url(filename)
    except Exception as e:
        print(f"[DB] snapshot upload error: {e}")
        return local_path


# ── workers ─────────────────────────────────────────────────────
def add_worker(employee_id: str, name: str, department: str,
               image_path: str, face_encoding: Optional[bytes]) -> Tuple[bool, str]:
    try:
        client = get_client()
        data = {
            "employee_id":   employee_id,
            "name":          name,
            "department":    department,
            "image_path":    image_path,
            "face_encoding": _enc_to_b64(face_encoding) if face_encoding else None,
            "created_at":    datetime.now().isoformat(),
        }
        client.table("workers").insert(data).execute()
        return True, f"Worker **{name}** ({employee_id}) registered ✅"
    except Exception as e:
        err = str(e)
        if "duplicate" in err.lower() or "unique" in err.lower():
            return False, f"Employee ID **{employee_id}** already exists."
        return False, f"DB error: {e}"


def get_all_workers() -> List[Dict]:
    try:
        client = get_client()
        res = (
            client.table("workers")
            .select("id, employee_id, name, department, image_path, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        return res.data or []
    except Exception as e:
        print(f"[DB] get_all_workers error: {e}")
        return []


def get_worker_face_encodings() -> Dict[str, tuple]:
    """
    Returns {employee_id: (name, department, numpy_encoding)}
    Only workers that have a stored face encoding are included.
    """
    try:
        client = get_client()
        res = (
            client.table("workers")
            .select("employee_id, name, department, face_encoding")
            .neq("face_encoding", "null")
            .execute()
        )
        out: Dict[str, tuple] = {}
        for row in (res.data or []):
            if not row.get("face_encoding"):
                continue
            try:
                enc = pickle.loads(_b64_to_enc(row["face_encoding"]))
                out[row["employee_id"]] = (row["name"], row["department"], enc)
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"[DB] get_worker_face_encodings error: {e}")
        return {}


def delete_worker(employee_id: str) -> Tuple[bool, str]:
    try:
        client = get_client()
        res = client.table("workers").delete().eq("employee_id", employee_id).execute()
        if res.data:
            return True, "Deleted."
        return False, "Not found."
    except Exception as e:
        return False, f"DB error: {e}"


# ── violations ──────────────────────────────────────────────────
def log_violation(employee_id: str, name: str, department: str,
                  violation_type: str, confidence: float,
                  snapshot_path: str = "",
                  camera_location: str = "Main Camera",
                  severity: str = "Low") -> bool:
    try:
        client = get_client()

        # Upload snapshot image to Supabase Storage
        image_url = ""
        if snapshot_path and os.path.exists(snapshot_path):
            image_url = _upload_snapshot(snapshot_path)

        data = {
            "timestamp":           datetime.now().isoformat(),
            "employee_id":         employee_id or "Unknown",
            "name":                name or "Unknown",
            "department":          department or "Unknown",
            "violation_type":      violation_type,
            "confidence":          round(confidence, 2),
            "image_snapshot_path": image_url or snapshot_path,
            "camera_location":     camera_location,
            "severity_level":      severity,
            "status":              "Open",
        }
        client.table("violation_logs").insert(data).execute()
        return True
    except Exception as e:
        print(f"[DB] log_violation error: {e}")
        return False


def get_violation_count_today(employee_id: str) -> int:
    try:
        client = get_client()
        today = datetime.now().strftime("%Y-%m-%d")
        res = (
            client.table("violation_logs")
            .select("id", count="exact")
            .eq("employee_id", employee_id)
            .gte("timestamp", f"{today}T00:00:00")
            .lte("timestamp", f"{today}T23:59:59")
            .execute()
        )
        return res.count or 0
    except Exception as e:
        print(f"[DB] get_violation_count_today error: {e}")
        return 0


def calc_severity(employee_id: str) -> str:
    """Auto‑severity: >5→High, 3‑5→Medium, <3→Low."""
    n = get_violation_count_today(employee_id)
    if n > 5:
        return "High"
    if n >= 3:
        return "Medium"
    return "Low"


def get_violations_df(start_date=None, end_date=None) -> pd.DataFrame:
    try:
        client = get_client()
        q = client.table("violation_logs").select("*").order("timestamp", desc=True)
        if start_date:
            q = q.gte("timestamp", f"{start_date}T00:00:00")
        if end_date:
            q = q.lte("timestamp", f"{end_date}T23:59:59")
        res = q.execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        print(f"[DB] get_violations_df error: {e}")
        return pd.DataFrame()


def get_dashboard_stats(date_str: Optional[str] = None) -> Dict:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        client = get_client()
        res = (
            client.table("violation_logs")
            .select("*")
            .gte("timestamp", f"{date_str}T00:00:00")
            .lte("timestamp", f"{date_str}T23:59:59")
            .execute()
        )
        rows = res.data or []
        df   = pd.DataFrame(rows)

        if df.empty:
            return dict(total=0, by_employee=[], by_dept=[],
                        by_severity={}, by_type=[])

        total       = len(df)
        by_employee = (
            df[df["name"] != "Unknown"]
            .groupby("name").size().reset_index(name="c")
            .sort_values("c", ascending=False).values.tolist()
        )
        by_dept = (
            df[df["department"] != "Unknown"]
            .groupby("department").size().reset_index(name="c")
            .sort_values("c", ascending=False).values.tolist()
        )
        by_severity = df.groupby("severity_level").size().to_dict()
        by_type = (
            df.groupby("violation_type").size().reset_index(name="c")
            .sort_values("c", ascending=False).values.tolist()
        )

        return dict(
            total=total,
            by_employee=by_employee,
            by_dept=by_dept,
            by_severity=by_severity,
            by_type=by_type,
        )
    except Exception as e:
        print(f"[DB] get_dashboard_stats error: {e}")
        return dict(total=0, by_employee=[], by_dept=[],
                    by_severity={}, by_type=[])


# Auto-init on first import
init_database()
