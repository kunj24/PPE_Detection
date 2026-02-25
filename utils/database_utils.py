"""
database_utils.py – SQLite backend for workers + violation logs.
All DB operations are centralised here so the rest of the app
only calls simple helper functions.
"""

import sqlite3
import os
import pickle
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, List, Dict

# Single DB file lives next to app.py
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database.db")


# ── initialisation ──────────────────────────────────────────────
def get_connection() -> sqlite3.Connection:
    """Return a new connection (auto-creates the file if absent)."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")      # safer for concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_database():
    """Create tables if they don't already exist."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS workers (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id     TEXT    UNIQUE NOT NULL,
            name            TEXT    NOT NULL,
            department      TEXT    NOT NULL,
            image_path      TEXT,
            face_encoding   BLOB,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS violation_logs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            employee_id         TEXT,
            name                TEXT,
            department          TEXT,
            violation_type      TEXT    NOT NULL,
            confidence          REAL    NOT NULL,
            image_snapshot_path TEXT,
            camera_location     TEXT    DEFAULT 'Main Camera',
            severity_level      TEXT    DEFAULT 'Low',
            status              TEXT    DEFAULT 'Open'
        )
    """)

    conn.commit()
    conn.close()


# ── workers ─────────────────────────────────────────────────────
def add_worker(employee_id: str, name: str, department: str,
               image_path: str, face_encoding: Optional[bytes]) -> Tuple[bool, str]:
    """Insert a new worker.  Returns (ok, message)."""
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO workers (employee_id, name, department, image_path, face_encoding) "
            "VALUES (?, ?, ?, ?, ?)",
            (employee_id, name, department, image_path, face_encoding),
        )
        conn.commit()
        conn.close()
        return True, f"Worker **{name}** ({employee_id}) registered ✅"
    except sqlite3.IntegrityError:
        return False, f"Employee ID **{employee_id}** already exists."
    except Exception as e:
        return False, f"DB error: {e}"


def get_all_workers() -> List[Dict]:
    """Return list of worker dicts (no encoding blob – too large for display)."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, employee_id, name, department, image_path, created_at "
        "FROM workers ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [
        dict(id=r[0], employee_id=r[1], name=r[2],
             department=r[3], image_path=r[4], created_at=r[5])
        for r in rows
    ]


def get_worker_face_encodings() -> Dict[str, tuple]:
    """
    {employee_id: (name, department, numpy_encoding)}
    Only workers with a stored encoding are returned.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT employee_id, name, department, face_encoding "
        "FROM workers WHERE face_encoding IS NOT NULL"
    ).fetchall()
    conn.close()

    out: Dict[str, tuple] = {}
    for eid, name, dept, blob in rows:
        try:
            enc = pickle.loads(blob)
            out[eid] = (name, dept, enc)
        except Exception:
            continue
    return out


def delete_worker(employee_id: str) -> Tuple[bool, str]:
    conn = get_connection()
    cur = conn.execute("DELETE FROM workers WHERE employee_id = ?", (employee_id,))
    conn.commit()
    conn.close()
    if cur.rowcount:
        return True, "Deleted."
    return False, "Not found."


# ── violations ──────────────────────────────────────────────────
def log_violation(employee_id: str, name: str, department: str,
                  violation_type: str, confidence: float,
                  snapshot_path: str = "",
                  camera_location: str = "Main Camera",
                  severity: str = "Low") -> bool:
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO violation_logs "
            "(employee_id, name, department, violation_type, confidence, "
            " image_snapshot_path, camera_location, severity_level) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (employee_id or "Unknown", name or "Unknown",
             department or "Unknown", violation_type, round(confidence, 2),
             snapshot_path, camera_location, severity),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB] log_violation error: {e}")
        return False


def get_violation_count_today(employee_id: str) -> int:
    """How many violations does this worker have today?"""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) FROM violation_logs "
        "WHERE employee_id = ? AND date(timestamp) = ?",
        (employee_id, today),
    ).fetchone()
    conn.close()
    return row[0] if row else 0


def calc_severity(employee_id: str) -> str:
    """Auto‑severity: >5→High, 3‑5→Medium, <3→Low."""
    n = get_violation_count_today(employee_id)
    if n > 5:
        return "High"
    if n >= 3:
        return "Medium"
    return "Low"


def get_violations_df(start_date=None, end_date=None) -> pd.DataFrame:
    conn = get_connection()
    q = "SELECT * FROM violation_logs"
    params: list = []
    clauses: list = []
    if start_date:
        clauses.append("date(timestamp) >= ?")
        params.append(str(start_date))
    if end_date:
        clauses.append("date(timestamp) <= ?")
        params.append(str(end_date))
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY timestamp DESC"
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    return df


def get_dashboard_stats(date_str: Optional[str] = None) -> Dict:
    """Quick aggregates for the dashboard cards."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()

    total = conn.execute(
        "SELECT COUNT(*) FROM violation_logs WHERE date(timestamp)=?", (date_str,)
    ).fetchone()[0]

    by_employee = conn.execute(
        "SELECT name, COUNT(*) c FROM violation_logs "
        "WHERE date(timestamp)=? AND name!='Unknown' GROUP BY name ORDER BY c DESC",
        (date_str,),
    ).fetchall()

    by_dept = conn.execute(
        "SELECT department, COUNT(*) c FROM violation_logs "
        "WHERE date(timestamp)=? AND department!='Unknown' GROUP BY department ORDER BY c DESC",
        (date_str,),
    ).fetchall()

    by_severity = conn.execute(
        "SELECT severity_level, COUNT(*) FROM violation_logs "
        "WHERE date(timestamp)=? GROUP BY severity_level",
        (date_str,),
    ).fetchall()

    by_type = conn.execute(
        "SELECT violation_type, COUNT(*) c FROM violation_logs "
        "WHERE date(timestamp)=? GROUP BY violation_type ORDER BY c DESC",
        (date_str,),
    ).fetchall()

    conn.close()
    return dict(
        total=total,
        by_employee=by_employee,
        by_dept=by_dept,
        by_severity=dict(by_severity),
        by_type=by_type,
    )


# Auto‑init on first import
init_database()
