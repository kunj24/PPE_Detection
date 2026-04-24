# -*- coding: utf-8 -*-
"""
Professional PPE Detection System v2
-------------------------------------
Pages (sidebar radio):
  1. Live Detection   - all 5 input sources, face-ID, smart 3-sec logging
  2. Worker Register  - capture / upload photo -> SQLite + face encoding
  3. Dashboard        - analytics, filters, CSV export
  4. Violation Logs   - enhanced table with worker name / dept / severity
"""

# --- stdlib + third-party imports ---
import streamlit as st
import cv2, os, time, tempfile, shutil
import numpy as np
import pandas as pd
from datetime import datetime

# --- Page config (MUST be first Streamlit call) ---
st.set_page_config(
    page_title="AI PPE Surveillance",
    layout="wide",
    page_icon="👷",
    initial_sidebar_state="expanded",
)

# --- Streamlit-extras (graceful if missing) ---
try:
    from streamlit_extras.colored_header import colored_header
    from streamlit_extras.metric_cards import style_metric_cards
    from streamlit_extras.stylable_container import stylable_container
    _EXTRAS = True
except ImportError:
    _EXTRAS = False

# --- Project utilities ---
from utils import database_utils as db
from utils import face_utils
from utils.detection_utils import PPEDetector
from utils.alarm_utils import AlarmSystem

# --- Ensure DB tables exist ---
db.init_database()


# ═══════════════════════════════════════════════════════════════
#  DARK‑THEME CSS  (same palette you already had)
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
:root{--bg:#0f1724;--card:#0b1220;--muted:#94a3b8;--accent:#00bcd4;
      --text:#e6eef8;--danger:#ff6b6b;--success:#51cf66}
.main{background:var(--bg)!important;color:var(--text)!important;
      font-family:'Segoe UI',sans-serif}
.stButton>button{background:linear-gradient(90deg,var(--accent),#0077b6)!important;
      color:#02121a!important;border-radius:8px;padding:10px 18px;font-weight:700}
.stSelectbox,.stTextInput,.stRadio>div{background:var(--card)!important;
      color:var(--text)!important;border-radius:8px;padding:10px;
      border:1px solid rgba(255,255,255,.04)}
.stDataFrame{background:rgba(255,255,255,.03)!important;border-radius:8px}
.violation-card{border-left:5px solid var(--danger);
      background:rgba(255,107,107,.08);padding:12px;margin-bottom:12px;
      border-radius:6px;color:var(--text)}
section[data-testid="stSidebar"]{background:#071025!important;color:var(--text)!important}
h1,h2,h3,h4{color:var(--text)!important}
img{border-radius:8px}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR – navigation + system info
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    if os.path.exists("home.jpeg"):
        st.image("home.jpeg", use_container_width=True)

    st.markdown("""
    <div style="margin-top:20px">
      <div style="display:flex;align-items:center;gap:10px;background:#003366;
           color:white;padding:10px 16px;border-radius:8px;font-weight:bold;
           font-size:1.1rem">⚙️ Navigation</div>
    </div>""", unsafe_allow_html=True)

    PAGE = st.radio(
        "Go to",
        ["🎥 Live Detection", "👤 Worker Registration",
         "📊 Dashboard", "📋 Violation Logs", "🗄️ Database Viewer"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    workers = db.get_all_workers()
    stats_today = db.get_dashboard_stats()
    c1, c2 = st.columns(2)
    c1.metric("Workers", len(workers))
    c2.metric("Violations today", stats_today["total"])

    st.markdown("---")
    st.caption("PPE Detection System v2 · SQLite backend")


# ═══════════════════════════════════════════════════════════════
#  PAGE 1 – LIVE DETECTION
# ═══════════════════════════════════════════════════════════════
def page_detection():
    if _EXTRAS:
        colored_header(label="👷 AI CCTV Surveillance System",
                       description="Real‑time PPE compliance monitoring with worker identification",
                       color_name="blue-70")
    else:
        st.title("👷 AI CCTV Surveillance System")

    # Settings expander
    with st.expander("⚙️ Detection settings", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        do_faces = sc1.checkbox("Enable face recognition", value=True)
        threshold = sc2.slider("Violation hold (sec)", 0.0, 10.0, 0.5, 0.1,
                               help="Violations must persist this long before logging. Set to 0 for instant logging.")
        cam_loc = sc3.text_input("Camera location", "Main Camera")

        # Alarm settings
        sa1, sa2, sa3 = st.columns(3)
        enable_alarm = sa1.checkbox("🔔 Enable Audio Alarm", value=True,
                                    help="Play beep sound when violations detected")
        alarm_cooldown = sa2.slider("Alarm cooldown (sec)", 1.0, 60.0, 5.0, 1.0,
                                   help="Minimum seconds between consecutive alarms (prevents spam)")
        alarm_freq = sa3.number_input("Tone frequency (Hz)", 500, 2000, 1000, 100,
                                     help="Beep pitch in Hertz")

    # Auto face detection info
    all_workers = db.get_all_workers()
    known_faces = db.get_worker_face_encodings()  # {eid: (name, dept, encoding)}

    if not all_workers:
        st.warning("No workers registered yet. Go to **Worker Registration** to add workers first. Violations will be logged as 'Unknown'.")
    elif not known_faces:
        st.info(f"{len(all_workers)} worker(s) registered but none have face encodings. Re-register with a clear face photo for auto-detection.")
    else:
        names = [v[0] for v in known_faces.values()]
        st.success(f"Auto Face Detection active: {len(known_faces)} worker(s) with face data ({', '.join(names)}). Faces will be matched automatically.")

    # Lazy‑init or update detector in session state
    if "detector" not in st.session_state:
        st.session_state.detector = PPEDetector(
            threshold_secs=threshold, camera_location=cam_loc)
    else:
        # Update threshold and camera if changed
        st.session_state.detector.threshold = threshold
        st.session_state.detector.camera = cam_loc

    # Initialize alarm system
    if "alarm_system" not in st.session_state:
        st.session_state.alarm_system = AlarmSystem(cooldown_secs=alarm_cooldown)
    else:
        st.session_state.alarm_system.cooldown = alarm_cooldown

    det: PPEDetector = st.session_state.detector
    alarm: AlarmSystem = st.session_state.alarm_system

    source_type = st.radio(
        "Select Input Source",
        ["Browser Webcam (Photo)", "Upload Video", "Upload Image",
         "RTSP IP Camera", "OpenCV Webcam (Local Only)"],
        horizontal=True,
    )

    # ── Browser webcam photo ────────────────────────────────────
    if source_type == "Browser Webcam (Photo)":
        st.info("📸 Captures a single photo (browser permission required)")
        img = st.camera_input("Take a photo for PPE detection")
        if img:
            frame = cv2.imdecode(
                np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
            annotated, stats = det.process_frame(frame, do_faces=do_faces)

            # Trigger alarm if violations logged
            if enable_alarm and stats["violations_logged"] > 0:
                alarm.play_alarm(frequency=int(alarm_freq))

            c1, c2 = st.columns([2, 1])
            c1.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Detection result", use_container_width=True)
            with c2:
                st.metric("Detections", stats["detections"])
                st.metric("Violations", stats["violations_in_frame"])
                st.metric("Workers ID'd", stats["workers_identified"])
                st.metric("Logged to DB", stats["violations_logged"])
                if stats["violations_in_frame"] > 0 and stats["violations_logged"] == 0:
                    st.warning(f"⏱️ {stats['pending_tracks']} violations pending (need {threshold:.1f}s hold time)")

    # ── Upload video ────────────────────────────────────────────
    elif source_type == "Upload Video":
        f = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if f:
            tmp = os.path.join(tempfile.gettempdir(), f.name)
            with open(tmp, "wb") as fp:
                fp.write(f.read())
            st.success("✅ Video uploaded – processing …")
            _stream_video(tmp, det, do_faces, alarm, enable_alarm, int(alarm_freq))

    # ── Upload image ────────────────────────────────────────────
    elif source_type == "Upload Image":
        f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if f:
            tmp = os.path.join(tempfile.gettempdir(), f.name)
            with open(tmp, "wb") as fp:
                fp.write(f.read())
            frame = cv2.imread(tmp)
            annotated, stats = det.process_frame(frame, do_faces=do_faces)

            # Trigger alarm if violations logged
            if enable_alarm and stats["violations_logged"] > 0:
                alarm.play_alarm(frequency=int(alarm_freq))

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="PPE Detection", use_container_width=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Detections", stats["detections"])
            mc2.metric("Violations", stats["violations_in_frame"])
            mc3.metric("Logged", stats["violations_logged"])
            mc4.metric("Pending", stats["pending_tracks"])
            if stats["violations_in_frame"] > 0 and stats["violations_logged"] == 0:
                st.warning(f"⏱️ Violations detected but not logged yet. Set threshold to 0s for instant logging, or keep violations in view for {threshold:.1f}+ seconds.")

    # ── RTSP ────────────────────────────────────────────────────
    elif source_type == "RTSP IP Camera":
        url = st.text_input("RTSP URL",
                            placeholder="rtsp://user:pass@192.168.1.100:554/stream1")
        if url and st.button("📡 Start stream", type="primary"):
            _stream_video(url, det, do_faces, alarm, enable_alarm, int(alarm_freq))

    # ── Local OpenCV webcam ─────────────────────────────────────
    elif source_type == "OpenCV Webcam (Local Only)":
        st.warning("⚠️ Only works when running Streamlit locally")
        if st.button("🎥 Start Webcam"):
            _stream_video(0, det, do_faces, alarm, enable_alarm, int(alarm_freq))


def _stream_video(source, det: PPEDetector, do_faces: bool, alarm: AlarmSystem = None, enable_alarm: bool = False, alarm_freq: int = 1000):
    """
    Stream frames with smooth continuous face labelling.

    Strategy:
    - YOLO runs every frame (fast, ~30ms)
    - Face detection runs in a background thread every 3rd frame
      so it never blocks rendering
    - Cached face list is drawn on EVERY frame → no flashing labels
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Cannot open video source"); return

    # Request the best quality the camera supports
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPEG = better quality at HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimal latency
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)   # enable autofocus if supported
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # enable auto-exposure

    frame_ph = st.empty()
    stat_ph  = st.empty()
    stop     = st.button("⏹ Stop Stream")
    n, t0    = 0, time.time()

    # Shared face cache updated by background thread
    import threading
    _face_cache      = []
    _face_cache_lock = threading.Lock()
    _face_running    = threading.Event()   # prevents overlapping face threads

    def _update_faces(frame_copy, prev_faces_copy):
        try:
            # Pass previous faces for spatial tracking to prevent name swapping
            new_faces = face_utils.identify_faces(
                frame_copy, det._known_faces, prev_faces=prev_faces_copy
            )
            with _face_cache_lock:
                _face_cache.clear()
                _face_cache.extend(new_faces)
        except Exception:
            pass
        finally:
            _face_running.clear()

    last_stats = dict(detections=0, violations_in_frame=0,
                      violations_logged=0, workers_identified=0,
                      pending_tracks=0)

    while cap.isOpened() and not stop:
        ok, frame = cap.read()
        if not ok:
            st.warning("Stream ended."); break

        n += 1

        # Kick off a new face-detection thread as soon as the previous one
        # finishes – this gives truly continuous face labels
        if do_faces and det._known_faces and not _face_running.is_set():
            _face_running.set()
            # Copy current faces for spatial tracking
            with _face_cache_lock:
                prev_faces_snapshot = list(_face_cache)
            threading.Thread(
                target=_update_faces,
                args=(frame.copy(), prev_faces_snapshot),
                daemon=True
            ).start()

        # Read current cached faces (never blocks)
        with _face_cache_lock:
            current_faces = list(_face_cache)

        # YOLO + draw cached face labels (no face detection here)
        annotated, stats = det.process_frame(
            frame,
            do_faces=False,          # detection handled above
            cached_faces=current_faces,
        )
        last_stats = stats

        # Trigger alarm if violations logged
        if enable_alarm and alarm and stats["violations_logged"] > 0:
            alarm.play_alarm(frequency=int(alarm_freq))

        frame_ph.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB", use_container_width=True,
        )

        # Update metrics every 5 frames
        if n % 5 == 0:
            fps = n / max(time.time() - t0, 0.001)
            with stat_ph.container():
                a, b, c, d, e, f = st.columns(6)
                a.metric("FPS",        f"{fps:.1f}")
                b.metric("Detections", stats["detections"])
                c.metric("Violations", stats["violations_in_frame"])
                d.metric("Workers",    stats["workers_identified"])
                e.metric("Logged",     stats["violations_logged"])
                f.metric("Pending",    stats["pending_tracks"])

    cap.release()



# ═══════════════════════════════════════════════════════════════
#  PAGE 2 – WORKER REGISTRATION
# ═══════════════════════════════════════════════════════════════
def page_register():
    if _EXTRAS:
        colored_header(label="👤 Worker Registration",
                       description="Register workers with a photo so the system can identify them automatically",
                       color_name="green-70")
    else:
        st.title("👤 Worker Registration")

    tab_new, tab_list = st.tabs(["➕ Register new worker", "📋 All workers"])

    # ── register ────────────────────────────────────────────────
    with tab_new:
        rc1, rc2 = st.columns(2)
        emp_id = rc1.text_input("Employee ID *", placeholder="EMP001")
        name   = rc1.text_input("Full Name *",   placeholder="Ravi Kumar")
        dept   = rc2.text_input("Department *",   placeholder="Construction")

        st.markdown("#### 📸 Capture or upload a photo")

        method = st.radio("Photo method", ["Camera", "Upload file"], horizontal=True)
        photo_path: str | None = None

        if method == "Camera":
            cam = st.camera_input("Take a photo")
            if cam:
                arr = cv2.imdecode(np.frombuffer(cam.read(), np.uint8), cv2.IMREAD_COLOR)
                photo_path = os.path.join(tempfile.gettempdir(), f"reg_{emp_id}.jpg")
                cv2.imwrite(photo_path, arr)
        else:
            up = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
            if up:
                photo_path = os.path.join(tempfile.gettempdir(), up.name)
                with open(photo_path, "wb") as fp:
                    fp.write(up.read())

        if st.button("✅ Register Worker", type="primary"):
            if not (emp_id and name and dept):
                st.error("Please fill all required fields."); return
            if not photo_path:
                st.error("Please provide a photo."); return

            with st.spinner("Processing …"):
                # Try to generate face encoding (None if lib missing)
                enc_blob, enc_msg = face_utils.generate_encoding_from_file(photo_path)
                if enc_blob is None and face_utils.FACE_REC_AVAILABLE:
                    st.error(enc_msg); return

                # Permanent copy of the photo
                perm = os.path.join("database", "workers",
                                    f"{emp_id}_{datetime.now():%Y%m%d_%H%M%S}.jpg")
                shutil.copy(photo_path, perm)

                ok, msg = db.add_worker(emp_id, name, dept, perm, enc_blob)
                if ok:
                    st.success(msg)
                    st.balloons()
                    # Refresh detector cache if it exists
                    if "detector" in st.session_state:
                        st.session_state.detector.reload_faces()
                else:
                    st.error(msg)

    # ── list ────────────────────────────────────────────────────
    with tab_list:
        wk = db.get_all_workers()
        if not wk:
            st.info("No workers registered yet."); return
        df = pd.DataFrame(wk)[["employee_id", "name", "department", "created_at"]]
        df.columns = ["Employee ID", "Name", "Department", "Registered"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        sel = st.selectbox("View details",
                           [w["employee_id"] for w in wk],
                           format_func=lambda x: f'{x} – {next(w["name"] for w in wk if w["employee_id"]==x)}')
        w = next(w for w in wk if w["employee_id"] == sel)
        ic1, ic2 = st.columns([1, 2])
        with ic1:
            if w["image_path"] and os.path.exists(w["image_path"]):
                st.image(w["image_path"], caption=w["name"], use_container_width=True)
            else:
                st.warning("Photo not found on disk")
        with ic2:
            st.markdown(f"**ID:** {w['employee_id']}")
            st.markdown(f"**Name:** {w['name']}")
            st.markdown(f"**Dept:** {w['department']}")
            st.markdown(f"**Since:** {w['created_at']}")

            st.markdown("---")
            st.markdown("##### :red[Danger Zone]")
            confirm = st.checkbox(f"I confirm I want to delete **{w['name']}** ({w['employee_id']})",
                                  key=f"del_confirm_{w['employee_id']}")
            if st.button("🗑️ Delete Worker", type="secondary",
                         disabled=not confirm, key=f"del_btn_{w['employee_id']}"):
                # Delete photo from disk
                if w.get("image_path") and os.path.exists(w["image_path"]):
                    os.remove(w["image_path"])
                ok, msg = db.delete_worker(w["employee_id"])
                if ok:
                    st.success(f"Worker **{w['name']}** ({w['employee_id']}) deleted successfully.")
                    # Refresh detector face cache
                    if "detector" in st.session_state:
                        st.session_state.detector.reload_faces()
                    st.rerun()
                else:
                    st.error(msg)


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 – DASHBOARD
# ═══════════════════════════════════════════════════════════════
def page_dashboard():
    if _EXTRAS:
        colored_header(label="📊 Analytics Dashboard",
                       description="Violation trends, department breakdown, severity distribution",
                       color_name="violet-70")
    else:
        st.title("📊 Analytics Dashboard")

    dc1, dc2 = st.columns([2, 1])
    chosen = dc1.date_input("Filter date", value=datetime.now())
    date_str = chosen.strftime("%Y-%m-%d")
    dc2.write("")  # spacer
    if dc2.button("🔄 Refresh"):
        st.rerun()

    s = db.get_dashboard_stats(date_str)

    # key metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total violations", s["total"])
    m2.metric("🔴 High severity", s["by_severity"].get("High", 0))
    m3.metric("Unique violators", len(s["by_employee"]))
    m4.metric("Departments hit", len(s["by_dept"]))

    st.markdown("---")
    gc1, gc2 = st.columns(2)

    with gc1:
        st.markdown("#### 👥 Top violators")
        if s["by_employee"]:
            df_e = pd.DataFrame(s["by_employee"], columns=["Name", "Count"])
            st.bar_chart(df_e.set_index("Name"))
        else:
            st.info("None")
    with gc2:
        st.markdown("#### 🏢 By department")
        if s["by_dept"]:
            df_d = pd.DataFrame(s["by_dept"], columns=["Dept", "Count"])
            st.bar_chart(df_d.set_index("Dept"))
        else:
            st.info("None")

    st.markdown("#### ⚠️ Violation types")
    if s["by_type"]:
        st.dataframe(pd.DataFrame(s["by_type"], columns=["Type", "Count"]),
                     use_container_width=True, hide_index=True)

    st.markdown("#### 🚨 Severity split")
    sv1, sv2, sv3 = st.columns(3)
    sv1.metric("🟢 Low",    s["by_severity"].get("Low", 0))
    sv2.metric("🟡 Medium", s["by_severity"].get("Medium", 0))
    sv3.metric("🔴 High",   s["by_severity"].get("High", 0))


# ═══════════════════════════════════════════════════════════════
#  PAGE 4 – VIOLATION LOGS (enhanced table)
# ═══════════════════════════════════════════════════════════════
def page_logs():
    if _EXTRAS:
        colored_header(label="📋 Violation Logs",
                       description="Full log with worker name, department, severity and snapshot",
                       color_name="red-70")
    else:
        st.title("📋 Violation Logs")

    lc1, lc2 = st.columns(2)
    d1 = lc1.date_input("From", value=datetime.now())
    d2 = lc2.date_input("To",   value=datetime.now())

    df = db.get_violations_df(start_date=d1, end_date=d2)

    if df.empty:
        st.info("No violations in the selected range."); return

    # Display nice table
    show_cols = ["timestamp", "employee_id", "name", "department",
                 "violation_type", "confidence", "severity_level", "status"]
    show = df[[c for c in show_cols if c in df.columns]].copy()
    show.columns = ["Time", "Emp ID", "Name", "Dept",
                    "Violation", "Confidence", "Severity", "Status"][:len(show.columns)]

    st.dataframe(show, use_container_width=True, hide_index=True)

    bc1, bc2 = st.columns(2)
    bc1.download_button(
        "📥 Download CSV", data=df.to_csv(index=False),
        file_name=f"violations_{d1}_{d2}.csv", mime="text/csv")

    st.markdown("### 📊 Violation Statistics")
    st.bar_chart(df["violation_type"].value_counts())


# ═══════════════════════════════════════════════════════════════
#  PAGE 5 – DATABASE VIEWER
# ═══════════════════════════════════════════════════════════════
def page_database():
    if _EXTRAS:
        colored_header(label="🗄️ Database Viewer",
                       description="View all database tables and entries (like MongoDB Compass)",
                       color_name="orange-70")
    else:
        st.title("🗄️ Database Viewer")

    st.info("💡 **Live view of your SQLite database** - all data is stored in `database.db`")

    # Tab selector for tables
    tab1, tab2, tab3 = st.tabs(["👥 Workers Table", "⚠️ Violations Table", "📊 Database Stats"])

    import sqlite3

    # --- WORKERS TABLE ---
    with tab1:
        st.markdown("### 👥 Workers Table")
        workers = db.get_all_workers()
        
        if not workers:
            st.warning("No workers registered yet. Go to Worker Registration to add one.")
        else:
            # Convert to DataFrame and display
            df = pd.DataFrame(workers)
            display_cols = [c for c in df.columns if c not in ('face_encoding',)]
            
            st.markdown(f"**Total entries:** {len(df)}")
            st.dataframe(
                df[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "image_path": st.column_config.TextColumn("Photo Path", width="medium"),
                    "employee_id": st.column_config.TextColumn("Employee ID", width="small"),
                }
            )
            
            # Download option
            csv = df[display_cols].to_csv(index=False)
            st.download_button("📥 Download Workers CSV", csv, "workers.csv", "text/csv")
            
            # Show a sample worker photo
            st.markdown("---")
            st.markdown("#### 📸 Preview Worker Photos")
            sel = st.selectbox("Select worker to preview", [w["employee_id"] for w in workers])
            worker = next(w for w in workers if w["employee_id"] == sel)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                if worker.get("image_path") and os.path.exists(worker["image_path"]):
                    st.image(worker["image_path"], caption=worker["name"])
                else:
                    st.warning("Photo not found")
            with c2:
                st.json({
                    "employee_id": worker["employee_id"],
                    "name": worker["name"],
                    "department": worker["department"],
                    "registered": worker["created_at"],
                })

    # --- VIOLATIONS TABLE ---
    with tab2:
        st.markdown("### ⚠️ Violation Logs Table")
        
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        limit = fc1.number_input("Show last N rows", 10, 1000, 50, 10)
        sort_order = fc2.selectbox("Sort by", ["Newest first", "Oldest first"])
        filter_type = fc3.selectbox("Filter by type", ["All"] + ["NO-Mask", "NO-Hardhat", "NO-Safety Vest"])
        
        # Query violations
        conn = sqlite3.connect("database.db")
        
        query = "SELECT * FROM violation_logs"
        if filter_type != "All":
            query += f" WHERE violation_type = '{filter_type}'"
        query += " ORDER BY timestamp " + ("DESC" if sort_order == "Newest first" else "ASC")
        query += f" LIMIT {limit}"
        
        df_viol = pd.read_sql_query(query, conn)
        conn.close()
        
        if df_viol.empty:
            st.warning("No violations found matching your filters.")
        else:
            # Count all violations (not just today)
            conn2 = sqlite3.connect("database.db")
            total_all = conn2.execute("SELECT COUNT(*) FROM violation_logs").fetchone()[0]
            conn2.close()
            st.markdown(f"**Showing:** {len(df_viol)} entries (of {total_all} total violations)")
            
            st.dataframe(
                df_viol,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time", format="DD/MM/YYYY HH:mm:ss"),
                    "confidence": st.column_config.NumberColumn("Conf", format="%.2f"),
                    "image_snapshot_path": st.column_config.TextColumn("Snapshot", width="medium")
                }
            )
            
            # Download
            st.download_button(
                "📥 Download Filtered CSV",
                df_viol.to_csv(index=False),
                f"violations_filtered_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "text/csv"
            )
            
            # Preview violation snapshot
            if len(df_viol) > 0:
                st.markdown("---")
                st.markdown("#### 📸 Preview Violation Snapshots")
                row_idx = st.slider("Select row to preview", 0, len(df_viol)-1, 0)
                selected_row = df_viol.iloc[row_idx]
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    snap_path = selected_row["image_snapshot_path"]
                    if os.path.exists(snap_path):
                        st.image(snap_path, caption=f"Violation #{selected_row['id']}")
                    else:
                        st.warning("Snapshot not found")
                with c2:
                    st.json(selected_row.to_dict())

    # --- DATABASE STATS ---
    with tab3:
        st.markdown("### 📊 Database Statistics & Schema")
        
        # Connection info
        st.markdown("#### 📁 File Information")
        db_path = os.path.abspath("database.db")
        db_size = os.path.getsize("database.db") if os.path.exists("database.db") else 0
        
        ic1, ic2 = st.columns(2)
        ic1.metric("Database File", "database.db")
        ic2.metric("File Size", f"{db_size / 1024:.1f} KB")
        st.code(db_path, language="")
        
        # Table schemas
        st.markdown("---")
        st.markdown("#### 📋 Table Schemas")
        
        conn = sqlite3.connect("database.db")
        cur = conn.cursor()
        
        # Workers schema
        st.markdown("**`workers` table:**")
        cur.execute("PRAGMA table_info(workers)")
        workers_schema = pd.DataFrame(cur.fetchall(), columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
        st.dataframe(workers_schema[["name", "type", "notnull", "pk"]], hide_index=True, use_container_width=True)
        
        # Violations schema
        st.markdown("**`violation_logs` table:**")
        cur.execute("PRAGMA table_info(violation_logs)")
        viol_schema = pd.DataFrame(cur.fetchall(), columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
        st.dataframe(viol_schema[["name", "type", "notnull", "pk"]], hide_index=True, use_container_width=True)
        
        conn.close()
        
        # Record counts
        st.markdown("---")
        st.markdown("#### 📊 Record Counts")
        wc1, wc2 = st.columns(2)
        wc1.metric("👥 Total Workers", len(db.get_all_workers()))
        wc2.metric("⚠️ Total Violations", db.get_dashboard_stats()["total"])
        
        # Quick SQL query executor
        st.markdown("---")
        st.markdown("#### 💻 Run Custom SQL Query")
        st.warning("⚠️ Use SELECT queries only to avoid data corruption")
        
        query_input = st.text_area(
            "SQL Query",
            "SELECT * FROM violation_logs ORDER BY timestamp DESC LIMIT 10;",
            height=100
        )
        
        if st.button("▶️ Execute Query", type="primary"):
            try:
                conn = sqlite3.connect("database.db")
                result = pd.read_sql_query(query_input, conn)
                conn.close()
                st.success(f"✅ Query returned {len(result)} rows")
                st.dataframe(result, use_container_width=True)
                
                # Download query result
                st.download_button(
                    "📥 Download Query Result",
                    result.to_csv(index=False),
                    "query_result.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"❌ Query failed: {e}")


# ═══════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════
if   PAGE == "🎥 Live Detection":       page_detection()
elif PAGE == "👤 Worker Registration":   page_register()
elif PAGE == "📊 Dashboard":             page_dashboard()
elif PAGE == "📋 Violation Logs":        page_logs()
elif PAGE == "🗄️ Database Viewer":      page_database()
