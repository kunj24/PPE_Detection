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
         "📊 Dashboard", "📋 Violation Logs"],
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

    # Manual worker selection (when face_recognition not available)
    workers = db.get_all_workers()
    manual_worker = None
    
    if not face_utils.FACE_REC_AVAILABLE and workers:
        st.info("ℹ️ Face recognition is disabled. Select worker manually below:")
        wcols = st.columns([2, 1])
        worker_opts = ["Unknown"] + [f"{w['employee_id']} - {w['name']} ({w['department']})" for w in workers]
        selected = wcols[0].selectbox("👤 Current worker being monitored", worker_opts)
        if selected != "Unknown":
            emp_id = selected.split(" - ")[0]
            manual_worker = next(w for w in workers if w["employee_id"] == emp_id)
        wcols[1].markdown("<br>", unsafe_allow_html=True)
        if wcols[1].button("🔄 Reload workers"):
            st.rerun()
    
    # Settings expander
    with st.expander("⚙️ Detection settings", expanded=False):
        sc1, sc2, sc3 = st.columns(3)
        do_faces = sc1.checkbox("Enable face recognition", value=True, 
                                disabled=not face_utils.FACE_REC_AVAILABLE,
                                help="Requires face_recognition library" if not face_utils.FACE_REC_AVAILABLE else None)
        threshold = sc2.slider("Violation hold (sec)", 0.0, 10.0, 0.5, 0.1,
                               help="Violations must persist this long before logging. Set to 0 for instant logging.")
        cam_loc = sc3.text_input("Camera location", "Main Camera")

    # Lazy‑init or update detector in session state
    if "detector" not in st.session_state:
        st.session_state.detector = PPEDetector(
            threshold_secs=threshold, camera_location=cam_loc)
    else:
        # Update threshold and camera if changed
        st.session_state.detector.threshold = threshold
        st.session_state.detector.camera = cam_loc
    
    # Store manual worker selection in session state
    st.session_state.manual_worker = manual_worker

    det: PPEDetector = st.session_state.detector

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
            _stream_video(tmp, det, do_faces)

    # ── Upload image ────────────────────────────────────────────
    elif source_type == "Upload Image":
        f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if f:
            tmp = os.path.join(tempfile.gettempdir(), f.name)
            with open(tmp, "wb") as fp:
                fp.write(f.read())
            frame = cv2.imread(tmp)
            annotated, stats = det.process_frame(frame, do_faces=do_faces)
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
            _stream_video(url, det, do_faces)

    # ── Local OpenCV webcam ─────────────────────────────────────
    elif source_type == "OpenCV Webcam (Local Only)":
        st.warning("⚠️ Only works when running Streamlit locally")
        if st.button("🎥 Start Webcam"):
            _stream_video(0, det, do_faces)


def _stream_video(source, det: PPEDetector, do_faces: bool):
    """Shared helper – streams *source* frame‑by‑frame."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("❌ Cannot open video source"); return

    frame_ph = st.empty()
    stat_ph  = st.empty()
    stop     = st.button("🛑 Stop Stream")
    n, t0    = 0, time.time()

    while cap.isOpened() and not stop:
        ok, frame = cap.read()
        if not ok:
            st.warning("⚠️ Stream ended."); break
        annotated, stats = det.process_frame(frame, do_faces=do_faces)
        frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                       channels="RGB", use_container_width=True)
        n += 1
        fps = n / max(time.time() - t0, 0.001)
        with stat_ph.container():
            a, b, c, d, e, f = st.columns(6)
            a.metric("FPS", f"{fps:.1f}")
            b.metric("Detections", stats["detections"])
            c.metric("Violations", stats["violations_in_frame"])
            d.metric("Workers", stats["workers_identified"])
            e.metric("Logged", stats["violations_logged"])
            f.metric("Pending", stats["pending_tracks"])
        time.sleep(0.01)
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
        if not face_utils.FACE_REC_AVAILABLE:
            st.warning(face_utils._UNAVAILABLE_MSG)

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
#  ROUTER
# ═══════════════════════════════════════════════════════════════
if   PAGE == "🎥 Live Detection":       page_detection()
elif PAGE == "👤 Worker Registration":   page_register()
elif PAGE == "📊 Dashboard":             page_dashboard()
elif PAGE == "📋 Violation Logs":        page_logs()
