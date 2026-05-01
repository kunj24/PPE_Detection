# -*- coding: utf-8 -*-
"""
Professional PPE Detection System v2  –  with Browser Voice Alarms
--------------------------------------------------------------------
Pages:
  1. Live Detection   – all 5 input sources, face-ID, voice alarms
  2. Worker Register  – capture / upload photo → DB + face encoding
  3. Dashboard        – analytics, filters, CSV export
  4. Violation Logs   – table with worker name / dept / severity
  5. Database Viewer  – raw table viewer + SQL console
"""

import streamlit as st
import cv2, os, time, tempfile, shutil
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="safety gear detection",
    layout="wide",
    page_icon="👷",
    initial_sidebar_state="expanded",
)

try:
    from streamlit_extras.colored_header import colored_header
    _EXTRAS = True
except ImportError:
    _EXTRAS = False

from utils import database_utils as db
from utils import face_utils
from utils.detection_utils import PPEDetector
from utils.alarm_utils import AlarmSystem, VIOLATION_MESSAGES, speak_in_browser

db.init_database()


# ═══════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=DM+Sans:wght@400;500;700&display=swap');

:root{
  --bg:#07131f;
  --bg2:#0a1e2f;
  --card:#0f2a3e;
  --card-soft:#143853;
  --accent:#f2b134;
  --accent-2:#2ec4b6;
  --text:#eef6ff;
  --muted:#a8c1d8;
  --danger:#ff6b6b;
  --success:#51cf66;
}

.stApp{
  background:
    radial-gradient(1200px 600px at 15% 0%, rgba(46,196,182,.22), transparent 60%),
    radial-gradient(900px 450px at 85% 10%, rgba(242,177,52,.18), transparent 60%),
    linear-gradient(145deg, var(--bg) 0%, var(--bg2) 100%);
  color:var(--text)!important;
}

.main, .stMarkdown, .stText, p, div, span, label{
  color:var(--text)!important;
  font-family:'DM Sans',sans-serif;
}

h1,h2,h3,h4{
  color:var(--text)!important;
  font-family:'Space Grotesk',sans-serif;
  letter-spacing:.2px;
}

.stButton>button{
  background:linear-gradient(95deg,var(--accent),#ffd36e)!important;
  color:#1d2530!important;
  border:0!important;
  border-radius:12px!important;
  padding:10px 18px!important;
  font-weight:700!important;
  box-shadow:0 8px 20px rgba(242,177,52,.25);
}
.stButton>button:hover{
  transform:translateY(-1px);
  box-shadow:0 10px 22px rgba(242,177,52,.32);
}

div[data-testid="stMetric"]{
  background:linear-gradient(140deg, rgba(20,56,83,.88), rgba(15,42,62,.88));
  border:1px solid rgba(168,193,216,.18);
  border-radius:14px;
  padding:12px 10px;
}

section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#06101a 0%, #0a1e2f 100%)!important;
  border-right:1px solid rgba(168,193,216,.14);
}

.stSelectbox > div > div,
.stTextInput > div > div,
.stTextArea textarea,
.stDateInput > div > div,
.stNumberInput > div > div{
  background:rgba(15,42,62,.95)!important;
  color:var(--text)!important;
  border:1px solid rgba(168,193,216,.2)!important;
  border-radius:10px!important;
}

.stTabs [data-baseweb="tab-list"]{
  gap:8px;
}
.stTabs [data-baseweb="tab"]{
  background:rgba(15,42,62,.75);
  border-radius:10px;
  border:1px solid rgba(168,193,216,.14);
}

.alarm-row{
  display:flex;
  align-items:center;
  gap:10px;
  background:rgba(255,107,107,.12);
  border-left:4px solid var(--danger);
  padding:10px 14px;
  border-radius:8px;
  margin:4px 0;
  font-size:0.92rem;
}
.alarm-icon{font-size:1.3rem}

div[data-testid="stExpander"]{
  border:1px solid rgba(168,193,216,.16);
  border-radius:12px;
  background:rgba(15,42,62,.72);
}

img{border-radius:10px}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:14px 12px;border:1px solid rgba(168,193,216,.22);
                border-radius:12px;background:rgba(15,42,62,.72);margin-bottom:14px">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.05rem;
                  font-weight:700;color:#eef6ff;letter-spacing:.3px">
        SAFETY GEAR DETECTION
      </div>
      <div style="font-size:.82rem;color:#a8c1d8;margin-top:2px">
        Smart compliance monitoring
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:20px">
      <div style="background:#003366;color:white;padding:10px 16px;
           border-radius:8px;font-weight:bold;font-size:1.1rem">
        ⚙️ Navigation
      </div>
    </div>""", unsafe_allow_html=True)

    PAGE = st.radio(
        "Go to",
        ["🎥 Live Detection", "👤 Worker Registration",
         "📊 Dashboard", "📋 Violation Logs", "🗄️ Database Viewer"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Workers",          len(db.get_all_workers()))
    c2.metric("Violations today", db.get_dashboard_stats()["total"])
    st.markdown("---")
    st.caption("PPE Detection System v2 · Supabase backend")


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def _fire_voice_alarms(alarm: AlarmSystem, stats: dict,
                       enabled: bool, rate: float, pitch: float) -> None:
    """
    For each unique violation type newly confirmed this frame,
    speak the corresponding warning once (respects per-type cooldown).
    """
    if not enabled or not stats.get("violations_this_frame"):
        return
    seen = set()
    for vtype in stats["violations_this_frame"]:
        if vtype not in seen:
            alarm.play_alarm(violation_type=vtype, rate=rate, pitch=pitch)
            seen.add(vtype)


def _show_alarm_banner(stats: dict) -> None:
    """Render a red banner for each violation type that fired this frame."""
    if not stats.get("violations_this_frame"):
        return
    for vtype in set(stats["violations_this_frame"]):
        msg = VIOLATION_MESSAGES.get(vtype, vtype)
        st.markdown(
            f'<div class="alarm-row">'
            f'<span class="alarm-icon">🔊</span>'
            f'<span><b>{vtype}</b> — {msg}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
#  PAGE 1 – LIVE DETECTION
# ═══════════════════════════════════════════════════════════════
def page_detection():
    if _EXTRAS:
        colored_header(label="👷 Safety gear detection system",
                       description="Real-time PPE monitoring with browser voice alerts",
                       color_name="blue-70")
    else:
        st.title("👷 Safety gear detection system")

    # ── Settings ────────────────────────────────────────────────
    with st.expander("⚙️ Detection & Voice Alarm Settings", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        do_faces  = sc1.checkbox("Enable face recognition", value=True)
        threshold = sc2.slider("Violation hold (sec)", 0.0, 10.0, 0.5, 0.1)
        cam_loc   = sc3.text_input("Camera location", "Main Camera")

        st.markdown("---")
        al1, al2, al3, al4 = st.columns(4)
        enable_alarm   = al1.checkbox("🔔 Enable Voice Alarm", value=True,
                                      help="Speaks warnings like 'Please wear your helmet'")
        alarm_cooldown = al2.slider("Cooldown (sec)", 1.0, 60.0, 6.0, 1.0,
                                    help="Gap between repeated warnings for the same violation")
        alarm_rate     = al3.slider("Speech speed", 0.5, 1.5, 0.9, 0.1,
                                    help="0.5 = slow, 1.0 = normal, 1.5 = fast")
        alarm_pitch    = al4.slider("Voice pitch", 0.5, 1.5, 1.0, 0.1,
                                    help="0.5 = deep, 1.0 = normal, 1.5 = high")

        # Test button
        if enable_alarm:
            if st.button("🔊 Test voice alarm"):
                speak_in_browser(
                    "Voice alarm test. Warning! Please wear your helmet immediately.",
                    rate=alarm_rate, pitch=alarm_pitch
                )

        # Show all configured messages
        if enable_alarm:
            st.markdown("**Voice messages configured:**")
            cols = st.columns(2)
            items = [(k, v) for k, v in VIOLATION_MESSAGES.items() if k != "DEFAULT"]
            for i, (vtype, msg) in enumerate(items):
                cols[i % 2].markdown(
                    f'<div class="alarm-row" style="font-size:0.82rem">'
                    f'🔴 <b>{vtype}</b><br>🔊 <i>{msg}</i></div>',
                    unsafe_allow_html=True,
                )

    # Face ID info
    all_workers = db.get_all_workers()
    known_faces = db.get_worker_face_encodings()
    if not all_workers:
        st.warning("No workers registered. Violations logged as 'Unknown'.")
    elif not known_faces:
        st.info(f"{len(all_workers)} worker(s) registered but no face encodings.")
    else:
        names = [v[0] for v in known_faces.values()]
        st.success(f"Face detection active: {len(known_faces)} worker(s) — {', '.join(names)}")

    # Detector / alarm init
    if "detector" not in st.session_state:
        st.session_state.detector = PPEDetector(
            threshold_secs=threshold, camera_location=cam_loc)
    else:
        st.session_state.detector.threshold = threshold
        st.session_state.detector.camera    = cam_loc

    if "alarm_system" not in st.session_state:
        st.session_state.alarm_system = AlarmSystem(cooldown_secs=alarm_cooldown)
    else:
        st.session_state.alarm_system.cooldown = alarm_cooldown

    det: PPEDetector  = st.session_state.detector
    alarm: AlarmSystem = st.session_state.alarm_system

    source_type = st.radio(
        "Select Input Source",
        ["Browser Webcam (Photo)", "Upload Video", "Upload Image",
         "RTSP IP Camera", "OpenCV Webcam (Local Only)"],
        horizontal=True,
    )

    # ── Browser webcam photo ─────────────────────────────────────
    if source_type == "Browser Webcam (Photo)":
        st.info("📸 Captures a single photo (browser permission required)")
        img = st.camera_input("Take a photo for PPE detection")
        if img:
            frame     = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
            annotated, stats = det.process_frame(frame, do_faces=do_faces)
            _fire_voice_alarms(alarm, stats, enable_alarm, alarm_rate, alarm_pitch)

            c1, c2 = st.columns([2, 1])
            c1.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Detection result", use_container_width=True)
            with c2:
                st.metric("Detections",   stats["detections"])
                st.metric("Violations",   stats["violations_in_frame"])
                st.metric("Workers ID'd", stats["workers_identified"])
                st.metric("Logged to DB", stats["violations_logged"])
                _show_alarm_banner(stats)
                if stats["violations_in_frame"] > 0 and stats["violations_logged"] == 0:
                    st.warning(f"⏱️ {stats['pending_tracks']} violation(s) pending hold time")

    # ── Upload video ─────────────────────────────────────────────
    elif source_type == "Upload Video":
        f = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if f:
            tmp = os.path.join(tempfile.gettempdir(), f.name)
            with open(tmp, "wb") as fp:
                fp.write(f.read())
            st.success("✅ Video uploaded – processing …")
            _stream_video(tmp, det, do_faces, alarm,
                          enable_alarm, alarm_rate, alarm_pitch)

    # ── Upload image ─────────────────────────────────────────────
    elif source_type == "Upload Image":
        f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if f:
            tmp = os.path.join(tempfile.gettempdir(), f.name)
            with open(tmp, "wb") as fp:
                fp.write(f.read())
            frame     = cv2.imread(tmp)
            annotated, stats = det.process_frame(frame, do_faces=do_faces)
            _fire_voice_alarms(alarm, stats, enable_alarm, alarm_rate, alarm_pitch)

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="PPE Detection", use_container_width=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Detections", stats["detections"])
            mc2.metric("Violations", stats["violations_in_frame"])
            mc3.metric("Logged",     stats["violations_logged"])
            mc4.metric("Pending",    stats["pending_tracks"])
            _show_alarm_banner(stats)

    # ── RTSP ─────────────────────────────────────────────────────
    elif source_type == "RTSP IP Camera":
        url = st.text_input("RTSP URL",
                            placeholder="rtsp://user:pass@192.168.1.100:554/stream1")
        if url and st.button("📡 Start stream", type="primary"):
            _stream_video(url, det, do_faces, alarm,
                          enable_alarm, alarm_rate, alarm_pitch)

    # ── Local OpenCV webcam ───────────────────────────────────────
    elif source_type == "OpenCV Webcam (Local Only)":
        st.warning("⚠️ Only works when running Streamlit locally")
        if st.button("🎥 Start Webcam"):
            _stream_video(0, det, do_faces, alarm,
                          enable_alarm, alarm_rate, alarm_pitch)


# ── Video / stream loop ──────────────────────────────────────────
def _stream_video(source, det: PPEDetector, do_faces: bool,
                  alarm: AlarmSystem, enable_alarm: bool,
                  alarm_rate: float, alarm_pitch: float):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Cannot open video source"); return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_ph = st.empty()
    stat_ph  = st.empty()
    alarm_ph = st.empty()
    stop     = st.button("⏹ Stop Stream")
    n, t0    = 0, time.time()

    import threading
    _face_cache      = []
    _face_cache_lock = threading.Lock()
    _face_running    = threading.Event()

    def _update_faces(frame_copy, prev_copy):
        try:
            nf = face_utils.identify_faces(
                frame_copy, det._known_faces, prev_faces=prev_copy)
            with _face_cache_lock:
                _face_cache.clear(); _face_cache.extend(nf)
        except Exception:
            pass
        finally:
            _face_running.clear()

    while cap.isOpened() and not stop:
        ok, frame = cap.read()
        if not ok:
            st.warning("Stream ended."); break

        n += 1

        if do_faces and det._known_faces and not _face_running.is_set():
            _face_running.set()
            with _face_cache_lock:
                prev_snap = list(_face_cache)
            threading.Thread(target=_update_faces,
                             args=(frame.copy(), prev_snap), daemon=True).start()

        with _face_cache_lock:
            cur_faces = list(_face_cache)

        annotated, stats = det.process_frame(
            frame, do_faces=False, cached_faces=cur_faces)

        # ── Speak violation-specific warnings ──
        _fire_voice_alarms(alarm, stats, enable_alarm, alarm_rate, alarm_pitch)

        frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                       channels="RGB", use_container_width=True)

        if n % 5 == 0:
            fps = n / max(time.time() - t0, 0.001)
            with stat_ph.container():
                cols = st.columns(6)
                cols[0].metric("FPS",        f"{fps:.1f}")
                cols[1].metric("Detections", stats["detections"])
                cols[2].metric("Violations", stats["violations_in_frame"])
                cols[3].metric("Workers",    stats["workers_identified"])
                cols[4].metric("Logged",     stats["violations_logged"])
                cols[5].metric("Pending",    stats["pending_tracks"])

            if stats.get("violations_this_frame"):
                with alarm_ph.container():
                    _show_alarm_banner(stats)
            else:
                alarm_ph.empty()

    cap.release()


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 – WORKER REGISTRATION
# ═══════════════════════════════════════════════════════════════
def page_register():
    if _EXTRAS:
        colored_header(label="👤 Worker Registration",
                       description="Register workers with a photo for auto-identification",
                       color_name="green-70")
    else:
        st.title("👤 Worker Registration")

    tab_new, tab_list = st.tabs(["➕ Register new worker", "📋 All workers"])

    with tab_new:
        rc1, rc2 = st.columns(2)
        emp_id = rc1.text_input("Employee ID *", placeholder="EMP001")
        name   = rc1.text_input("Full Name *",   placeholder="Ravi Kumar")
        dept   = rc2.text_input("Department *",   placeholder="Construction")

        st.markdown("#### 📸 Capture or upload a photo")
        method = st.radio("Photo method", ["Camera", "Upload file"], horizontal=True)
        photo_path = None

        if method == "Camera":
            cam = st.camera_input("Take a photo")
            if cam:
                arr = cv2.imdecode(np.frombuffer(cam.read(), np.uint8), cv2.IMREAD_COLOR)
                photo_path = os.path.join(tempfile.gettempdir(), f"reg_{emp_id}.jpg")
                cv2.imwrite(photo_path, arr)
        else:
            up = st.file_uploader("Upload photo", type=["jpg","jpeg","png"])
            if up:
                photo_path = os.path.join(tempfile.gettempdir(), up.name)
                with open(photo_path, "wb") as fp:
                    fp.write(up.read())

        if st.button("✅ Register Worker", type="primary"):
            if not (emp_id and name and dept):
                st.error("Fill all required fields."); return
            if not photo_path:
                st.error("Provide a photo."); return
            with st.spinner("Processing …"):
                enc_blob, enc_msg = face_utils.generate_encoding_from_file(photo_path)
                if enc_blob is None and face_utils.FACE_REC_AVAILABLE:
                    st.error(enc_msg); return
                perm = os.path.join("database", "workers",
                                    f"{emp_id}_{datetime.now():%Y%m%d_%H%M%S}.jpg")
                shutil.copy(photo_path, perm)
                ok, msg = db.add_worker(emp_id, name, dept, perm, enc_blob)
                if ok:
                    st.success(msg); st.balloons()
                    if "detector" in st.session_state:
                        st.session_state.detector.reload_faces()
                else:
                    st.error(msg)

    with tab_list:
        wk = db.get_all_workers()
        if not wk:
            st.info("No workers registered yet."); return
        df = pd.DataFrame(wk)[["employee_id","name","department","created_at"]]
        df.columns = ["Employee ID","Name","Department","Registered"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        sel = st.selectbox("View details", [w["employee_id"] for w in wk],
            format_func=lambda x: f'{x} – {next(w["name"] for w in wk if w["employee_id"]==x)}')
        w = next(w for w in wk if w["employee_id"] == sel)
        ic1, ic2 = st.columns([1,2])
        with ic1:
            if w["image_path"] and os.path.exists(w["image_path"]):
                st.image(w["image_path"], caption=w["name"], use_container_width=True)
            else:
                st.warning("Photo not found")
        with ic2:
            st.markdown(f"**ID:** {w['employee_id']}")
            st.markdown(f"**Name:** {w['name']}")
            st.markdown(f"**Dept:** {w['department']}")
            st.markdown(f"**Since:** {w['created_at']}")
            st.markdown("---")
            st.markdown("##### :red[Danger Zone]")
            confirm = st.checkbox(f"Confirm delete **{w['name']}**",
                                  key=f"del_{w['employee_id']}")
            if st.button("🗑️ Delete", type="secondary",
                         disabled=not confirm, key=f"btn_{w['employee_id']}"):
                if w.get("image_path") and os.path.exists(w["image_path"]):
                    os.remove(w["image_path"])
                ok, msg = db.delete_worker(w["employee_id"])
                if ok:
                    st.success(f"Deleted {w['name']}")
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
                       description="Violation trends, department breakdown, severity",
                       color_name="violet-70")
    else:
        st.title("📊 Analytics Dashboard")

    dc1, dc2 = st.columns([2,1])
    chosen   = dc1.date_input("Filter date", value=datetime.now())
    date_str = chosen.strftime("%Y-%m-%d")
    if dc2.button("🔄 Refresh"):
        st.rerun()

    s = db.get_dashboard_stats(date_str)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total violations", s["total"])
    m2.metric("🔴 High severity", s["by_severity"].get("High", 0))
    m3.metric("Unique violators", len(s["by_employee"]))
    m4.metric("Departments hit",  len(s["by_dept"]))

    st.markdown("---")
    gc1, gc2 = st.columns(2)
    with gc1:
        st.markdown("#### 👥 Top violators")
        if s["by_employee"]:
            st.bar_chart(pd.DataFrame(s["by_employee"],
                         columns=["Name","Count"]).set_index("Name"))
        else:
            st.info("None")
    with gc2:
        st.markdown("#### 🏢 By department")
        if s["by_dept"]:
            st.bar_chart(pd.DataFrame(s["by_dept"],
                         columns=["Dept","Count"]).set_index("Dept"))
        else:
            st.info("None")

    if s["by_type"]:
        st.markdown("#### ⚠️ Violation types")
        st.dataframe(pd.DataFrame(s["by_type"], columns=["Type","Count"]),
                     use_container_width=True, hide_index=True)

    st.markdown("#### 🚨 Severity split")
    sv1, sv2, sv3 = st.columns(3)
    sv1.metric("🟢 Low",    s["by_severity"].get("Low", 0))
    sv2.metric("🟡 Medium", s["by_severity"].get("Medium", 0))
    sv3.metric("🔴 High",   s["by_severity"].get("High", 0))


# ═══════════════════════════════════════════════════════════════
#  PAGE 4 – VIOLATION LOGS
# ═══════════════════════════════════════════════════════════════
def page_logs():
    if _EXTRAS:
        colored_header(label="📋 Violation Logs",
                       description="Full log with worker name, department, severity",
                       color_name="red-70")
    else:
        st.title("📋 Violation Logs")

    lc1, lc2 = st.columns(2)
    d1 = lc1.date_input("From", value=datetime.now())
    d2 = lc2.date_input("To",   value=datetime.now())
    df = db.get_violations_df(start_date=d1, end_date=d2)

    if df.empty:
        st.info("No violations in the selected range."); return

    show_cols = ["timestamp","employee_id","name","department",
                 "violation_type","confidence","severity_level","status"]
    show = df[[c for c in show_cols if c in df.columns]].copy()
    show.columns = ["Time","Emp ID","Name","Dept",
                    "Violation","Confidence","Severity","Status"][:len(show.columns)]
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.download_button("📥 Download CSV", df.to_csv(index=False),
                       f"violations_{d1}_{d2}.csv", "text/csv")
    st.markdown("### 📊 Violation Statistics")
    st.bar_chart(df["violation_type"].value_counts())


# ═══════════════════════════════════════════════════════════════
#  PAGE 5 – DATABASE VIEWER
# ═══════════════════════════════════════════════════════════════
def page_database():
    if _EXTRAS:
        colored_header(label="🗄️ Database Explorer",
                       description="View all database tables from Supabase",
                       color_name="orange-70")
    else:
        st.title("🗄️ Database Explorer")

    st.info("💡 Live view of your remote Supabase database")
    
    c_ref, _ = st.columns([2, 5])
    if c_ref.button("🔄 Refresh Data", type="primary"):
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["👥 Workers", "⚠️ Violations", "📊 Stats"])

    import sqlite3

    with tab1:
        workers = db.get_all_workers()
        if not workers:
            st.warning("No workers registered yet.")
        else:
            df = pd.DataFrame(workers)
            display_cols = [c for c in df.columns if c != "face_encoding"]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
            st.download_button("📥 CSV", df[display_cols].to_csv(index=False),
                               "workers.csv", "text/csv")

    with tab2:
        fc1, fc2, fc3 = st.columns(3)
        limit       = fc1.selectbox("Last N rows", [10, 50, 100, 200, 500, 1000, 2000, 5000], index=3)
        sort_order  = fc2.selectbox("Sort", ["Newest first","Oldest first"])
        filter_type = fc3.selectbox("Filter type",
                                    ["All","NO-Mask","NO-Hardhat","NO-Safety Vest"])
        
        try:
            client = db.get_client()
            q = client.table("violation_logs").select("*")
            if filter_type != "All":
                q = q.eq("violation_type", filter_type)
                
            q = q.order("timestamp", desc=("Newest" in sort_order)).limit(limit)
            res = q.execute()
            
            df_v = pd.DataFrame(res.data or [])
            if df_v.empty:
                st.warning("No violations found.")
            else:
                st.dataframe(df_v, use_container_width=True, hide_index=True)
                st.download_button("📥 CSV", df_v.to_csv(index=False),
                                   "violations.csv", "text/csv")
        except Exception as e:
            st.error(f"Error fetching from Supabase: {e}")

    with tab3:
        db_size = os.path.getsize("database.db") if os.path.exists("database.db") else 0
        c1, c2  = st.columns(2)
        c1.metric("File size",         f"{db_size/1024:.1f} KB")
        c2.metric("Total violations",  db.get_dashboard_stats()["total"])

        st.markdown("#### 💻 Custom SQL")
        q_in = st.text_area("Query",
            "SELECT * FROM violation_logs ORDER BY timestamp DESC LIMIT 10;", height=80)
        if st.button("▶️ Run SQL"):
            try:
                conn = sqlite3.connect("database.db")
                res  = pd.read_sql_query(q_in, conn); conn.close()
                st.success(f"{len(res)} rows returned"); st.dataframe(res)
            except Exception as e:
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════
if   PAGE == "🎥 Live Detection":       page_detection()
elif PAGE == "👤 Worker Registration":   page_register()
elif PAGE == "📊 Dashboard":             page_dashboard()
elif PAGE == "📋 Violation Logs":        page_logs()
elif PAGE == "🗄️ Database Viewer":      page_database()