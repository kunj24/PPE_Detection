# AI-Powered Industrial Safety Surveillance System

## Project Overview

This project is a real-time industrial safety monitoring system built with **Streamlit**, **YOLOv8**, **OpenCV**, and **Supabase**. It detects Personal Protective Equipment (PPE) violations such as missing helmets, masks, and safety vests, then records those events and shows them in a dashboard.

### What problem does it solve?

In factories, construction sites, and industrial plants, supervisors often need to check whether workers are wearing the correct safety gear. Manual monitoring is slow, expensive, and easy to miss. This project automates that work by watching a camera feed, detecting PPE violations, and logging the result.

### Real-world use case

A construction site installs one camera at the gate and another near the work area. The system watches the live feed, identifies workers, checks for missing PPE, stores violations, and shows analytics in a dashboard. Supervisors can also get voice alerts in the browser when a violation is found.

### Key features

- Real-time PPE detection from webcam, video, image, and RTSP streams
- Face recognition for identifying registered workers
- Voice alerts in the browser using speech synthesis
- Violation logging with timestamps and confidence scores
- Dashboard with statistics, logs, and exports
- Supabase backend support with a local SQLite fallback for offline testing

---

## System Architecture

### High-level flow

```text
+---------------------------+
|      User / Operator      |
+-------------+-------------+
              |
              v
+---------------------------+
|        Streamlit UI       |
|  app.py (page routing)    |
+-------------+-------------+
              |
              v
+---------------------------+
|   Detection + Face Logic  |
| utils/detection_utils.py  |
| utils/face_utils.py       |
+-------------+-------------+
              |
     +--------+--------+
     |                 |
     v                 v
+-----------+   +------------------+
| YOLOv8    |   | Face Recognition |
| best.pt   |   | Haar + histogram |
+-----------+   +------------------+
     |                 |
     +--------+--------+
              |
              v
+---------------------------+
| Logging / Storage Layer   |
| utils/database_utils.py   |
| Supabase or SQLite        |
+-------------+-------------+
              |
              v
+---------------------------+
| Dashboard + CSV Export    |
| Alerts + Metrics + Logs   |
+---------------------------+
```

### Frontend, backend, and database interaction

- **Frontend:** Streamlit builds the pages for live detection, worker registration, dashboard, violation logs, and database viewer.
- **Backend:** Python handles frame processing, face detection, violation timing, and logging.
- **Database:** Supabase stores workers and violation records. If Supabase credentials are missing, the app uses local SQLite so the project can still run.

### Technologies used and why

| Technology | Why it is used |
| --- | --- |
| Streamlit | Fast way to build a web dashboard in Python |
| YOLOv8 (Ultralytics) | Accurate and fast PPE object detection |
| OpenCV | Camera input, image processing, and face detection |
| Supabase | Cloud PostgreSQL backend and storage |
| SQLite | Local fallback for development and offline work |
| Pandas | Tabular analytics and data export |
| Python | Main application language |

---

## Full Project Structure

```text
PPE_Detection/
├── app.py
├── best.pt
├── yolov8n.pt
├── .env
├── .env.example
├── database/
│   ├── workers/
│   └── violations/
├── database.db
├── PROJECT_DOCUMENTATION.md
├── README.md
├── requirements.txt
├── run.bat
├── supabase_schema.sql
├── view_database.py
├── violation_logs.csv
├── utils/
│   ├── alarm_utils.py
│   ├── database_utils.py
│   ├── detection_utils.py
│   ├── face_utils.py
│   └── __init__.py
└── static assets
    ├── home.jpeg
    ├── sample.mp4
    ├── system-architecture.png
    └── user-workflow.png
```

### Purpose of each important file

- `app.py` - Main Streamlit application and page router.
- `utils/detection_utils.py` - YOLOv8 PPE detection and violation tracking.
- `utils/face_utils.py` - Face detection, encoding, and worker matching.
- `utils/database_utils.py` - Database abstraction for Supabase and SQLite.
- `utils/alarm_utils.py` - Browser-based speech alerts for PPE violations.
- `supabase_schema.sql` - SQL schema for workers and violation logs.
- `run.bat` - Windows launcher for starting the app in one click.
- `view_database.py` - Simple command-line tool to inspect local SQLite data.
- `requirements.txt` - Python package dependencies.
- `.env.example` - Sample environment file for Supabase settings.

---

## Step-by-Step Working

### 1. User opens the app

The user starts the application using Streamlit. The sidebar allows switching between pages:

- Live Detection
- Worker Registration
- Dashboard
- Violation Logs
- Database Viewer

### 2. User provides input

Depending on the selected page, the user can:

- Capture a photo from the browser webcam
- Upload a video
- Upload an image
- Start an RTSP camera stream
- Register a worker by uploading or capturing a face photo

### 3. Frame processing begins

On the live detection page, each frame is passed into `PPEDetector.process_frame()`.

This method:

- Runs YOLOv8 on the frame
- Detects PPE violations
- Uses face recognition to match a worker
- Applies a hold-time threshold so the same issue is not logged too quickly
- Creates a snapshot for evidence
- Sends the log to Supabase or SQLite

### 4. Face matching happens

If face recognition is enabled, the system tries to identify the worker inside the detected PPE box. This helps the app store the worker name, employee ID, and department with the violation.

### 5. Violation is logged

When a violation stays visible long enough, the app logs it to the database.

Stored fields include:

- Timestamp
- Employee ID
- Name
- Department
- Violation type
- Confidence
- Snapshot path
- Camera location
- Severity
- Status

### 6. Browser voice alert plays

The browser receives a speech alert using the Web Speech API. This gives a human-sounding warning like:

- "Warning! Worker is not wearing a helmet. Please wear your helmet immediately."

### 7. Dashboard updates

The dashboard page reads the stored violations and shows:

- Total violations
- High severity cases
- Unique violators
- Department breakdown
- Violation type breakdown

### 8. Database viewer helps debugging

The database viewer page provides a raw look at workers and violation records. It also includes a custom SQL box for local SQLite queries.

---

## Complete Code Walkthrough

This section explains the most important code sections in simple terms. The full source lives in the project files listed above.

### 1. Main app entry point - `app.py`

`app.py` creates the UI, sets the pages, and connects everything together.

```python
st.set_page_config(
    page_title="AI PPE Surveillance",
    layout="wide",
    page_icon="👷",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    PAGE = st.radio(
        "Go to",
        ["🎥 Live Detection", "👤 Worker Registration",
         "📊 Dashboard", "📋 Violation Logs", "🗄️ Database Viewer"],
        label_visibility="collapsed",
    )
```

#### What this does

- Sets the browser tab title and layout.
- Builds the sidebar navigation.
- Stores the selected page in `PAGE`.

#### Important helper functions in `app.py`

```python
def _fire_voice_alarms(alarm: AlarmSystem, stats: dict,
                       enabled: bool, rate: float, pitch: float) -> None:
```

This function checks whether a new violation happened in the current frame and then tells the alarm system to speak the correct warning.

```python
def _show_alarm_banner(stats: dict) -> None:
```

This draws a red warning banner inside the UI for each violation type that happened in the frame.

### 2. Live detection page - `page_detection()`

```python
def page_detection():
    source_type = st.radio(
        "Select Input Source",
        ["Browser Webcam (Photo)", "Upload Video", "Upload Image",
         "RTSP IP Camera", "OpenCV Webcam (Local Only)"],
        horizontal=True,
    )
```

#### What this does

It lets the user choose the input source. The app can work with:

- Browser webcam photo
- Uploaded video
- Uploaded image
- RTSP IP camera
- Local OpenCV webcam

#### How the live detection works

- The selected frame is sent to the detector.
- YOLO finds PPE classes.
- The face matcher tries to identify a worker.
- The system decides whether a violation should be logged.
- The annotated image and metrics are shown to the user.

### 3. Video stream loop - `_stream_video()`

```python
def _stream_video(source, det: PPEDetector, do_faces: bool,
                  alarm: AlarmSystem, enable_alarm: bool,
                  alarm_rate: float, alarm_pitch: float):
```

#### What this does

This function opens the camera or video stream and processes every frame inside a loop.

### 4. Worker registration - `page_register()`

This page saves new workers and their face encodings.

```python
enc_blob, enc_msg = face_utils.generate_encoding_from_file(photo_path)
ok, msg = db.add_worker(emp_id, name, dept, perm, enc_blob)
```

#### Simple explanation

- The user enters employee ID, name, and department.
- The user uploads or captures a photo.
- The app creates a face encoding.
- The worker is saved in the database.

### 5. Dashboard - `page_dashboard()`

This page reads violation data and shows charts.

```python
s = db.get_dashboard_stats(date_str)
```

The result includes:

- Total number of violations
- Violations by employee
- Violations by department
- Violations by type
- Violations by severity

### 6. Database viewer - `page_database()`

This page helps inspect raw data.

- Workers table
- Violations table
- Stats panel
- Custom SQL query box

This is useful while debugging or testing the database layer.

---

## Backend Code - Detection, Face Matching, and Database

### A. PPE detection - `utils/detection_utils.py`

```python
class PPEDetector:
    def __init__(self, model_path: str = "best.pt",
                 threshold_secs: float = 3.0,
                 camera_location: str = "Main Camera"):
```

#### What the class does

This class keeps the detection state between frames.

- Loads the YOLO model
- Stores the violation hold time
- Tracks which violations are already being watched
- Loads known worker faces

#### Main method

```python
def process_frame(self, frame: np.ndarray, *,
                  do_faces: bool = True,
                  cached_faces: List[Dict] = None,
                  manual_worker: Dict = None,
                  prev_faces: List[Dict] = None) -> Tuple[np.ndarray, dict]:
```

#### Step-by-step inside `process_frame()`

1. Run YOLO on the frame.
2. Remove duplicate boxes that overlap too much.
3. Find faces in the frame.
4. Match faces with known workers.
5. Check if a violation class has stayed visible long enough.
6. Save a snapshot and log the event.
7. Return the annotated frame and a stats dictionary.

### B. Face detection and matching - `utils/face_utils.py`

This module uses OpenCV rather than heavy face-recognition libraries.

#### Main ideas

- Detect faces with Haar cascade
- Crop the face area
- Build a numeric encoding from color histogram and texture grid
- Compare encodings using cosine similarity
- Keep the best match only if it is confident enough

#### Important functions

```python
def generate_encoding_from_file(image_path: str) -> Tuple[Optional[bytes], str]:
```

Loads a photo from disk and generates a face encoding.

```python
def identify_faces(frame_bgr: np.ndarray,
                   known: Dict[str, tuple],
                   threshold: float = 0.62,
                   min_margin: float = 0.10,
                   enforce_unique: bool = True,
                   prev_faces: List[Dict] = None) -> List[Dict]:
```

Finds faces in the current frame and tries to match them with registered workers.

```python
def draw_face_labels(frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
```

Draws names and IDs on top of the video frame.

### C. Database layer - `utils/database_utils.py`

This module is the storage abstraction.

- If Supabase is configured, it uses cloud PostgreSQL and storage.
- If Supabase is not configured, it uses local SQLite.

#### Important functions

```python
def init_database():
```

Initializes the selected backend.

```python
def add_worker(employee_id: str, name: str, department: str,
               image_path: str, face_encoding: Optional[bytes]) -> Tuple[bool, str]:
```

Stores a new worker.

```python
def log_violation(employee_id: str, name: str, department: str,
                  violation_type: str, confidence: float,
                  snapshot_path: str = "",
                  camera_location: str = "Main Camera",
                  severity: str = "Low") -> bool:
```

Stores a PPE violation and evidence image.

```python
def get_dashboard_stats(date_str: Optional[str] = None) -> Dict:
```

Returns summary counts for the dashboard.

### D. Browser voice alarms - `utils/alarm_utils.py`

This module speaks the alert in the browser.

```python
def speak_in_browser(message: str, rate: float = 0.9,
                     pitch: float = 1.0, volume: float = 1.0):
```

Uses the browser's speech synthesis feature.

```python
class AlarmSystem:
```

This class adds a cooldown so the same warning does not repeat too quickly.

---

## Database Schema

The schema in `supabase_schema.sql` creates two tables.

### Workers table

- `employee_id` - Unique worker ID
- `name` - Full name
- `department` - Department name
- `image_path` - Stored photo path
- `face_encoding` - Encoded face data
- `created_at` - Registration time

### Violation logs table

- `timestamp` - When the violation happened
- `employee_id` - Worker ID
- `name` - Worker name
- `department` - Department name
- `violation_type` - Example: `NO-Hardhat`
- `confidence` - Detection confidence
- `image_snapshot_path` - Evidence image URL or local path
- `camera_location` - Camera name
- `severity_level` - Low, Medium, or High
- `status` - Open or closed

---

## Setup and Installation Guide

### Prerequisites

- Windows 10 or Windows 11
- Python 3.13 or compatible installed
- Git
- Internet connection for installing dependencies

### Step 1: Clone the project

```bash
git clone https://github.com/kunj24/PPE_Detection.git
cd PPE_Detection
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 4: Configure Supabase or local mode

If you want cloud storage and cloud logging, copy `.env.example` to `.env` and fill in:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-key-here
```

If you leave these values empty, the app uses local SQLite.

### Step 5: Run database setup

If using Supabase:

- Open `supabase_schema.sql`
- Run it in the Supabase SQL editor
- Create a public storage bucket named `violations`

If using local mode:

- The app will create `database.db`
- The required folders are `database/workers` and `database/violations`

### Step 6: Run the project

#### Option A - one-click launcher on Windows

```bash
run.bat
```

#### Option B - direct Streamlit command

```bash
streamlit run app.py
```

#### Option C - use the venv explicitly

```bash
.\venv\Scripts\python.exe -m streamlit run app.py
```

---

## Key Concepts Used

### 1. Object detection

YOLOv8 finds PPE-related classes inside each frame. The model returns boxes, class names, and confidence values.

### 2. Threshold-based logging

A violation is not always logged immediately. The code waits for the issue to stay visible for a few seconds. This reduces false alarms.

### 3. Duplicate box suppression

Sometimes a model draws more than one box around the same object. The detector removes overlapping duplicates before using the result.

### 4. Face matching and tracking

The system tries to map a worker face to the detected violation box. It also uses simple tracking so the same person can stay identified across frames.

### 5. Browser speech synthesis

Instead of playing a local beep, the app speaks the warning directly in the browser using JavaScript speech synthesis.

### 6. Local fallback pattern

The database layer supports two modes:

- Supabase when credentials are present
- SQLite when credentials are not present

This is a practical design pattern for development and deployment.

---

## Debugging and Common Errors

### Problem 1: `No module named streamlit`

**Cause:** Dependencies were installed in the wrong Python environment.

**Fix:**

```bash
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Problem 2: `FileNotFoundError: best.pt`

**Cause:** The model file is missing or the working directory is wrong.

**Fix:**

- Make sure `best.pt` exists in the project root
- Run the app from the repo root

### Problem 3: Missing `database/workers` folder

**Cause:** The app tries to save registration photos before the folder exists.

**Fix:**

```bash
mkdir database\workers
mkdir database\violations
```

### Problem 4: Supabase connection error

**Cause:** `.env` is missing or the URL/key is wrong.

**Fix:**

- Check `.env`
- Confirm `SUPABASE_URL`
- Confirm `SUPABASE_KEY`
- Make sure the schema has been run

### Problem 5: Face not recognized

**Cause:** The worker photo is low quality, the face is not visible, or the threshold is too strict.

**Fix:**

- Register a clearer front-facing photo
- Improve lighting
- Reduce the face threshold in `face_utils.py` if needed

### Problem 6: Voice alarm not speaking

**Cause:** Browser audio permissions are blocked.

**Fix:**

- Allow audio permissions in the browser
- Click the test voice alarm button first
- Refresh the page after permission changes

### Problem 7: App is slow on video

**Cause:** YOLO and face matching run on every frame.

**Fix:**

- Reduce input resolution
- Use a faster device or GPU
- Lower frame rate for testing

---

## Sample Inputs and Outputs

### Sample input 1: Worker registration

**Input:**

- Employee ID: `EMP001`
- Name: `Ravi Kumar`
- Department: `Construction`
- Photo: uploaded from camera

**Output:**

- Worker saved in the database
- Face encoding generated
- Worker appears in the All Workers list

### Sample input 2: PPE violation image

**Input:**

- Uploaded image of a worker not wearing a helmet

**Output:**

- Frame annotated with a red detection box
- Dashboard count increases
- Violation log saved
- Voice alert speaks the warning

### Sample output in the dashboard

- Total violations: 12
- High severity: 4
- Unique violators: 3
- Departments hit: 2

### Sample violation log row

| Time | Emp ID | Name | Dept | Violation | Confidence | Severity | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-01 20:44 | EMP001 | Ravi Kumar | Construction | NO-Hardhat | 0.91 | High | Open |

---

## Future Improvements

- Add SMS, email, or WhatsApp alerts
- Store violation video clips, not just images
- Add multi-camera support
- Add user authentication and role-based access
- Improve the face matcher with a stronger model
- Add a mobile-friendly supervisor dashboard
- Add export to PDF reports
- Add analytics for weekly and monthly trends

## Scalability Ideas

- Move image and video storage to cloud object storage
- Use a message queue for background logging
- Split the Streamlit UI and detection worker into separate services
- Add Docker support for easy deployment
- Use a managed PostgreSQL instance for production
- Add caching for frequently accessed dashboard data

---

## Final Notes

This project is a full example of how computer vision, web UI, and database storage can be combined into a practical safety system.

The important learning goals are:

- How a real-time AI app is structured
- How frame-by-frame detection works
- How to connect UI, backend, and database layers
- How to make a system robust with fallbacks and logging

For the actual source code, inspect these files directly:

- [app.py](app.py)
- [utils/detection_utils.py](utils/detection_utils.py)
- [utils/face_utils.py](utils/face_utils.py)
- [utils/database_utils.py](utils/database_utils.py)
- [utils/alarm_utils.py](utils/alarm_utils.py)
- [supabase_schema.sql](supabase_schema.sql)
- [run.bat](run.bat)
