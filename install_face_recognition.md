# Installing face_recognition on Windows

The `dlib` library (required by `face_recognition`) needs C++ compilation on Windows. Here are **3 working options**:

---

## ✅ Option 1: Install Visual Studio Build Tools (Recommended for pip/venv users)

1. Download **Visual Studio Build Tools**: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

2. Run the installer and select:
   - ✅ **"Desktop development with C++"** workload
   - This is ~7GB but enables compiling all Python C extensions

3. After installation, restart your terminal and run:
   ```powershell
   & e:\AI-PPE-Detection-main\PPE_DETECTION\.venv\Scripts\Activate.ps1
   pip install dlib
   pip install face-recognition
   ```

4. Verify it works:
   ```powershell
   python -c "import face_recognition; print('✅ face_recognition installed')"
   ```

5. Restart Streamlit:
   ```powershell
   streamlit run app.py
   ```

---

## ✅ Option 2: Use Conda (Easiest - no C++ compiler needed)

Conda has prebuilt binaries for everything:

```powershell
# Install Miniconda if you don't have it: https://docs.conda.io/en/latest/miniconda.html

# Create new environment
conda create -n ppe python=3.11 -y
conda activate ppe

# Install face_recognition and dlib (prebuilt!)
conda install -c conda-forge dlib face-recognition -y

# Install other dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ✅ Option 3: Continue WITHOUT face recognition (Current state)

The app **already works** without `face_recognition`:
- ✅ Worker registration still saves photos
- ✅ Violation detection works normally  
- ✅ All other features functional
- ❌ Face-based worker identification is disabled

**Nothing breaks** - you just won't see worker names in the Live Detection view automatically.

---

## 🔧 Troubleshooting

If you get errors about missing `vcruntime140.dll` or similar after installing:
- Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe

---

**My recommendation:** Use **Option 2 (Conda)** if you want face recognition now, or **Option 3** (continue without it) if you just need the core PPE detection working.
