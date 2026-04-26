"""
alarm_utils.py - Browser-based voice alarm system for PPE violations.

Uses the browser's built-in Web Speech API (SpeechSynthesis) via
Streamlit's st.components.v1.html() — no extra libraries needed,
works on all platforms, produces real human-sounding speech.

Usage:
    from utils.alarm_utils import AlarmSystem
    alarm = AlarmSystem(cooldown_secs=5.0)
    alarm.play_alarm(violation_type="NO-Hardhat")   # speaks the warning
"""

import time
import threading
import streamlit.components.v1 as components
from typing import Dict, Optional


# ── Violation → spoken message mapping ──────────────────────────
VIOLATION_MESSAGES: Dict[str, str] = {
    "NO-Hardhat":      "Warning! Worker is not wearing a helmet. Please wear your helmet immediately.",
    "NO-Mask":         "Warning! Worker is not wearing a face mask. Please wear your mask immediately.",
    "NO-Safety Vest":  "Warning! Worker is not wearing a safety vest. Please wear your safety vest immediately.",
    "DEFAULT":         "Warning! A safety violation has been detected. Please wear all required PPE immediately.",
}


def speak_in_browser(message: str, rate: float = 0.9,
                     pitch: float = 1.0, volume: float = 1.0):
    """
    Inject a JS snippet that uses the browser's SpeechSynthesis API
    to speak the message in a real human voice.

    Works in Chrome, Edge, Firefox, Safari — no Python install needed.
    """
    safe_msg = message.replace("'", "\\'")

    js = f"""
    <script>
    (function() {{
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance('{safe_msg}');
        msg.rate   = {rate};
        msg.pitch  = {pitch};
        msg.volume = {volume};
        msg.lang   = 'en-US';

        // Pick the best available English voice
        function speakNow() {{
            var voices = window.speechSynthesis.getVoices();
            var pick = voices.find(function(v) {{
                return v.name.includes('Google US English') ||
                       v.name.includes('Microsoft Zira')   ||
                       v.name.includes('Microsoft David')  ||
                       v.name.includes('Samantha')          ||
                       v.name.includes('Karen')             ||
                       v.name.includes('Daniel')            ||
                       (v.lang === 'en-US' && v.localService);
            }});
            if (!pick) {{
                pick = voices.find(function(v) {{ return v.lang.startsWith('en'); }});
            }}
            if (pick) msg.voice = pick;
            window.speechSynthesis.speak(msg);
        }}

        // Voices may not be loaded yet on first call
        if (window.speechSynthesis.getVoices().length > 0) {{
            speakNow();
        }} else {{
            window.speechSynthesis.onvoiceschanged = speakNow;
        }}
    }})();
    </script>
    """
    components.html(js, height=0)


class AlarmSystem:
    """
    Voice alarm with per-violation-type cooldown.

    Uses speak_in_browser() → Web Speech API → real human voice.
    No beeps. No extra Python packages. Works on any OS.
    """

    def __init__(self, cooldown_secs: float = 5.0):
        self.cooldown = cooldown_secs
        # {violation_type: last_alarm_timestamp}
        self._last_alarm: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _should_alarm(self, key: str) -> bool:
        with self._lock:
            now  = time.time()
            last = self._last_alarm.get(key, 0.0)
            if now - last >= self.cooldown:
                self._last_alarm[key] = now
                return True
            return False

    def play_alarm(self,
                   violation_type: str = "DEFAULT",
                   rate: float = 0.9,
                   pitch: float = 1.0,
                   volume: float = 1.0,
                   # legacy params kept for backward-compat — ignored
                   frequency: int = 1000,
                   duration_ms: int = 500) -> None:
        """
        Speak the violation-specific warning through the browser.

        Args:
            violation_type : YOLO class name, e.g. "NO-Hardhat"
            rate           : speech speed  (0.5 slow → 1.5 fast)
            pitch          : voice pitch   (0.5 low  → 1.5 high)
            volume         : loudness      (0.0 mute → 1.0 max)
        """
        if not self._should_alarm(violation_type):
            return
        message = VIOLATION_MESSAGES.get(violation_type,
                                         VIOLATION_MESSAGES["DEFAULT"])
        speak_in_browser(message, rate=rate, pitch=pitch, volume=volume)

    def play_custom_message(self, message: str,
                            cooldown_key: str = "custom",
                            rate: float = 0.9) -> None:
        """Speak any custom message with its own cooldown key."""
        if not self._should_alarm(cooldown_key):
            return
        speak_in_browser(message, rate=rate)

    def reset_cooldown(self, violation_type: Optional[str] = None) -> None:
        with self._lock:
            if violation_type:
                self._last_alarm.pop(violation_type, None)
            else:
                self._last_alarm.clear()

    @staticmethod
    def get_message(violation_type: str) -> str:
        return VIOLATION_MESSAGES.get(violation_type,
                                      VIOLATION_MESSAGES["DEFAULT"])

    @staticmethod
    def add_message(violation_type: str, message: str) -> None:
        """Register a custom spoken message for a new violation type."""
        VIOLATION_MESSAGES[violation_type] = message