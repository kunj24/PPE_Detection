"""
alarm_utils.py - Audio alarm system for PPE violations.

Provides:
  - Generate system beep tones
  - Play alarms with cooldown threshold (no spam)
  - Configurable alarm duration and frequency
"""

import time
import threading
import numpy as np
import platform
from typing import Optional


class AlarmSystem:
    """
    Audio alarm with cooldown threshold to prevent alarm spam.

    When a violation is detected, plays a beep sound.
    Subsequent alarms within the cooldown period are suppressed.
    """

    def __init__(self, cooldown_secs: float = 5.0):
        """
        Args:
            cooldown_secs: Minimum seconds between consecutive alarms
        """
        self.cooldown = cooldown_secs
        self._last_alarm_time = 0.0
        self._alarm_lock = threading.Lock()
        self._stop_event = threading.Event()

    def should_alarm(self) -> bool:
        """Check if enough time has passed since last alarm."""
        with self._alarm_lock:
            now = time.time()
            if now - self._last_alarm_time >= self.cooldown:
                self._last_alarm_time = now
                return True
            return False

    def play_alarm(self,
                   frequency: int = 1000,
                   duration_ms: int = 500,
                   sample_rate: int = 22050) -> None:
        """
        Generate and play a beep tone asynchronously.

        Args:
            frequency: Hz (default 1000 = A6 note)
            duration_ms: milliseconds
            sample_rate: audio sample rate
        """
        if not self.should_alarm():
            return  # cooldown active, skip

        # Run in daemon thread so it doesn't block
        threading.Thread(
            target=self._play_beep,
            args=(frequency, duration_ms),
            daemon=True
        ).start()

    def _play_beep(self, frequency: int, duration_ms: int) -> None:
        """Play beep using platform-specific method."""
        if platform.system() == "Windows":
            self._play_with_winsound(frequency, duration_ms)
        else:
            self._play_with_pyaudio(frequency, duration_ms, 22050)

    def _play_with_winsound(self, frequency: int, duration_ms: int) -> None:
        """Windows-only beep using built-in winsound."""
        try:
            import winsound
            winsound.Beep(frequency, duration_ms)
        except Exception as e:
            print(f"[ALARM] Beep failed: {e}")

    def _play_with_pyaudio(self,
                          frequency: int,
                          duration_ms: int,
                          sample_rate: int) -> None:
        """Cross-platform beep using PyAudio (if available)."""
        try:
            import pyaudio
            import wave
            import tempfile
            import os

            # Generate sine wave
            num_samples = int(sample_rate * duration_ms / 1000.0)
            t = np.linspace(0, duration_ms / 1000.0, num_samples, False)
            waveform = np.sin(2 * np.pi * frequency * t).astype(np.float32)

            # Convert to 16-bit PCM
            audio_data = (waveform * 32767).astype(np.int16)

            # Write to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                with wave.open(tmp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)           # mono
                    wav_file.setsampwidth(2)           # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data.tobytes())

                # Play the WAV file
                with wave.open(tmp_path, 'rb') as wav_file:
                    p = pyaudio.PyAudio()
                    stream = p.open(
                        format=p.get_format_from_width(wav_file.getsampwidth()),
                        channels=wav_file.getnchannels(),
                        rate=wav_file.getframerate(),
                        output=True
                    )

                    data = wav_file.readframes(1024)
                    while data and not self._stop_event.is_set():
                        stream.write(data)
                        data = wav_file.readframes(1024)

                    stream.stop_stream()
                    stream.close()
                    p.terminate()
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except ImportError:
            print("[ALARM] PyAudio not available, skipping non-Windows audio")
        except Exception as e:
            print(f"[ALARM] PyAudio beep failed: {e}")

    def stop(self) -> None:
        """Signal any playing alarm to stop."""
        self._stop_event.set()

    def reset_cooldown(self) -> None:
        """Reset cooldown timer (allows immediate next alarm)."""
        with self._alarm_lock:
            self._last_alarm_time = 0.0
