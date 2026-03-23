"""
PandAI Assistant — Real-time AI conversation assistant
Captures mic + system audio, transcribes with faster-whisper (CUDA),
suggests responses via Claude API, displayed as a transparent overlay.
"""

import sys
import os
import threading
import queue
import time
import json
import numpy as np
import sounddevice as sd
import anthropic
from scipy.signal import resample_poly
from math import gcd
from faster_whisper import WhisperModel
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit,
    QScrollArea, QFrame, QSizeGrip, QSlider, QStackedWidget,
    QSizePolicy, QTabWidget, QCheckBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPoint, QSize, QPropertyAnimation, QEasingCurve,
    QObject
)
from PyQt6.QtGui import (
    QColor, QPainter, QBrush, QPen, QFont, QFontDatabase,
    QLinearGradient, QPalette, QCursor, QIcon
)

# ── HOTKEY SIGNALER ───────────────────────────────────────────────────────────
class _HotkeySignaler(QObject):
    """Bridges the keyboard library's background thread into Qt's main thread."""
    triggered = pyqtSignal()

# ── VERSION ──────────────────────────────────────────────────────────────────
GITHUB_REPO = "AHappyPandaaa/pandai-assistant"

def _local_version():
    """Read version from version.txt next to this file."""
    try:
        vfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version.txt")
        with open(vfile) as f:
            return f.read().strip()
    except Exception:
        return None

# ── CONFIG ──────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
WINDOW_SECONDS = 8     # longer window = more context = fewer wrong words
STEP_SECONDS   = 3     # step forward every N seconds (balance speed vs repetition)
WHISPER_MODEL  = "medium"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"
CONFIG_FILE   = os.path.join(os.path.expanduser("~"), ".pandai_assistant.json")
SESSIONS_FILE = os.path.join(os.path.expanduser("~"), ".pandai_sessions.json")

ANALYSIS_SYSTEM_PROMPT = (
    "You are a real-time conversation intelligence assistant. "
    "The user has selected a piece of transcript to analyze.\n"
    "Provide:\n"
    "1. A concise depth analysis (2-4 sentences) of what is being discussed\n"
    "2. Connected topics — 2-4 adjacent concepts or ideas that relate to this, "
    "even if not explicitly mentioned in the conversation\n"
    "3. 2-3 smart follow-up questions the user could raise to go deeper\n\n"
    "Respond ONLY in JSON:\n"
    '{"analysis":"...","connected":[{"icon":"...","title":"...","detail":"..."}],"followups":["..."]}'
)

HIGHLIGHT_SYSTEM_PROMPT = (
    "Extract the 3-8 most important, substantive phrases from this conversation transcript. "
    "Focus on specific topics, technical terms, names, decisions, or key concepts. "
    "Ignore small talk, filler words, and simple greetings. "
    "Return ONLY a JSON array of short phrase strings (2-6 words each), "
    'e.g.: ["API rate limiting", "pricing model", "Q3 deadline"]'
)

# ── CONFIG PERSISTENCE ───────────────────────────────────────────────────────
def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

def load_sessions():
    try:
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_sessions(sessions):
    try:
        with open(SESSIONS_FILE, "w") as f:
            json.dump(sessions[-50:], f, indent=2)
    except Exception:
        pass

# ── AUDIO DEVICE HELPERS ─────────────────────────────────────────────────────
def get_audio_devices():
    """Return list of (index, name, max_input_channels) for all input devices."""
    devices = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                devices.append((i, d["name"], d["max_input_channels"]))
    except Exception:
        pass
    return devices

# ── CUDA DLL PATH HELPER ─────────────────────────────────────────────────────
def _add_cuda_dll_paths():
    """
    On Windows, pip installs CUDA libs into site-packages/nvidia/*/bin/
    but doesn't add them to PATH. We find and add them manually so
    ctranslate2/faster-whisper can locate cublas64_12.dll etc.
    """
    import site, os
    added = []
    for sp in site.getsitepackages():
        nvidia_root = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for pkg in os.listdir(nvidia_root):
            bin_dir = os.path.join(nvidia_root, pkg, "bin")
            if os.path.isdir(bin_dir) and bin_dir not in os.environ.get("PATH", ""):
                os.add_dll_directory(bin_dir)   # Python 3.8+ Windows API
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                added.append(bin_dir)
    return added

# ── WHISPER LOADER THREAD ────────────────────────────────────────────────────
def _nvidia_gpu_available():
    """Check if an NVIDIA GPU is present and CUDA is usable."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False

class WhisperLoader(QThread):
    done = pyqtSignal(object, str)   # model, error
    status = pyqtSignal(str)         # progress messages

    def run(self):
        # Check for NVIDIA GPU first — skip CUDA entirely if not present
        has_nvidia = _nvidia_gpu_available()

        if has_nvidia:
            self.status.emit("⏳  Loading Whisper on GPU (CUDA)...")
            try:
                _add_cuda_dll_paths()
            except Exception as e:
                print(f"DLL path warning: {e}")

            try:
                model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
                self.done.emit(model, "")
                return
            except Exception as e:
                print(f"CUDA failed, falling back to CPU: {e}")

        # CPU fallback — either no NVIDIA GPU or CUDA failed
        try:
            self.status.emit("⏳  Loading Whisper on CPU...")
            model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            msg = "" if has_nvidia else "No NVIDIA GPU detected — running on CPU"
            self.done.emit(model, msg)
        except Exception as e:
            self.done.emit(None, str(e))
        except Exception as e2:
            self.done.emit(None, str(e2))

# ── UPDATE CHECKER ───────────────────────────────────────────────────────────
class UpdateChecker(QThread):
    update_available = pyqtSignal(str)  # emits remote version string

    def run(self):
        local_ver = _local_version()
        if not local_ver:
            return
        try:
            import urllib.request
            url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/version.txt"
            req = urllib.request.Request(url, headers={"User-Agent": "PandAI-Assistant"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                remote_ver = resp.read().decode().strip()
            if remote_ver and remote_ver != local_ver:
                self.update_available.emit(remote_ver)
        except Exception:
            pass  # silently ignore — no network, etc.

# ── AUTO UPDATER ──────────────────────────────────────────────────────────────
class AutoUpdater(QThread):
    """Downloads the latest code as a ZIP from GitHub and extracts it in-place."""
    finished = pyqtSignal(bool, str)  # (success, message)

    def run(self):
        import urllib.request, zipfile, tempfile, shutil
        app_dir = os.path.dirname(os.path.abspath(__file__))
        zip_url = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/main.zip"
        try:
            # Download zip to a temp file
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp_path = tmp.name
            urllib.request.urlretrieve(zip_url, tmp_path)

            # Extract into a temp directory
            with tempfile.TemporaryDirectory() as extract_dir:
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    zf.extractall(extract_dir)

                # The zip contains a single top-level folder, e.g. "pandai-assistant-main"
                entries = os.listdir(extract_dir)
                if not entries:
                    raise RuntimeError("Downloaded zip was empty")
                src_dir = os.path.join(extract_dir, entries[0])

                # Copy every file from the zip into the app directory
                for item in os.listdir(src_dir):
                    src = os.path.join(src_dir, item)
                    dst = os.path.join(app_dir, item)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)

            os.unlink(tmp_path)

            # Install any new dependencies silently
            req_file = os.path.join(app_dir, "requirements.txt")
            if os.path.exists(req_file):
                import subprocess
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"],
                    timeout=120
                )

            self.finished.emit(True, "Update complete! Restart to apply.")
        except Exception as e:
            self.finished.emit(False, f"Update failed: {e}")

# ── HELPERS ──────────────────────────────────────────────────────────────────
def get_device_native_rate(device_idx):
    """Return native sample rate for a device, defaulting to 44100."""
    try:
        info = sd.query_devices(device_idx) if device_idx is not None else sd.query_devices(kind="input")
        return int(info["default_samplerate"])
    except Exception:
        return 44100

def resample_to_whisper(audio: np.ndarray, native_rate: int) -> np.ndarray:
    """Resample audio from native_rate to SAMPLE_RATE (16000) for Whisper."""
    if native_rate == SAMPLE_RATE:
        return audio
    g = gcd(native_rate, SAMPLE_RATE)
    up = SAMPLE_RATE // g
    down = native_rate // g
    return resample_poly(audio, up, down).astype(np.float32)

# ── AUDIO CAPTURE THREAD ─────────────────────────────────────────────────────
class AudioCapture(QThread):
    """
    Sliding-window audio capture.
    - Collects audio continuously from mic (+ optional system loopback)
    - Every STEP_SECONDS, emits a WINDOW_SECONDS-long chunk for Whisper
    - This gives near-realtime transcription while keeping enough context
      for Whisper to be accurate.
    """
    mic_chunk_ready = pyqtSignal(np.ndarray)  # float32 mono at SAMPLE_RATE — mic source
    sys_chunk_ready = pyqtSignal(np.ndarray)  # float32 mono at SAMPLE_RATE — system source
    level_changed   = pyqtSignal(float)        # 0.0–1.0 mic RMS level for VU meter

    def __init__(self, mic_idx=None, sys_idx=None):
        super().__init__()
        self.mic_idx  = mic_idx
        self.sys_idx  = sys_idx
        self._stop    = threading.Event()
        self._ring_mic = []   # rolling buffer at native rate
        self._ring_sys = []
        self._lock    = threading.Lock()
        self._mic_rate = get_device_native_rate(mic_idx)
        self._sys_rate = get_device_native_rate(sys_idx) if sys_idx is not None else SAMPLE_RATE
        # Timer fires every STEP_SECONDS to emit a window
        self._last_emit = 0.0
        self._last_level_emit = 0.0

    def stop(self):
        self._stop.set()

    def run(self):
        streams = []
        try:
            def mic_cb(indata, frames, time_info, status):
                mono = indata[:, 0].copy()
                with self._lock:
                    self._ring_mic.extend(mono.tolist())
                    # Keep only the last WINDOW_SECONDS worth at native rate
                    max_samples = self._mic_rate * (WINDOW_SECONDS + 1)
                    if len(self._ring_mic) > max_samples:
                        self._ring_mic = self._ring_mic[-max_samples:]
                # Throttled level emission (~10 fps) for VU meter
                now = time.time()
                if now - self._last_level_emit >= 0.1:
                    rms = float(np.sqrt(np.mean(mono ** 2)))
                    self.level_changed.emit(min(1.0, rms * 10))
                    self._last_level_emit = now

            def sys_cb(indata, frames, time_info, status):
                mono = indata[:, 0].copy()
                with self._lock:
                    self._ring_sys.extend(mono.tolist())
                    max_samples = self._sys_rate * (WINDOW_SECONDS + 1)
                    if len(self._ring_sys) > max_samples:
                        self._ring_sys = self._ring_sys[-max_samples:]

            mic_kwargs = dict(samplerate=self._mic_rate, channels=1, dtype="float32", callback=mic_cb)
            if self.mic_idx is not None:
                mic_kwargs["device"] = self.mic_idx
            streams.append(sd.InputStream(**mic_kwargs))

            if self.sys_idx is not None:
                sys_kwargs = dict(device=self.sys_idx, samplerate=self._sys_rate,
                                  channels=1, dtype="float32", callback=sys_cb)
                streams.append(sd.InputStream(**sys_kwargs))

            for s in streams:
                s.start()

            self._last_emit = time.time()
            while not self._stop.is_set():
                now = time.time()
                if now - self._last_emit >= STEP_SECONDS:
                    self._emit_window()
                    self._last_emit = now
                time.sleep(0.05)

        except Exception as e:
            print(f"Audio capture error: {e}")
        finally:
            for s in streams:
                try: s.stop(); s.close()
                except Exception: pass

    def _emit_window(self):
        with self._lock:
            window_native = self._mic_rate * WINDOW_SECONDS
            has_mic = len(self._ring_mic) >= self._mic_rate * 0.5

            mic_chunk = None
            if has_mic:
                mic_native = np.array(self._ring_mic[-window_native:], dtype=np.float32)
                mic_chunk  = resample_to_whisper(mic_native, self._mic_rate)

            sys_chunk = None
            if self._ring_sys:
                window_sys = self._sys_rate * WINDOW_SECONDS
                sys_native = np.array(self._ring_sys[-window_sys:], dtype=np.float32)
                sys_chunk  = resample_to_whisper(sys_native, self._sys_rate)

        if mic_chunk is not None:
            self.mic_chunk_ready.emit(mic_chunk)
        if sys_chunk is not None:
            self.sys_chunk_ready.emit(sys_chunk)

# ── LEVEL METER WIDGET ───────────────────────────────────────────────────────
class LevelMeter(QWidget):
    """Full-width audio level bar with Apple-style gradient fill."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0
        self.setFixedHeight(3)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_level(self, level: float):
        self._level = max(0.0, min(1.0, level))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        # Track
        painter.fillRect(0, 0, w, h, QColor(84, 84, 88, 60))
        # Active gradient fill
        if self._level > 0:
            fill_w = max(4, int(w * self._level))
            grad = QLinearGradient(0, 0, fill_w, 0)
            grad.setColorAt(0.0, QColor("#34C759"))
            grad.setColorAt(0.65, QColor("#00C7FF"))
            grad.setColorAt(1.0, QColor("#FF453A") if self._level > 0.85 else QColor("#00C7FF"))
            painter.fillRect(0, 0, fill_w, h, QBrush(grad))

# ── TRANSCRIPTION THREAD ─────────────────────────────────────────────────────
class TranscribeWorker(QThread):
    result = pyqtSignal(str, str)  # (text, source) — source is "mic" or "sys"

    def __init__(self, model, audio: np.ndarray, source: str = "mic", no_speech_thresh: float = 0.6):
        super().__init__()
        self.model  = model
        self.audio  = audio
        self.source = source
        self.no_speech_thresh = no_speech_thresh

    def run(self):
        try:
            segments, info = self.model.transcribe(
                self.audio,
                language="en",
                beam_size=3,
                best_of=2,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 400,
                    "speech_pad_ms": 200,
                    "threshold": 0.4,
                },
                no_speech_threshold=self.no_speech_thresh,
                compression_ratio_threshold=2.0,
                log_prob_threshold=-0.8,
            )
            parts = []
            for s in segments:
                txt = s.text.strip()
                if not txt:
                    continue
                if hasattr(s, 'no_speech_prob') and s.no_speech_prob > self.no_speech_thresh:
                    continue
                parts.append(txt)
            text = " ".join(parts).strip()
            if text:
                self.result.emit(text, self.source)
        except Exception as e:
            print(f"Transcription error: {e}")

# ── CLAUDE ANALYSIS WORKER ────────────────────────────────────────────────────
class ClaudeAnalysisWorker(QThread):
    """On-demand deep analysis of a user-selected transcript excerpt."""
    result = pyqtSignal(dict)
    error  = pyqtSignal(str)

    def __init__(self, api_key, selection, context, briefing=""):
        super().__init__()
        self.api_key   = api_key
        self.selection = selection
        self.context   = context
        self.briefing  = briefing

    def run(self):
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            user_content = (
                f'Selected text:\n"{self.selection}"\n\n'
                f'Full conversation context:\n"{self.context[-1500:]}"'
            )
            if self.briefing:
                user_content = f"Session context: {self.briefing}\n\n{user_content}"
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                system=ANALYSIS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = msg.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            self.result.emit(data)
        except json.JSONDecodeError:
            self.error.emit("Unexpected response format — please try again.")
        except Exception as e:
            self.error.emit(str(e))


# ── HIGHLIGHT WORKER ──────────────────────────────────────────────────────────
class HighlightWorker(QThread):
    """Background pass every ~12s — extracts key phrases to highlight in transcript."""
    phrases_ready = pyqtSignal(list)

    def __init__(self, api_key, transcript):
        super().__init__()
        self.api_key    = api_key
        self.transcript = transcript

    def run(self):
        if not self.api_key or len(self.transcript.strip().split()) < 15:
            return
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                system=HIGHLIGHT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f'Transcript:\n"{self.transcript[-1000:]}"'}],
            )
            raw = msg.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            phrases = json.loads(raw)
            if isinstance(phrases, list):
                self.phrases_ready.emit([p for p in phrases if isinstance(p, str)])
        except Exception:
            pass  # highlights are best-effort, never block the UI


# ── SESSION TITLE WORKER ──────────────────────────────────────────────────────
class SessionTitleWorker(QThread):
    """Generates a short auto-title for a completed session."""
    title_ready = pyqtSignal(str)

    def __init__(self, api_key, transcript, briefing=""):
        super().__init__()
        self.api_key    = api_key
        self.transcript = transcript
        self.briefing   = briefing

    def run(self):
        if not self.api_key or len(self.transcript.strip().split()) < 10:
            return
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            ctx = (f"Briefing: {self.briefing}\n" if self.briefing else "")
            ctx += f"Transcript excerpt:\n{self.transcript[:600]}"
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=30,
                system="Generate a short (4-7 word) title for this conversation session. Return ONLY the title, no quotes or punctuation.",
                messages=[{"role": "user", "content": ctx}],
            )
            title = msg.content[0].text.strip().strip('"\'')
            if title:
                self.title_ready.emit(title)
        except Exception:
            pass

# ── SETTINGS DIALOG ───────────────────────────────────────────────────────────
from PyQt6.QtWidgets import QDialog, QLineEdit, QDialogButtonBox, QGridLayout, QGroupBox, QScrollArea as _QScrollArea

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = dict(config)
        self.setWindowTitle("PandAI Assistant — Settings")
        self.setFixedSize(500, 660)
        self._apply_theme_styles()
        self._build_ui()

    def _apply_theme_styles(self):
        dark = (self.config.get("theme", "dark") != "light")
        if dark:
            dlg_bg     = "#1C1C1E"
            dlg_fg     = "#F2F2F7"
            lbl_col    = "rgba(235,235,245,0.5)"
            inp_bg     = "#2C2C2E"
            inp_br     = "rgba(255,255,255,0.1)"
            inp_fg     = "#F2F2F7"
            focus_br   = "rgba(0,199,255,0.5)"
            grp_br     = "rgba(255,255,255,0.08)"
            grp_col    = "rgba(235,235,245,0.4)"
            chk_fg     = "#F2F2F7"
            chk_ind_bg = "#3A3A3C"
            chk_ind_br = "rgba(255,255,255,0.18)"
            cancel_bg  = "rgba(255,255,255,0.08)"
            cancel_br  = "rgba(255,255,255,0.12)"
            cancel_fg  = "rgba(235,235,245,0.45)"
            scroll_h   = "rgba(255,255,255,0.18)"
            accent     = "#00C7FF"
        else:
            dlg_bg     = "#F2F2F7"
            dlg_fg     = "#1C1C1E"
            lbl_col    = "rgba(60,60,67,0.6)"
            inp_bg     = "#FFFFFF"
            inp_br     = "rgba(60,60,67,0.13)"
            inp_fg     = "#1C1C1E"
            focus_br   = "rgba(0,122,255,0.5)"
            grp_br     = "rgba(60,60,67,0.1)"
            grp_col    = "rgba(60,60,67,0.45)"
            chk_fg     = "#1C1C1E"
            chk_ind_bg = "#FFFFFF"
            chk_ind_br = "rgba(60,60,67,0.2)"
            cancel_bg  = "rgba(0,0,0,0.05)"
            cancel_br  = "rgba(60,60,67,0.13)"
            cancel_fg  = "rgba(60,60,67,0.5)"
            scroll_h   = "rgba(60,60,67,0.22)"
            accent     = "#007AFF"

        self.setStyleSheet(f"""
            QDialog {{ background: {dlg_bg}; color: {dlg_fg}; font-family: 'Segoe UI', sans-serif; }}
            QLabel {{ color: {lbl_col}; font-size: 10pt; }}
            QLineEdit {{
                background: {inp_bg}; border: 1px solid {inp_br};
                border-radius: 8px; padding: 8px 10px;
                color: {inp_fg}; font-size: 10pt;
            }}
            QLineEdit:focus {{ border-color: {focus_br}; }}
            QComboBox {{
                background: {inp_bg}; border: 1px solid {inp_br};
                border-radius: 8px; padding: 8px 10px;
                color: {inp_fg}; font-size: 10pt;
            }}
            QComboBox QAbstractItemView {{
                background: {inp_bg}; color: {inp_fg};
                selection-background-color: rgba(0,122,255,0.12);
            }}
            QTextEdit {{
                background: {inp_bg}; border: 1px solid {inp_br};
                border-radius: 8px; padding: 6px;
                color: {inp_fg}; font-size: 9pt;
            }}
            QPushButton {{
                background: {accent};
                border: none; border-radius: 8px; padding: 10px;
                color: white; font-size: 10pt; font-weight: 600;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
            QPushButton#cancel {{
                background: {cancel_bg};
                border: 1px solid {cancel_br};
                color: {cancel_fg};
            }}
            QPushButton#show_btn {{
                background: {cancel_bg};
                border: 1px solid {cancel_br};
                color: {cancel_fg};
                border-radius: 8px; padding: 6px; font-size: 10px;
            }}
            QGroupBox {{
                border: 1px solid {grp_br};
                border-radius: 10px; margin-top: 8px; padding: 12px;
                color: {grp_col}; font-size: 9pt; font-weight: 600;
            }}
            QCheckBox {{ color: {chk_fg}; font-size: 10pt; spacing: 8px; }}
            QCheckBox::indicator {{
                width: 18px; height: 18px;
                border: 1px solid {chk_ind_br}; border-radius: 5px;
                background: {chk_ind_bg};
            }}
            QCheckBox::indicator:checked {{
                background: {accent}; border-color: transparent;
            }}
            QSlider::groove:horizontal {{
                height: 3px; background: {inp_br}; border-radius: 1px;
            }}
            QSlider::handle:horizontal {{
                width: 14px; height: 14px; margin: -6px 0;
                background: {accent}; border-radius: 7px;
            }}
            QScrollBar:vertical {{ background: transparent; width: 3px; }}
            QScrollBar::handle:vertical {{ background: {scroll_h}; border-radius: 1px; min-height: 20px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        # API Key
        api_group = QGroupBox("Anthropic API Key")
        api_layout = QVBoxLayout(api_group)
        self.api_input = QLineEdit(self.config.get("api_key", ""))
        self.api_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_input.setPlaceholderText("sk-ant-...")
        hint = QLabel("Get yours at console.anthropic.com — stored locally only")
        hint.setStyleSheet("color: #475569; font-size: 9pt;")
        show_btn = QPushButton("👁 Show")
        show_btn.setFixedWidth(70)
        show_btn.setStyleSheet("background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); color: #94a3b8; border-radius:6px; padding:6px; font-size:11px;")
        show_btn.clicked.connect(lambda: self.api_input.setEchoMode(
            QLineEdit.EchoMode.Normal if self.api_input.echoMode() == QLineEdit.EchoMode.Password
            else QLineEdit.EchoMode.Password
        ))
        row = QHBoxLayout()
        row.addWidget(self.api_input)
        row.addWidget(show_btn)
        api_layout.addLayout(row)
        api_layout.addWidget(hint)
        layout.addWidget(api_group)

        # Devices
        dev_group = QGroupBox("Audio Devices")
        dev_layout = QGridLayout(dev_group)
        dev_layout.addWidget(QLabel("Microphone (your voice):"), 0, 0)
        dev_layout.addWidget(QLabel("System audio (incoming):"), 1, 0)
        dev_layout.addWidget(QLabel("Capture mode:"), 2, 0)
        self.mic_combo = QComboBox()
        self.sys_combo = QComboBox()
        self.sys_combo.addItem("None — mic only", None)
        self.capture_mode_combo = QComboBox()
        self.capture_mode_combo.addItem("Both — mic + inbound audio", "both")
        self.capture_mode_combo.addItem("Inbound only — capture their voice", "inbound")
        self.capture_mode_combo.addItem("Mic only — capture my voice", "mic")
        devices = get_audio_devices()
        saved_mic = self.config.get("mic_device")
        saved_sys = self.config.get("sys_device")
        saved_mode = self.config.get("capture_mode", "both")
        for idx, name, _ in devices:
            self.mic_combo.addItem(name, idx)
            self.sys_combo.addItem(name, idx)
        # Restore saved selections
        for i in range(self.mic_combo.count()):
            if self.mic_combo.itemData(i) == saved_mic:
                self.mic_combo.setCurrentIndex(i)
        for i in range(self.sys_combo.count()):
            if self.sys_combo.itemData(i) == saved_sys:
                self.sys_combo.setCurrentIndex(i)
        for i in range(self.capture_mode_combo.count()):
            if self.capture_mode_combo.itemData(i) == saved_mode:
                self.capture_mode_combo.setCurrentIndex(i)
        dev_layout.addWidget(self.mic_combo, 0, 1)
        dev_layout.addWidget(self.sys_combo, 1, 1)
        dev_layout.addWidget(self.capture_mode_combo, 2, 1)
        vb_row = QHBoxLayout()
        hint2 = QLabel("System audio: select  CABLE Output (VB-Audio Virtual Cable)  from the list above.")
        hint2.setStyleSheet("color: #475569; font-size: 9pt;")
        hint2.setWordWrap(True)
        vb_btn = QPushButton("Get VB-Cable (free) ↗")
        vb_btn.setObjectName("cancel")
        vb_btn.setFixedHeight(24)
        vb_btn.setStyleSheet("font-size: 8pt; padding: 2px 8px;")
        import webbrowser
        vb_btn.clicked.connect(lambda: webbrowser.open("https://vb-audio.com/Cable/"))
        vb_row.addWidget(hint2, 1)
        vb_row.addWidget(vb_btn)
        dev_layout.addLayout(vb_row, 3, 0, 1, 2)
        layout.addWidget(dev_group)

        # Hotkey
        hotkey_group = QGroupBox("Keyboard Shortcut")
        hotkey_layout = QVBoxLayout(hotkey_group)
        self.hotkey_input = QLineEdit(self.config.get("hotkey", "ctrl+shift+space"))
        self.hotkey_input.setPlaceholderText("e.g. ctrl+shift+space")
        hotkey_hint = QLabel("Global hotkey to show/hide the overlay (uses the 'keyboard' library syntax)")
        hotkey_hint.setStyleSheet("color: #475569; font-size: 9pt;")
        hotkey_hint.setWordWrap(True)
        hotkey_layout.addWidget(self.hotkey_input)
        hotkey_layout.addWidget(hotkey_hint)
        # Show hotkey registration status from parent window
        hotkey_err = getattr(self.parent(), "_hotkey_error", None) if self.parent() else None
        if hotkey_err:
            hotkey_status = QLabel(f"⚠  Hotkey error: {hotkey_err}")
            hotkey_status.setStyleSheet("color: #f87171; font-size: 9pt;")
            hotkey_status.setWordWrap(True)
            hotkey_layout.addWidget(hotkey_status)
        else:
            hotkey_ok = QLabel(f"✓  Active: {self.config.get('hotkey', 'ctrl+shift+space')}")
            hotkey_ok.setStyleSheet("color: #6ee7b7; font-size: 9pt;")
            hotkey_layout.addWidget(hotkey_ok)
        layout.addWidget(hotkey_group)

        # Privacy
        priv_group = QGroupBox("Privacy")
        priv_layout = QVBoxLayout(priv_group)
        self.stealth_check = QCheckBox("Stealth mode — hide from screen sharing")
        self.stealth_check.setChecked(self.config.get("stealth_mode", False))
        stealth_hint = QLabel("Window stays visible to you but won't appear in screen captures or recordings (Windows 10 2004+)")
        stealth_hint.setStyleSheet("color: #475569; font-size: 9pt;")
        stealth_hint.setWordWrap(True)
        priv_layout.addWidget(self.stealth_check)
        priv_layout.addWidget(stealth_hint)
        layout.addWidget(priv_group)

        # Transcription sensitivity
        sens_group = QGroupBox("Transcription Sensitivity")
        sens_layout = QVBoxLayout(sens_group)
        sens_row = QHBoxLayout()
        sens_low = QLabel("Strict")
        sens_low.setStyleSheet("color: #475569; font-size: 9pt;")
        sens_high = QLabel("Permissive")
        sens_high.setStyleSheet("color: #475569; font-size: 9pt;")
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(1, 10)
        self.sens_slider.setValue(self.config.get("transcription_sensitivity", 7))
        self.sens_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sens_slider.setTickInterval(1)
        sens_row.addWidget(sens_low)
        sens_row.addWidget(self.sens_slider, 1)
        sens_row.addWidget(sens_high)
        sens_hint = QLabel("Strict: only transcribes high-confidence speech. Permissive: captures more, including quieter or less clear audio.")
        sens_hint.setStyleSheet("color: #475569; font-size: 9pt;")
        sens_hint.setWordWrap(True)
        sens_layout.addLayout(sens_row)
        sens_layout.addWidget(sens_hint)
        layout.addWidget(sens_group)

        # Appearance
        appear_group = QGroupBox("Appearance")
        appear_layout = QHBoxLayout(appear_group)
        appear_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("🌙  Dark", "dark")
        self.theme_combo.addItem("☀️  Light", "light")
        saved_theme = self.config.get("theme", "dark")
        for i in range(self.theme_combo.count()):
            if self.theme_combo.itemData(i) == saved_theme:
                self.theme_combo.setCurrentIndex(i)
        def _on_theme_changed():
            self.config["theme"] = self.theme_combo.currentData()
            self._apply_theme_styles()
        self.theme_combo.currentIndexChanged.connect(_on_theme_changed)
        appear_layout.addWidget(self.theme_combo, 1)
        layout.addWidget(appear_group)

        # What's New
        whats_new_btn = QPushButton("📋  What's New")
        whats_new_btn.setObjectName("cancel")
        whats_new_btn.clicked.connect(self._show_changelog)
        layout.addWidget(whats_new_btn)

        # Buttons
        btn_row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save & Close")
        save.clicked.connect(self._save)
        btn_row.addWidget(cancel)
        btn_row.addWidget(save)
        layout.addLayout(btn_row)

    def _show_changelog(self):
        dark = (self.config.get("theme", "dark") != "light")
        dlg_bg  = "#1C1C1E" if dark else "#F2F2F7"
        dlg_fg  = "#F2F2F7" if dark else "#1C1C1E"
        txt_bg  = "#2C2C2E" if dark else "#FFFFFF"
        txt_br  = "rgba(255,255,255,0.08)" if dark else "rgba(60,60,67,0.1)"
        accent  = "#00C7FF" if dark else "#007AFF"

        dlg = QDialog(self)
        dlg.setWindowTitle("What's New")
        dlg.setFixedSize(480, 520)
        dlg.setStyleSheet(f"""
            QDialog {{ background: {dlg_bg}; font-family: 'Segoe UI', sans-serif; }}
            QTextEdit {{
                background: {txt_bg}; color: {dlg_fg};
                border: 1px solid {txt_br}; border-radius: 10px;
                font-size: 10pt; padding: 10px;
            }}
            QPushButton {{
                background: {accent};
                border: none; border-radius: 8px; padding: 10px;
                color: white; font-size: 10pt; font-weight: 600;
            }}
        """)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)
        text = QTextEdit()
        text.setReadOnly(True)
        v.addWidget(text)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        v.addWidget(close_btn)

        # Try fetching from GitHub, fall back to local file
        def _fetch():
            try:
                import urllib.request
                url = "https://raw.githubusercontent.com/AHappyPandaaa/pandai-assistant/main/CHANGELOG.md"
                with urllib.request.urlopen(url, timeout=5) as r:
                    return r.read().decode("utf-8")
            except Exception:
                pass
            try:
                local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CHANGELOG.md")
                with open(local, encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return "Could not load changelog."

        text.setPlainText(_fetch())
        dlg.exec()

    def _save(self):
        self.config["api_key"]                   = self.api_input.text().strip()
        self.config["mic_device"]                = self.mic_combo.currentData()
        self.config["sys_device"]                = self.sys_combo.currentData()
        self.config["capture_mode"]              = self.capture_mode_combo.currentData()
        self.config["hotkey"]                    = self.hotkey_input.text().strip() or "ctrl+shift+space"
        self.config["stealth_mode"]              = self.stealth_check.isChecked()
        self.config["transcription_sensitivity"] = self.sens_slider.value()
        self.config["theme"]                     = self.theme_combo.currentData()
        self.accept()

    def get_config(self):
        return self.config

# ── MAIN OVERLAY WINDOW ───────────────────────────────────────────────────────
class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.config       = load_config()
        self.whisper      = None
        self.capture      = None
        self.transcribe_q = []   # pending TranscribeWorker refs
        self.claude_q     = []   # pending ClaudeAnalysisWorker refs
        self.full_transcript = ""
        self._drag_pos    = None
        self._paused      = False
        self._briefing    = ""           # pre-session context
        self._display_transcript = ""    # labelled transcript for the UI box
        # Per-source dedup state for speaker labels
        self._last_words_mic = []
        self._last_words_sys = []
        # Selection & analysis
        self._current_selection      = ""     # text user has selected
        self._pending_display_update = False  # freeze transcript updates during selection
        self._highlight_phrases      = set()  # currently highlighted phrases
        # Session tracking
        self._current_session  = None   # dict for the in-progress session
        self._session_start    = None   # datetime when recording started
        self._analyses         = []     # analysis entries for current session
        self._hotkey_signaler = _HotkeySignaler()
        self._hotkey_signaler.triggered.connect(
            self._toggle_visibility, Qt.ConnectionType.QueuedConnection
        )
        self._hotkey_error = None
        self._hotkey_handle = None
        self._register_hotkey()
        self._opacity_val = 1.0
        self._theme = self.config.get("theme", "dark")

        self._setup_window()
        self._build_ui()
        self._load_sessions_into_tab()
        self._apply_styles()
        self._set_opacity(int(self._opacity_val * 100))
        self._load_whisper()

    # ── WINDOW SETUP ──────────────────────────────────────────────────────────
    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(420, 500)
        self.resize(460, 700)
        # Position: right side of primary screen
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - 480, 60)

    # ── UI BUILD ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.card = QFrame()
        self.card.setObjectName("card")
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)
        root.addWidget(self.card)

        # ── Header
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(42)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 10, 0)
        hl.setSpacing(8)

        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("dot_inactive")
        self.status_dot.setFixedWidth(14)

        title = QLabel("PANDAI ASSISTANT")
        title.setObjectName("header_title")

        settings_btn = QPushButton("⚙")
        settings_btn.setObjectName("icon_btn")
        settings_btn.setFixedSize(28, 28)
        settings_btn.clicked.connect(self._open_settings)

        clear_btn = QPushButton("↺")
        clear_btn.setObjectName("icon_btn")
        clear_btn.setFixedSize(28, 28)
        clear_btn.setToolTip("Clear session — wipes transcript and suggestions")
        clear_btn.clicked.connect(self._clear_session)

        self.export_btn = QPushButton("↓")
        self.export_btn.setObjectName("icon_btn")
        self.export_btn.setFixedSize(28, 28)
        self.export_btn.setToolTip("Export session transcript and suggestions to .txt")
        self.export_btn.clicked.connect(self._export_session)

        hide_btn = QPushButton("✕")
        hide_btn.setObjectName("icon_btn")
        hide_btn.setFixedSize(28, 28)
        hide_btn.clicked.connect(self.close)

        hl.addWidget(self.status_dot)
        hl.addWidget(title)
        hl.addStretch()
        hl.addWidget(clear_btn)
        hl.addWidget(self.export_btn)
        hl.addWidget(settings_btn)
        hl.addWidget(hide_btn)
        card_layout.addWidget(header)


        # ── Controls bar
        ctrl = QFrame()
        ctrl.setObjectName("ctrl_bar")
        cl = QVBoxLayout(ctrl)
        cl.setContentsMargins(12, 6, 12, 6)
        cl.setSpacing(5)

        row1 = QHBoxLayout()
        self.rec_btn = QPushButton("▶  Start Listening")
        self.rec_btn.setObjectName("rec_btn")
        self.rec_btn.setFixedHeight(30)
        self.rec_btn.clicked.connect(self._toggle_recording)

        self.pause_btn = QPushButton("⏸  Pause")
        self.pause_btn.setObjectName("pause_btn")
        self.pause_btn.setFixedHeight(30)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.pause_btn.hide()

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setFixedWidth(70)
        self.opacity_slider.setToolTip("Opacity")
        self.opacity_slider.valueChanged.connect(self._set_opacity)

        row1.addWidget(self.rec_btn)
        row1.addWidget(self.pause_btn)
        row1.addStretch()
        row1.addWidget(QLabel("◑"))
        row1.addWidget(self.opacity_slider)
        cl.addLayout(row1)

        # Full-width VU meter strip — shown only when recording
        self.level_meter = LevelMeter()
        self.level_meter.setToolTip("Mic input level")
        self.level_meter.hide()
        cl.addWidget(self.level_meter)

        card_layout.addWidget(ctrl)

        # ── Briefing panel (collapsible)
        self._briefing_panel = QFrame()
        self._briefing_panel.setObjectName("briefing_panel")
        bp_layout = QVBoxLayout(self._briefing_panel)
        bp_layout.setContentsMargins(12, 6, 12, 6)
        bp_layout.setSpacing(4)
        bp_header = QHBoxLayout()
        bp_label = QLabel("📋  SESSION BRIEFING")
        bp_label.setObjectName("section_label")
        self._briefing_toggle = QPushButton("▸ Expand")
        self._briefing_toggle.setObjectName("briefing_toggle_btn")
        self._briefing_toggle.setFixedHeight(18)
        self._briefing_toggle.clicked.connect(self._toggle_briefing)
        bp_header.addWidget(bp_label)
        bp_header.addStretch()
        bp_header.addWidget(self._briefing_toggle)
        bp_layout.addLayout(bp_header)
        self._briefing_edit = QTextEdit()
        self._briefing_edit.setObjectName("briefing_edit")
        self._briefing_edit.setPlaceholderText("Paste job description, meeting agenda, client background… used as context for every suggestion.")
        self._briefing_edit.setFixedHeight(70)
        self._briefing_edit.textChanged.connect(lambda: setattr(self, '_briefing', self._briefing_edit.toPlainText()))
        self._briefing_edit.hide()
        bp_layout.addWidget(self._briefing_edit)
        card_layout.addWidget(self._briefing_panel)

        # ── Whisper status banner
        self.whisper_banner = QLabel("⏳  Loading Whisper medium model (first run may take a moment)...")
        self.whisper_banner.setObjectName("banner_loading")
        self.whisper_banner.setWordWrap(True)
        self.whisper_banner.setContentsMargins(12, 6, 12, 6)
        card_layout.addWidget(self.whisper_banner)

        # ── Update available banner (hidden until needed)
        self.update_banner = QFrame()
        self.update_banner.setObjectName("banner_update_frame")
        _ub_row = QHBoxLayout(self.update_banner)
        _ub_row.setContentsMargins(12, 4, 8, 4)
        _ub_row.setSpacing(8)
        self._update_banner_label = QLabel("")
        self._update_banner_label.setObjectName("banner_update_label")
        self._update_banner_label.setWordWrap(True)
        self._update_now_btn = QPushButton("Update Now")
        self._update_now_btn.setObjectName("update_now_btn")
        self._update_now_btn.setFixedWidth(90)
        self._update_now_btn.clicked.connect(self._do_update)
        _ub_row.addWidget(self._update_banner_label, 1)
        _ub_row.addWidget(self._update_now_btn)
        self.update_banner.hide()
        card_layout.addWidget(self.update_banner)

        # ── Transcript
        t_frame = QFrame()
        t_frame.setObjectName("section_frame")
        tl = QVBoxLayout(t_frame)
        tl.setContentsMargins(12, 8, 12, 6)
        tl.setSpacing(4)

        t_header_row = QHBoxLayout()
        t_header_row.addWidget(self._section_label("TRANSCRIPT"))
        t_header_row.addStretch()
        self.analyze_btn = QPushButton("🔍  Analyze Selection")
        self.analyze_btn.setObjectName("analyze_btn")
        self.analyze_btn.setFixedHeight(26)
        self.analyze_btn.hide()
        self.analyze_btn.clicked.connect(self._analyze_selection)
        t_header_row.addWidget(self.analyze_btn)
        tl.addLayout(t_header_row)

        self.transcript_box = QTextEdit()
        self.transcript_box.setObjectName("transcript_box")
        self.transcript_box.setReadOnly(True)
        self.transcript_box.setMinimumHeight(180)
        self.transcript_box.setMaximumHeight(280)
        self.transcript_box.setPlaceholderText("Press Start Listening to begin…\n\nSelect any text in the transcript to analyze it.")
        self.transcript_box.selectionChanged.connect(self._on_selection_changed)
        tl.addWidget(self.transcript_box)
        card_layout.addWidget(t_frame)

        # ── Tab widget: Live suggestions | History
        self.tabs = QTabWidget()
        self.tabs.setObjectName("main_tabs")

        # ── TAB 1: Live ──────────────────────────────────────────────────────
        live_widget = QWidget()
        live_widget.setObjectName("content_widget")
        self.content_layout = QVBoxLayout(live_widget)
        self.content_layout.setContentsMargins(12, 6, 12, 12)
        self.content_layout.setSpacing(10)

        # Analysis section
        self.content_layout.addWidget(self._section_label("ANALYSIS"))
        self.response_frame = QFrame()
        self.response_frame.setObjectName("response_frame")
        self.response_layout = QVBoxLayout(self.response_frame)
        self.response_layout.setContentsMargins(10, 8, 10, 8)
        self.response_label = QLabel("Select text in the transcript above, then tap Analyze to go deep on any topic…")
        self.response_label.setObjectName("placeholder_text")
        self.response_label.setWordWrap(True)
        self.response_layout.addWidget(self.response_label)
        self.content_layout.addWidget(self.response_frame)

        # Connected topics section
        self.content_layout.addWidget(self._section_label("CONNECTED TOPICS"))
        self.topics_frame = QFrame()
        self.topics_frame.setObjectName("topics_frame")
        self.topics_layout = QVBoxLayout(self.topics_frame)
        self.topics_layout.setContentsMargins(0, 0, 0, 0)
        self.topics_layout.setSpacing(4)
        placeholder2 = QLabel("Related topics will surface here...")
        placeholder2.setObjectName("placeholder_text")
        self.topics_layout.addWidget(placeholder2)
        self.content_layout.addWidget(self.topics_frame)

        # Follow-ups section
        self.followups_label = self._section_label("YOU COULD ASK")
        self.followups_label.hide()
        self.content_layout.addWidget(self.followups_label)
        self.followups_frame = QFrame()
        self.followups_frame.setObjectName("followups_frame")
        self.followups_layout = QVBoxLayout(self.followups_frame)
        self.followups_layout.setContentsMargins(0, 0, 0, 0)
        self.followups_layout.setSpacing(4)
        self.followups_frame.hide()
        self.content_layout.addWidget(self.followups_frame)
        self.content_layout.addStretch()

        live_scroll = QScrollArea()
        live_scroll.setObjectName("scroll_area")
        live_scroll.setWidgetResizable(True)
        live_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        live_scroll.setFrameShape(QFrame.Shape.NoFrame)
        live_scroll.setWidget(live_widget)
        self.tabs.addTab(live_scroll, "💬  Live")

        # ── TAB 2: Sessions ──────────────────────────────────────────────────
        sessions_outer = QWidget()
        sessions_outer.setObjectName("content_widget")
        sessions_outer_layout = QVBoxLayout(sessions_outer)
        sessions_outer_layout.setContentsMargins(0, 0, 0, 0)

        self.sessions_scroll = QScrollArea()
        self.sessions_scroll.setObjectName("scroll_area")
        self.sessions_scroll.setWidgetResizable(True)
        self.sessions_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sessions_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.sessions_widget = QWidget()
        self.sessions_widget.setObjectName("content_widget")
        self.sessions_layout = QVBoxLayout(self.sessions_widget)
        self.sessions_layout.setContentsMargins(12, 8, 12, 12)
        self.sessions_layout.setSpacing(8)
        self.sessions_layout.addWidget(self._make_sessions_placeholder())
        self.sessions_layout.addStretch()

        self.sessions_scroll.setWidget(self.sessions_widget)
        sessions_outer_layout.addWidget(self.sessions_scroll)
        self.tabs.addTab(sessions_outer, "📁  Sessions")

        card_layout.addWidget(self.tabs, 1)

        # ── Resize grip
        grip_row = QHBoxLayout()
        grip_row.addStretch()
        grip = QSizeGrip(self)
        grip.setFixedSize(16, 16)
        grip_row.addWidget(grip)
        grip_row.setContentsMargins(0, 0, 4, 4)
        card_layout.addLayout(grip_row)

    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("section_label")
        return lbl

    def _load_saved_history(self):
        pass  # replaced by _load_sessions_into_tab

    def _add_history_entry(self, data: dict, snippet: str, time_str: str = "", persist: bool = True):
        pass  # replaced by session-based persistence


    # ── SELECTION & ON-DEMAND ANALYSIS ───────────────────────────────────────
    def _on_selection_changed(self):
        sel = self.transcript_box.textCursor().selectedText().strip()
        if sel:
            self._current_selection = sel
            preview = sel[:40] + ("…" if len(sel) > 40 else "")
            self.analyze_btn.setText(f"🔍  Analyze  \"{preview}\"")
            self.analyze_btn.show()
            self._pending_display_update = True  # freeze transcript refresh
        else:
            self._current_selection = ""
            self.analyze_btn.hide()
            if self._pending_display_update:
                self._pending_display_update = False
                self._flush_transcript_display()

    def _analyze_selection(self):
        api_key = self.config.get("api_key", "")
        if not api_key:
            self._set_response_text("🔑 Add your Anthropic API key in Settings to enable analysis.", error=True)
            return
        if not self._current_selection:
            return

        selection = self._current_selection
        self._current_selection = ""
        self.analyze_btn.hide()
        self._pending_display_update = False

        self.tabs.setCurrentIndex(0)  # switch to Analysis tab

        worker = ClaudeAnalysisWorker(
            api_key, selection, self.full_transcript, briefing=self._briefing
        )
        worker.result.connect(lambda d: self._render_analysis(d, selection))
        worker.error.connect(lambda e: self._set_response_text(f"⚠ {e}", error=True))
        self.claude_q.append(worker)
        worker.finished.connect(lambda: self.claude_q.remove(worker) if worker in self.claude_q else None)
        preview = selection[:50] + ("…" if len(selection) > 50 else "")
        self._set_response_text(f"Analyzing \"{preview}\"…", thinking=True)
        worker.start()

    def _render_analysis(self, data: dict, selection: str):
        import datetime
        analysis_entry = {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "selection": selection,
            "analysis":  data.get("analysis", ""),
            "connected": data.get("connected", []),
            "followups": data.get("followups", []),
        }
        self._analyses.append(analysis_entry)

        self._set_response_text(data.get("analysis", ""))

        # Connected topics
        while self.topics_layout.count():
            item = self.topics_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for t in data.get("connected", []):
            self.topics_layout.addWidget(self._make_topic_card(t))

        # Follow-up questions
        while self.followups_layout.count():
            item = self.followups_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        followups = data.get("followups", [])
        if followups:
            self.followups_label.show()
            self.followups_frame.show()
            d = (self._theme != "light")
            chip_bg  = "rgba(0,199,255,0.09)"  if d else "rgba(0,122,255,0.07)"
            chip_br  = "rgba(0,199,255,0.22)"  if d else "rgba(0,122,255,0.18)"
            chip_c   = "#00C7FF"               if d else "#007AFF"
            chip_hov = "rgba(0,199,255,0.18)"  if d else "rgba(0,122,255,0.14)"
            for q in followups:
                chip = QPushButton(q)
                chip.setObjectName("followup_chip")
                chip.setFont(QFont("Segoe UI", 9))
                chip.setStyleSheet(f"""
                    QPushButton {{
                        background: {chip_bg}; border: 1px solid {chip_br};
                        border-radius: 12px; color: {chip_c};
                        padding: 5px 14px; text-align: left; font-weight: 500;
                    }}
                    QPushButton:hover {{ background: {chip_hov}; border-color: {chip_c}44; }}
                """)
                chip.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                chip.clicked.connect(lambda _, t=q: QApplication.clipboard().setText(t))
                chip.setToolTip("Click to copy")
                chip.setMaximumWidth(400)
                self.followups_layout.addWidget(chip)
        else:
            self.followups_label.hide()
            self.followups_frame.hide()

    # ── HIGHLIGHT ENGINE ──────────────────────────────────────────────────────
    def _schedule_highlight_pass(self):
        if not hasattr(self, '_highlight_pass_timer'):
            self._highlight_pass_timer = QTimer()
            self._highlight_pass_timer.timeout.connect(self._run_highlight_pass)
        if not self._highlight_pass_timer.isActive():
            self._highlight_pass_timer.start(12000)  # every 12s

    def _stop_highlight_timer(self):
        if hasattr(self, '_highlight_pass_timer'):
            self._highlight_pass_timer.stop()

    def _run_highlight_pass(self):
        api_key = self.config.get("api_key", "")
        if not api_key or not self.full_transcript.strip():
            return
        worker = HighlightWorker(api_key, self.full_transcript)
        worker.phrases_ready.connect(self._apply_highlights)
        worker.start()

    def _apply_highlights(self, phrases: list):
        """Highlight key phrases without disturbing any active selection."""
        self._highlight_phrases = set(phrases)
        if self._pending_display_update:
            return

        from PyQt6.QtGui import QTextCharFormat, QTextCursor as _TC
        doc = self.transcript_box.document()
        saved_cursor = self.transcript_box.textCursor()
        saved_anchor = saved_cursor.anchor()
        saved_pos    = saved_cursor.position()

        # Clear existing highlights
        full_c = _TC(doc)
        full_c.select(_TC.SelectionType.Document)
        clear_fmt = QTextCharFormat()
        clear_fmt.setBackground(QColor(0, 0, 0, 0))
        full_c.mergeCharFormat(clear_fmt)

        # Apply new highlights
        d = (self._theme != "light")
        hl_fmt = QTextCharFormat()
        hl_fmt.setBackground(QColor(0, 199, 255, 38) if d else QColor(0, 122, 255, 28))

        text = self.transcript_box.toPlainText()
        for phrase in phrases:
            start = 0
            while True:
                idx = text.lower().find(phrase.lower(), start)
                if idx == -1:
                    break
                c = _TC(doc)
                c.setPosition(idx)
                c.setPosition(idx + len(phrase), _TC.MoveMode.KeepAnchor)
                c.mergeCharFormat(hl_fmt)
                start = idx + 1

        # Restore selection
        restored = self.transcript_box.textCursor()
        restored.setPosition(saved_anchor)
        restored.setPosition(saved_pos, _TC.MoveMode.KeepAnchor)
        self.transcript_box.setTextCursor(restored)

    def _flush_transcript_display(self):
        """Apply any buffered transcript update after user clears selection."""
        self.transcript_box.setPlainText(self._display_transcript.strip()[-2000:])
        cursor = self.transcript_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.transcript_box.setTextCursor(cursor)
        if self._highlight_phrases:
            self._apply_highlights(list(self._highlight_phrases))

    # ── SESSION LIFECYCLE ─────────────────────────────────────────────────────
    def _start_session(self):
        import datetime, uuid
        self._session_start = datetime.datetime.now()
        self._analyses = []
        self._highlight_phrases = set()
        self._current_session = {
            "id":               str(uuid.uuid4()),
            "title":            self._session_start.strftime("Session %b %d, %Y  %H:%M"),
            "date":             self._session_start.isoformat(),
            "duration_seconds": 0,
            "briefing":         self._briefing,
            "transcript":       "",
            "analyses":         [],
        }

    def _end_session(self):
        if not self._current_session:
            return
        import datetime
        duration = int((datetime.datetime.now() - self._session_start).total_seconds()) if self._session_start else 0
        self._current_session.update({
            "duration_seconds": duration,
            "transcript":       self._display_transcript.strip(),
            "analyses":         self._analyses,
        })
        self._stop_highlight_timer()
        if self._display_transcript.strip():
            sessions = load_sessions()
            sessions.append(self._current_session)
            save_sessions(sessions)
            self._load_sessions_into_tab()
            # Auto-title in background
            api_key = self.config.get("api_key", "")
            if api_key:
                sid = self._current_session["id"]
                tw = SessionTitleWorker(api_key, self.full_transcript, self._briefing)
                tw.title_ready.connect(lambda t: self._update_session_title(sid, t))
                tw.start()
        self._current_session = None
        self._session_start   = None

    def _update_session_title(self, session_id: str, title: str):
        sessions = load_sessions()
        for s in sessions:
            if s.get("id") == session_id:
                s["title"] = title
                break
        save_sessions(sessions)
        self._load_sessions_into_tab()

    # ── SESSIONS TAB ──────────────────────────────────────────────────────────
    def _make_sessions_placeholder(self):
        lbl = QLabel("Completed sessions will appear here.\n\nStart and stop a recording to save a session.")
        lbl.setObjectName("placeholder_text")
        lbl.setFont(QFont("Segoe UI", 10))
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setContentsMargins(12, 24, 12, 24)
        return lbl

    def _load_sessions_into_tab(self):
        while self.sessions_layout.count():
            item = self.sessions_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        sessions = load_sessions()
        if not sessions:
            self.sessions_layout.addWidget(self._make_sessions_placeholder())
            self.sessions_layout.addStretch()
            return
        for session in reversed(sessions[-30:]):
            self.sessions_layout.addWidget(self._make_session_card(session))
        self.sessions_layout.addStretch()

    def _make_session_card(self, session: dict):
        import datetime
        d = (self._theme != "light")
        card_bg  = "rgba(44,44,46,0.5)"      if d else "rgba(255,255,255,0.78)"
        card_br  = "rgba(255,255,255,0.06)"  if d else "rgba(60,60,67,0.09)"
        card_hbg = "rgba(58,58,60,0.7)"      if d else "rgba(255,255,255,0.96)"
        card_hbr = "rgba(0,199,255,0.18)"    if d else "rgba(0,122,255,0.18)"
        title_c  = "#F2F2F7"                 if d else "#1C1C1E"
        meta_c   = "rgba(235,235,245,0.38)"  if d else "rgba(60,60,67,0.45)"
        tog_bg   = "rgba(255,255,255,0.07)"  if d else "rgba(0,0,0,0.05)"
        tog_br   = "rgba(255,255,255,0.1)"   if d else "rgba(60,60,67,0.13)"
        tog_c    = "rgba(235,235,245,0.4)"   if d else "rgba(60,60,67,0.45)"
        tog_hc   = "#00C7FF"                 if d else "#007AFF"
        tog_hbr  = "rgba(0,199,255,0.3)"     if d else "rgba(0,122,255,0.3)"
        resp_c   = "rgba(235,235,245,0.82)"  if d else "#1C1C1E"
        t_bg     = "rgba(44,44,46,0.5)"      if d else "rgba(255,255,255,0.7)"

        card = QFrame()
        card.setObjectName("session_card")
        card.setStyleSheet(f"""
            QFrame#session_card {{
                background: {card_bg}; border: 1px solid {card_br}; border-radius: 10px;
            }}
            QFrame#session_card:hover {{ background: {card_hbg}; border-color: {card_hbr}; }}
        """)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(12, 10, 12, 10)
        cl.setSpacing(4)

        hrow = QHBoxLayout()
        title_lbl = QLabel(session.get("title", "Untitled Session"))
        title_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        title_lbl.setStyleSheet(f"color: {title_c};")
        try:
            dt = datetime.datetime.fromisoformat(session.get("date", ""))
            date_str = dt.strftime("%b %d, %Y  %H:%M")
        except Exception:
            date_str = session.get("date", "")[:16]
        dur_s = session.get("duration_seconds", 0)
        dur_str = f"{dur_s//60}m {dur_s%60}s" if dur_s >= 60 else f"{dur_s}s"
        n = len(session.get("analyses", []))
        meta_lbl = QLabel(f"{date_str}  ·  {dur_str}  ·  {n} analysis{'es' if n != 1 else ''}")
        meta_lbl.setFont(QFont("Segoe UI", 8))
        meta_lbl.setStyleSheet(f"color: {meta_c};")

        toggle_btn = QPushButton("▾ Expand")
        toggle_btn.setFont(QFont("Segoe UI", 8))
        toggle_btn.setFixedWidth(75)
        toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background: {tog_bg}; border: 1px solid {tog_br};
                border-radius: 6px; color: {tog_c}; padding: 2px 8px;
            }}
            QPushButton:hover {{ color: {tog_hc}; border-color: {tog_hbr}; }}
        """)
        hrow.addWidget(title_lbl, 1)
        hrow.addWidget(toggle_btn)
        cl.addLayout(hrow)
        cl.addWidget(meta_lbl)

        # Collapsible detail
        detail = QWidget()
        detail.setVisible(False)
        dl = QVBoxLayout(detail)
        dl.setContentsMargins(0, 8, 0, 0)
        dl.setSpacing(8)

        transcript = session.get("transcript", "").strip()
        if transcript:
            tl = self._section_label("TRANSCRIPT")
            dl.addWidget(tl)
            t_box = QTextEdit()
            t_box.setReadOnly(True)
            t_box.setPlainText(transcript)
            t_box.setFixedHeight(120)
            t_box.setStyleSheet(f"""
                QTextEdit {{
                    background: {t_bg}; border: 1px solid {card_br};
                    border-radius: 8px; color: {resp_c}; font-size: 9pt; padding: 4px;
                }}
            """)
            dl.addWidget(t_box)

        for entry in session.get("analyses", []):
            af = QFrame()
            af.setStyleSheet(f"""
                QFrame {{
                    background: {"rgba(0,199,255,0.07)" if d else "rgba(0,122,255,0.06)"};
                    border-left: 2px solid {"rgba(0,199,255,0.35)" if d else "rgba(0,122,255,0.35)"};
                    border-radius: 6px;
                }}
            """)
            al = QVBoxLayout(af)
            al.setContentsMargins(10, 8, 10, 8)
            al.setSpacing(3)
            sel_lbl = QLabel(f"🔍  \"{entry.get('selection', '')}\"  ·  {entry.get('timestamp', '')}")
            sel_lbl.setFont(QFont("Segoe UI", 8))
            sel_lbl.setStyleSheet(f"color: {tog_c};")
            sel_lbl.setWordWrap(True)
            analysis_lbl = QLabel(entry.get("analysis", ""))
            analysis_lbl.setFont(QFont("Segoe UI", 9))
            analysis_lbl.setStyleSheet(f"color: {resp_c};")
            analysis_lbl.setWordWrap(True)
            al.addWidget(sel_lbl)
            al.addWidget(analysis_lbl)
            dl.addWidget(af)

        cl.addWidget(detail)

        def _toggle(btn=toggle_btn, w=detail):
            vis = not w.isVisible()
            w.setVisible(vis)
            btn.setText("▴ Collapse" if vis else "▾ Expand")
        toggle_btn.clicked.connect(_toggle)
        return card

    # ── STYLES ────────────────────────────────────────────────────────────────
    def _apply_styles(self):
        d = (self._theme != "light")

        # ── Apple-inspired colour palette ───────────────────────────────────
        if d:
            c_card      = "rgba(28,28,30,0.96)"
            b_main      = "rgba(255,255,255,0.09)"
            c_header    = "rgba(255,255,255,0.04)"
            c_ctrl      = "rgba(255,255,255,0.03)"
            t_title     = "rgba(235,235,245,0.45)"
            t_primary   = "#F2F2F7"
            t_second    = "rgba(235,235,245,0.6)"
            t_section   = "rgba(235,235,245,0.3)"
            t_place     = "rgba(235,235,245,0.22)"
            accent      = "#00C7FF"
            accent_bg   = "rgba(0,199,255,0.1)"
            accent_br   = "rgba(0,199,255,0.22)"
            accent_str  = "#00C7FF"
            b_input     = "rgba(255,255,255,0.08)"
            c_input     = "rgba(44,44,46,0.7)"
            c_dropdown  = "#2C2C2E"
            t_input     = "#F2F2F7"
            c_trans     = "rgba(44,44,46,0.65)"
            t_trans     = "rgba(235,235,245,0.85)"
            c_scroll    = "rgba(255,255,255,0.18)"
            dot_active  = "#34C759"
            dot_inact   = "rgba(235,235,245,0.2)"
            c_resp_txt  = "#F2F2F7"
            groove_bg   = "rgba(255,255,255,0.12)"
            icon_hov_bg = "rgba(255,255,255,0.09)"
            rec_idle_bg = "rgba(52,199,89,0.13)"
            rec_idle_br = "rgba(52,199,89,0.35)"
            rec_idle_c  = "#34C759"
            rec_act_bg  = "rgba(255,69,58,0.13)"
            rec_act_br  = "rgba(255,69,58,0.35)"
            rec_act_c   = "#FF453A"
        else:
            c_card      = "rgba(242,242,247,0.97)"
            b_main      = "rgba(60,60,67,0.13)"
            c_header    = "rgba(0,0,0,0.03)"
            c_ctrl      = "rgba(0,0,0,0.02)"
            t_title     = "rgba(60,60,67,0.45)"
            t_primary   = "#1C1C1E"
            t_second    = "rgba(60,60,67,0.65)"
            t_section   = "rgba(60,60,67,0.38)"
            t_place     = "rgba(60,60,67,0.3)"
            accent      = "#007AFF"
            accent_bg   = "rgba(0,122,255,0.08)"
            accent_br   = "rgba(0,122,255,0.2)"
            accent_str  = "#007AFF"
            b_input     = "rgba(60,60,67,0.13)"
            c_input     = "rgba(255,255,255,0.88)"
            c_dropdown  = "#FFFFFF"
            t_input     = "#1C1C1E"
            c_trans     = "rgba(255,255,255,0.88)"
            t_trans     = "#1C1C1E"
            c_scroll    = "rgba(60,60,67,0.22)"
            dot_active  = "#34C759"
            dot_inact   = "rgba(60,60,67,0.25)"
            c_resp_txt  = "#1C1C1E"
            groove_bg   = "rgba(60,60,67,0.13)"
            icon_hov_bg = "rgba(0,0,0,0.06)"
            rec_idle_bg = "rgba(52,199,89,0.1)"
            rec_idle_br = "rgba(52,199,89,0.3)"
            rec_idle_c  = "#248A3D"
            rec_act_bg  = "rgba(255,69,58,0.1)"
            rec_act_br  = "rgba(255,69,58,0.3)"
            rec_act_c   = "#D70015"
        # ───────────────────────────────────────────────────────────────────

        self.setStyleSheet(f"""
            QWidget {{ font-family: 'Segoe UI', sans-serif; }}

            #card {{
                background: {c_card};
                border: 1px solid {b_main};
                border-radius: 18px;
            }}

            #header {{
                background: {c_header};
                border-bottom: 1px solid {b_main};
                border-top-left-radius: 18px;
                border-top-right-radius: 18px;
            }}

            #header_title {{
                font-size: 9pt; font-weight: 700;
                letter-spacing: 2px; color: {t_title};
            }}

            #dot_inactive {{ color: {dot_inact}; font-size: 9pt; }}
            #dot_active   {{ color: {dot_active}; font-size: 9pt; }}

            #analyze_btn {{
                background: {accent_bg};
                border: 1px solid {accent_br};
                border-radius: 8px; color: {accent};
                font-size: 8pt; font-weight: 600; padding: 0 10px;
            }}
            #analyze_btn:hover {{ background: rgba(99,102,241,0.22); }}

            #icon_btn {{
                background: transparent; border: none;
                color: {t_second}; font-size: 11pt; border-radius: 7px;
            }}
            #icon_btn:hover {{ background: {icon_hov_bg}; color: {t_primary}; }}

            #ctrl_bar {{
                border-bottom: 1px solid {b_main};
                background: {c_ctrl};
            }}

            #rec_btn {{
                background: {rec_idle_bg};
                border: 1px solid {rec_idle_br};
                border-radius: 10px; color: {rec_idle_c};
                font-size: 9pt; font-weight: 600; padding: 0 14px;
            }}
            #rec_btn:hover {{ background: rgba(52,199,89,0.22); }}
            #rec_btn[recording=true] {{
                background: {rec_act_bg};
                border-color: {rec_act_br};
                color: {rec_act_c};
            }}
            #rec_btn[recording=true]:hover {{ background: rgba(255,69,58,0.22); }}

            #pause_btn {{
                background: transparent;
                border: 1px solid {b_input};
                border-radius: 10px; color: {t_second};
                font-size: 9pt; font-weight: 600; padding: 0 10px;
            }}
            #pause_btn:hover {{ background: {icon_hov_bg}; color: {t_primary}; }}

            #device_combo {{
                background: {c_input};
                border: 1px solid {b_input};
                border-radius: 8px; color: {t_input}; font-size: 9pt;
                padding: 2px 6px;
            }}
            #device_combo QAbstractItemView {{
                background: {c_dropdown}; color: {t_input};
                selection-background-color: {accent_bg};
            }}

            #briefing_panel {{
                background: rgba(99,102,241,0.06);
                border-bottom: 1px solid rgba(99,102,241,0.18);
            }}
            #briefing_edit {{
                background: {c_input};
                border: 1px solid {b_input};
                border-radius: 8px; color: {t_input};
                font-size: 9pt; padding: 4px 6px;
            }}
            #briefing_toggle_btn {{
                background: transparent; border: none;
                color: rgba(120,120,255,0.75); font-size: 8pt; padding: 0;
            }}
            #briefing_toggle_btn:hover {{ color: rgba(160,160,255,1.0); }}

            #banner_loading {{
                background: rgba(255,159,10,0.1);
                border-bottom: 1px solid rgba(255,159,10,0.28);
                color: #FF9F0A; font-size: 9pt; padding: 5px 14px;
            }}
            #banner_ok {{
                background: rgba(52,199,89,0.09);
                border-bottom: 1px solid rgba(52,199,89,0.28);
                color: #34C759; font-size: 9pt; padding: 5px 14px;
            }}
            #banner_error {{
                background: rgba(255,69,58,0.09);
                border-bottom: 1px solid rgba(255,69,58,0.28);
                color: #FF453A; font-size: 9pt; padding: 5px 14px;
            }}
            #banner_hidden {{ max-height: 0px; padding: 0px; border: none; }}
            #banner_update_frame {{
                background: rgba(191,90,242,0.1);
                border-bottom: 1px solid rgba(191,90,242,0.25);
            }}
            #banner_update_label {{
                color: #BF5AF2; font-size: 9pt;
            }}
            #update_now_btn {{
                background: rgba(191,90,242,0.18);
                color: #BF5AF2; font-size: 8pt; font-weight: 600;
                border: 1px solid rgba(191,90,242,0.35);
                border-radius: 6px; padding: 3px 10px;
            }}
            #update_now_btn:hover {{ background: rgba(191,90,242,0.3); }}
            #update_now_btn:disabled {{ color: rgba(191,90,242,0.4); }}

            #section_frame {{ background: transparent; }}

            #section_label {{
                font-size: 8pt; font-weight: 600;
                letter-spacing: 1px; color: {t_section};
                padding: 4px 0 2px 0;
            }}

            #transcript_box {{
                background: {c_trans};
                border: 1px solid {b_input};
                border-radius: 10px; color: {t_trans};
                font-size: 10pt; padding: 6px 8px;
                selection-background-color: {accent_bg};
            }}
            QScrollBar:vertical {{ background: transparent; width: 3px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: {c_scroll}; border-radius: 1px; min-height: 20px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

            #scroll_area {{ background: transparent; border: none; }}
            #content_widget {{ background: transparent; }}

            #response_frame {{
                background: {accent_bg};
                border: 1px solid {accent_br};
                border-left: 3px solid {accent_str};
                border-radius: 12px;
            }}

            #topics_frame {{ background: transparent; }}
            #followups_frame {{ background: transparent; }}

            #placeholder_text {{ color: {t_place}; font-size: 10pt; font-style: italic; }}

            QLabel#response_text {{
                color: {c_resp_txt}; font-size: 10pt;
            }}

            QLabel {{ color: {t_second}; }}

            QSlider::groove:horizontal {{
                height: 3px; background: {groove_bg}; border-radius: 1px;
            }}
            QSlider::handle:horizontal {{
                width: 13px; height: 13px; margin: -5px 0;
                background: {accent}; border-radius: 6px;
            }}

            QTabWidget#main_tabs::pane {{
                border: none;
                border-top: 1px solid {b_main};
                background: transparent;
            }}
            QTabWidget#main_tabs > QTabBar {{
                background: transparent;
            }}
            QTabWidget#main_tabs > QTabBar::tab {{
                background: transparent;
                border: none;
                border-bottom: 2px solid transparent;
                padding: 8px 20px;
                color: {t_section};
                font-size: 9pt; font-weight: 600;
            }}
            QTabWidget#main_tabs > QTabBar::tab:selected {{
                color: {accent};
                border-bottom: 2px solid {accent};
            }}
            QTabWidget#main_tabs > QTabBar::tab:hover:!selected {{
                color: {t_second};
            }}
        """)

    # ── DEVICES ───────────────────────────────────────────────────────────────


    # ── WHISPER LOADING ───────────────────────────────────────────────────────
    def _load_whisper(self):
        self.loader = WhisperLoader()
        self.loader.done.connect(self._on_whisper_loaded)
        self.loader.status.connect(lambda msg: (
            self.whisper_banner.setText(msg) or self._apply_styles()
        ))
        self.loader.start()

        self._update_checker = UpdateChecker()
        self._update_checker.update_available.connect(self._on_update_available)
        self._update_checker.start()

    def _on_update_available(self, remote_sha: str):
        self._update_banner_label.setText(f'⬆  Update available ({remote_sha})')
        self.update_banner.show()
        self._apply_styles()

    def _do_update(self):
        self._update_now_btn.setEnabled(False)
        self._update_banner_label.setText("⏳  Downloading update...")
        self._auto_updater = AutoUpdater()
        self._auto_updater.finished.connect(self._on_update_done)
        self._auto_updater.start()

    def _on_update_done(self, success: bool, message: str):
        if success:
            self._update_banner_label.setText(f"✓  {message}")
            self._update_now_btn.setText("Restart")
            self._update_now_btn.setEnabled(True)
            self._update_now_btn.clicked.disconnect()
            self._update_now_btn.clicked.connect(self._restart_app)
        else:
            self._update_banner_label.setText(f"✗  {message}")
            self._update_now_btn.setText("Retry")
            self._update_now_btn.setEnabled(True)

    def _restart_app(self):
        python = sys.executable
        if sys.platform == "win32":
            pythonw = os.path.join(os.path.dirname(python), "pythonw.exe")
            if os.path.exists(pythonw):
                python = pythonw
        import subprocess
        subprocess.Popen([python] + sys.argv)
        self.close()

    def closeEvent(self, event):
        """Ensure all threads are stopped before the process exits."""
        # Stop audio capture and wait for its thread to finish
        if self.capture:
            self.capture.stop()
            self.capture.wait(3000)
            self.capture = None

        # Stop in-flight transcription and Claude workers
        for w in list(self.transcribe_q):
            w.quit()
            w.wait(1000)
        self.transcribe_q.clear()
        for w in list(self.claude_q):
            w.quit()
            w.wait(1000)
        self.claude_q.clear()

        # Stop any background utility threads
        for attr in ("_update_checker", "_auto_updater", "_whisper_loader"):
            t = getattr(self, attr, None)
            if t and t.isRunning():
                t.quit()
                t.wait(2000)

        QApplication.quit()
        event.accept()

    def _on_whisper_loaded(self, model, err):
        self.whisper = model
        if model:
            msg = f"✓  Whisper {WHISPER_MODEL} ready"
            if err:
                msg += f"  ({err})"
            self.whisper_banner.setObjectName("banner_ok")
            self.whisper_banner.setText(msg)
            self.whisper_banner.setStyleSheet("")
            self._apply_styles()
            QTimer.singleShot(4000, lambda: self.whisper_banner.setObjectName("banner_hidden") or self._apply_styles())
        else:
            self.whisper_banner.setObjectName("banner_error")
            self.whisper_banner.setText(f"✗  Whisper failed to load: {err}")
            self._apply_styles()

    # ── RECORDING ─────────────────────────────────────────────────────────────
    def _toggle_recording(self):
        if self.capture:
            self._stop_recording()
            self._end_session()
        else:
            self._start_recording()
            self._start_session()

    def _start_recording(self):
        if not self.whisper:
            self.whisper_banner.setText("⏳  Still loading Whisper model, please wait...")
            self.whisper_banner.setObjectName("banner_loading")
            self._apply_styles()
            return

        capture_mode = self.config.get("capture_mode", "both")
        mic_idx = self.config.get("mic_device") if capture_mode != "inbound" else None
        sys_idx = self.config.get("sys_device") if capture_mode != "mic" else None

        self.capture = AudioCapture(mic_idx=mic_idx, sys_idx=sys_idx)
        self.capture.mic_chunk_ready.connect(lambda a: self._on_audio_chunk(a, "mic"))
        self.capture.sys_chunk_ready.connect(lambda a: self._on_audio_chunk(a, "sys"))
        self.capture.level_changed.connect(self.level_meter.set_level)
        self.capture.start()

        self.rec_btn.setText("■  Stop Listening")
        self.rec_btn.setProperty("recording", True)
        self.rec_btn.setStyle(self.rec_btn.style())
        self.status_dot.setObjectName("dot_active")
        self.status_dot.setStyle(self.status_dot.style())
        self.transcript_box.setPlaceholderText("Listening — updating every 2s...")
        self.pause_btn.setText("⏸  Pause")
        self.pause_btn.show()
        self.level_meter.show()

    def _stop_recording(self):
        if self.capture:
            self.capture.stop()
            self.capture = None
        self._paused = False
        self.rec_btn.setText("▶  Start Listening")
        self.rec_btn.setProperty("recording", False)
        self.rec_btn.setStyle(self.rec_btn.style())
        self.status_dot.setObjectName("dot_inactive")
        self.status_dot.setStyle(self.status_dot.style())
        self.pause_btn.hide()
        self.level_meter.set_level(0.0)
        self.level_meter.hide()

    def _toggle_pause(self):
        if self._paused:
            self._resume_recording()
        else:
            self._pause_recording()

    def _pause_recording(self):
        if self.capture:
            self.capture.stop()
            self.capture = None
        self._paused = True
        self.pause_btn.setText("▶  Resume")
        self.level_meter.set_level(0.0)
        self.status_dot.setObjectName("dot_inactive")
        self.status_dot.setStyle(self.status_dot.style())
        self.transcript_box.setPlaceholderText("Paused — session context preserved. Click Resume to continue.")

    def _resume_recording(self):
        self._paused = False
        capture_mode = self.config.get("capture_mode", "both")
        mic_idx = self.config.get("mic_device") if capture_mode != "inbound" else None
        sys_idx = self.config.get("sys_device") if capture_mode != "mic" else None
        self.capture = AudioCapture(mic_idx=mic_idx, sys_idx=sys_idx)
        self.capture.mic_chunk_ready.connect(lambda a: self._on_audio_chunk(a, "mic"))
        self.capture.sys_chunk_ready.connect(lambda a: self._on_audio_chunk(a, "sys"))
        self.capture.level_changed.connect(self.level_meter.set_level)
        self.capture.start()
        self.pause_btn.setText("⏸  Pause")
        self.status_dot.setObjectName("dot_active")
        self.status_dot.setStyle(self.status_dot.style())
        self.transcript_box.setPlaceholderText("Listening — updating every 2s...")

    def _on_audio_chunk(self, audio: np.ndarray, source: str = "mic"):
        if not self.whisper:
            return
        # Allow one worker per source to run concurrently
        running_sources = [w.source for w in self.transcribe_q]
        if source in running_sources:
            return
        sensitivity = self.config.get("transcription_sensitivity", 7)
        no_speech_thresh = round(0.9 - (sensitivity / 10) * 0.6, 2)
        worker = TranscribeWorker(self.whisper, audio, source=source, no_speech_thresh=no_speech_thresh)
        worker.result.connect(self._on_transcription)
        self.transcribe_q.append(worker)
        worker.finished.connect(lambda: self.transcribe_q.remove(worker) if worker in self.transcribe_q else None)
        worker.start()

    def _on_transcription(self, text: str, source: str = "mic"):
        import difflib
        text = text.strip()
        if not text:
            return

        # Per-source dedup state
        last_words = self._last_words_mic if source == "mic" else self._last_words_sys
        curr_words = text.split()

        overlap = 0
        max_check = min(len(last_words), len(curr_words), 20)
        for n in range(max_check, 0, -1):
            if last_words[-n:] == curr_words[:n]:
                overlap = n
                break

        if overlap == 0 and last_words:
            ratio = difflib.SequenceMatcher(
                None,
                " ".join(last_words[-len(curr_words):]).lower(),
                " ".join(curr_words).lower()
            ).ratio()
            if ratio > 0.85:
                if source == "mic":
                    self._last_words_mic = curr_words
                else:
                    self._last_words_sys = curr_words
                return

        new_words = curr_words[overlap:]
        if source == "mic":
            self._last_words_mic = curr_words
        else:
            self._last_words_sys = curr_words

        if not new_words:
            return

        new_text = " ".join(new_words)
        label = "🎤 You" if source == "mic" else "👤 Them"

        # full_transcript (for Claude) — plain text, no labels
        self.full_transcript += " " + new_text
        if len(self.full_transcript) > 2000:
            self.full_transcript = self.full_transcript[-2000:]

        # Display transcript — labelled, shown in the box
        self._display_transcript += f"\n{label}: {new_text}"
        if len(self._display_transcript) > 3000:
            self._display_transcript = self._display_transcript[-3000:]

        # Update display (freeze if user has active selection)
        if not self._pending_display_update:
            self.transcript_box.setPlainText(self._display_transcript.strip()[-2000:])
            cursor = self.transcript_box.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.transcript_box.setTextCursor(cursor)
            if self._highlight_phrases:
                self._apply_highlights(list(self._highlight_phrases))

        # Keep highlight timer ticking while speech is coming in
        self._schedule_highlight_pass()

        else:
            self.followups_label.hide()
            self.followups_frame.hide()

    def _set_response_text(self, text, error=False, thinking=False):
        while self.response_layout.count():
            item = self.response_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        d = (self._theme != "light")
        resp_col    = "#F2F2F7"                    if d else "#1C1C1E"
        think_col   = "rgba(235,235,245,0.3)"      if d else "rgba(60,60,67,0.35)"
        err_col     = "#FF453A"                    if d else "#D70015"
        copy_br     = "rgba(255,255,255,0.12)"     if d else "rgba(60,60,67,0.15)"
        copy_c      = "rgba(235,235,245,0.38)"     if d else "rgba(60,60,67,0.42)"
        copy_hov_br = "rgba(0,199,255,0.45)"       if d else "rgba(0,122,255,0.45)"
        copy_hov_c  = "#00C7FF"                    if d else "#007AFF"

        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setFont(QFont("Segoe UI", 10))
        if error:
            lbl.setStyleSheet(f"color: {err_col}; font-style: italic;")
        elif thinking:
            lbl.setStyleSheet(f"color: {think_col}; font-style: italic;")
        else:
            lbl.setStyleSheet(f"color: {resp_col};")
            self.response_layout.addWidget(lbl)
            copy_btn = QPushButton("Copy")
            copy_btn.setFont(QFont("Segoe UI", 9))
            copy_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    border: 1px solid {copy_br};
                    border-radius: 10px; color: {copy_c};
                    padding: 3px 12px; max-width: 70px; font-size: 9pt;
                }}
                QPushButton:hover {{ border-color: {copy_hov_br}; color: {copy_hov_c}; }}
            """)
            def _do_copy(btn=copy_btn, t=text, _br=copy_br, _c=copy_c, _hbr=copy_hov_br, _hc=copy_hov_c):
                QApplication.clipboard().setText(t)
                btn.setText("✓ Copied!")
                btn.setStyleSheet(f"QPushButton {{ background: transparent; border: 1px solid rgba(52,199,89,0.5); border-radius: 10px; color: #34C759; padding: 3px 12px; max-width: 70px; font-size: 9pt; }}")
                QTimer.singleShot(1500, lambda: btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent; border: 1px solid {_br};
                        border-radius: 10px; color: {_c};
                        padding: 3px 12px; max-width: 70px; font-size: 9pt;
                    }}
                    QPushButton:hover {{ border-color: {_hbr}; color: {_hc}; }}
                """) or btn.setText("Copy"))
            copy_btn.clicked.connect(_do_copy)
            self.response_layout.addWidget(copy_btn)
            return
        self.response_layout.addWidget(lbl)

    def _make_topic_card(self, topic: dict):
        d = (self._theme != "light")
        card_bg     = "rgba(44,44,46,0.55)"       if d else "rgba(255,255,255,0.82)"
        card_br     = "rgba(255,255,255,0.07)"    if d else "rgba(60,60,67,0.1)"
        card_hov_bg = "rgba(58,58,60,0.75)"       if d else "rgba(255,255,255,0.96)"
        card_hov_br = "rgba(0,199,255,0.22)"      if d else "rgba(0,122,255,0.22)"
        title_col   = "#F2F2F7"                   if d else "#1C1C1E"
        detail_col  = "rgba(235,235,245,0.48)"    if d else "rgba(60,60,67,0.55)"

        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: {card_bg};
                border: 1px solid {card_br};
                border-radius: 10px;
            }}
            QFrame:hover {{
                background: {card_hov_bg};
                border-color: {card_hov_br};
            }}
        """)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        header_row = QHBoxLayout()
        icon = QLabel(topic.get("icon", "🔗"))
        icon.setFont(QFont("Segoe UI", 11))
        icon.setFixedWidth(22)
        icon.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel(topic.get("title", ""))
        title.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {title_col};")
        title.setWordWrap(True)
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        header_row.addWidget(icon)
        header_row.addWidget(title, 1)
        layout.addLayout(header_row)

        detail = topic.get("detail", "")
        if detail:
            detail_lbl = QLabel(detail)
            detail_lbl.setFont(QFont("Segoe UI", 9))
            detail_lbl.setStyleSheet(f"color: {detail_col}; padding-left: 22px;")
            detail_lbl.setWordWrap(True)
            detail_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            layout.addWidget(detail_lbl)
        return card

    # ── CLEAR SESSION ─────────────────────────────────────────────────────────
    def _clear_session(self):
        # Stop recording if active (also handles paused state)
        was_recording = self.capture is not None or self._paused
        if was_recording:
            self._stop_recording()

        # Wipe transcript state
        self.full_transcript = ""
        self._last_transcribed = ""
        self._last_words = []
        self._last_words_mic = []
        self._last_words_sys = []
        self._display_transcript = ""
        self._briefing = ""
        self._briefing_edit.clear()

        # Reset session/analysis state
        self._current_session = None
        self._session_start = None
        self._analyses = []
        self._highlight_phrases = set()
        self._current_selection = ""
        self._pending_display_update = False
        self._stop_highlight_timer()

        # Cancel any in-flight workers
        for w in self.transcribe_q:
            try: w.terminate()
            except Exception: pass
        self.transcribe_q.clear()
        for w in self.claude_q:
            try: w.terminate()
            except Exception: pass
        self.claude_q.clear()

        # Reset UI
        self.transcript_box.clear()
        self.transcript_box.setPlaceholderText("Session cleared — press Start Listening to begin...")

        self._set_response_text("Session cleared. Ready for a new conversation.", thinking=True)

        # Clear topics
        while self.topics_layout.count():
            item = self.topics_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        placeholder = QLabel("Related topics will surface here...")
        placeholder.setObjectName("placeholder_text")
        placeholder.setFont(QFont("Segoe UI", 10))
        self.topics_layout.addWidget(placeholder)

        # Clear followups
        while self.followups_layout.count():
            item = self.followups_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.followups_label.hide()
        self.followups_frame.hide()

        # Refresh sessions tab
        self._load_sessions_into_tab()

        # Restart recording if it was running
        if was_recording:
            QTimer.singleShot(300, self._start_recording)


    # ── SETTINGS ──────────────────────────────────────────────────────────────
    def _export_session(self):
        import datetime
        now = datetime.datetime.now()
        session_title = (self._current_session or {}).get("title", "Session")
        filename = os.path.join(
            os.path.expanduser("~"), "Documents",
            f"PandAI_{now.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        sep = "=" * 52
        briefing = self._briefing.strip()
        lines = [
            "PandAI Assistant — Session Export",
            f"Title : {session_title}",
            f"Date  : {now.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if briefing:
            lines += ["", f"Briefing: {briefing}"]
        lines += [
            "",
            sep, "TRANSCRIPT", sep,
            self.full_transcript.strip() or "(no transcript recorded)",
        ]
        if self._analyses:
            lines += ["", sep, "ANALYSES", sep]
            for a in self._analyses:
                lines += [
                    f"\n[{a.get('timestamp', '')}]",
                    f"Selection : {a.get('selection', '')}",
                    f"Analysis  : {a.get('analysis', '')}",
                ]
                if a.get("connected"):
                    lines.append("Connected : " + " | ".join(
                        f"{c.get('icon','')} {c.get('title','')}" for c in a["connected"]
                    ))
                if a.get("followups"):
                    lines.append("Follow-ups: " + " | ".join(a["followups"]))
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.export_btn.setText("✓")
            self.export_btn.setToolTip(f"Saved: {filename}")
            QTimer.singleShot(2000, lambda: (
                self.export_btn.setText("↓"),
                self.export_btn.setToolTip("Export session transcript and analyses to .txt")
            ))
        except Exception:
            self.export_btn.setText("✗")
            QTimer.singleShot(2000, lambda: self.export_btn.setText("↓"))

    def _toggle_briefing(self):
        visible = self._briefing_edit.isVisible()
        self._briefing_edit.setVisible(not visible)
        self._briefing_toggle.setText("▾ Collapse" if not visible else "▸ Expand")

    def _register_hotkey(self):
        try:
            import keyboard
            if self._hotkey_handle is not None:
                try:
                    keyboard.remove_hotkey(self._hotkey_handle)
                except Exception:
                    pass
            hotkey = self.config.get("hotkey", "ctrl+shift+space")
            self._hotkey_handle = keyboard.add_hotkey(hotkey, self._hotkey_signaler.triggered.emit)
            self._hotkey_error = None
        except Exception as e:
            self._hotkey_error = str(e)

    def _toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def _open_settings(self):
        dlg = SettingsDialog(self.config, self)
        if dlg.exec():
            self.config = dlg.get_config()
            save_config(self.config)
            self._theme = self.config.get("theme", "dark")
            self._apply_styles()
            self._set_opacity(int(self._opacity_val * 100))
            self._apply_stealth_mode(self.config.get("stealth_mode", False))
            self._register_hotkey()

    # ── STEALTH MODE ──────────────────────────────────────────────────────────
    def _apply_stealth_mode(self, enabled: bool):
        """Hide the window from screen capture using Windows display affinity."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            WDA_NONE               = 0x00000000
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            hwnd = int(self.winId())
            affinity = WDA_EXCLUDEFROMCAPTURE if enabled else WDA_NONE
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, affinity)
        except Exception:
            pass

    # ── OPACITY ───────────────────────────────────────────────────────────────
    def _set_opacity(self, val):
        self._opacity_val = val / 100.0
        dark = (self._theme != "light")
        solid_bg   = "rgb(28,28,30)"              if dark else "rgb(242,242,247)"
        border_col = "rgba(255,255,255,0.09)"     if dark else "rgba(60,60,67,0.13)"
        base_r, base_g, base_b = (28, 28, 30)    if dark else (242, 242, 247)
        if val >= 100:
            self.card.setStyleSheet(f"""
                #card {{
                    background: {solid_bg};
                    border: 1px solid {border_col};
                    border-radius: 18px;
                }}
            """)
            self.setWindowOpacity(1.0)
        else:
            alpha = int(245 * (val / 100))
            self.card.setStyleSheet(f"""
                #card {{
                    background: rgba({base_r},{base_g},{base_b},{alpha});
                    border: 1px solid {border_col};
                    border-radius: 18px;
                }}
            """)
            self.setWindowOpacity(max(0.2, val / 100.0))

    # ── DRAG TO MOVE ──────────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton and self._drag_pos:
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None

    # ── PAINT (rounded transparent bg) ───────────────────────────────────────
    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def _is_admin():
    """Return True if the current process has admin/elevated privileges."""
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False

def _relaunch_as_admin():
    """Trigger a UAC prompt and relaunch this script elevated.
    Returns True if the elevated process was launched (caller should exit).
    Returns False if the user denied the prompt or it failed."""
    import ctypes
    exe = sys.executable
    pythonw = os.path.join(os.path.dirname(exe), "pythonw.exe")
    if os.path.exists(pythonw):
        exe = pythonw  # suppress console in the elevated process too
    script = os.path.abspath(sys.argv[0])
    ret = ctypes.windll.shell32.ShellExecuteW(None, "runas", exe, f'"{script}"', None, 1)
    return ret > 32  # > 32 = success; <= 32 = denied or error

def _write_crash_log(exc_type, exc_value, exc_tb):
    import traceback, datetime
    log_path = os.path.join(os.path.expanduser("~"), "Desktop", "pandai_assistant_crash.txt")
    lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    msg = f"[{datetime.datetime.now()}]\n{''.join(lines)}\n"
    try:
        with open(log_path, "a") as f:
            f.write(msg)
    except Exception:
        pass
    print(msg)
    # Also show a Qt message box if app is running
    try:
        from PyQt6.QtWidgets import QMessageBox
        box = QMessageBox()
        box.setWindowTitle("PandAI Assistant — Crash")
        box.setText("An error occurred. See Desktop/pandai_assistant_crash.txt for details.")
        box.setDetailedText("".join(lines))
        box.exec()
    except Exception:
        pass

if __name__ == "__main__":
    if sys.platform == "win32":
        # Request admin elevation — needed for global hotkey (keyboard library)
        if not _is_admin():
            if _relaunch_as_admin():
                sys.exit(0)
            # User denied UAC — continue anyway; hotkey warning will show in Settings
        # Suppress the console window by re-launching with pythonw.exe
        elif os.path.basename(sys.executable).lower() == "python.exe":
            _pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
            if os.path.exists(_pythonw):
                import subprocess
                subprocess.Popen([_pythonw] + sys.argv)
                sys.exit(0)

    sys.excepthook = _write_crash_log

    app = QApplication(sys.argv)
    app.setApplicationName("PandAI Assistant")
    app.setQuitOnLastWindowClosed(True)

    window = OverlayWindow()
    window.show()
    window._apply_stealth_mode(window.config.get("stealth_mode", False))
    sys.exit(app.exec())
