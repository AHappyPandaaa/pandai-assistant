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
CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".pandai_assistant.json")

MODES = ["general", "interview", "technical", "sales", "casual"]
MODE_LABELS = {
    "general":   "General",
    "interview": "Interview",
    "technical": "Technical",
    "sales":     "Client Call",
    "casual":    "Casual",
}
MODE_COLORS = {
    "general":   "#00d4ff",
    "interview": "#10b981",
    "technical": "#f59e0b",
    "sales":     "#a78bfa",
    "casual":    "#f97316",
}
SYSTEM_PROMPTS = {
    "general": (
        "You are a real-time conversation assistant. Analyze the transcript and respond ONLY in JSON:\n"
        '{"response":"2-3 sentence suggested reply","topics":[{"icon":"...","title":"...","detail":"..."}],"followups":["...","..."]}'
    ),
    "interview": (
        "You help in job interviews. Give a concise STAR-format answer suggestion. Respond ONLY in JSON:\n"
        '{"response":"STAR answer","topics":[{"icon":"⭐","title":"competency","detail":"example"}],"followups":["question to ask interviewer"]}'
    ),
    "technical": (
        "You assist in technical discussions. Respond ONLY in JSON:\n"
        '{"response":"precise technical response","topics":[{"icon":"⚙️","title":"concept","detail":"explanation"}],"followups":["follow-up question"]}'
    ),
    "sales": (
        "You assist in client/sales calls. Respond ONLY in JSON:\n"
        '{"response":"value-focused response","topics":[{"icon":"💼","title":"talking point","detail":"detail"}],"followups":["discovery question"]}'
    ),
    "casual": (
        "You help in casual conversation. Respond ONLY in JSON:\n"
        '{"response":"natural reply","topics":[{"icon":"💬","title":"topic","detail":"detail"}],"followups":["follow-up question"]}'
    ),
}

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
    chunk_ready = pyqtSignal(np.ndarray)  # float32 mono at SAMPLE_RATE

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
            if len(self._ring_mic) < self._mic_rate * 0.5:
                return  # not enough audio yet

            mic_native = np.array(self._ring_mic[-window_native:], dtype=np.float32)
            mic_chunk  = resample_to_whisper(mic_native, self._mic_rate)

            if self._ring_sys:
                window_sys = self._sys_rate * WINDOW_SECONDS
                sys_native = np.array(self._ring_sys[-window_sys:], dtype=np.float32)
                sys_chunk  = resample_to_whisper(sys_native, self._sys_rate)
                min_len = min(len(mic_chunk), len(sys_chunk))
                mixed = np.clip(mic_chunk[:min_len] * 0.6 + sys_chunk[:min_len] * 0.6, -1.0, 1.0)
            else:
                mixed = mic_chunk

        self.chunk_ready.emit(mixed)

# ── TRANSCRIPTION THREAD ─────────────────────────────────────────────────────
class TranscribeWorker(QThread):
    result = pyqtSignal(str)

    def __init__(self, model, audio: np.ndarray):
        super().__init__()
        self.model = model
        self.audio = audio

    def run(self):
        try:
            segments, info = self.model.transcribe(
                self.audio,
                language="en",
                beam_size=3,           # lower = faster, less over-generation
                best_of=2,             # pick best of N candidates
                temperature=0.0,       # deterministic — reduces hallucinations
                condition_on_previous_text=False,  # prevents repetition loops
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 400,
                    "speech_pad_ms": 200,          # pad edges so words aren't clipped
                    "threshold": 0.4,              # slightly lower = catches softer speech (headset)
                },
                no_speech_threshold=0.6,           # discard segments that are probably silence
                compression_ratio_threshold=2.0,   # discard garbled/repetitive output
                log_prob_threshold=-0.8,           # discard low-confidence segments
            )
            # Filter out low-confidence segments before joining
            parts = []
            for s in segments:
                # Skip segments that are suspiciously short or repeated filler
                txt = s.text.strip()
                if not txt:
                    continue
                if hasattr(s, 'no_speech_prob') and s.no_speech_prob > 0.6:
                    continue
                parts.append(txt)
            text = " ".join(parts).strip()
            if text:
                self.result.emit(text)
        except Exception as e:
            print(f"Transcription error: {e}")

# ── CLAUDE WORKER ─────────────────────────────────────────────────────────────
class ClaudeWorker(QThread):
    result = pyqtSignal(dict)
    error  = pyqtSignal(str)

    def __init__(self, api_key, mode, transcript):
        super().__init__()
        self.api_key    = api_key
        self.mode       = mode
        self.transcript = transcript

    def run(self):
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                system=SYSTEM_PROMPTS.get(self.mode, SYSTEM_PROMPTS["general"]),
                messages=[{"role": "user", "content": f'Transcript:\n"{self.transcript[-800:]}"'}],
            )
            raw = msg.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            self.result.emit(data)
        except json.JSONDecodeError:
            self.error.emit("Claude returned unexpected format — retrying next chunk.")
        except Exception as e:
            self.error.emit(str(e))

# ── SETTINGS DIALOG ───────────────────────────────────────────────────────────
from PyQt6.QtWidgets import QDialog, QLineEdit, QDialogButtonBox, QGridLayout, QGroupBox

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = dict(config)
        self.setWindowTitle("PandAI Assistant — Settings")
        self.setFixedSize(500, 570)
        self.setStyleSheet("""
            QDialog { background: #0d0e12; color: #e2e8f0; font-family: Segoe UI; }
            QLabel { color: #94a3b8; font-size: 10pt; }
            QLabel.heading { color: #e2e8f0; font-size: 10pt; font-weight: bold; }
            QLineEdit {
                background: #1e2029; border: 1px solid rgba(255,255,255,0.1);
                border-radius: 6px; padding: 7px 10px;
                color: #e2e8f0; font-size: 10pt;
            }
            QLineEdit:focus { border-color: rgba(0,212,255,0.4); }
            QComboBox {
                background: #1e2029; border: 1px solid rgba(255,255,255,0.1);
                border-radius: 6px; padding: 7px 10px;
                color: #e2e8f0; font-size: 10pt;
            }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0ea5e9,stop:1 #6366f1);
                border: none; border-radius: 8px; padding: 10px;
                color: white; font-size: 10pt; font-weight: bold;
            }
            QPushButton:hover { opacity: 0.85; }
            QPushButton#cancel {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.1);
                color: #94a3b8;
            }
            QGroupBox {
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 8px; margin-top: 8px; padding: 12px;
                color: #64748b; font-size: 9pt;
            }
            QCheckBox { color: #e2e8f0; font-size: 10pt; spacing: 8px; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid rgba(255,255,255,0.2); border-radius: 4px;
                background: #1e2029;
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0ea5e9,stop:1 #6366f1);
                border-color: transparent;
            }
        """)
        self._build_ui()

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
        hint2 = QLabel("For system audio, select your VB-Cable or loopback device")
        hint2.setStyleSheet("color: #475569; font-size: 9pt;")
        dev_layout.addWidget(hint2, 3, 0, 1, 2)
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

    def _save(self):
        self.config["api_key"]      = self.api_input.text().strip()
        self.config["mic_device"]   = self.mic_combo.currentData()
        self.config["sys_device"]   = self.sys_combo.currentData()
        self.config["capture_mode"] = self.capture_mode_combo.currentData()
        self.config["hotkey"]       = self.hotkey_input.text().strip() or "ctrl+shift+space"
        self.config["stealth_mode"] = self.stealth_check.isChecked()
        self.accept()

    def get_config(self):
        return self.config

# ── MAIN OVERLAY WINDOW ───────────────────────────────────────────────────────
class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.config       = load_config()
        self.mode         = self.config.get("mode", "general")
        self.whisper      = None
        self.capture      = None
        self.transcribe_q = []   # pending TranscribeWorker refs
        self.claude_q     = []   # pending ClaudeWorker refs
        self.full_transcript = ""
        self._last_suggestion_data = None
        self._drag_pos    = None
        self._export_entries = []   # list of suggestion dicts for session export
        self._hotkey_signaler = _HotkeySignaler()
        self._hotkey_signaler.triggered.connect(self._toggle_visibility)
        self._register_hotkey()
        self._opacity_val = 1.0

        self._setup_window()
        self._build_ui()
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
        self.setMinimumSize(400, 500)
        self.resize(420, 680)
        # Position: right side of primary screen
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - 440, 60)

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

        self.mode_btn = QPushButton(MODE_LABELS[self.mode])
        self.mode_btn.setObjectName("mode_btn")
        self.mode_btn.setFixedHeight(22)
        self.mode_btn.clicked.connect(self._cycle_mode)
        self._update_mode_btn()

        settings_btn = QPushButton("⚙")
        settings_btn.setObjectName("icon_btn")
        settings_btn.setFixedSize(26, 26)
        settings_btn.clicked.connect(self._open_settings)

        clear_btn = QPushButton("↺")
        clear_btn.setObjectName("icon_btn")
        clear_btn.setFixedSize(26, 26)
        clear_btn.setToolTip("Clear session — wipes transcript and suggestions")
        clear_btn.clicked.connect(self._clear_session)

        self.export_btn = QPushButton("↓")
        self.export_btn.setObjectName("icon_btn")
        self.export_btn.setFixedSize(26, 26)
        self.export_btn.setToolTip("Export session transcript and suggestions to .txt")
        self.export_btn.clicked.connect(self._export_session)

        hide_btn = QPushButton("✕")
        hide_btn.setObjectName("icon_btn")
        hide_btn.setFixedSize(26, 26)
        hide_btn.clicked.connect(self.close)

        hl.addWidget(self.status_dot)
        hl.addWidget(title)
        hl.addStretch()
        hl.addWidget(self.mode_btn)
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
        self.rec_btn.setFixedHeight(28)
        self.rec_btn.clicked.connect(self._toggle_recording)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setFixedWidth(70)
        self.opacity_slider.setToolTip("Opacity")
        self.opacity_slider.valueChanged.connect(self._set_opacity)

        row1.addWidget(self.rec_btn)
        row1.addStretch()
        row1.addWidget(QLabel("◑"))
        row1.addWidget(self.opacity_slider)
        cl.addLayout(row1)

        card_layout.addWidget(ctrl)

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
        tl.addWidget(self._section_label("LIVE TRANSCRIPT"))
        self.transcript_box = QTextEdit()
        self.transcript_box.setObjectName("transcript_box")
        self.transcript_box.setReadOnly(True)
        self.transcript_box.setFixedHeight(80)
        self.transcript_box.setPlaceholderText("Press Start Listening to begin...")
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

        # Response section
        self.content_layout.addWidget(self._section_label("SUGGESTED RESPONSE"))
        self.response_frame = QFrame()
        self.response_frame.setObjectName("response_frame")
        self.response_layout = QVBoxLayout(self.response_frame)
        self.response_layout.setContentsMargins(10, 8, 10, 8)
        self.response_label = QLabel("Responses will appear as conversation unfolds...")
        self.response_label.setObjectName("placeholder_text")
        self.response_label.setWordWrap(True)
        self.response_layout.addWidget(self.response_label)
        self.content_layout.addWidget(self.response_frame)

        # Topics section
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

        # ── TAB 2: History ───────────────────────────────────────────────────
        history_outer = QWidget()
        history_outer.setObjectName("content_widget")
        history_outer_layout = QVBoxLayout(history_outer)
        history_outer_layout.setContentsMargins(0, 0, 0, 0)

        self.history_scroll = QScrollArea()
        self.history_scroll.setObjectName("scroll_area")
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.history_widget = QWidget()
        self.history_widget.setObjectName("content_widget")
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.setContentsMargins(12, 8, 12, 12)
        self.history_layout.setSpacing(8)
        self.history_layout.addWidget(self._make_history_placeholder())
        self.history_layout.addStretch()

        self.history_scroll.setWidget(self.history_widget)
        history_outer_layout.addWidget(self.history_scroll)
        self.tabs.addTab(history_outer, "🕘  History")

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

    def _make_history_placeholder(self):
        lbl = QLabel("Previous suggestions will appear here as the conversation progresses...")
        lbl.setObjectName("placeholder_text")
        lbl.setFont(QFont("Segoe UI", 10))
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setContentsMargins(12, 24, 12, 24)
        return lbl

    def _add_history_entry(self, data: dict, snippet: str):
        """Add a collapsed history card for a past suggestion."""
        import datetime
        self._export_entries.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "snippet": snippet,
            "response": data.get("response", ""),
            "topics": data.get("topics", []),
            "followups": data.get("followups", []),
        })
        # Remove placeholder if present
        if self.history_layout.count() > 0:
            first = self.history_layout.itemAt(0).widget()
            if first and first.objectName() == "placeholder_text":
                self.history_layout.removeWidget(first)
                first.deleteLater()

        # Build card
        card = QFrame()
        card.setStyleSheet("""
            QFrame#history_card {
                background: rgba(255,255,255,6);
                border: 1px solid rgba(255,255,255,15);
                border-radius: 8px;
            }
            QFrame#history_card:hover {
                border-color: rgba(0,212,255,50);
                background: rgba(255,255,255,10);
            }
        """)
        card.setObjectName("history_card")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(4)

        # Header row: snippet + expand toggle
        header_row = QHBoxLayout()
        snippet_lbl = QLabel(snippet[:80] + ("…" if len(snippet) > 80 else ""))
        snippet_lbl.setFont(QFont("Segoe UI", 9))
        snippet_lbl.setStyleSheet("color: #64748b; font-style: italic;")
        snippet_lbl.setWordWrap(False)

        toggle_btn = QPushButton("▾ Show")
        toggle_btn.setFont(QFont("Segoe UI", 8))
        toggle_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid rgba(255,255,255,20);
                border-radius: 6px;
                color: #64748b;
                padding: 2px 8px;
            }
            QPushButton:hover { color: #00d4ff; border-color: rgba(0,212,255,60); }
        """)
        toggle_btn.setFixedWidth(65)
        header_row.addWidget(snippet_lbl, 1)
        header_row.addWidget(toggle_btn)
        card_layout.addLayout(header_row)

        # Collapsible detail area
        detail_widget = QWidget()
        detail_widget.setVisible(False)
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 6, 0, 0)
        detail_layout.setSpacing(6)

        # Response
        resp_lbl = QLabel(data.get("response", ""))
        resp_lbl.setFont(QFont("Segoe UI", 10))
        resp_lbl.setStyleSheet("color: #cbd5e1;")
        resp_lbl.setWordWrap(True)
        resp_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        copy_btn = QPushButton("Copy response")
        copy_btn.setFont(QFont("Segoe UI", 8))
        copy_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid rgba(255,255,255,20);
                border-radius: 6px; color: #64748b;
                padding: 2px 8px; max-width: 100px;
            }
            QPushButton:hover { border-color: #00d4ff; color: #00d4ff; }
        """)
        response_text = data.get("response", "")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(response_text))

        detail_layout.addWidget(resp_lbl)
        detail_layout.addWidget(copy_btn)

        # Topics
        for t in data.get("topics", []):
            t_row = QHBoxLayout()
            t_icon = QLabel(t.get("icon", "🔗"))
            t_icon.setFont(QFont("Segoe UI", 10))
            t_icon.setFixedWidth(20)
            t_title = QLabel(f"<b>{t.get('title','')}</b> — {t.get('detail','')}")
            t_title.setFont(QFont("Segoe UI", 9))
            t_title.setStyleSheet("color: #94a3b8;")
            t_title.setWordWrap(True)
            t_title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            t_row.addWidget(t_icon)
            t_row.addWidget(t_title, 1)
            detail_layout.addLayout(t_row)

        card_layout.addWidget(detail_widget)

        # Toggle logic
        def toggle():
            visible = not detail_widget.isVisible()
            detail_widget.setVisible(visible)
            toggle_btn.setText("▴ Hide" if visible else "▾ Show")
        toggle_btn.clicked.connect(toggle)

        # Insert before the stretch at the end
        insert_pos = max(0, self.history_layout.count() - 1)
        self.history_layout.insertWidget(insert_pos, card)

        # Scroll history to bottom
        QTimer.singleShot(50, lambda: self.history_scroll.verticalScrollBar().setValue(
            self.history_scroll.verticalScrollBar().maximum()
        ))

    # ── STYLES ────────────────────────────────────────────────────────────────
    def _apply_styles(self):
        self.setStyleSheet(f"""
            QWidget {{ font-family: 'Segoe UI', sans-serif; }}

            #card {{
                background: rgba(10,11,15,235);
                border: 1px solid rgba(255,255,255,18);
                border-radius: 16px;
            }}

            #header {{
                background: rgba(255,255,255,8);
                border-bottom: 1px solid rgba(255,255,255,18);
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
            }}

            #header_title {{
                font-size: 9pt; font-weight: 700;
                letter-spacing: 2px; color: #64748b;
            }}

            #dot_inactive {{ color: #475569; font-size: 8pt; }}
            #dot_active   {{ color: #00d4ff; font-size: 8pt; }}

            #mode_btn {{
                font-size: 8pt; font-weight: 600;
                border-radius: 11px; padding: 2px 10px;
                border: 1px solid rgba(0,212,255,100);
                background: rgba(0,212,255,30); color: #00d4ff;
            }}
            #mode_btn:hover {{ background: rgba(0,212,255,55); }}

            #icon_btn {{
                background: transparent; border: none;
                color: #64748b; font-size: 10pt; border-radius: 6px;
            }}
            #icon_btn:hover {{ background: rgba(255,255,255,20); color: #e2e8f0; }}

            #ctrl_bar {{
                border-bottom: 1px solid rgba(255,255,255,18);
                background: rgba(255,255,255,5);
            }}

            #rec_btn {{
                background: rgba(255,255,255,10);
                border: 1px solid rgba(255,255,255,18);
                border-radius: 8px; color: #e2e8f0;
                font-size: 9pt; font-weight: 600; padding: 0 12px;
            }}
            #rec_btn:hover {{ background: rgba(255,255,255,20); }}
            #rec_btn[recording=true] {{
                background: rgba(239,68,68,25);
                border-color: rgba(239,68,68,120);
                color: #fca5a5;
            }}

            #device_combo {{
                background: rgba(255,255,255,10);
                border: 1px solid rgba(255,255,255,18);
                border-radius: 6px; color: #e2e8f0; font-size: 9pt;
                padding: 2px 6px;
            }}
            #device_combo QAbstractItemView {{
                background: #1e2029; color: #e2e8f0;
                selection-background-color: rgba(0,212,255,60);
            }}

            #banner_loading {{
                background: rgba(245,158,11,20);
                border-bottom: 1px solid rgba(245,158,11,60);
                color: #fcd34d; font-size: 9pt; padding: 5px 12px;
            }}
            #banner_ok {{
                background: rgba(16,185,129,15);
                border-bottom: 1px solid rgba(16,185,129,50);
                color: #6ee7b7; font-size: 9pt; padding: 5px 12px;
            }}
            #banner_error {{
                background: rgba(239,68,68,15);
                border-bottom: 1px solid rgba(239,68,68,50);
                color: #fca5a5; font-size: 9pt; padding: 5px 12px;
            }}
            #banner_hidden {{ max-height: 0px; padding: 0px; border: none; }}
            #banner_update_frame {{
                background: rgba(124,58,237,20);
                border-bottom: 1px solid rgba(124,58,237,60);
            }}
            #banner_update_label {{
                color: #c4b5fd; font-size: 9pt;
            }}
            #update_now_btn {{
                background: rgba(124,58,237,180);
                color: white; font-size: 8pt; font-weight: bold;
                border: 1px solid rgba(124,58,237,200);
                border-radius: 4px; padding: 3px 8px;
            }}
            #update_now_btn:hover {{ background: rgba(124,58,237,220); }}
            #update_now_btn:disabled {{ background: rgba(124,58,237,80); color: rgba(255,255,255,120); }}

            #section_frame {{ background: transparent; }}

            #section_label {{
                font-size: 7pt; font-weight: 700;
                letter-spacing: 2px; color: #475569;
                padding: 4px 0 2px 0;
            }}

            #transcript_box {{
                background: rgba(255,255,255,8);
                border: 1px solid rgba(255,255,255,18);
                border-radius: 8px; color: #cbd5e1;
                font-size: 10pt; padding: 4px 6px;
                selection-background-color: rgba(0,212,255,60);
            }}
            QScrollBar:vertical {{ background: transparent; width: 4px; }}
            QScrollBar::handle:vertical {{ background: rgba(255,255,255,40); border-radius: 2px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

            #scroll_area {{ background: transparent; }}
            #content_widget {{ background: transparent; }}

            #response_frame {{
                background: rgba(0,212,255,13);
                border: 1px solid rgba(0,212,255,40);
                border-radius: 10px;
                border-left: 3px solid #00d4ff;
            }}

            #topics_frame {{ background: transparent; }}
            #followups_frame {{ background: transparent; }}

            #placeholder_text {{ color: #475569; font-size: 10pt; font-style: italic; }}

            QLabel#response_text {{
                color: #cbd5e1; font-size: 10pt;
                line-height: 1.6;
            }}

            QSlider::groove:horizontal {{
                height: 3px; background: rgba(255,255,255,25); border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 12px; height: 12px; margin: -5px 0;
                background: #00d4ff; border-radius: 6px;
            }}

            QTabWidget#main_tabs::pane {{
                border: none;
                background: transparent;
            }}
            QTabWidget#main_tabs > QTabBar::tab {{
                background: rgba(255,255,255,5);
                border: 1px solid rgba(255,255,255,12);
                border-bottom: none;
                border-radius: 6px 6px 0 0;
                padding: 5px 14px;
                margin-right: 2px;
                color: #64748b;
                font-size: 9pt;
                font-weight: 600;
            }}
            QTabWidget#main_tabs > QTabBar::tab:selected {{
                background: rgba(0,212,255,18);
                border-color: rgba(0,212,255,60);
                color: #00d4ff;
            }}
            QTabWidget#main_tabs > QTabBar::tab:hover:!selected {{
                background: rgba(255,255,255,10);
                color: #94a3b8;
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
        else:
            self._start_recording()

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
        self.capture.chunk_ready.connect(self._on_audio_chunk)
        self.capture.start()

        self.rec_btn.setText("■  Stop Listening")
        self.rec_btn.setProperty("recording", True)
        self.rec_btn.setStyle(self.rec_btn.style())
        self.status_dot.setObjectName("dot_active")
        self.status_dot.setStyle(self.status_dot.style())
        self.transcript_box.setPlaceholderText("Listening — updating every 2s...")

    def _stop_recording(self):
        if self.capture:
            self.capture.stop()
            self.capture = None
        self.rec_btn.setText("▶  Start Listening")
        self.rec_btn.setProperty("recording", False)
        self.rec_btn.setStyle(self.rec_btn.style())
        self.status_dot.setObjectName("dot_inactive")
        self.status_dot.setStyle(self.status_dot.style())

    def _on_audio_chunk(self, audio: np.ndarray):
        if not self.whisper:
            return
        # Skip if a transcription is already in progress — avoid GPU queue pileup
        if self.transcribe_q:
            return
        worker = TranscribeWorker(self.whisper, audio)
        worker.result.connect(self._on_transcription)
        self.transcribe_q.append(worker)
        worker.finished.connect(lambda: self.transcribe_q.remove(worker) if worker in self.transcribe_q else None)
        worker.start()

    def _on_transcription(self, text: str):
        text = text.strip()
        if not text:
            return

        # Deduplicate: sliding window means Whisper re-transcribes overlap.
        # Use difflib to find the longest matching suffix/prefix across windows.
        if not hasattr(self, '_last_transcribed'):
            self._last_transcribed = ""
        if not hasattr(self, '_last_words'):
            self._last_words = []

        import difflib
        curr_words = text.split()
        last_words = self._last_words

        # Find how many words at the END of last output match the START of current
        overlap = 0
        max_check = min(len(last_words), len(curr_words), 20)
        for n in range(max_check, 0, -1):
            if last_words[-n:] == curr_words[:n]:
                overlap = n
                break

        # If no prefix overlap found, try fuzzy — check if current is mostly
        # contained in last (Whisper re-ran same audio)
        if overlap == 0 and last_words:
            ratio = difflib.SequenceMatcher(
                None,
                " ".join(last_words[-len(curr_words):]).lower(),
                " ".join(curr_words).lower()
            ).ratio()
            if ratio > 0.85:
                # Almost identical — fully duplicated window, skip
                self._last_words = curr_words
                self._last_transcribed = text
                return

        new_words = curr_words[overlap:]
        if not new_words:
            self._last_words = curr_words
            self._last_transcribed = text
            return

        new_text = " ".join(new_words)
        self._last_words = curr_words
        self._last_transcribed = text
        self.full_transcript += " " + new_text
        if len(self.full_transcript) > 2000:
            self.full_transcript = self.full_transcript[-2000:]

        self.transcript_box.setPlainText(self.full_transcript.strip()[-500:])
        cursor = self.transcript_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.transcript_box.setTextCursor(cursor)

        # Debounce Claude calls — only call if no new transcription for 1.5s
        if hasattr(self, '_claude_timer'):
            self._claude_timer.stop()
        self._claude_timer = QTimer()
        self._claude_timer.setSingleShot(True)
        self._claude_timer.timeout.connect(self._fetch_suggestions)
        self._claude_timer.start(1500)

        # Silence watchdog — if no new speech for 12s, auto-archive current
        # exchange to history and start fresh (captures natural turn-taking)
        if hasattr(self, '_silence_timer'):
            self._silence_timer.stop()
        self._silence_timer = QTimer()
        self._silence_timer.setSingleShot(True)
        self._silence_timer.timeout.connect(self._auto_archive)
        self._silence_timer.start(12000)

    # ── AUTO ARCHIVE ON SILENCE ───────────────────────────────────────────────
    def _auto_archive(self):
        """Called after a silence gap — archives current exchange to history
        and resets the transcript so the next question starts fresh."""
        if not self.full_transcript.strip():
            return
        # Need at least 10 words to be worth archiving
        if len(self.full_transcript.strip().split()) < 10:
            return

        # Save current response to history if we have one
        if hasattr(self, '_last_suggestion_data') and self._last_suggestion_data:
            snippet = self.full_transcript.strip().split()[-12:]
            self._add_history_entry(self._last_suggestion_data, " ".join(snippet))

        # Flash the transcript box to signal the reset
        self.transcript_box.setStyleSheet("QTextEdit { border: 1px solid rgba(0,212,255,80); }")
        QTimer.singleShot(400, lambda: self.transcript_box.setStyleSheet(""))

        # Reset transcript state for next exchange
        self.full_transcript = ""
        self._last_suggestion_data = None
        self._last_transcribed = ""
        self._last_words = []
        self.transcript_box.clear()
        self.transcript_box.setPlaceholderText("Listening for next question...")
        self._set_response_text("Waiting for next question...", thinking=True)

    # ── AI SUGGESTIONS ────────────────────────────────────────────────────────
    def _fetch_suggestions(self):
        api_key = self.config.get("api_key", "")
        if not api_key:
            self._set_response_text("🔑 Add your Anthropic API key in Settings (⚙) to enable suggestions.", error=True)
            return

        worker = ClaudeWorker(api_key, self.mode, self.full_transcript)
        worker.result.connect(self._render_suggestions)
        worker.error.connect(lambda e: self._set_response_text(f"⚠ {e}", error=True))
        self.claude_q.append(worker)
        worker.finished.connect(lambda: self.claude_q.remove(worker) if worker in self.claude_q else None)
        self._set_response_text("Analyzing conversation...", thinking=True)
        worker.start()

    def _render_suggestions(self, data: dict):
        # Store for potential auto-archive use
        self._last_suggestion_data = data
        # Save to history before rendering live view
        snippet = self.full_transcript.strip().split()[-12:]  # last ~12 words as snippet
        self._add_history_entry(data, " ".join(snippet))

        # Response
        self._set_response_text(data.get("response", ""))

        # Topics
        while self.topics_layout.count():
            item = self.topics_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for t in data.get("topics", []):
            card = self._make_topic_card(t)
            self.topics_layout.addWidget(card)

        # Follow-ups
        while self.followups_layout.count():
            item = self.followups_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        followups = data.get("followups", [])
        if followups:
            self.followups_label.show()
            self.followups_frame.show()
            for q in followups:
                chip = QPushButton(q)
                chip.setObjectName("followup_chip")
                chip.setFont(QFont("Segoe UI", 9))
                chip.setStyleSheet("""
                    QPushButton {
                        background: rgba(124,58,237,20); border: 1px solid rgba(124,58,237,80);
                        border-radius: 12px; color: #a78bfa;
                        padding: 5px 12px; text-align: left;
                    }
                    QPushButton:hover { background: rgba(124,58,237,50); }
                """)
                chip.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                chip.clicked.connect(lambda _, text=q: QApplication.clipboard().setText(text))
                chip.setToolTip("Click to copy")
                # Note: QPushButton has no setWordWrap — use a fixed max width instead
                chip.setMaximumWidth(370)
                self.followups_layout.addWidget(chip)
        else:
            self.followups_label.hide()
            self.followups_frame.hide()

    def _set_response_text(self, text, error=False, thinking=False):
        while self.response_layout.count():
            item = self.response_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setFont(QFont("Segoe UI", 10))
        if error:
            lbl.setStyleSheet("color: #fca5a5; font-style: italic;")
        elif thinking:
            lbl.setStyleSheet("color: #64748b; font-style: italic;")
        else:
            lbl.setStyleSheet("color: #cbd5e1;")
            self.response_layout.addWidget(lbl)
            copy_btn = QPushButton("Copy")
            copy_btn.setFont(QFont("Segoe UI", 9))
            copy_btn.setStyleSheet("""
                QPushButton {
                    background: transparent; border: 1px solid rgba(255,255,255,25);
                    border-radius: 10px; color: #64748b; padding: 3px 10px;
                    max-width: 60px;
                }
                QPushButton:hover { border-color: #00d4ff; color: #00d4ff; }
            """)
            def _do_copy(btn=copy_btn, t=text):
                QApplication.clipboard().setText(t)
                btn.setText("✓ Copied!")
                btn.setStyleSheet(btn.styleSheet() + "QPushButton { color: #10b981; border-color: #10b981; }")
                QTimer.singleShot(1500, lambda: (btn.setText("Copy"), btn.setStyleSheet("""
                    QPushButton {
                        background: transparent; border: 1px solid rgba(255,255,255,25);
                        border-radius: 10px; color: #64748b; padding: 3px 10px;
                        max-width: 60px;
                    }
                    QPushButton:hover { border-color: #00d4ff; color: #00d4ff; }
                """)))
            copy_btn.clicked.connect(_do_copy)
            self.response_layout.addWidget(copy_btn)
            return
        self.response_layout.addWidget(lbl)

    def _make_topic_card(self, topic: dict):
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,8);
                border: 1px solid rgba(255,255,255,18);
                border-radius: 8px;
            }
            QFrame:hover { background: rgba(255,255,255,15); border-color: rgba(0,212,255,60); }
        """)
        card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum   # shrink/grow vertically to fit content
        )
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
        title.setStyleSheet("color: #e2e8f0;")
        title.setWordWrap(True)   # titles can wrap too
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        header_row.addWidget(icon)
        header_row.addWidget(title, 1)
        layout.addLayout(header_row)

        detail = topic.get("detail", "")
        if detail:
            detail_lbl = QLabel(detail)
            detail_lbl.setFont(QFont("Segoe UI", 9))
            detail_lbl.setStyleSheet("color: #64748b; padding-left: 22px;")
            detail_lbl.setWordWrap(True)
            detail_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            layout.addWidget(detail_lbl)
        return card

    # ── CLEAR SESSION ─────────────────────────────────────────────────────────
    def _clear_session(self):
        # Stop recording if active
        was_recording = self.capture is not None
        if was_recording:
            self._stop_recording()

        # Wipe transcript state
        self.full_transcript = ""
        self._last_suggestion_data = None
        self._last_transcribed = ""
        self._last_words = []
        self._export_entries.clear()
        if hasattr(self, '_claude_timer'):
            self._claude_timer.stop()

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

        # Clear history tab
        while self.history_layout.count():
            item = self.history_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.history_layout.addWidget(self._make_history_placeholder())
        self.history_layout.addStretch()

        # Restart recording if it was running
        if was_recording:
            QTimer.singleShot(300, self._start_recording)

    # ── MODE ──────────────────────────────────────────────────────────────────
    def _cycle_mode(self):
        idx = MODES.index(self.mode)
        self.mode = MODES[(idx + 1) % len(MODES)]
        self.config["mode"] = self.mode
        save_config(self.config)
        self._update_mode_btn()

    def _update_mode_btn(self):
        self.mode_btn.setText(MODE_LABELS[self.mode])
        c = MODE_COLORS[self.mode]
        self.mode_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 8pt; font-weight: 600;
                border-radius: 11px; padding: 2px 10px;
                border: 1px solid {c}99;
                background: {c}33; color: {c};
            }}
            QPushButton:hover {{ background: {c}55; }}
        """)

    # ── SETTINGS ──────────────────────────────────────────────────────────────
    def _export_session(self):
        import datetime
        now = datetime.datetime.now()
        filename = os.path.join(
            os.path.expanduser("~"), "Documents",
            f"PandAI_Session_{now.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        sep = "=" * 52
        lines = [
            "PandAI Assistant — Session Export",
            f"Date : {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Mode : {MODE_LABELS.get(self.mode, self.mode)}",
            "",
            sep, "TRANSCRIPT", sep,
            self.full_transcript.strip() or "(no transcript recorded)",
            "",
            sep, "SUGGESTIONS", sep,
        ]
        if not self._export_entries:
            lines.append("(no suggestions generated this session)")
        for entry in self._export_entries:
            lines += [
                f"\n[{entry['time']}]",
                f"Context : {entry['snippet']}",
                f"Response: {entry['response']}",
            ]
            if entry["topics"]:
                lines.append("Topics  : " + " | ".join(t.get("title", "") for t in entry["topics"]))
            if entry["followups"]:
                lines.append("Ask next: " + " | ".join(entry["followups"]))
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            # Brief feedback on the button
            self.export_btn.setText("✓")
            self.export_btn.setToolTip(f"Saved: {filename}")
            QTimer.singleShot(2000, lambda: (
                self.export_btn.setText("↓"),
                self.export_btn.setToolTip("Export session transcript and suggestions to .txt")
            ))
        except Exception as e:
            self.export_btn.setText("✗")
            QTimer.singleShot(2000, lambda: self.export_btn.setText("↓"))

    def _register_hotkey(self):
        try:
            import keyboard
            keyboard.unhook_all_hotkeys()
            hotkey = self.config.get("hotkey", "ctrl+shift+space")
            keyboard.add_hotkey(hotkey, self._hotkey_signaler.triggered.emit)
        except Exception:
            pass

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
        if val >= 100:
            # Fully opaque — solid background, no transparency
            self.card.setStyleSheet("""
                #card {
                    background: rgb(10,11,15);
                    border: 1px solid rgba(255,255,255,18);
                    border-radius: 16px;
                }
            """)
            self.setWindowOpacity(1.0)
        else:
            # Restore transparent card background
            alpha = int(235 * (val / 100))
            self.card.setStyleSheet(f"""
                #card {{
                    background: rgba(10,11,15,{alpha});
                    border: 1px solid rgba(255,255,255,18);
                    border-radius: 16px;
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
    # On Windows, re-launch with pythonw.exe to suppress the console window
    if sys.platform == "win32" and os.path.basename(sys.executable).lower() == "python.exe":
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
