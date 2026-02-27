# config.py - Central configuration for the VoiceBot system
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
AUDIO_DIR = BASE_DIR / "audio_output"

# Create directories if they don't exist
for d in [MODEL_DIR, DATA_DIR, LOGS_DIR, AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── ASR (Whisper) ──────────────────────────────────────────────────────────────
ASR_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")   # tiny/base/small/medium/large
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")            # cpu / cuda

# ── Intent Classifier ─────────────────────────────────────────────────────────
INTENT_BASE_MODEL = os.getenv("INTENT_BASE_MODEL", "distilbert-base-uncased")
INTENT_MODEL_PATH = str(MODEL_DIR / "intent_model")
INTENT_LABELS_PATH = str(MODEL_DIR / "intent_labels.json")
INTENT_MAX_LEN = 128
INTENT_BATCH_SIZE = 16
INTENT_EPOCHS = 5
INTENT_LR = 2e-5
INTENT_CONFIDENCE_THRESHOLD = 0.50  # below this → "unknown" fallback

# ── Response Generator ────────────────────────────────────────────────────────
INTENTS_JSON_PATH = str(DATA_DIR / "intents.json")

# ── TTS (gTTS) ────────────────────────────────────────────────────────────────
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en")
TTS_SLOW = False                                       # True = slower speech
TTS_OUTPUT_FORMAT = "mp3"                              # mp3 / wav

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "AI VoiceBot - Customer Support"
API_VERSION = "1.0.0"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = str(LOGS_DIR / "voicebot.log")
