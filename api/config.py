"""SpeechScore API configuration."""

import os

# Server
HOST = os.getenv("SPEECHSCORE_HOST", "0.0.0.0")
PORT = int(os.getenv("SPEECHSCORE_PORT", "8000"))

# ---------------------------------------------------------------------------
# Analysis Module Definitions (Milestone 3)
# ---------------------------------------------------------------------------
# All valid module names that callers can request
VALID_MODULES = {
    "transcription",    # Whisper STT (always required)
    "diarization",      # Speaker diarization (resemblyzer + clustering)
    "text",             # core text analysis (analysis3_module)
    "advanced_text",    # readability, discourse, entities, perspective
    "rhetorical",       # anaphora, repetition, rhythm
    "sentiment",        # 3-class sentiment + emotional arc
    "audio",            # pitch, intensity, jitter/shimmer/HNR, pauses, WPM
    "advanced_audio",   # formants, pitch contour, vocal fry, rate variability, emphasis
    "quality",          # audio quality gate (SNR, confidence)
    "all",              # shorthand for everything
}

# The full set when "all" is requested
ALL_MODULES = VALID_MODULES - {"all"}

# Named presets → module sets
MODULE_PRESETS = {
    "quick":     {"transcription", "quality"},
    "text_only": {"transcription", "quality", "text", "advanced_text", "rhetorical", "sentiment"},
    "audio_only": {"transcription", "quality", "audio", "advanced_audio"},
    "full":      ALL_MODULES.copy(),
    "clinical":  {"transcription", "quality", "audio", "advanced_audio"},
    "coaching":  {"transcription", "quality", "text", "audio", "sentiment"},
}

# Quality gate modes
VALID_QUALITY_GATES = {"block", "warn", "skip"}
DEFAULT_QUALITY_GATE = "warn"

# Limits
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_DURATION_SEC = 600  # 10 minutes

# Accepted audio formats (extension -> MIME types)
ALLOWED_EXTENSIONS = {"mp3", "wav", "aac", "ogg", "m4a", "flac", "webm"}
ALLOWED_CONTENT_TYPES = {
    "audio/mpeg", "audio/mp3",
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/aac",
    "audio/ogg", "audio/vorbis",
    "audio/m4a", "audio/x-m4a", "audio/mp4",
    "audio/flac", "audio/x-flac",
    "audio/webm",
    "application/octet-stream",  # fallback for some clients
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /home/jsuppe/speech3
API_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "speechscore.db")
LOG_DIR = os.path.join(API_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "api.log")
ASYNC_UPLOAD_DIR = os.path.join(API_DIR, "tmp_uploads")

# Whisper
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# Job Queue
MAX_QUEUE_DEPTH = int(os.getenv("SPEECHSCORE_MAX_QUEUE", "20"))
JOB_EXPIRATION_HOURS = int(os.getenv("SPEECHSCORE_JOB_EXPIRY_HOURS", "24"))
JOB_CLEANUP_INTERVAL_SEC = 3600  # 1 hour
WEBHOOK_TIMEOUT_SEC = 10
