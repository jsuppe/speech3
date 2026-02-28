"""
SpeakFit GPU Worker - Celery Tasks
Processes transcription jobs from Redis queue
"""
import os
import logging
from celery import Celery
from faster_whisper import WhisperModel
import tempfile
from google.cloud import storage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("speakfit", broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 min max per task
    worker_prefetch_multiplier=1,  # One task at a time (GPU bound)
)

# Initialize Whisper model (loaded once per worker)
_whisper_model = None

def get_whisper_model():
    """Lazy-load Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper large-v3 model...")
        _whisper_model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",
        )
        logger.info("Whisper model loaded.")
    return _whisper_model

# Cloud Storage client
_storage_client = None

def get_storage_client():
    """Lazy-load storage client."""
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


@app.task(bind=True, max_retries=3)
def transcribe_audio(self, speech_id: int, audio_url: str, analyze_confidence: bool = False):
    """
    Transcribe audio file using Whisper.
    
    Args:
        speech_id: Database ID of the speech record
        audio_url: GCS URL (gs://bucket/path) or HTTP URL
        analyze_confidence: Whether to run confidence analysis
        
    Returns:
        dict with transcript, segments, and optional confidence metrics
    """
    try:
        logger.info(f"Processing speech {speech_id}: {audio_url}")
        
        # Download audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            temp_path = f.name
            
            if audio_url.startswith("gs://"):
                # Download from GCS
                bucket_name = audio_url.split("/")[2]
                blob_path = "/".join(audio_url.split("/")[3:])
                client = get_storage_client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.download_to_filename(temp_path)
            else:
                # Download from HTTP
                import httpx
                with httpx.Client() as client:
                    response = client.get(audio_url)
                    f.write(response.content)
        
        # Transcribe
        model = get_whisper_model()
        segments, info = model.transcribe(
            temp_path,
            beam_size=5,
            language="en",
            word_timestamps=True,
            vad_filter=True,
        )
        
        # Collect results
        transcript_parts = []
        segment_data = []
        
        for segment in segments:
            transcript_parts.append(segment.text)
            segment_data.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in (segment.words or [])
                ],
            })
        
        result = {
            "speech_id": speech_id,
            "transcript": " ".join(transcript_parts).strip(),
            "segments": segment_data,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }
        
        # Optional confidence analysis
        if analyze_confidence:
            from api.confidence_analysis import analyze_confidence_quick
            confidence_result = analyze_confidence_quick(temp_path, result["transcript"])
            result["confidence"] = confidence_result
        
        # Cleanup
        os.unlink(temp_path)
        
        logger.info(f"Completed speech {speech_id}: {len(result['transcript'])} chars")
        return result
        
    except Exception as e:
        logger.error(f"Error processing speech {speech_id}: {e}")
        # Retry with exponential backoff
        self.retry(exc=e, countdown=2 ** self.request.retries)


@app.task
def health_check():
    """Health check task for monitoring."""
    model = get_whisper_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": True,
    }
