"""
Pipeline runner — wraps voice_pipeline.py logic into callable functions.
Models are loaded once at module level and reused across requests.
All analysis results are automatically persisted to SQLite (speechscore.db).
"""

import os
import sys
import time
import logging
import subprocess
import tempfile
import numpy as np

# Ensure speech3 root is on the path
SPEECH3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SPEECH3_DIR not in sys.path:
    sys.path.insert(0, SPEECH3_DIR)

# Suppress noisy library logs during import
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("speechscore.pipeline")

# ---------------------------------------------------------------------------
# Global model holders — populated by preload_models()
# ---------------------------------------------------------------------------
_whisper_model = None
_models_loaded = False
_speech_db = None


def get_db():
    """Lazy-load the SpeechDB singleton."""
    global _speech_db
    if _speech_db is None:
        from speech_db import SpeechDB
        _speech_db = SpeechDB()
        logger.info(f"SpeechDB initialized at {_speech_db.db_path}")
    return _speech_db


def preload_models():
    """Load all heavy models once. Call at startup."""
    global _whisper_model, _models_loaded

    if _models_loaded:
        return

    logger.info("Pre-loading models...")

    # Whisper
    from faster_whisper import WhisperModel
    t0 = time.time()
    _whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    logger.info(f"Whisper large-v3 loaded in {time.time() - t0:.1f}s")

    # Pre-import heavy modules so first request is fast
    t1 = time.time()
    import audio_quality  # noqa: F401
    import analysis3_module  # noqa: F401
    import advanced_text_analysis  # noqa: F401
    import rhetorical_analysis  # noqa: F401
    import sentiment_analysis  # noqa: F401
    import audio_analysis  # noqa: F401
    import advanced_audio_analysis  # noqa: F401
    logger.info(f"Analysis modules imported in {time.time() - t1:.1f}s")

    _models_loaded = True
    logger.info("All models pre-loaded.")


def is_model_loaded() -> bool:
    return _models_loaded and _whisper_model is not None


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def convert_to_wav(input_path: str) -> str:
    """Convert audio to 16kHz mono WAV. Returns path to temp WAV file."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[:500]}")
        return wav_path
    except Exception:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise


def run_pipeline(
    audio_path: str,
    progress_callback=None,
    modules: set = None,
    language: str = None,
    quality_gate: str = "warn",
    preset: str = None,
    modules_requested: list = None,
) -> dict:
    """
    Run the speech analysis pipeline on an audio file.

    Args:
        audio_path: Path to WAV audio file.
        progress_callback: Optional callable(stage_name: str) called before each stage.
        modules: Set of module names to run. None = all modules.
        language: Language code for Whisper (None = auto-detect).
        quality_gate: "warn" (default), "block", or "skip".
        preset: Name of preset used (for metadata only).
        modules_requested: Original module list from request (for metadata).
    """
    from . import config as api_config

    if not _models_loaded:
        raise RuntimeError("Models not loaded. Call preload_models() first.")

    # Default to all modules if not specified
    if modules is None:
        modules = api_config.ALL_MODULES.copy()

    # Transcription is always required
    modules.add("transcription")

    def _report(stage: str):
        if progress_callback:
            try:
                progress_callback(stage)
            except Exception:
                pass

    def _want(mod: str) -> bool:
        """Check if a module should run."""
        return mod in modules

    t0 = time.time()
    modules_run = []
    timing = {}

    # --- Audio Quality ---
    quality = None
    is_unusable = False
    is_low = False

    if _want("quality"):
        _report("audio_quality")
        tq = time.time()
        from audio_quality import load_audio_as_pcm, assess_quality
        audio_pcm, sr = load_audio_as_pcm(audio_path)
        quality = assess_quality(audio_pcm, sr)
        timing["quality_check_sec"] = round(time.time() - tq, 2)
        modules_run.append("audio_quality")

        is_unusable = quality.get("confidence") == "UNUSABLE"
        is_low = quality.get("confidence") == "LOW"

        # Quality gate: block mode
        if quality_gate == "block" and (is_unusable or is_low):
            gate_label = quality.get("confidence", "UNKNOWN")
            raise QualityGateError(
                f"Audio quality is {gate_label}. "
                f"Analysis blocked by quality_gate=block. "
                f"SNR: {quality.get('snr_db', 'N/A')} dB, "
                f"Confidence: {quality.get('confidence', 'N/A')}"
            )
    elif quality_gate == "skip":
        # Explicitly skipping quality check
        pass
    else:
        # quality module not in requested set but quality_gate != skip
        # Still run quality for warn mode if not explicitly excluded
        pass

    # --- Transcription ---
    _report("transcription")
    tt = time.time()

    # Build Whisper transcribe kwargs
    whisper_kwargs = {
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 300},
        "word_timestamps": True,  # Enable word-level timestamps
    }
    if language:
        whisper_kwargs["language"] = language
    else:
        whisper_kwargs["language"] = "en"  # original default

    segments_gen, info = _whisper_model.transcribe(audio_path, **whisper_kwargs)

    segment_list = []
    full_text_parts = []
    for seg in segments_gen:
        # Extract word-level timestamps if available
        words = []
        if hasattr(seg, 'words') and seg.words:
            for w in seg.words:
                words.append({
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "word": w.word.strip(),
                    "probability": round(w.probability, 3) if hasattr(w, 'probability') else None,
                })
        
        segment_list.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "words": words,  # Include word-level data
        })
        full_text_parts.append(seg.text.strip())

    transcript = " ".join(full_text_parts)
    timing["transcription_sec"] = round(time.time() - tt, 2)
    modules_run.append("transcription")

    # --- Text Analysis ---
    analysis = None
    if _want("text"):
        _report("text_analysis")
        ta = time.time()
        from analysis3_module import analyze_transcript
        if is_unusable:
            analysis = {"metrics": {}, "key_terms": [], "skipped": "Audio quality too low"}
        else:
            analysis = analyze_transcript(transcript) if transcript else {"metrics": {}, "key_terms": []}
            if is_low:
                analysis["warning"] = "Low audio quality — results may be unreliable."
        timing["text_analysis_sec"] = round(time.time() - ta, 2)
        modules_run.append("text_analysis")

    # --- Advanced Text ---
    advanced_text = None
    if _want("advanced_text"):
        _report("advanced_text_analysis")
        tat = time.time()
        from advanced_text_analysis import analyze_advanced_text
        if is_unusable:
            advanced_text = {"skipped": "Audio quality too low"}
        else:
            advanced_text = analyze_advanced_text(transcript) if transcript else {}
        timing["advanced_text_sec"] = round(time.time() - tat, 2)
        modules_run.append("advanced_text_analysis")

    # --- Rhetoric ---
    rhetoric = None
    if _want("rhetorical"):
        _report("rhetorical_analysis")
        tr = time.time()
        from rhetorical_analysis import analyze_rhetoric
        if is_unusable:
            rhetoric = {"skipped": "Audio quality too low"}
        else:
            rhetoric = analyze_rhetoric(transcript, segment_list) if transcript else {}
        timing["rhetoric_sec"] = round(time.time() - tr, 2)
        modules_run.append("rhetorical_analysis")

    # --- Sentiment ---
    sentiment_result = None
    if _want("sentiment"):
        _report("sentiment_analysis")
        ts = time.time()
        from sentiment_analysis import analyze_sentiment
        if is_unusable:
            sentiment_result = {"skipped": "Audio quality too low"}
        else:
            sentiment_result = analyze_sentiment(transcript, segment_list) if transcript else {
                "overall": {"label": "NEUTRAL"}, "per_segment": [], "emotional_arc": "flat"
            }
            if is_low:
                sentiment_result["warning"] = "Low audio quality — transcript errors may affect sentiment accuracy."
        timing["sentiment_sec"] = round(time.time() - ts, 2)
        modules_run.append("sentiment_analysis")

    # --- Audio Analysis ---
    audio_analysis_result = None
    if _want("audio"):
        _report("audio_analysis")
        tab = time.time()
        from audio_analysis import analyze_audio
        audio_duration = segment_list[-1]["end"] if segment_list else 0
        if is_unusable:
            audio_analysis_result = {"skipped": "Audio quality too low"}
        else:
            audio_analysis_result = analyze_audio(audio_path, segment_list, audio_duration)
            if is_low:
                audio_analysis_result["warning"] = "Low audio quality — audio metrics may be affected."
        timing["audio_analysis_sec"] = round(time.time() - tab, 2)
        modules_run.append("audio_analysis")

    # --- Advanced Audio ---
    advanced_audio = None
    if _want("advanced_audio"):
        _report("advanced_audio_analysis")
        tac = time.time()
        from advanced_audio_analysis import analyze_advanced_audio
        if is_unusable:
            advanced_audio = {"skipped": "Audio quality too low"}
        else:
            advanced_audio = analyze_advanced_audio(audio_path, segment_list)
        timing["advanced_audio_sec"] = round(time.time() - tac, 2)
        modules_run.append("advanced_audio_analysis")

    total_time = time.time() - t0
    timing["total_sec"] = round(total_time, 2)

    # Compute audio duration from segments or ffprobe
    computed_duration = segment_list[-1]["end"] if segment_list else get_audio_duration(audio_path)

    # Build result — only include sections for modules that were run
    result = {
        "audio_file": os.path.basename(audio_path),
        "audio_duration_sec": round(computed_duration, 2),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "transcript": transcript,
        "segments": segment_list,
        "processing": {
            "duration_s": round(computed_duration, 2),
            "processing_time_s": round(total_time, 2),
            "modules_requested": modules_requested or sorted(modules),
            "modules_run": modules_run,
            "preset": preset,
            "quality_gate": quality_gate,
            "language_requested": language,
            "audio_quality": quality,
            "timing": timing,
        }
    }

    # Conditionally include analysis sections
    if quality is not None:
        result["audio_quality"] = quality
    if analysis is not None:
        result["speech3_text_analysis"] = analysis
    if advanced_text is not None:
        result["advanced_text_analysis"] = advanced_text
    if rhetoric is not None:
        result["rhetorical_analysis"] = rhetoric
    if sentiment_result is not None:
        result["sentiment"] = sentiment_result
    if audio_analysis_result is not None:
        result["audio_analysis"] = audio_analysis_result
    if advanced_audio is not None:
        result["advanced_audio_analysis"] = advanced_audio

    return result


def _generate_spectrogram_bytes(audio_path: str) -> bytes | None:
    """Generate spectrogram PNG from audio file using songsee."""
    try:
        import shutil
        if not shutil.which("songsee"):
            return None
        fd, png_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        result = subprocess.run(
            ["songsee", audio_path, "-o", png_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and os.path.exists(png_path):
            with open(png_path, "rb") as f:
                return f.read()
        return None
    except Exception as e:
        logger.warning(f"Spectrogram generation failed: {e}")
        return None
    finally:
        if 'png_path' in locals() and os.path.exists(png_path):
            os.unlink(png_path)


def persist_result(
    original_audio_path: str,
    result: dict,
    filename: str = None,
    preset: str = None,
    title: str = None,
    speaker: str = None,
    products: list = None,
    category: str = None,
    source_url: str = None,
    spectrogram_path: str = None,
    user_id: int = None,
) -> int | None:
    """
    Persist analysis result + audio (as Opus) to SQLite.
    Returns speech_id or None on error.
    """
    try:
        db = get_db()
        audio_hash = db.get_audio_hash(original_audio_path)
        duration = db.get_audio_duration(original_audio_path)

        # Auto-generate title from filename if not provided
        if not title:
            title = filename or os.path.basename(original_audio_path)
            # Strip extension and clean up
            title = os.path.splitext(title)[0].replace("_", " ").replace("-", " ").strip()
            if not title:
                title = f"Upload {time.strftime('%Y-%m-%d %H:%M')}"

        # Check for duplicate audio
        existing = db.find_speech_by_hash(audio_hash)
        if existing:
            # Still store new analysis against existing speech
            speech_id = existing["id"]
            logger.info(f"Duplicate audio — appending analysis to speech_id={speech_id}")
        else:
            # Create new speech record
            speech_id = db.add_speech(
                title=title,
                speaker=speaker,
                source_url=source_url,
                source_file=filename or os.path.basename(original_audio_path),
                duration_sec=duration,
                category=category,
                products=products or ["SpeechScore"],
                audio_hash=audio_hash,
                user_id=user_id,
            )

            # Store audio as Opus
            try:
                db.store_audio(speech_id, original_audio_path)
            except Exception as e:
                logger.warning(f"Failed to encode audio to Opus: {e}")

        # Store transcription
        transcript_text = result.get("transcript", "")
        segments = result.get("segments", [])
        if transcript_text:
            db.store_transcription(
                speech_id,
                text=transcript_text,
                segments=segments,
                language=result.get("language"),
            )

        # Store analysis (remap to expected structure for indexed extraction)
        # The result from run_pipeline uses top-level keys that store_analysis expects
        db.store_analysis(speech_id, result, preset=preset)

        # Store spectrogram — use provided path or auto-generate
        if spectrogram_path and os.path.exists(spectrogram_path):
            db.store_spectrogram_from_file(speech_id, spectrogram_path)
        else:
            # Auto-generate spectrogram from the original audio
            png_data = _generate_spectrogram_bytes(original_audio_path)
            if png_data:
                db.store_spectrogram(speech_id, png_data, spec_type="combined", fmt="png")
                logger.info(f"Auto-generated spectrogram for speech_id={speech_id} ({len(png_data)/1024:.0f} KB)")

        logger.info(f"Persisted analysis → speech_id={speech_id}, title='{title}'")
        return speech_id

    except Exception as e:
        logger.exception(f"Failed to persist analysis result: {e}")
        return None


class QualityGateError(Exception):
    """Raised when quality_gate=block and audio quality is too low."""
    pass
