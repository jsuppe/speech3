"""
Pipeline runner — wraps voice_pipeline.py logic into callable functions.
Models are loaded once at module level and reused across requests.
All analysis results are automatically persisted to SQLite (speechscore.db).

@arch component=pipeline_runner layer=api type=module
@arch depends=[Whisper STT, audio_analysis, text_analysis, diarization, SQLite DB]
@arch external=[faster_whisper, spacy, numpy, torch]
@arch description="Orchestrates all speech analysis modules"

@arch.flow run_analysis
@arch.step 1: load_audio "Load and convert audio file"
@arch.step 2: transcribe "Run Whisper STT"
@arch.step 3: analyze_audio "Extract audio features"
@arch.step 4: analyze_text "Process transcript"
@arch.step 5: score "Calculate overall score"
@arch.step 6: persist "Save to database"
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


def generate_title_from_transcript(transcript: str, max_words: int = 8) -> str:
    """
    Generate a concise title from transcript content.
    Uses the first meaningful sentence or phrase, cleaned up for display.
    
    Args:
        transcript: Full transcript text
        max_words: Maximum words in the title (default 8)
    
    Returns:
        A clean, readable title string
    """
    if not transcript or not transcript.strip():
        return None
    
    text = transcript.strip()
    
    # Try to get the first sentence
    import re
    
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    first_sentence = sentences[0].strip() if sentences else text
    
    # Clean up the text
    # Remove filler words from the start
    filler_starts = ['um', 'uh', 'so', 'well', 'okay', 'ok', 'like', 'you know', 'i mean']
    words = first_sentence.split()
    
    # Skip leading fillers
    while words and words[0].lower().rstrip(',') in filler_starts:
        words.pop(0)
    
    if not words:
        # Fallback to original if all words were fillers
        words = first_sentence.split()
    
    # Take up to max_words
    title_words = words[:max_words]
    
    # Join and clean
    title = ' '.join(title_words)
    
    # Remove trailing punctuation except for sentence-ending ones
    title = title.rstrip(',;:')
    
    # Add ellipsis if we truncated
    if len(words) > max_words:
        title = title.rstrip('.!?') + '...'
    
    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:] if len(title) > 1 else title.upper()
    
    # Ensure reasonable length (not too short)
    if len(title) < 10 and len(text) > len(title):
        # Title too short, try getting more words
        words = text.split()[:max_words + 3]
        title = ' '.join(words)
        if len(text.split()) > len(words):
            title = title.rstrip('.!?,;:') + '...'
        title = title[0].upper() + title[1:] if len(title) > 1 else title.upper()
    
    return title if title else None

# ---------------------------------------------------------------------------
# Global model holders — populated by preload_models()
# ---------------------------------------------------------------------------
import threading

_whisper_model = None
_models_loaded = False
_models_lock = threading.Lock()
_speech_db = None
_db_lock = threading.Lock()


def get_db():
    """Lazy-load the SpeechDB singleton (thread-safe)."""
    global _speech_db
    if _speech_db is None:
        with _db_lock:
            if _speech_db is None:  # Double-check after acquiring lock
                from speech_db import SpeechDB
                _speech_db = SpeechDB()
                logger.info(f"SpeechDB initialized at {_speech_db.db_path}")
    return _speech_db


def preload_models():
    """Load all heavy models once. Call at startup (thread-safe)."""
    global _whisper_model, _models_loaded

    if _models_loaded:
        return

    with _models_lock:
        # Double-check after acquiring lock to prevent race condition
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
        import analysis3_module
        analysis3_module.preload_models()  # Load spaCy + SentenceTransformer
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
    """Get audio duration in seconds using ffprobe.
    
    Tries multiple methods since browser-recorded WebM files often
    don't have duration in the format header.
    """
    # Method 1: Try format duration (fast)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True, timeout=10
        )
        duration = float(result.stdout.strip())
        if duration > 0:
            return duration
    except Exception:
        pass
    
    # Method 2: Try stream duration (for files without format duration)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a:0",
             "-show_entries", "stream=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True, timeout=10
        )
        duration = float(result.stdout.strip())
        if duration > 0:
            return duration
    except Exception:
        pass
    
    # Method 3: Decode to get actual duration (slower but reliable for browser WebM)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             "-i", audio_path],
            capture_output=True, text=True, timeout=30
        )
        duration = float(result.stdout.strip())
        if duration > 0:
            return duration
    except Exception:
        pass
    
    # Method 4: Use ffmpeg to decode and count frames (last resort)
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-f", "null", "-"],
            capture_output=True, text=True, timeout=60
        )
        # Parse duration from ffmpeg stderr output
        import re
        match = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", result.stderr)
        if match:
            h, m, s, cs = match.groups()
            return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100
        
        # Try to find "time=" at the end
        match = re.search(r"time=(\d+):(\d+):(\d+)\.(\d+)", result.stderr)
        if match:
            h, m, s, cs = match.groups()
            return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100
    except Exception:
        pass
    
    logger.warning(f"Could not determine duration for {audio_path}")
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
    known_speaker: str = None,
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
        known_speaker: Speaker name from metadata (used for diarization if single speaker).
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

    # --- Speaker Diarization ---
    diarization_result = None
    speaker_registration_result = None
    speaker_match_result = None
    if _want("diarization") and segment_list:
        _report("diarization")
        td = time.time()
        try:
            from diarization import run_diarization, get_speaker_stats, register_diarized_speakers, match_speakers_to_profiles
            # Run diarization and update segment_list in place
            # Pass known_speaker so single-speaker recordings use the actual name
            segment_list = run_diarization(audio_path, segment_list, known_speaker=known_speaker)
            
            # Try to match diarized speakers against stored voice profiles
            if user_id:
                try:
                    from speech_db import get_db
                    db = get_db()
                    segment_list, speaker_match_result = match_speakers_to_profiles(
                        db=db,
                        user_id=user_id,
                        audio_path=audio_path,
                        segments=segment_list,
                        match_threshold=0.80,
                    )
                    if speaker_match_result.get("matches"):
                        logger.info(f"Speaker matching: {len(speaker_match_result['matches'])} matches found")
                except Exception as e:
                    logger.warning(f"Speaker profile matching failed: {e}")
            
            diarization_result = get_speaker_stats(segment_list)
            timing["diarization_sec"] = round(time.time() - td, 2)
            modules_run.append("diarization")
            
            # Register diarized speakers as voice fingerprints for future matching
            if user_id and diarization_result.get("num_speakers", 0) > 0:
                try:
                    if db is None:
                        from speech_db import get_db
                        db = get_db()
                    speaker_registration_result = register_diarized_speakers(
                        db=db,
                        user_id=user_id,
                        audio_path=audio_path,
                        segments=segment_list,
                        speech_id=None,  # Will be set after persist
                        min_duration_per_speaker=2.0,  # At least 2 seconds of speech
                    )
                    if speaker_registration_result.get("registered") or speaker_registration_result.get("matched"):
                        logger.info(f"Speaker registration: {speaker_registration_result}")
                except Exception as e:
                    logger.warning(f"Speaker registration failed: {e}")
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")
            diarization_result = {"error": str(e)}

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
    if diarization_result is not None:
        result["diarization"] = diarization_result
    if speaker_match_result is not None and speaker_match_result.get("matches"):
        result["speaker_matches"] = speaker_match_result
    if speaker_registration_result is not None:
        result["speaker_profiles"] = speaker_registration_result
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
    profile: str = "general",
) -> int | None:
    """
    Persist analysis result + audio (as Opus) to SQLite.
    Returns speech_id or None on error.
    """
    try:
        db = get_db()
        
        # Check if audio file still exists (may have been cleaned up)
        audio_exists = os.path.exists(original_audio_path)
        if not audio_exists:
            logger.warning(f"Audio file no longer exists: {original_audio_path}")
        
        # Get audio hash and duration (use result data as fallback)
        if audio_exists:
            audio_hash = db.get_audio_hash(original_audio_path)
            duration = db.get_audio_duration(original_audio_path)
        else:
            # Generate a unique hash from filename + timestamp
            import hashlib
            audio_hash = hashlib.sha256(f"{filename or original_audio_path}_{time.time()}".encode()).hexdigest()[:32]
            # Try to get duration from result
            duration = result.get("audio_metrics", {}).get("duration_sec") or result.get("processing", {}).get("duration_sec") or 0

        # Auto-generate title from transcript if not provided
        if not title:
            transcript_text = result.get("transcript", "")
            if transcript_text:
                title = generate_title_from_transcript(transcript_text)
            
            # Fallback to filename if transcript-based title failed
            if not title:
                title = filename or os.path.basename(original_audio_path)
                # Strip extension and clean up
                title = os.path.splitext(title)[0].replace("_", " ").replace("-", " ").strip()
            
            # Final fallback
            if not title:
                title = f"Upload {time.strftime('%Y-%m-%d %H:%M')}"
            
            logger.info(f"Auto-generated title from transcript: '{title}'")
        
        # Speaker fingerprinting: try to identify speaker from voice
        speaker_fingerprint_result = None
        if user_id and audio_exists:
            try:
                from speaker_fingerprint import process_audio_for_speaker
                speaker_fingerprint_result = process_audio_for_speaker(
                    db=db,
                    user_id=user_id,
                    audio_path=original_audio_path,
                    provided_speaker=speaker,
                    speech_id=None,  # Will update after speech is created
                    auto_register=True,
                    match_threshold=0.80,
                )
                
                if speaker_fingerprint_result.get("embedding_extracted"):
                    if speaker:
                        # Speaker was provided - we registered their voice
                        if speaker_fingerprint_result.get("is_new_speaker"):
                            logger.info(f"Registered new voice profile for speaker: '{speaker}'")
                        else:
                            logger.info(f"Updated voice profile for speaker: '{speaker}'")
                    else:
                        # Try to identify speaker from voice
                        identified = speaker_fingerprint_result.get("speaker_identified")
                        if identified:
                            similarity = speaker_fingerprint_result.get("similarity", 0)
                            speaker = identified
                            logger.info(f"Voice match: '{speaker}' ({similarity:.0%} confidence)")
                            # Add to result for API response
                            result["speaker_detected"] = {
                                "name": speaker,
                                "confidence": round(similarity, 2),
                                "method": "voice_fingerprint",
                            }
            except Exception as e:
                logger.warning(f"Speaker fingerprinting failed: {e}")
        
        # Fallback: Auto-set speaker from user account if still not provided
        if not speaker and user_id:
            try:
                from .user_auth import get_user_by_id
                user = get_user_by_id(user_id)
                if user and user.get("name"):
                    speaker = user["name"]
                    logger.info(f"Auto-set speaker from user account: '{speaker}'")
            except Exception as e:
                logger.warning(f"Failed to get user name for speaker: {e}")

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
                profile=profile,
            )

            # Store audio as Opus (if file still exists)
            if audio_exists:
                try:
                    db.store_audio(speech_id, original_audio_path)
                except Exception as e:
                    logger.warning(f"Failed to encode audio to Opus: {e}")
            else:
                logger.warning(f"Skipping audio storage - file no longer exists")

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
        elif audio_exists:
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
