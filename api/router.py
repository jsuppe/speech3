"""API route handlers for SpeechScore.

@arch component=router layer=api type=router
@arch depends=[pipeline_runner, job_manager, auth, user_auth]
@arch description="All API endpoints - analyze, speeches, projects, coach"

@arch.flow analyze_speech
@arch.step 1: receive_request "Receive audio upload request"
@arch.step 2: validate_upload "Validate file type and size"
@arch.step 3: queue_or_process "Queue async job or process sync"
@arch.step 4: run_pipeline "Execute analysis pipeline"
@arch.step 5: persist_results "Save to database"
@arch.step 6: return_response "Return results to client"
"""

import os
import sys
import time
import uuid
import json
import logging
import tempfile

from fastapi import APIRouter, UploadFile, File, Query, Request, Depends, Response, Body, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from . import config
from .models import (
    HealthResponse, ErrorResponse,
    JobStatusResponse, JobListResponse, JobSummary,
    QueueStatusResponse,
    CreateKeyRequest, CreateKeyResponse, KeyInfo, KeyListResponse,
    UpdateKeyRequest,
    UsageResponse, UsagePeriod, UsageAllTime,
    AdminUsageEntry, AdminUsageResponse,
)
from . import pipeline_runner
from .job_manager import manager as job_manager, QueueFullError
from .auth import auth_required, admin_required
from .user_auth import get_current_user, decode_token, flexible_auth
from .rate_limiter import rate_limiter, TIER_LIMITS
from .usage_limits import get_usage_limiter, TIER_LIMITS as USAGE_TIER_LIMITS
from .metrics import metrics
from .usage import usage_tracker

# Ensure speech3 root is on the path for scoring import
SPEECH3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SPEECH3_DIR not in sys.path:
    sys.path.insert(0, SPEECH3_DIR)

logger = logging.getLogger("speechscore.api")

router = APIRouter(prefix="/v1")

# Architecture trace helper
def arch_trace(component: str, action: str, request_id: str = "", **details):
    """Log an ARCH trace marker for sequence diagram generation"""
    detail_str = ' '.join(f'{k}={v}' for k, v in details.items())
    req_str = f'request_id={request_id}' if request_id else ''
    logger.info(f"ARCH:{component}.{action} {req_str} {detail_str}".strip())


def _error(status_code: int, code: str, message: str):
    """Return a structured JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}}
    )


# ---------------------------------------------------------------------------
# Health (NO auth required)
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health(request: Request):
    """Check API health, GPU availability, and Whisper model status. No authentication required."""
    import subprocess

    gpu_available = False
    gpu_name = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_available = True
            gpu_name = result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    uptime = time.time() - request.app.state.start_time

    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        whisper_model_loaded=pipeline_runner.is_model_loaded(),
        whisper_model="large-v3" if pipeline_runner.is_model_loaded() else None,
        uptime_sec=round(uptime, 1)
    )


@router.get("/metrics", tags=["Health"])
async def get_metrics(key_data: dict = Depends(auth_required)):
    """
    Get API metrics (latency, error rates, queue stats, pipeline performance).
    Requires admin API key.
    """
    if key_data.get("tier") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get metrics snapshot
    metrics_data = metrics.get_metrics()
    
    # Add queue stats
    queue_stats = job_manager.get_queue_stats()
    metrics_data["queue"].update({
        "current_depth": queue_stats["queue_depth"],
        "max_depth": queue_stats["max_queue_depth"],
        "estimated_wait_sec": queue_stats["estimated_wait_sec"],
    })
    
    return metrics_data


# ---------------------------------------------------------------------------
# Analyze (sync + async modes) — auth required
# ---------------------------------------------------------------------------

async def _validate_and_save_upload(audio: UploadFile) -> tuple:
    """
    Validate and save uploaded audio file to a temp location.
    Returns (tmp_path, ext, total_bytes) or raises an error response.
    """
    if not audio.filename:
        return None, None, _error(400, "MISSING_FILENAME", "No filename provided.")

    ext = audio.filename.rsplit(".", 1)[-1].lower() if "." in audio.filename else ""
    if ext not in config.ALLOWED_EXTENSIONS:
        return None, None, _error(
            400, "INVALID_FILE_TYPE",
            f"File type '.{ext}' not allowed. Accepted: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
        )

    suffix = f".{ext}"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=config.ASYNC_UPLOAD_DIR)
    os.close(fd)

    total_bytes = 0
    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > config.MAX_FILE_SIZE_BYTES:
                    os.remove(tmp_path)
                    return None, None, _error(
                        413, "FILE_TOO_LARGE",
                        f"File exceeds {config.MAX_FILE_SIZE_MB}MB limit."
                    )
                f.write(chunk)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if total_bytes == 0:
        os.remove(tmp_path)
        return None, None, _error(400, "EMPTY_FILE", "Uploaded file is empty.")

    return tmp_path, ext, total_bytes


def _resolve_modules(
    modules_param: Optional[str],
    preset_param: Optional[str],
) -> tuple:
    """
    Resolve modules/preset parameters into a set of module names.
    Returns (modules_set, preset_name, modules_requested_list, error_response_or_None).
    """
    # Preset takes precedence
    if preset_param:
        preset_lower = preset_param.strip().lower()
        if preset_lower not in config.MODULE_PRESETS:
            return None, None, None, _error(
                400, "INVALID_PRESET",
                f"Unknown preset '{preset_param}'. Valid presets: {', '.join(sorted(config.MODULE_PRESETS.keys()))}"
            )
        resolved = config.MODULE_PRESETS[preset_lower].copy()
        return resolved, preset_lower, sorted(resolved), None

    # Individual modules
    if modules_param:
        requested = [m.strip().lower() for m in modules_param.split(",") if m.strip()]
        # Check for "all"
        if "all" in requested:
            return config.ALL_MODULES.copy(), None, ["all"], None
        # Validate
        invalid = [m for m in requested if m not in config.VALID_MODULES]
        if invalid:
            return None, None, None, _error(
                400, "INVALID_MODULES",
                f"Unknown module(s): {', '.join(invalid)}. "
                f"Valid modules: {', '.join(sorted(config.VALID_MODULES))}"
            )
        return set(requested), None, sorted(requested), None

    # Default: all modules
    return config.ALL_MODULES.copy(), None, ["all"], None


@router.post("/analyze", tags=["Analysis"])
async def analyze(
    request: Request,
    audio: UploadFile = File(...),
    mode: Optional[str] = Query(None, description="Set to 'async' for async processing"),
    webhook_url: Optional[str] = Query(None, description="URL to POST results to when complete"),
    modules: Optional[str] = Query(None, description="Comma-separated module names (e.g. 'transcription,audio'). Use 'all' for everything."),
    preset: Optional[str] = Form(None, description="Named preset: quick, text_only, audio_only, full, clinical, coaching"),
    quality_gate: Optional[str] = Form(None, description="Quality gate mode: warn (default), block, skip"),
    language: Optional[str] = Form(None, description="Language code for Whisper (e.g. 'en', 'es'). Default: auto-detect."),
    speaker: Optional[str] = Form(None, description="Speaker name (used for diarization when single speaker detected)"),
    title: Optional[str] = Form(None, description="Title for the speech"),
    profile: Optional[str] = Form("general", description="Scoring profile: general, oratory, reading_fluency, presentation, clinical, dementia"),
    exercise_type: Optional[str] = Form(None, description="Exercise type for voice_health profile (e.g. 'sustained_ah', 'humming', 'projection')"),
    target_duration: Optional[float] = Form(None, description="Target duration in seconds for sustained exercises"),
    reference_text: Optional[str] = Form(None, description="Reference text for reading practice (stored for later comparison)"),
    auth: dict = Depends(flexible_auth),
):
    """
    Analyze an uploaded audio file through the speech pipeline.

    Default: synchronous (blocks until complete, returns full result).
    With mode=async: returns immediately with a job_id for polling.

    Use `modules` or `preset` to select which analysis modules run.
    If both are provided, `preset` takes precedence.
    
    For voice_health profile, pass exercise_type and target_duration for enhanced feedback.
    """
    key_data = auth.get("key_data")
    user_id = auth.get("user_id")
    # Use user_id for rate limiting JWT users, key_id for API keys
    key_id = key_data["key_id"] if key_data else f"user:{user_id}"
    
    # Generate request ID for tracing
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"Analyze request {request_id}: profile={profile}, user_id={user_id}")
    arch_trace("router", "receive_analyze", request_id, mode=mode or "sync")

    # Ensure upload dir exists
    os.makedirs(config.ASYNC_UPLOAD_DIR, exist_ok=True)

    # --- Resolve modules ---
    resolved_modules, resolved_preset, modules_requested, mod_error = _resolve_modules(modules, preset)
    if mod_error is not None:
        return mod_error

    # --- Validate quality_gate ---
    qg = config.DEFAULT_QUALITY_GATE
    if quality_gate is not None:
        qg = quality_gate.strip().lower()
        if qg not in config.VALID_QUALITY_GATES:
            return _error(
                400, "INVALID_QUALITY_GATE",
                f"Unknown quality_gate '{quality_gate}'. Valid: {', '.join(sorted(config.VALID_QUALITY_GATES))}"
            )

    # --- Validate language (basic sanity) ---
    lang = None
    if language is not None:
        lang = language.strip().lower()
        if not lang or len(lang) > 10:
            return _error(400, "INVALID_LANGUAGE", "Language code must be a short ISO code (e.g. 'en', 'es').")

    # Validate and save upload
    tmp_path, ext, total_bytes_or_error = await _validate_and_save_upload(audio)
    if tmp_path is None:
        return total_bytes_or_error

    total_bytes = total_bytes_or_error
    logger.info(
        f"Received file: {audio.filename} ({total_bytes / 1024:.0f} KB) "
        f"[mode={mode or 'sync'}, preset={resolved_preset}, "
        f"modules={modules_requested}, qg={qg}, lang={lang}, key={key_id}]"
    )

    # --- ASYNC MODE ---
    if mode == "async":
        try:
            job = job_manager.create_job(
                filename=audio.filename,
                upload_path=tmp_path,
                webhook_url=webhook_url,
                modules=resolved_modules,
                preset=resolved_preset,
                quality_gate=qg,
                language=lang,
                modules_requested=modules_requested,
                key_id=key_id,
            )
        except QueueFullError as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return _error(429, "QUEUE_FULL", str(e))

        return JSONResponse(
            status_code=202,
            content={
                "job_id": job.job_id,
                "status": job.status,
                "created_at": job.created_at,
            }
        )

    # --- SYNC MODE (default, backward compatible) ---
    tmp_wav = None
    try:
        request_start = time.time()
        arch_trace("router", "sync_start", request_id, file=audio.filename)

        # Try to get duration (may fail for browser-recorded WebM)
        duration = pipeline_runner.get_audio_duration(tmp_path)
        
        # Convert to WAV first - this validates the audio better than duration check
        # Browser WebM files often don't have duration in header but are valid
        try:
            tmp_wav = pipeline_runner.convert_to_wav(tmp_path)
        except RuntimeError as e:
            return _error(422, "CONVERSION_FAILED", f"Could not process audio file: {str(e)}")
        
        # If we couldn't get duration from original, try from converted WAV
        if duration <= 0:
            duration = pipeline_runner.get_audio_duration(tmp_wav)
        
        # Final check - if still no duration, the file is truly corrupt
        if duration <= 0:
            return _error(422, "CORRUPT_AUDIO", "Could not read audio file. It may be corrupt.")
        
        if duration > config.MAX_DURATION_SEC:
            return _error(
                422, "AUDIO_TOO_LONG",
                f"Audio duration {duration:.0f}s exceeds {config.MAX_DURATION_SEC}s limit."
            )

        # --- USAGE LIMIT CHECK ---
        audio_minutes = duration / 60.0
        if user_id:
            db = pipeline_runner.get_db()
            usage_limiter = get_usage_limiter(db.conn)
            
            # Check recording count limit
            can_record, limit_error = usage_limiter.check_can_record(user_id)
            if not can_record:
                if tmp_wav and os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "RECORDING_LIMIT_REACHED",
                            "message": limit_error["reason"],
                            "details": limit_error,
                        }
                    }
                )
            
            # Check audio minutes limit
            can_use_audio, audio_error = usage_limiter.check_audio_minutes(user_id, audio_minutes)
            if not can_use_audio:
                if tmp_wav and os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "AUDIO_LIMIT_REACHED",
                            "message": audio_error["reason"],
                            "details": audio_error,
                        }
                    }
                )

        # Record audio minutes in rate limiter and usage tracker
        rate_limiter.record_audio_minutes(key_id, audio_minutes)
        usage_tracker.record_audio_minutes(key_id, audio_minutes)

        # Run pipeline with module selection
        arch_trace("pipeline", "start", request_id, modules=str(len(resolved_modules)))
        try:
            result = pipeline_runner.run_pipeline(
                tmp_wav,
                modules=resolved_modules,
                language=lang,
                quality_gate=qg,
                preset=resolved_preset,
                modules_requested=modules_requested,
                known_speaker=speaker,
                voice_health_mode=(profile == "voice_health"),
                exercise_type=exercise_type,
                target_duration=target_duration,
            )
        except pipeline_runner.QualityGateError as e:
            return _error(422, "QUALITY_GATE_BLOCKED", str(e))
        except Exception as e:
            logger.exception("Pipeline error")
            return _error(500, "PIPELINE_ERROR", f"Analysis failed: {str(e)[:500]}")

        # Check for empty transcript (silent/no speech)
        transcript = result.get("transcript", "").strip()
        if not transcript:
            return _error(
                422, "NO_SPEECH_DETECTED",
                "No speech was detected in the recording. Please speak clearly and try again."
            )

        # Override filename
        result["audio_file"] = audio.filename

        processing_time = time.time() - request_start
        result["processing"]["processing_time_s"] = round(processing_time, 2)
        result["processing"]["timing"]["total_sec"] = round(processing_time, 2)

        # Persist to SQLite (non-blocking — errors logged but don't fail the request)
        speech_id = pipeline_runner.persist_result(
            original_audio_path=tmp_path,
            result=result,
            filename=audio.filename,
            preset=resolved_preset,
            user_id=user_id,
            speaker=speaker,
            title=title,
            profile=profile or "general",
            reference_text=reference_text,
        )
        if speech_id is not None:
            result["speech_id"] = speech_id
            
            # Record usage after successful save
            if user_id:
                try:
                    db = pipeline_runner.get_db()
                    usage_limiter = get_usage_limiter(db.conn)
                    usage_limiter.record_usage(user_id, audio_minutes)
                except Exception as e:
                    logger.warning(f"Failed to record usage for user {user_id}: {e}")

        logger.info(
            f"Completed: {audio.filename} | "
            f"duration={duration:.1f}s | "
            f"processing={processing_time:.1f}s | "
            f"modules={modules_requested} | "
            f"size={total_bytes / 1024:.0f}KB | "
            f"key={key_id} | "
            f"speech_id={speech_id}"
        )

        return result

    except Exception as e:
        logger.exception("Unexpected error in /v1/analyze")
        return _error(500, "INTERNAL_ERROR", f"Unexpected error: {str(e)[:300]}")
    finally:
        for path in [tmp_path, tmp_wav]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Streaming Analysis (Server-Sent Events)
# ---------------------------------------------------------------------------

from fastapi.responses import StreamingResponse
import asyncio
import queue
import threading

@router.post("/analyze/stream", tags=["Analysis"])
async def analyze_stream(
    request: Request,
    audio: UploadFile = File(...),
    modules: Optional[str] = Query(None, description="Comma-separated module names"),
    preset: Optional[str] = Form(None, description="Named preset"),
    quality_gate: Optional[str] = Form(None, description="Quality gate mode: warn, block, skip"),
    language: Optional[str] = Form(None, description="Language code for Whisper"),
    speaker: Optional[str] = Form(None, description="Speaker name (used for diarization)"),
    title: Optional[str] = Form(None, description="Title for the speech"),
    profile: Optional[str] = Form("general", description="Scoring profile: general, oratory, reading_fluency, presentation, clinical"),
    auth: dict = Depends(flexible_auth),
):
    """
    Analyze audio with real-time progress streaming via Server-Sent Events (SSE).
    
    Events emitted:
    - `progress`: {stage: "transcription" | "text_analysis" | etc, progress: 0-100}
    - `complete`: Full analysis result
    - `error`: Error message if analysis fails
    
    Connect with EventSource or fetch with streaming body.
    """
    key_data = auth.get("key_data")
    user_id = auth.get("user_id")
    # Use user_id for rate limiting JWT users, key_id for API keys
    key_id = key_data["key_id"] if key_data else f"user:{user_id}"

    os.makedirs(config.ASYNC_UPLOAD_DIR, exist_ok=True)

    # Resolve modules
    resolved_modules, resolved_preset, modules_requested, mod_error = _resolve_modules(modules, preset)
    if mod_error is not None:
        return mod_error

    # Validate quality_gate
    qg = config.DEFAULT_QUALITY_GATE
    if quality_gate is not None:
        qg = quality_gate.strip().lower()
        if qg not in config.VALID_QUALITY_GATES:
            return _error(400, "INVALID_QUALITY_GATE", f"Unknown quality_gate '{quality_gate}'")

    # Validate language
    lang = None
    if language is not None:
        lang = language.strip().lower()
        if not lang or len(lang) > 10:
            return _error(400, "INVALID_LANGUAGE", "Language code must be a short ISO code")

    # Save upload
    tmp_path, ext, total_bytes_or_error = await _validate_and_save_upload(audio)
    if tmp_path is None:
        return total_bytes_or_error

    total_bytes = total_bytes_or_error
    original_filename = audio.filename

    logger.info(f"SSE stream: {original_filename} ({total_bytes / 1024:.0f} KB) [preset={resolved_preset}]")

    # Module stages for progress tracking
    ALL_STAGES = [
        "audio_quality", "transcription", "text_analysis", "advanced_text_analysis",
        "rhetorical_analysis", "sentiment_analysis", "audio_analysis", "advanced_audio_analysis"
    ]
    
    async def event_stream():
        progress_queue = queue.Queue()
        result_holder = {"result": None, "error": None}
        tmp_wav = None

        def progress_callback(stage: str):
            """Called by pipeline_runner before each stage."""
            # Calculate progress percentage based on stage
            try:
                stage_idx = ALL_STAGES.index(stage)
                progress = int((stage_idx / len(ALL_STAGES)) * 100)
            except ValueError:
                progress = 50
            progress_queue.put(("progress", {"stage": stage, "progress": progress}))

        def run_pipeline():
            nonlocal tmp_wav
            try:
                # Try to get duration (may fail for browser-recorded WebM)
                duration = pipeline_runner.get_audio_duration(tmp_path)

                # Convert to WAV first - validates audio better than duration check
                # Browser WebM files often don't have duration in header but are valid
                progress_queue.put(("progress", {"stage": "converting", "progress": 5}))
                try:
                    tmp_wav = pipeline_runner.convert_to_wav(tmp_path)
                except RuntimeError as e:
                    result_holder["error"] = f"Could not process audio file: {str(e)}"
                    return
                
                # If we couldn't get duration from original, try from converted WAV
                if duration <= 0:
                    duration = pipeline_runner.get_audio_duration(tmp_wav)
                
                # Final check
                if duration <= 0:
                    result_holder["error"] = "Could not read audio file. It may be corrupt."
                    return
                if duration > config.MAX_DURATION_SEC:
                    result_holder["error"] = f"Audio duration {duration:.0f}s exceeds {config.MAX_DURATION_SEC}s limit."
                    return

                # Record usage
                audio_minutes = duration / 60.0
                rate_limiter.record_audio_minutes(key_id, audio_minutes)
                usage_tracker.record_audio_minutes(key_id, audio_minutes)

                # Run pipeline with progress callback
                result = pipeline_runner.run_pipeline(
                    tmp_wav,
                    progress_callback=progress_callback,
                    modules=resolved_modules,
                    language=lang,
                    quality_gate=qg,
                    preset=resolved_preset,
                    modules_requested=modules_requested,
                    known_speaker=speaker,
                )

                # Override filename
                result["audio_file"] = original_filename

                # Persist to SQLite
                speech_id = pipeline_runner.persist_result(
                    original_audio_path=tmp_path,
                    result=result,
                    filename=original_filename,
                    preset=resolved_preset,
                    user_id=user_id,
                    speaker=speaker,
                    title=title,
                    profile=profile or "general",
                )
                if speech_id is not None:
                    result["speech_id"] = speech_id

                result_holder["result"] = result

            except pipeline_runner.QualityGateError as e:
                result_holder["error"] = str(e)
            except Exception as e:
                logger.exception("Pipeline error in SSE stream")
                result_holder["error"] = f"Analysis failed: {str(e)[:500]}"
            finally:
                progress_queue.put(("done", None))

        # Start pipeline in background thread
        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()

        # Stream events
        try:
            while True:
                try:
                    event_type, data = progress_queue.get(timeout=0.5)
                except queue.Empty:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
                    continue

                if event_type == "done":
                    break
                elif event_type == "progress":
                    yield f"event: progress\ndata: {json.dumps(data)}\n\n"

            # Wait for thread to finish (long timeout for large files)
            thread.join(timeout=600)

            # Send final result or error
            if result_holder["error"]:
                yield f"event: error\ndata: {json.dumps({'error': result_holder['error']})}\n\n"
            elif result_holder["result"]:
                # Send 100% progress
                yield f"event: progress\ndata: {json.dumps({'stage': 'complete', 'progress': 100})}\n\n"
                yield f"event: complete\ndata: {json.dumps(result_holder['result'])}\n\n"

        finally:
            # Wait for thread to fully complete before cleanup
            if thread.is_alive():
                thread.join(timeout=60)
            
            # Cleanup temp files
            for path in [tmp_path, tmp_wav]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ---------------------------------------------------------------------------
# Job endpoints — auth required
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job(job_id: str, key_data: dict = Depends(auth_required)):
    """Get status and result of a specific async job. Returns full analysis result when status is 'complete'."""
    job = job_manager.get_job(job_id)
    if not job:
        return _error(404, "JOB_NOT_FOUND", f"Job '{job_id}' not found or expired.")

    # Get queue position if queued
    queue_position = None
    if job.status == "queued":
        queue_position = job_manager.get_queue_position(job_id)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        filename=job.filename,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=job.progress,
        queue_position=queue_position,
        result=job.result,
        error=job.error,
    )


@router.get("/jobs", response_model=JobListResponse, tags=["Jobs"])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status: queued|processing|complete|failed"),
    auth: dict = Depends(flexible_auth),
):
    """List recent async jobs (last 50). Optionally filter by status."""
    valid_statuses = {"queued", "processing", "complete", "failed"}
    if status and status not in valid_statuses:
        return _error(400, "INVALID_STATUS",
                      f"Invalid status filter. Must be one of: {', '.join(sorted(valid_statuses))}")

    jobs = job_manager.list_jobs(status_filter=status)
    summaries = [
        JobSummary(**j.to_summary())
        for j in jobs
    ]
    return JobListResponse(jobs=summaries, total=len(summaries))


# ---------------------------------------------------------------------------
# Queue status — auth required
# ---------------------------------------------------------------------------

@router.get("/queue", response_model=QueueStatusResponse, tags=["Jobs"])
async def queue_status(key_data: dict = Depends(auth_required)):
    """Get current queue depth, active job, and estimated wait time."""
    stats = job_manager.get_queue_stats()
    return QueueStatusResponse(**stats)


# ---------------------------------------------------------------------------
# Usage endpoint — auth required
# ---------------------------------------------------------------------------

@router.get("/usage", response_model=UsageResponse, tags=["Usage"])
async def get_usage(key_data: dict = Depends(auth_required)):
    """Get usage stats (requests, audio minutes) for your API key."""
    key_id = key_data["key_id"]
    tier = key_data.get("tier", "free")
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

    # Daily stats from rate limiter
    daily = rate_limiter.get_daily_stats(key_id)

    # All-time stats from usage tracker
    all_time = usage_tracker.get_usage(key_id)

    rpm = limits["requests_per_minute"]
    daily_audio = limits["audio_minutes_per_day"]

    return UsageResponse(
        key_id=key_id,
        key_name=key_data.get("name", ""),
        tier=tier,
        current_period=UsagePeriod(
            requests_today=daily["requests_today"],
            audio_minutes_today=daily["audio_minutes_today"],
            requests_limit_per_minute=int(rpm) if rpm != float("inf") else "unlimited",
            audio_minutes_limit_per_day=daily_audio if daily_audio != float("inf") else "unlimited",
        ),
        all_time=UsageAllTime(
            total_requests=all_time["total_requests"],
            total_audio_minutes=all_time["total_audio_minutes"],
        ),
    )


# ---------------------------------------------------------------------------
# Admin: Key Management
# ---------------------------------------------------------------------------

@router.post("/admin/keys", response_model=CreateKeyResponse, tags=["Admin"])
async def create_api_key(
    body: CreateKeyRequest,
    key_data: dict = Depends(admin_required),
):
    """Create a new API key. Returns the plaintext key ONCE — store it securely."""
    from .auth import create_key, VALID_TIERS

    if body.tier not in VALID_TIERS:
        return _error(400, "INVALID_TIER",
                      f"Invalid tier '{body.tier}'. Must be one of: {', '.join(sorted(VALID_TIERS))}")

    plaintext, entry = create_key(body.name, body.tier)

    return CreateKeyResponse(
        key_id=entry["key_id"],
        name=entry["name"],
        tier=entry["tier"],
        api_key=plaintext,
    )


@router.get("/admin/keys", response_model=KeyListResponse, tags=["Admin"])
async def list_api_keys(key_data: dict = Depends(admin_required)):
    """List all API keys with usage summaries. Key hashes are never exposed."""
    from .auth import get_all_keys

    keys = get_all_keys()
    result = []
    for k in keys:
        usage = usage_tracker.get_usage(k["key_id"])
        result.append(KeyInfo(
            key_id=k["key_id"],
            name=k["name"],
            tier=k["tier"],
            created_at=k["created_at"],
            enabled=k["enabled"],
            usage=usage,
        ))

    return KeyListResponse(keys=result, total=len(result))


@router.patch("/admin/keys/{key_id}", tags=["Admin"])
async def update_api_key(
    key_id: str,
    body: UpdateKeyRequest,
    key_data: dict = Depends(admin_required),
):
    """Update an API key — enable/disable, change tier, or rename."""
    from .auth import update_key

    updates = {}
    if body.name is not None:
        updates["name"] = body.name
    if body.tier is not None:
        updates["tier"] = body.tier
    if body.enabled is not None:
        updates["enabled"] = body.enabled

    if not updates:
        return _error(400, "NO_UPDATES", "No fields to update.")

    updated = update_key(key_id, **updates)
    if not updated:
        return _error(404, "KEY_NOT_FOUND", f"API key '{key_id}' not found.")

    return {"key_id": updated["key_id"], "name": updated["name"],
            "tier": updated["tier"], "enabled": updated["enabled"],
            "message": "Key updated."}


@router.delete("/admin/keys/{key_id}", tags=["Admin"])
async def delete_api_key(
    key_id: str,
    key_data: dict = Depends(admin_required),
):
    """Permanently delete an API key. This action cannot be undone."""
    from .auth import delete_key

    if delete_key(key_id):
        return {"message": f"Key '{key_id}' deleted."}
    return _error(404, "KEY_NOT_FOUND", f"API key '{key_id}' not found.")


# ---------------------------------------------------------------------------
# Database: Speech lookup & stats
# ---------------------------------------------------------------------------

@router.get("/speeches", tags=["Database"])
async def list_speeches(
    category: Optional[str] = Query(None),
    product: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    unassigned: bool = Query(False, description="Only return speeches not assigned to any project"),
    auth: dict = Depends(flexible_auth),
):
    """List speeches stored in the database. Users see only their own speeches; API keys see all."""
    db = pipeline_runner.get_db()
    user_id = auth.get("user_id")
    speeches = db.list_speeches(
        category=category, product=product, limit=limit, offset=offset, 
        user_id=user_id, unassigned=unassigned
    )
    total = db.count_speeches(user_id=user_id, unassigned=unassigned)
    return {"speeches": speeches, "total": total, "limit": limit, "offset": offset}


@router.get("/speeches/{speech_id}", tags=["Database"])
async def get_speech(speech_id: int, auth: dict = Depends(flexible_auth)):
    """Get a speech record with its latest analysis metrics."""
    db = pipeline_runner.get_db()
    speech = db.get_speech(speech_id)
    if not speech:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found.")

    # Users can only see their own speeches
    user_id = auth.get("user_id")
    if user_id and speech.get("user_id") and speech["user_id"] != user_id:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found.")

    analysis = db.get_analysis(speech_id)
    transcription = db.get_transcription(speech_id)
    audio = db.get_audio(speech_id)

    result = {
        "speech": speech,
        "has_audio": audio is not None,
        "audio_format": audio["format"] if audio else None,
        "audio_size_bytes": audio["size_bytes"] if audio else None,
    }
    if transcription:
        result["transcription"] = {
            "text": transcription["text"],
            "word_count": transcription["word_count"],
            "language": transcription["language"],
        }
    if analysis:
        # Return indexed metrics (not the full JSON blob — use /speeches/{id}/analysis for that)
        result["metrics"] = {
            k: analysis[k] for k in [
                "wpm", "pitch_mean_hz", "pitch_std_hz", "jitter_pct", "shimmer_pct",
                "hnr_db", "vocal_fry_ratio", "lexical_diversity", "syntactic_complexity",
                "fk_grade_level", "repetition_score", "sentiment", "quality_level",
            ] if analysis.get(k) is not None
        }

    return result


@router.get("/speeches/{speech_id}/full", tags=["Database"])
async def get_speech_full(
    speech_id: int,
    profile: str = Query("general", description="Scoring profile"),
    auth: dict = Depends(flexible_auth),
):
    """
    Get speech with all data in one call: speech metadata, transcription, analysis, and score.
    Reduces round-trips for mobile apps.
    """
    import json as json_module
    from scoring import score_speech, SCORING_PROFILES
    
    db = pipeline_runner.get_db()
    speech = db.get_speech(speech_id)
    if not speech:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found.")

    # Users can only see their own speeches
    user_id = auth.get("user_id")
    if user_id and speech.get("user_id") and speech["user_id"] != user_id:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found.")

    analysis = db.get_analysis(speech_id)
    transcription = db.get_transcription(speech_id)
    audio = db.get_audio(speech_id)
    
    # Auto-run pronunciation analysis if profile is pronunciation and data is missing
    if profile == "pronunciation" and speech.get("pronunciation_mean_confidence") is None:
        if audio:
            try:
                from pronunciation import PronunciationAnalyzer
                import tempfile
                import os
                import numpy as np
                
                # Write audio to temp file
                with tempfile.NamedTemporaryFile(suffix=f".{audio['format']}", delete=False) as f:
                    f.write(audio["data"])
                    temp_path = f.name
                
                try:
                    analyzer = PronunciationAnalyzer()
                    result = analyzer.analyze(temp_path, transcription["text"] if transcription else None)
                    
                    # Extract stats from PronunciationAnalysis object
                    word_scores = [w.pronunciation_score for w in result.words]
                    mean_conf = float(np.mean(word_scores)) if word_scores else 0.0
                    std_conf = float(np.std(word_scores)) if word_scores else 0.0
                    problem_ratio = len(result.problem_words) / len(result.words) if result.words else 0.0
                    words_analyzed = len(result.words)
                    
                    # Store results in speeches table
                    db.conn.execute("""
                        UPDATE speeches SET
                            pronunciation_mean_confidence = ?,
                            pronunciation_std_confidence = ?,
                            pronunciation_problem_ratio = ?,
                            pronunciation_words_analyzed = ?,
                            pronunciation_analyzed_at = datetime('now')
                        WHERE id = ?
                    """, (
                        mean_conf,
                        std_conf,
                        problem_ratio,
                        words_analyzed,
                        speech_id,
                    ))
                    db.conn.commit()
                    
                    # Update speech dict with new data
                    speech["pronunciation_mean_confidence"] = mean_conf
                    speech["pronunciation_std_confidence"] = std_conf
                    speech["pronunciation_problem_ratio"] = problem_ratio
                    speech["pronunciation_words_analyzed"] = words_analyzed
                finally:
                    os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Auto pronunciation analysis failed for speech {speech_id}: {e}")
    
    # Get or compute score
    score = None
    cached_json = speech.get("cached_score_json")
    cached_profile = speech.get("cached_score_profile")
    
    if cached_json and cached_profile == profile:
        try:
            score = json_module.loads(cached_json)
            score["cached"] = True
        except:
            pass
    
    if not score and profile in SCORING_PROFILES:
        score = score_speech(db, speech_id, profile)
        if score:
            score["cached"] = False
            # Cache it
            try:
                db.cache_speech_score_full(
                    speech_id,
                    json_module.dumps(score),
                    score.get("composite_score"),
                    profile
                )
            except:
                pass

    result = {
        "speech": speech,
        "has_audio": audio is not None,
        "audio_format": audio["format"] if audio else None,
        "audio_size_bytes": audio["size_bytes"] if audio else None,
    }
    
    if transcription:
        result["transcription"] = {
            "text": transcription["text"],
            "word_count": transcription["word_count"],
            "language": transcription["language"],
        }
    
    if analysis:
        result["analysis"] = {
            k: analysis[k] for k in [
                "wpm", "pitch_mean_hz", "pitch_std_hz", "jitter_pct", "shimmer_pct",
                "hnr_db", "vocal_fry_ratio", "lexical_diversity", "syntactic_complexity",
                "fk_grade_level", "repetition_score", "sentiment", "quality_level",
            ] if analysis.get(k) is not None
        }
    
    if score:
        result["score"] = score
    
    # Add grammar analysis (run on demand if not stored)
    grammar_json = speech.get("grammar_analysis_json")
    if grammar_json:
        try:
            result["grammar_analysis"] = json_module.loads(grammar_json)
        except:
            pass
    elif transcription and transcription.get("text"):
        try:
            from grammar_analysis import analyze_grammar
            grammar_result = analyze_grammar(transcription["text"])
            result["grammar_analysis"] = grammar_result
            # Store for future
            try:
                db.conn.execute(
                    "UPDATE speeches SET grammar_analysis_json = ? WHERE id = ?",
                    (json_module.dumps(grammar_result), speech_id)
                )
                db.conn.commit()
            except:
                pass
        except Exception as e:
            logger.warning(f"Grammar analysis failed: {e}")
    
    # Add presentation analysis for presentation profile
    if profile == "presentation" and transcription and transcription.get("text"):
        try:
            from presentation_analysis import analyze_presentation
            
            # Get timing data if available
            wpm_segments = None
            pause_data = None
            pitch_data = None
            
            if audio_result:
                pitch_data = {
                    "pitch_mean_hz": audio_result.get("pitch_mean_hz"),
                    "pitch_std_hz": audio_result.get("pitch_std_hz"),
                    "pitch_range_hz": audio_result.get("pitch_range_hz"),
                }
                pause_data = {
                    "pause_ratio": audio_result.get("pause_ratio"),
                    "pause_count": audio_result.get("pause_count", 0),
                    "avg_pause_duration": audio_result.get("avg_pause_duration"),
                }
            
            # Get WPM segments if available
            if result.get("wpm_segments"):
                wpm_segments = result["wpm_segments"]
            
            presentation_result = analyze_presentation(
                transcript=transcription["text"],
                wpm_segments=wpm_segments,
                pitch_data=pitch_data,
                pause_data=pause_data,
            )
            result["presentation_analysis"] = presentation_result
        except Exception as e:
            logger.warning(f"Presentation analysis failed: {e}")
    
    # Add AI summary for the current profile (if available)
    try:
        from ai_summary import get_summary
        summary = get_summary(speech_id, profile, config.DB_PATH)
        if summary:
            result["ai_summary"] = summary
    except Exception as e:
        logger.warning(f"Failed to get AI summary: {e}")
    
    return result


@router.delete("/speeches/{speech_id}", tags=["Database"])
async def delete_speech(
    speech_id: int,
    auth: dict = Depends(flexible_auth),
):
    """
    Delete a speech and all associated data (audio, transcription, analysis).
    Users can only delete their own speeches.
    """
    db = pipeline_runner.get_db()
    
    # Get user_id from auth (None for API key auth = admin)
    user_id = auth.get("user_id")
    
    # Delete the speech (delete_speech handles ownership check)
    deleted = db.delete_speech(
        speech_id=speech_id,
        user_id=user_id,  # Pass None for API key auth (allows admin deletes)
    )
    
    if not deleted:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found or access denied.")
    
    return {
        "message": "Speech deleted successfully.",
        "speech_id": speech_id,
    }


@router.patch("/speeches/{speech_id}", tags=["Database"])
async def update_speech(
    speech_id: int,
    title: Optional[str] = Body(None, description="New title for the speech"),
    speaker: Optional[str] = Body(None, description="Speaker name"),
    category: Optional[str] = Body(None, description="Category (e.g. presentation, casual)"),
    tags: Optional[List[str]] = Body(None, description="Tags for the speech"),
    notes: Optional[str] = Body(None, description="Notes about the speech"),
    profile: Optional[str] = Body(None, description="Scoring profile (general, oratory, reading_fluency, presentation, clinical, dementia)"),
    project_id: Optional[int] = Body(None, description="Project ID to assign speech to (null to unassign)"),
    auth: dict = Depends(flexible_auth),
):
    """
    Update speech metadata (title, speaker, category, tags, notes, profile, project_id).
    Users can only update their own speeches.
    """
    db = pipeline_runner.get_db()
    
    # Get user_id from auth (None for API key auth = admin)
    user_id = auth.get("user_id")
    
    # Update the speech (update_speech handles ownership check)
    updated = db.update_speech(
        speech_id=speech_id,
        title=title,
        speaker=speaker,
        category=category,
        tags=tags,
        notes=notes,
        profile=profile,
        project_id=project_id,
        user_id=user_id,  # Pass None for API key auth (allows admin updates)
    )
    
    if not updated:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found or access denied.")
    
    return {
        "message": "Speech updated successfully.",
        "speech": updated,
    }


@router.get("/speeches/{speech_id}/analysis", tags=["Database"])
async def get_speech_analysis(speech_id: int, auth: dict = Depends(flexible_auth)):
    """Get the full analysis JSON for a speech."""
    db = pipeline_runner.get_db()
    result = db.get_full_analysis(speech_id)
    if not result:
        return _error(404, "NOT_FOUND", f"No analysis found for speech {speech_id}.")
    return result


@router.get("/speeches/{speech_id}/emotional-arc", tags=["Database"])
async def get_emotional_arc(speech_id: int, auth: dict = Depends(flexible_auth)):
    """
    Get normalized emotional arc data for real-time playback visualization.
    
    Returns intensity points normalized to the speech's own emotional range,
    with phase detection (Setup, Build, Peak, Resolution).
    """
    import json as json_module
    from sentiment_analysis import compute_normalized_arc
    
    db = pipeline_runner.get_db()
    
    # Get the speech and check it exists
    speech = db.get_speech(speech_id)
    if not speech:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found.")
    
    # Get transcript segments from transcriptions table
    cursor = db.conn.execute("""
        SELECT segments_json FROM transcriptions WHERE speech_id = ?
    """, (speech_id,))
    row = cursor.fetchone()
    
    if not row or not row[0]:
        return _error(400, "NO_TRANSCRIPT", "No transcript segments available for this speech.")
    
    try:
        segments = json_module.loads(row[0])
    except:
        return _error(500, "PARSE_ERROR", "Failed to parse transcript segments.")
    
    if not segments or len(segments) < 2:
        return _error(400, "INSUFFICIENT_DATA", "Need at least 2 segments for arc analysis.")
    
    # Check if we have cached sentiment per-segment data
    cursor = db.conn.execute("""
        SELECT analysis_json FROM analyses WHERE speech_id = ?
    """, (speech_id,))
    analysis_row = cursor.fetchone()
    
    per_segment = None
    if analysis_row and analysis_row[0]:
        try:
            analysis = json_module.loads(analysis_row[0])
            sentiment = analysis.get("sentiment", {})
            per_segment = sentiment.get("per_segment", [])
        except:
            pass
    
    # If no cached sentiment, compute it now
    if not per_segment:
        from sentiment_analysis import analyze_sentiment
        
        # Build text from segments
        full_text = " ".join(s.get("text", "") for s in segments)
        result = analyze_sentiment(full_text, segments)
        per_segment = result.get("per_segment", [])
    
    if not per_segment or len(per_segment) < 2:
        return _error(400, "INSUFFICIENT_DATA", "Not enough sentiment data for arc analysis.")
    
    # Compute normalized arc
    arc_data = compute_normalized_arc(per_segment)
    
    return {
        "speech_id": speech_id,
        "title": speech.get("title", "Untitled"),
        "duration_sec": speech.get("duration_sec", 0),
        **arc_data,
    }


@router.get("/speeches/{speech_id}/group", tags=["Database"])
async def get_speech_group_analysis(speech_id: int, auth: dict = Depends(flexible_auth)):
    """Get group/meeting analysis for a speech with profile='group'.
    
    Returns:
    - speakers: List of unique speakers detected
    - speaker_summaries: Summary of what each speaker said (word count, text preview)
    - diarized_transcript: Full transcript with speaker labels and timestamps
    - stats: Speaking time per speaker, turn count
    """
    db = pipeline_runner.get_db()
    speech = db.get_speech(speech_id)
    if not speech:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found.")
    
    # Get full analysis
    result = db.get_full_analysis(speech_id)
    if not result:
        return _error(404, "NOT_FOUND", f"No analysis found for speech {speech_id}.")
    
    # Use diarized segments if available (they have speaker labels)
    diarization = result.get("diarization", {})
    if isinstance(diarization, dict) and diarization.get("segments"):
        segments = diarization.get("segments", [])
    else:
        segments = result.get("segments", [])
    
    # Collect speaker data
    speaker_data = {}
    diarized_transcript = []
    
    for seg in segments:
        speaker = seg.get("speaker") or "Speaker 1"
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        duration = end - start
        
        if speaker not in speaker_data:
            speaker_data[speaker] = {
                "name": speaker,
                "word_count": 0,
                "speaking_time_sec": 0,
                "turn_count": 0,
                "segments": []
            }
        
        speaker_data[speaker]["word_count"] += len(text.split())
        speaker_data[speaker]["speaking_time_sec"] += duration
        speaker_data[speaker]["turn_count"] += 1
        speaker_data[speaker]["segments"].append(text)
        
        diarized_transcript.append({
            "speaker": speaker,
            "text": text,
            "start": start,
            "end": end,
        })
    
    # Build speaker summaries
    speaker_summaries = []
    for speaker, data in sorted(speaker_data.items()):
        full_text = " ".join(data["segments"])
        speaker_summaries.append({
            "name": speaker,
            "word_count": data["word_count"],
            "speaking_time_sec": round(data["speaking_time_sec"], 1),
            "turn_count": data["turn_count"],
            "text_preview": full_text[:300] + "..." if len(full_text) > 300 else full_text,
            "full_text": full_text,
        })
    
    speakers = list(speaker_data.keys())
    total_duration = speech.get("duration_sec", 0)
    
    return {
        "speech_id": speech_id,
        "title": speech.get("title", ""),
        "speakers": speakers,
        "speaker_count": len(speakers),
        "total_duration_sec": total_duration,
        "speaker_summaries": speaker_summaries,
        "diarized_transcript": diarized_transcript,
    }


@router.get("/speeches/{speech_id}/audio", tags=["Database"])
async def get_speech_audio(
    speech_id: int,
    download: bool = Query(False, description="Force download instead of inline playback"),
    auth: dict = Depends(flexible_auth)
):
    """Stream or download the stored audio for a speech."""
    db = pipeline_runner.get_db()
    audio = db.get_audio(speech_id)
    if not audio:
        return _error(404, "NOT_FOUND", f"No audio stored for speech {speech_id}.")

    fmt = audio["format"]
    mime = {"opus": "audio/opus", "mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg"}.get(fmt, "application/octet-stream")
    filename = audio.get("original_filename", f"speech_{speech_id}.{fmt}")
    if not filename.endswith(f".{fmt}"):
        filename = f"{os.path.splitext(filename)[0]}.{fmt}"

    disposition = "attachment" if download else "inline"
    
    return Response(
        content=audio["data"],
        media_type=mime,
        headers={
            "Content-Disposition": f'{disposition}; filename="{filename}"',
            "Accept-Ranges": "bytes",
        },
    )


@router.get("/speeches/{speech_id}/spectrogram", tags=["Database"])
async def get_speech_spectrogram(
    speech_id: int,
    type: str = Query("combined", description="Spectrogram type: combined, spectrogram, mel, loudness"),
    auth: dict = Depends(flexible_auth),
):
    """Download stored spectrogram image for a speech."""
    db = pipeline_runner.get_db()
    spec = db.get_spectrogram(speech_id, spec_type=type)
    if not spec:
        return _error(404, "NOT_FOUND", f"No {type} spectrogram for speech {speech_id}.")

    fmt = spec["format"]
    mime = {"png": "image/png", "jpg": "image/jpeg"}.get(fmt, "application/octet-stream")

    return Response(
        content=spec["data"],
        media_type=mime,
        headers={"Content-Disposition": f'inline; filename="speech_{speech_id}_{type}.{fmt}"'},
    )


@router.get("/db/stats", tags=["Database"])
async def db_stats(auth: dict = Depends(flexible_auth)):
    """Get database statistics: speech count, audio storage, product distribution."""
    db = pipeline_runner.get_db()
    return db.stats()


# ---------------------------------------------------------------------------
# Admin: Usage Dashboard
# ---------------------------------------------------------------------------

@router.get("/admin/usage", response_model=AdminUsageResponse, tags=["Admin"])
async def admin_usage(key_data: dict = Depends(admin_required)):
    """Dashboard view — usage summaries for all API keys."""
    from .auth import get_all_keys

    keys = get_all_keys()
    all_usage = usage_tracker.get_all_usage()

    entries = []
    for k in keys:
        kid = k["key_id"]
        u = all_usage.get(kid, {})
        entries.append(AdminUsageEntry(
            key_id=kid,
            key_name=k["name"],
            tier=k["tier"],
            enabled=k["enabled"],
            total_requests=u.get("total_requests", 0),
            total_audio_minutes=u.get("total_audio_minutes", 0.0),
            last_used=u.get("last_used"),
        ))

    return AdminUsageResponse(keys=entries, total=len(entries))


# ---------------------------------------------------------------------------
# Scoring & Grading (Task #38)
# ---------------------------------------------------------------------------

@router.get("/speeches/{speech_id}/score", tags=["Scoring"])
async def get_speech_score(
    speech_id: int,
    profile: str = Query("general", description="Scoring profile: general, oratory, reading_fluency, presentation, clinical"),
    refresh: bool = Query(False, description="Force refresh the cached score"),
    auth: dict = Depends(flexible_auth),
):
    """
    Get a scored assessment of a speech with letter grades and numeric scores.

    Scoring is percentile-based: each metric is compared against the full database distribution.
    Different profiles weight metrics differently for various use cases.
    Results are cached for fast subsequent loads.
    """
    import json as json_module
    from scoring import score_speech, SCORING_PROFILES

    if profile not in SCORING_PROFILES:
        return _error(400, "INVALID_PROFILE",
                      f"Unknown profile '{profile}'. Valid profiles: {', '.join(sorted(SCORING_PROFILES.keys()))}")

    db = pipeline_runner.get_db()
    speech = db.get_speech(speech_id)
    if not speech:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found.")

    # Check for cached score
    cached_json = speech.get("cached_score_json")
    cached_profile = speech.get("cached_score_profile")
    
    if not refresh and cached_json and cached_profile == profile:
        try:
            cached_result = json_module.loads(cached_json)
            cached_result["cached"] = True
            return cached_result
        except:
            pass  # Fall through to regenerate
    
    # Generate fresh score
    result = score_speech(db, speech_id, profile)
    if not result:
        return _error(404, "NO_ANALYSIS", f"No analysis found for speech {speech_id}.")

    # Cache the result
    try:
        db.cache_speech_score_full(
            speech_id,
            json_module.dumps(result),
            result.get("composite_score"),
            profile
        )
    except Exception as e:
        pass  # Don't fail the request if caching fails
    
    result["cached"] = False
    return result


@router.get("/scoring/profiles", tags=["Scoring"])
async def list_scoring_profiles(key_data: dict = Depends(auth_required)):
    """List all available scoring profiles with their descriptions and metrics."""
    from scoring import get_available_profiles
    return {"profiles": get_available_profiles()}


@router.get("/progress", tags=["Scoring"])
async def get_progress_data(
    limit: int = Query(50, ge=1, le=200),
    auth: dict = Depends(flexible_auth),
):
    """
    Get progress data for all user speeches with scores.
    Returns speeches with their composite scores for trend visualization.
    """
    from scoring import score_speech

    db = pipeline_runner.get_db()
    user_id = auth.get("user_id")
    speeches = db.list_speeches(limit=limit, offset=0, user_id=user_id)

    progress_data = []
    for speech in speeches:
        speech_id = speech["id"]
        profile = speech.get("profile", "general")

        # Get score for this speech
        try:
            score_result = score_speech(db, speech_id, profile)
            composite_score = score_result.get("composite_score") if score_result else None
        except Exception:
            composite_score = None

        progress_data.append({
            "id": speech_id,
            "title": speech.get("title"),
            "created_at": speech.get("created_at"),
            "profile": profile,
            "composite_score": composite_score,
        })

    return {"progress": progress_data, "total": len(progress_data)}


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

def _count_unidentified_speakers(db, project_id: int) -> int:
    """Count unique unidentified speakers across all speeches in a project."""
    # Get all speeches in this project
    speeches = db.conn.execute(
        "SELECT id FROM speeches WHERE project_id = ?", (project_id,)
    ).fetchall()
    
    unidentified = set()
    for (speech_id,) in speeches:
        # Get diarization from analysis
        result = db.get_full_analysis(speech_id)
        if not result:
            continue
        
        diarization = result.get("diarization", {})
        if isinstance(diarization, dict):
            segments = diarization.get("segments", [])
        else:
            segments = []
        
        for seg in segments:
            speaker = seg.get("speaker", "")
            # Check if speaker is unidentified (generic names)
            if speaker and not seg.get("speaker_matched"):
                # Unidentified speakers have generic names like "Speaker 1", "First Speaker", etc.
                if (speaker.startswith("Speaker ") or 
                    speaker in ("First Speaker", "Second Speaker", "Third Speaker", 
                               "Fourth Speaker", "Fifth Speaker", "Sixth Speaker",
                               "Seventh Speaker", "Eighth Speaker", "Ninth Speaker", 
                               "Tenth Speaker", "Unknown") or
                    speaker.startswith("Speaker 1")):
                    unidentified.add(speaker)
    
    return len(unidentified)


@router.get("/projects", tags=["Projects"])
async def list_projects(auth: dict = Depends(flexible_auth)):
    """List all projects for the current user."""
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required for projects.")
    
    db = pipeline_runner.get_db()
    projects = db.list_projects(user_id)
    
    # Add speech count to each project
    # NOTE: unidentified_speakers count disabled for now - the old logic counted
    # raw diarization labels which was incorrect. Will re-enable with proper
    # speaker fingerprint matching.
    for p in projects:
        p["speech_count"] = db.count_project_speeches(p["id"])
        p["unidentified_speakers"] = 0  # TODO: implement proper speaker fingerprint matching
    
    return {"projects": projects}


@router.post("/projects", tags=["Projects"])
async def create_project(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    profile: str = Form("general"),
    type: str = Form("standard"),
    reminder_enabled: bool = Form(False),
    reminder_frequency: str = Form("daily"),
    reminder_days: Optional[str] = Form(None),  # JSON string of int array
    reminder_time: str = Form("09:00"),
    auth: dict = Depends(flexible_auth),
):
    """Create a new project. Type can be 'standard' or 'group_meeting'."""
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required for projects.")
    
    project_type = type if type in ("standard", "group_meeting") else "standard"
    reminder_freq = reminder_frequency if reminder_frequency in ("daily", "weekly", "monthly") else "daily"
    
    # Parse reminder_days from JSON string
    parsed_days = []
    if reminder_days:
        try:
            parsed_days = json.loads(reminder_days)
            if not isinstance(parsed_days, list):
                parsed_days = []
        except:
            parsed_days = []
    
    db = pipeline_runner.get_db()
    project_id = db.create_project(
        user_id, name, description, profile, project_type,
        reminder_enabled=reminder_enabled,
        reminder_frequency=reminder_freq,
        reminder_days=parsed_days,
        reminder_time=reminder_time,
    )
    project = db.get_project(project_id)
    
    return {"project": project}


@router.get("/projects/{project_id}", tags=["Projects"])
async def get_project(project_id: int, auth: dict = Depends(flexible_auth)):
    """Get a project by ID."""
    user_id = auth.get("user_id")
    db = pipeline_runner.get_db()
    project = db.get_project(project_id)
    
    if not project:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    # Check ownership
    if user_id and project.get("user_id") != user_id:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    project["speech_count"] = db.count_project_speeches(project_id)
    project["speeches"] = db.get_project_speeches(project_id)
    
    return {"project": project}


@router.put("/projects/{project_id}", tags=["Projects"])
@router.patch("/projects/{project_id}", tags=["Projects"])
async def update_project(
    project_id: int,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    profile: Optional[str] = Form(None),
    type: Optional[str] = Form(None),
    reminder_enabled: Optional[bool] = Form(None),
    reminder_frequency: Optional[str] = Form(None),
    reminder_days: Optional[str] = Form(None),  # JSON string of int array
    reminder_time: Optional[str] = Form(None),
    auth: dict = Depends(flexible_auth),
):
    """Update a project."""
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required.")
    
    db = pipeline_runner.get_db()
    project = db.get_project(project_id)
    
    if not project or project.get("user_id") != user_id:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    # Validate type if provided
    if type is not None and type not in ("standard", "group_meeting"):
        type = None  # Ignore invalid type
    
    # Validate reminder_frequency if provided
    if reminder_frequency is not None and reminder_frequency not in ("daily", "weekly", "monthly"):
        reminder_frequency = None
    
    # Parse reminder_days if provided as JSON string
    parsed_days = None
    if reminder_days is not None:
        try:
            import json
            parsed_days = json.loads(reminder_days)
        except:
            parsed_days = None
    
    db.update_project(
        project_id, name=name, description=description, profile=profile, type=type,
        reminder_enabled=reminder_enabled, reminder_frequency=reminder_frequency,
        reminder_days=parsed_days, reminder_time=reminder_time,
    )
    return {"project": db.get_project(project_id)}


@router.delete("/projects/{project_id}", tags=["Projects"])
async def delete_project(
    project_id: int,
    delete_speeches: bool = Query(True, description="Also delete all speeches in the project"),
    auth: dict = Depends(flexible_auth),
):
    """Delete a project and optionally all its speeches."""
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required.")
    
    db = pipeline_runner.get_db()
    project = db.get_project(project_id)
    
    if not project or project.get("user_id") != user_id:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    deleted_count = db.delete_project(project_id, delete_speeches=delete_speeches)
    return {"deleted": True, "speeches_deleted": deleted_count}


@router.post("/speeches/{speech_id}/project", tags=["Projects"])
async def assign_speech_to_project(
    speech_id: int,
    project_id: int = Form(...),
    auth: dict = Depends(flexible_auth),
):
    """Assign a speech to a project."""
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required.")
    
    db = pipeline_runner.get_db()
    
    # Verify speech ownership
    speech = db.get_speech(speech_id)
    if not speech or speech.get("user_id") != user_id:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found.")
    
    # Verify project ownership
    project = db.get_project(project_id)
    if not project or project.get("user_id") != user_id:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    db.assign_speech_to_project(speech_id, project_id)
    return {"assigned": True, "speech_id": speech_id, "project_id": project_id}


@router.get("/projects/{project_id}/speeches", tags=["Projects"])
async def get_project_speeches(
    project_id: int,
    auth: dict = Depends(flexible_auth),
):
    """Get all speeches in a project."""
    user_id = auth.get("user_id")
    db = pipeline_runner.get_db()
    project = db.get_project(project_id)
    
    if not project:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    if user_id and project.get("user_id") != user_id:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    speeches = db.get_project_speeches(project_id)
    return {"speeches": speeches}


@router.get("/projects/{project_id}/meeting", tags=["Projects"])
async def get_project_meeting_data(
    project_id: int,
    auth: dict = Depends(flexible_auth),
):
    """Get meeting data for a group_meeting project.
    Returns participants, chronological transcripts, and meeting stats.
    """
    user_id = auth.get("user_id")
    db = pipeline_runner.get_db()
    project = db.get_project(project_id)
    
    if not project:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    if user_id and project.get("user_id") != user_id:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    meeting_data = db.get_meeting_data(project_id)
    return {"meeting": meeting_data}


@router.get("/projects/{project_id}/participant/{participant_name}/summary", tags=["Projects"])
async def get_participant_summary(
    project_id: int,
    participant_name: str,
    force_refresh: bool = Query(False, description="Force regenerate the summary"),
    auth: dict = Depends(flexible_auth),
):
    """Get an LLM-generated summary of a participant's contributions in a meeting.
    
    Returns:
    - key_points: Main points this person raised
    - todos: To-do items they mentioned  
    - opinions: Opinions they expressed
    - action_items: Action items assigned to/by them
    
    Summary is cached and auto-regenerates when new recordings are added.
    """
    import httpx
    
    user_id = auth.get("user_id")
    db = pipeline_runner.get_db()
    project = db.get_project(project_id)
    
    if not project:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    if user_id and project.get("user_id") != user_id:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    if project.get("type") != "group_meeting":
        return _error(400, "BAD_REQUEST", "Participant summaries only available for group meeting projects.")
    
    # Get meeting data
    meeting_data = db.get_meeting_data(project_id)
    
    # Find the participant
    participant_summary = None
    for ps in meeting_data.get("participant_summaries", []):
        if ps.get("name") == participant_name:
            participant_summary = ps
            break
    
    if not participant_summary:
        return _error(404, "NOT_FOUND", f"Participant '{participant_name}' not found in this meeting.")
    
    # Check cache (stored in DB or in-memory)
    cache_key = f"participant_summary_{project_id}_{participant_name}"
    cached = db.get_cache(cache_key)
    
    # Check if cache is valid (meeting hasn't been updated since)
    if cached and not force_refresh:
        cached_at = cached.get("generated_at")
        meeting_updated = project.get("updated_at") or project.get("created_at")
        if cached_at and cached_at >= meeting_updated:
            return {"summary": cached, "cached": True}
    
    # Build full meeting transcript for context
    full_transcript = []
    for t in meeting_data.get("transcripts", []):
        full_transcript.append(f"{t['speaker']}: {t['transcript']}")
    full_meeting_text = "\n".join(full_transcript)
    
    # Participant's specific contributions
    participant_text = participant_summary.get("full_text", "")
    
    # Generate summary using Ollama
    prompt = f"""Analyze this meeting participant's contributions. The meeting has {meeting_data.get('participant_count', 2)} participants.

FULL MEETING TRANSCRIPT (for context):
{full_meeting_text[:4000]}

PARTICIPANT "{participant_name}" SAID:
{participant_text[:2000]}

Generate a structured summary of what "{participant_name}" contributed to this meeting. Return JSON with these fields:
- key_points: array of 2-5 main points they raised
- todos: array of any to-do items they mentioned (empty if none)
- opinions: array of opinions they expressed on topics discussed
- action_items: array of action items assigned to or by them (empty if none)
- speaking_style: one sentence about their communication style

Return ONLY valid JSON, no other text."""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
                timeout=60.0
            )
            result = response.json()
            llm_response = result.get("response", "{}")
            
            # Parse the JSON response
            import json as json_module
            try:
                summary = json_module.loads(llm_response)
            except json_module.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    summary = json_module.loads(json_match.group())
                else:
                    summary = {
                        "key_points": ["Unable to parse summary"],
                        "todos": [],
                        "opinions": [],
                        "action_items": [],
                        "speaking_style": "Analysis unavailable"
                    }
            
            # Add metadata
            from datetime import datetime
            summary["participant_name"] = participant_name
            summary["word_count"] = participant_summary.get("word_count", 0)
            summary["recording_count"] = participant_summary.get("recording_count", 0)
            summary["generated_at"] = datetime.now().isoformat()
            
            # Cache the result
            db.set_cache(cache_key, summary)
            
            return {"summary": summary, "cached": False}
            
    except Exception as e:
        logger.error(f"LLM summary generation failed: {e}")
        return _error(500, "LLM_ERROR", f"Failed to generate summary: {str(e)}")


@router.get("/projects/{project_id}/progress", tags=["Projects"])
async def get_project_progress(
    project_id: int,
    auth: dict = Depends(flexible_auth),
):
    """Get progress data for a specific project.
    
    Uses cached scores when available for performance.
    Only recalculates if cache miss or profile changed.
    """
    from scoring import score_speech
    
    user_id = auth.get("user_id")
    db = pipeline_runner.get_db()
    project = db.get_project(project_id)
    
    if not project:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    if user_id and project.get("user_id") != user_id:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found.")
    
    speeches = db.get_project_speeches(project_id)
    profile = project.get("profile", "general")
    
    progress_data = []
    for speech in speeches:
        speech_id = speech["id"]
        cached_score = speech.get("cached_score")
        cached_profile = speech.get("cached_score_profile")
        
        # Use cached score if available and profile matches
        if cached_score is not None and cached_profile == profile:
            composite_score = cached_score
        else:
            # Calculate and cache the score
            try:
                score_result = score_speech(db, speech_id, profile)
                composite_score = score_result.get("composite_score") if score_result else None
                # Cache the score
                if composite_score is not None:
                    db.cache_speech_score(speech_id, composite_score, profile)
            except Exception:
                composite_score = None
        
        progress_data.append({
            "id": speech_id,
            "title": speech.get("title"),
            "created_at": speech.get("created_at"),
            "composite_score": composite_score,
        })
    
    return {
        "project": {"id": project_id, "name": project.get("name"), "profile": profile},
        "progress": progress_data,
        "total": len(progress_data)
    }


# ---------------------------------------------------------------------------
# Benchmark (Task #39)
# ---------------------------------------------------------------------------

@router.get("/speeches/{speech_id}/benchmark", tags=["Comparison"])
async def benchmark_speech(
    speech_id: int,
    auth: dict = Depends(flexible_auth),
):
    """
    Benchmark a speech against the entire database.
    Returns percentile rankings and scores across all profiles.
    """
    from scoring import benchmark_speech as _benchmark

    db = pipeline_runner.get_db()
    speech = db.get_speech(speech_id)
    if not speech:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found.")

    result = _benchmark(db, speech_id)
    if not result:
        return _error(404, "NO_ANALYSIS", f"No analysis found for speech {speech_id}.")

    return result


# ---------------------------------------------------------------------------
# Comparison (Task #39)
# ---------------------------------------------------------------------------

@router.post("/compare", tags=["Comparison"])
async def compare_speeches(
    body: dict = Body(..., example={"speech_ids": [1, 2, 3], "profile": "general"}),
    auth: dict = Depends(flexible_auth),
):
    """
    Compare 2+ speeches side-by-side with metrics and scoring.

    Request body:
    - speech_ids: list of 2+ speech IDs to compare
    - profile: scoring profile to use (default: "general")
    """
    from scoring import score_speech, SCORING_PROFILES

    speech_ids = body.get("speech_ids", [])
    profile = body.get("profile", "general")

    if not speech_ids or len(speech_ids) < 2:
        return _error(400, "INVALID_REQUEST", "Provide at least 2 speech_ids to compare.")

    if len(speech_ids) > 10:
        return _error(400, "TOO_MANY", "Maximum 10 speeches can be compared at once.")

    if profile not in SCORING_PROFILES:
        return _error(400, "INVALID_PROFILE",
                      f"Unknown profile '{profile}'. Valid: {', '.join(sorted(SCORING_PROFILES.keys()))}")

    db = pipeline_runner.get_db()

    comparisons = []
    for sid in speech_ids:
        speech = db.get_speech(sid)
        if not speech:
            continue

        analysis = db.get_analysis(sid)
        transcription = db.get_transcription(sid)
        score = score_speech(db, sid, profile)

        entry = {
            "speech_id": sid,
            "title": speech.get("title", ""),
            "speaker": speech.get("speaker"),
            "category": speech.get("category"),
            "duration_sec": speech.get("duration_sec"),
            "metrics": {},
            "score": score,
        }

        if analysis:
            for col in [
                "wpm", "pitch_mean_hz", "pitch_std_hz", "jitter_pct", "shimmer_pct",
                "hnr_db", "vocal_fry_ratio", "lexical_diversity", "syntactic_complexity",
                "fk_grade_level", "repetition_score", "sentiment", "quality_level",
                "articulation_rate", "pitch_range_hz", "connectedness",
                "vowel_space_index", "snr_db",
            ]:
                val = analysis.get(col)
                if val is not None:
                    entry["metrics"][col] = val

        if transcription:
            entry["word_count"] = transcription.get("word_count")
            entry["language"] = transcription.get("language")

        comparisons.append(entry)

    if len(comparisons) < 2:
        return _error(404, "INSUFFICIENT_DATA",
                      "Could not find enough speeches with data for comparison.")

    return {
        "profile": profile,
        "speeches": comparisons,
        "count": len(comparisons),
    }


# ---------------------------------------------------------------------------
# Batch Analysis (Task #46)
# ---------------------------------------------------------------------------

# In-memory batch store
_batches: Dict[str, dict] = {}


@router.post("/batch/analyze", tags=["Batch"])
async def batch_analyze(
    request: Request,
    preset: Optional[str] = Query("full", description="Analysis preset for all files"),
    language: Optional[str] = Query(None, description="Language code for Whisper"),
    auth: dict = Depends(flexible_auth),
):
    """
    Submit multiple audio files for batch analysis.
    Returns a batch_id with individual job IDs for tracking.

    Upload multiple files as multipart form data with field name 'files'.
    """
    form = await request.form()
    files = form.getlist("files")

    if not files:
        return _error(400, "NO_FILES", "No files provided. Upload with field name 'files'.")

    if len(files) > 20:
        return _error(400, "TOO_MANY_FILES", "Maximum 20 files per batch.")

    # Validate preset
    if preset:
        preset_lower = preset.strip().lower()
        if preset_lower not in config.MODULE_PRESETS:
            return _error(400, "INVALID_PRESET",
                          f"Unknown preset '{preset}'. Valid: {', '.join(sorted(config.MODULE_PRESETS.keys()))}")
    else:
        preset_lower = "full"

    # Resolve modules
    resolved_modules = config.MODULE_PRESETS.get(preset_lower, config.ALL_MODULES.copy())
    modules_requested = sorted(resolved_modules)

    # Validate language
    lang = None
    if language:
        lang = language.strip().lower()
        if not lang or len(lang) > 10:
            return _error(400, "INVALID_LANGUAGE", "Language code must be a short ISO code.")

    batch_id = str(uuid.uuid4())
    jobs = []

    os.makedirs(config.ASYNC_UPLOAD_DIR, exist_ok=True)

    for file_item in files:
        if not hasattr(file_item, 'filename') or not file_item.filename:
            continue

        ext = file_item.filename.rsplit(".", 1)[-1].lower() if "." in file_item.filename else ""
        if ext not in config.ALLOWED_EXTENSIONS:
            jobs.append({
                "filename": file_item.filename,
                "job_id": None,
                "status": "rejected",
                "error": f"Invalid file type '.{ext}'",
            })
            continue

        # Save to temp
        suffix = f".{ext}"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=config.ASYNC_UPLOAD_DIR)
        os.close(fd)

        content = await file_item.read()
        if len(content) > config.MAX_FILE_SIZE_BYTES:
            os.remove(tmp_path)
            jobs.append({
                "filename": file_item.filename,
                "job_id": None,
                "status": "rejected",
                "error": f"File exceeds {config.MAX_FILE_SIZE_MB}MB limit",
            })
            continue

        if len(content) == 0:
            os.remove(tmp_path)
            jobs.append({
                "filename": file_item.filename,
                "job_id": None,
                "status": "rejected",
                "error": "Empty file",
            })
            continue

        with open(tmp_path, "wb") as f:
            f.write(content)

        # Submit as async job
        try:
            job = job_manager.create_job(
                filename=file_item.filename,
                upload_path=tmp_path,
                webhook_url=None,
                modules=resolved_modules.copy(),
                preset=preset_lower,
                quality_gate="warn",
                language=lang,
                modules_requested=modules_requested,
                key_id=auth["key_data"]["key_id"] if auth.get("key_data") else "user",
            )
            jobs.append({
                "filename": file_item.filename,
                "job_id": job.job_id,
                "status": "queued",
                "error": None,
            })
        except QueueFullError as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            jobs.append({
                "filename": file_item.filename,
                "job_id": None,
                "status": "rejected",
                "error": str(e),
            })

    # Store batch info
    _batches[batch_id] = {
        "batch_id": batch_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preset": preset_lower,
        "language": lang,
        "total_files": len(files),
        "jobs": jobs,
    }

    return JSONResponse(
        status_code=202,
        content={
            "batch_id": batch_id,
            "total_files": len(files),
            "queued": sum(1 for j in jobs if j["status"] == "queued"),
            "rejected": sum(1 for j in jobs if j["status"] == "rejected"),
            "jobs": jobs,
        }
    )


@router.get("/batch/{batch_id}", tags=["Batch"])
async def get_batch_status(
    batch_id: str,
    auth: dict = Depends(flexible_auth),
):
    """Check status of a batch analysis job. Returns individual job statuses."""
    batch = _batches.get(batch_id)
    if not batch:
        return _error(404, "BATCH_NOT_FOUND", f"Batch '{batch_id}' not found or expired.")

    # Update job statuses from job manager
    updated_jobs = []
    for job_info in batch["jobs"]:
        if job_info.get("job_id"):
            job = job_manager.get_job(job_info["job_id"])
            if job:
                entry = {
                    "filename": job_info["filename"],
                    "job_id": job_info["job_id"],
                    "status": job.status,
                    "error": job.error,
                    "speech_id": job.result.get("speech_id") if job.result else None,
                }
                updated_jobs.append(entry)
            else:
                updated_jobs.append(job_info)
        else:
            updated_jobs.append(job_info)

    total = len(updated_jobs)
    complete = sum(1 for j in updated_jobs if j["status"] == "complete")
    failed = sum(1 for j in updated_jobs if j["status"] in ("failed", "rejected"))
    processing = sum(1 for j in updated_jobs if j["status"] == "processing")
    queued = sum(1 for j in updated_jobs if j["status"] == "queued")

    overall_status = "complete" if (complete + failed) == total else \
                     "processing" if processing > 0 else \
                     "queued" if queued > 0 else "complete"

    return {
        "batch_id": batch_id,
        "status": overall_status,
        "created_at": batch["created_at"],
        "preset": batch["preset"],
        "total": total,
        "complete": complete,
        "failed": failed,
        "processing": processing,
        "queued": queued,
        "jobs": updated_jobs,
    }


# ---------------------------------------------------------------------------
# Speech Coach AI
# ---------------------------------------------------------------------------

import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
COACH_MODEL = "llama3.1:8b"

COACH_SYSTEM_PROMPT = """You are a proactive, professional speech coach AI assistant in the SpeakFit app. Your role is to help users improve their public speaking skills.

Your approach:
1. Be encouraging but honest - celebrate wins and gently address areas for growth
2. Be PROACTIVE - ask follow-up questions, check on goals, prompt reflection
3. Keep responses conversational (2-3 paragraphs unless more detail needed)
4. Remember context from previous conversations
5. CELEBRATE milestones and achievements enthusiastically!

Things you should proactively do:
- Reference their onboarding survey (if available) - tailor advice to their primary goal, skill level, and focus areas
- Check in on progress toward their stated goals
- Suggest specific exercises based on their focus areas (pacing, clarity, filler words, confidence, etc.)
- Ask for feedback on the app features
- Celebrate improvements you notice in their metrics
- Acknowledge streaks and achievements ("Great job on your 7-day streak!")
- Challenge users to beat their personal bests
- Mention daily XP goals if they're close
- For pronunciation-focused users, emphasize clarity and articulation exercises
- For presentation-focused users, emphasize structure and engagement
- For confidence-focused users, be extra encouraging and positive

Gamification awareness:
- Users earn XP for recordings (25 XP) and coach chats (5 XP)
- Daily goal is 50 XP
- Streaks track consecutive days of practice
- Achievements unlock for milestones (first recording, streaks, improvements)
- When you see a new achievement or streak milestone, celebrate it!
- Encourage users to maintain their streak

When discussing metrics:
- WPM (words per minute): 120-150 is ideal for most contexts
- HNR (Harmonics-to-Noise Ratio): Higher is better, indicates voice clarity
- Jitter/Shimmer: Lower is better, high values may indicate vocal strain
- Vocal Fry: Common in casual speech, reduce for formal presentations
- Pitch Variation: More variation = more engaging delivery

Always end with either actionable advice OR a question to engage the user."""


def _build_user_context(db, user_id: int) -> str:
    """Build comprehensive user context for the coach."""
    context_parts = []
    
    try:
        # Get user's recording stats
        cursor = db.conn.execute("""
            SELECT 
                COUNT(*) as total_recordings,
                AVG(CASE WHEN profile = 'pronunciation' THEN pronunciation_mean_confidence END) as avg_pronunciation,
                COUNT(DISTINCT profile) as profiles_used,
                MAX(created_at) as last_recording
            FROM speeches 
            WHERE user_id = ?
        """, (user_id,))
        stats = cursor.fetchone()
        
        if stats and stats[0] > 0:
            context_parts.append(f"User has {stats[0]} total recordings")
            if stats[1]:
                context_parts.append(f"Average pronunciation clarity: {stats[1]*100:.1f}%")
            context_parts.append(f"Uses {stats[2]} different profiles")
            if stats[3]:
                context_parts.append(f"Last recording: {stats[3]}")
        
        # Get projects
        cursor = db.conn.execute("""
            SELECT name, type FROM projects WHERE user_id = ? LIMIT 5
        """, (user_id,))
        projects = cursor.fetchall()
        if projects:
            project_names = [p[0] for p in projects]
            context_parts.append(f"Projects: {', '.join(project_names)}")
        
        # Get recent scores trend
        cursor = db.conn.execute("""
            SELECT cached_score, created_at 
            FROM speeches 
            WHERE user_id = ? AND cached_score IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 5
        """, (user_id,))
        recent_scores = cursor.fetchall()
        if len(recent_scores) >= 2:
            avg_recent = sum(s[0] for s in recent_scores) / len(recent_scores)
            context_parts.append(f"Recent average score: {avg_recent:.1f}")
        
        # Get common problem areas
        cursor = db.conn.execute("""
            SELECT grammar_analysis_json 
            FROM speeches 
            WHERE user_id = ? AND grammar_analysis_json IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 5
        """, (user_id,))
        grammar_rows = cursor.fetchall()
        total_fillers = 0
        for row in grammar_rows:
            try:
                import json
                data = json.loads(row[0])
                total_fillers += data.get('filler_count', 0)
            except:
                pass
        if total_fillers > 10:
            context_parts.append(f"Common issue: Uses filler words frequently ({total_fillers} in last 5 recordings)")
        
        # Get gamification stats
        try:
            from gamification import get_gamification_status, get_unnotified_achievements
            import sqlite3 as sqlite3_module
            conn = sqlite3_module.connect(config.DB_PATH)
            status = get_gamification_status(conn, user_id)
            
            # Streak info
            if status.current_streak > 0:
                context_parts.append(f"🔥 Current streak: {status.current_streak} days")
                if status.current_streak >= 7:
                    context_parts.append("(Celebrate this streak milestone!)")
                elif status.current_streak == status.longest_streak and status.current_streak > 1:
                    context_parts.append("(This is their longest streak ever!)")
            
            # Level and XP
            context_parts.append(f"Level {status.level} ({status.total_xp} XP total)")
            
            # Daily progress
            daily_progress = status.daily_xp / status.daily_goal * 100
            if status.daily_xp >= status.daily_goal:
                context_parts.append("✅ Daily XP goal already met!")
            elif daily_progress >= 50:
                context_parts.append(f"Daily XP: {status.daily_xp}/{status.daily_goal} ({daily_progress:.0f}% - almost there!)")
            
            # Recent achievements (celebrate these!)
            if status.recent_achievements:
                recent_names = [f"{a.icon} {a.name}" for a in status.recent_achievements[:3]]
                context_parts.append(f"Recent achievements: {', '.join(recent_names)}")
            
            # Check for unnotified achievements (big celebration!)
            unnotified = get_unnotified_achievements(conn, user_id)
            if unnotified:
                achievement_names = [f"{a.icon} {a.name}" for a in unnotified]
                context_parts.append(f"🎉 NEW ACHIEVEMENTS TO CELEBRATE: {', '.join(achievement_names)}")
            
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to get gamification context: {e}")
        
        # Get user's survey responses (personalization data)
        try:
            cursor = db.conn.execute("""
                SELECT primary_goal, skill_level, focus_areas, context, time_commitment, native_language
                FROM user_surveys
                WHERE user_id = ?
            """, (user_id,))
            survey = cursor.fetchone()
            
            if survey:
                context_parts.append("\n📋 USER PROFILE (from onboarding survey):")
                context_parts.append(f"  Primary goal: {survey[0].replace('_', ' ')}")
                context_parts.append(f"  Skill level: {survey[1]}")
                
                # Parse focus areas JSON
                try:
                    import json
                    focus_areas = json.loads(survey[2])
                    if focus_areas:
                        context_parts.append(f"  Focus areas: {', '.join(a.replace('_', ' ') for a in focus_areas)}")
                except:
                    pass
                
                context_parts.append(f"  Context: {survey[3].replace('_', ' ')}")
                context_parts.append(f"  Time commitment: {survey[4]}")
                
                if survey[5]:
                    context_parts.append(f"  Native language: {survey[5]}")
                
                context_parts.append("(Tailor your advice to their goals and focus areas!)")
        except Exception as e:
            logger.warning(f"Failed to get survey context: {e}")
        
    except Exception as e:
        logger.warning(f"Failed to build user context: {e}")
    
    return "\n".join(context_parts) if context_parts else ""


@router.get("/coach/history", tags=["Coach"])
async def get_coach_history(
    limit: int = Query(50, description="Max messages to return"),
    speech_id: Optional[int] = Query(None, description="Filter by speech ID"),
    auth: dict = Depends(flexible_auth),
):
    """Get the user's chat history with the coach."""
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required")
    
    db = pipeline_runner.get_db()
    messages = db.get_coach_messages(user_id, limit=limit, speech_id=speech_id)
    
    return {
        "messages": messages,
        "count": len(messages),
    }


@router.get("/coach/unread", tags=["Coach"])
async def get_coach_unread(
    auth: dict = Depends(flexible_auth),
):
    """Check if there are unread coach messages (proactive messages)."""
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required")
    
    db = pipeline_runner.get_db()
    
    # Check for unread proactive messages
    try:
        cursor = db.conn.execute("""
            SELECT COUNT(*) FROM coach_messages 
            WHERE user_id = ? AND role = 'assistant' AND read_at IS NULL
        """, (user_id,))
        unread = cursor.fetchone()[0]
        return {"unread_count": unread}
    except:
        return {"unread_count": 0}


@router.post("/coach/mark-read", tags=["Coach"])
async def mark_coach_read(
    auth: dict = Depends(flexible_auth),
):
    """Mark all coach messages as read."""
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required")
    
    db = pipeline_runner.get_db()
    try:
        db.conn.execute("""
            UPDATE coach_messages 
            SET read_at = datetime('now')
            WHERE user_id = ? AND role = 'assistant' AND read_at IS NULL
        """, (user_id,))
        db.conn.commit()
        return {"success": True}
    except Exception as e:
        return _error(500, "ERROR", str(e))


@router.post("/coach/chat", tags=["Coach"])
async def coach_chat(
    speech_id: Optional[int] = Body(None, description="Speech ID for context"),
    message: str = Body(..., description="User message to the coach"),
    history: Optional[List[Dict[str, str]]] = Body(None, description="Previous messages [{role, content}]"),
    auth: dict = Depends(flexible_auth),
):
    """
    Chat with the AI Speech Coach. Optionally provide a speech_id for context.
    Returns coaching advice based on the speech analysis and user question.
    """
    db = pipeline_runner.get_db()
    user_id = auth.get("user_id")
    
    # Build user context (overall stats, projects, trends)
    user_context = ""
    if user_id:
        user_context = _build_user_context(db, user_id)
    
    # Build context from speech analysis if provided
    context = ""
    if speech_id:
        # Check ownership
        user_id = auth.get("user_id")
        speech = db.get_speech(speech_id)
        if not speech:
            return _error(404, "NOT_FOUND", f"Speech {speech_id} not found")
        if user_id and speech.get("user_id") != user_id:
            return _error(403, "FORBIDDEN", "You don't have access to this speech")
        
        # Get analysis data
        analysis = db.get_full_analysis(speech_id)
        if analysis:
            # Build context summary
            context_parts = [f"Speech: {speech.get('title', 'Untitled')}"]
            
            if analysis.get("transcription"):
                # Truncate long transcripts
                transcript = analysis["transcription"][:500]
                if len(analysis["transcription"]) > 500:
                    transcript += "..."
                context_parts.append(f"Transcript excerpt: {transcript}")
            
            if analysis.get("text"):
                text = analysis["text"]
                context_parts.append(f"Word count: {text.get('word_count', 'N/A')}")
                context_parts.append(f"Filler words: {text.get('filler_count', 0)}")
            
            if analysis.get("audio"):
                audio = analysis["audio"]
                context_parts.append(f"Duration: {audio.get('duration_sec', 0):.1f}s")
                context_parts.append(f"WPM: {audio.get('wpm', 'N/A')}")
                context_parts.append(f"Pitch (avg): {audio.get('pitch_mean', 'N/A')} Hz")
                context_parts.append(f"HNR: {audio.get('hnr', 'N/A')} dB")
                context_parts.append(f"Jitter: {audio.get('jitter', 'N/A')}")
                context_parts.append(f"Shimmer: {audio.get('shimmer', 'N/A')}")
            
            if analysis.get("advanced_audio"):
                adv = analysis["advanced_audio"]
                if adv.get("vocal_fry"):
                    context_parts.append(f"Vocal fry: {adv['vocal_fry'].get('fry_percentage', 0):.1f}%")
                if adv.get("speech_rate_variability"):
                    srv = adv["speech_rate_variability"]
                    context_parts.append(f"Rate variability: {srv.get('cv', 0):.2f}")
            
            context = "\n".join(context_parts)
    
    # Build the prompt
    prompt_parts = [COACH_SYSTEM_PROMPT]
    
    if user_context:
        prompt_parts.append(f"\n--- User Overview ---\n{user_context}\n---")
    
    if context:
        prompt_parts.append(f"\n--- Current Speech Analysis ---\n{context}\n---")
    
    # Add history if provided (from DB if not passed)
    if not history and user_id:
        # Load recent chat history from database
        db_history = db.get_coach_messages(user_id, limit=10, speech_id=speech_id)
        if db_history:
            history = [{"role": m["role"], "content": m["content"]} for m in db_history]
    
    if history:
        for msg in history[-6:]:  # Last 6 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"\nUser: {content}")
            else:
                prompt_parts.append(f"\nCoach: {content}")
    
    prompt_parts.append(f"\nUser: {message}")
    prompt_parts.append("\nCoach:")
    
    full_prompt = "\n".join(prompt_parts)
    
    # Call Ollama
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": COACH_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            coach_response = result.get("response", "").strip()
            
            # Save messages to database
            user_id = auth.get("user_id")
            if user_id:
                db.save_coach_message(user_id, "user", message, speech_id)
                db.save_coach_message(user_id, "assistant", coach_response, speech_id, COACH_MODEL)
                
                # Award XP for coach chat and check achievements
                try:
                    from gamification import award_xp, check_achievements
                    conn = sqlite3.connect(config.DB_PATH)
                    award_xp(conn, user_id, 'coach_chat', metadata={'speech_id': speech_id})
                    check_achievements(conn, user_id)
                    conn.close()
                except Exception as e:
                    logger.warning(f"Gamification update failed: {e}")
            
            return {
                "response": coach_response,
                "speech_id": speech_id,
                "model": COACH_MODEL,
            }
    except httpx.TimeoutException:
        return _error(504, "TIMEOUT", "Coach AI took too long to respond")
    except httpx.HTTPError as e:
        logger.error(f"Ollama error: {e}")
        return _error(502, "AI_ERROR", "Failed to get response from coach AI")
    except Exception as e:
        logger.error(f"Coach chat error: {e}")
        return _error(500, "INTERNAL_ERROR", str(e))


@router.post("/coach/meeting", tags=["Coach"])
async def meeting_coach_chat(
    project_id: int = Body(..., description="Meeting project ID for context"),
    message: str = Body(..., description="User message to the coach"),
    history: Optional[List[Dict[str, str]]] = Body(None, description="Previous messages [{role, content}]"),
    auth: dict = Depends(flexible_auth),
):
    """
    Chat with the AI Coach about a group meeting.
    The coach has context of all meeting transcripts and participants.
    """
    db = pipeline_runner.get_db()
    user_id = auth.get("user_id")
    
    # Verify project exists and user has access
    project = db.get_project(project_id)
    if not project:
        return _error(404, "NOT_FOUND", f"Project {project_id} not found")
    if user_id and project.get("user_id") != user_id:
        return _error(403, "FORBIDDEN", "You don't have access to this project")
    if project.get("type") != "group_meeting":
        return _error(400, "INVALID_PROJECT", "This endpoint is for group meeting projects only")
    
    # Get meeting data for context
    meeting_data = db.get_meeting_data(project_id)
    
    # Build meeting context
    context_parts = [
        f"Meeting: {project.get('name', 'Untitled Meeting')}",
        f"Participants: {', '.join(meeting_data.get('participants', ['Unknown']))}",
        f"Total recordings: {meeting_data.get('recording_count', 0)}",
        f"Total duration: {meeting_data.get('total_duration_sec', 0):.0f} seconds",
    ]
    
    # Add transcript excerpts (limit to avoid token overflow)
    transcripts = meeting_data.get("transcripts", [])
    if transcripts:
        context_parts.append("\n--- Meeting Transcript ---")
        total_chars = 0
        max_chars = 2000  # Limit context size
        for t in transcripts:
            speaker = t.get("speaker", "Unknown")
            text = t.get("transcript", "")
            if total_chars + len(text) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 50:
                    context_parts.append(f"{speaker}: {text[:remaining]}...")
                context_parts.append("(transcript truncated)")
                break
            context_parts.append(f"{speaker}: {text}")
            total_chars += len(text)
        context_parts.append("--- End Transcript ---")
    
    context = "\n".join(context_parts)
    
    # Build the prompt
    meeting_system_prompt = """You are an AI Meeting Coach for SpeechScore. You help users understand and analyze their group meetings.

Your capabilities:
- Summarize meeting discussions and key points
- Identify action items and decisions made
- Analyze participant contributions and dynamics
- Provide insights on meeting effectiveness
- Answer questions about specific topics discussed

Be concise, helpful, and reference specific parts of the meeting when relevant."""

    prompt_parts = [meeting_system_prompt]
    prompt_parts.append(f"\n--- Meeting Context ---\n{context}\n---")
    
    # Add history if provided
    if history:
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"\nUser: {content}")
            else:
                prompt_parts.append(f"\nCoach: {content}")
    
    prompt_parts.append(f"\nUser: {message}")
    prompt_parts.append("\nCoach:")
    
    full_prompt = "\n".join(prompt_parts)
    
    # Call Ollama
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": COACH_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            coach_response = result.get("response", "").strip()
            
            # Save messages to database (using project_id as context)
            if user_id:
                db.save_coach_message(user_id, "user", message, None, None, project_id)
                db.save_coach_message(user_id, "assistant", coach_response, None, COACH_MODEL, project_id)
            
            return {
                "response": coach_response,
                "project_id": project_id,
                "model": COACH_MODEL,
            }
    except httpx.TimeoutException:
        return _error(504, "TIMEOUT", "Coach AI took too long to respond")
    except httpx.HTTPError as e:
        logger.error(f"Ollama error: {e}")
        return _error(502, "AI_ERROR", "Failed to get response from coach AI")
    except Exception as e:
        logger.error(f"Meeting coach error: {e}")
        return _error(500, "INTERNAL_ERROR", str(e))


# ---------------------------------------------------------------------------
# Kids Mode - Word Help (AI-Powered Pronunciation Guidance)
# ---------------------------------------------------------------------------

KIDS_WORD_HELP_PROMPT = """You are a friendly reading helper for children ages 4-8. 
A child is reading aloud and tapped on a word they need help with.

Your job:
1. Explain how to say the word in a fun, simple way
2. Break it into easy sound chunks if helpful
3. Give a short tip they can remember
4. Keep your response SHORT (2-3 sentences max)
5. Use encouraging, kid-friendly language
6. If relevant, mention a rhyme or similar word they might know

DO NOT:
- Use technical linguistic terms
- Give long explanations
- Be condescending
- Use complicated words in your explanation

Examples of good responses:
- "This word is 'through' - say it like 'threw' the ball! The 'ough' makes an 'oo' sound."
- "Say 'beau-ti-ful' - three parts! BYOO-tih-full. Like saying 'beautiful butterfly'!"
- "This is 'knight' - the K is silent! Say it like 'night' when it's dark outside."
"""


@router.post("/kids/word-help", tags=["Kids"])
async def kids_word_help(
    word: str = Body(..., description="The word the child tapped on"),
    sentence: str = Body(..., description="The full sentence containing the word"),
    book_title: Optional[str] = Body(None, description="Title of the book being read"),
    difficulty: Optional[str] = Body(None, description="Book difficulty level"),
    auth: dict = Depends(flexible_auth),
):
    """
    Get kid-friendly pronunciation help for a word.
    Uses Ollama to generate contextual, age-appropriate guidance.
    """
    
    # Build the prompt
    context_parts = [f"Word: {word}", f"Sentence: {sentence}"]
    if book_title:
        context_parts.append(f"Book: {book_title} ({difficulty or 'unknown'} level)")
    
    prompt = f"""{KIDS_WORD_HELP_PROMPT}

{chr(10).join(context_parts)}

How should the child say "{word}"? (Keep it short and fun!)"""

    # Call Ollama
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 150,  # Keep responses short
                    }
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
                return _error(502, "AI_ERROR", "Failed to get help from reading assistant")
            
            result = response.json()
            help_text = result.get("response", "").strip()
            
            if not help_text:
                # Fallback for empty response
                help_text = f"Let's sound it out: {word}. Try saying each part slowly!"
            
            return {
                "word": word,
                "tip": help_text,
                "sentence": sentence,
            }
            
    except httpx.TimeoutException:
        logger.error("Ollama timeout for kids word help")
        return _error(504, "TIMEOUT", "Reading helper took too long")
    except httpx.RequestError as e:
        logger.error(f"Ollama error: {e}")
        return _error(502, "AI_ERROR", "Failed to get help from reading assistant")
    except Exception as e:
        logger.error(f"Kids word help error: {e}")
        return _error(500, "INTERNAL_ERROR", str(e))


# ---------------------------------------------------------------------------
# Dementia Conversation Analysis
# ---------------------------------------------------------------------------

from .dementia_analysis import analyze_dementia_markers, calculate_preposition_rate

class DementiaAnalysisRequest(BaseModel):
    """Request model for dementia conversation analysis."""
    speech_id: Optional[int] = None
    segments: Optional[List[Dict[str, Any]]] = None
    speaker_labels: Optional[Dict[str, str]] = None
    # Tunable analysis parameters
    repetition_threshold: Optional[float] = 0.85  # Similarity threshold (0.5-1.0)
    min_segment_gap: Optional[int] = 2  # Min segments apart for repetition

@router.post("/dementia/analyze", tags=["Dementia"])
async def analyze_dementia_conversation(
    request: DementiaAnalysisRequest,
    auth: dict = Depends(flexible_auth),
):
    """
    Analyze a conversation for dementia markers.
    
    Metrics scored per speaker:
    1. Prepositions per 10 words (lower rates correlate with cognitive decline)
    2. Repeated concepts/questions (memory issues indicator)
    
    Provide either:
    - speech_id: ID of a previously analyzed speech with diarization
    - segments: List of {text, speaker, start, end} objects
    """
    user_id = auth.get("user_id")
    
    segments = request.segments
    
    # If speech_id provided, load segments from database
    if request.speech_id and not segments:
        db = pipeline_runner.get_db()
        
        # Check ownership
        speech = db.get_speech(request.speech_id)
        if not speech:
            return _error(404, "NOT_FOUND", "Speech not found")
        
        if speech.get("user_id") and speech.get("user_id") != user_id:
            return _error(403, "FORBIDDEN", "Not authorized to access this speech")
        
        # Get analysis result (contains transcript and optionally diarization)
        conn = db.conn
        cursor = conn.cursor()
        cursor.execute("""
            SELECT result 
            FROM analyses 
            WHERE speech_id = ? 
            ORDER BY created_at DESC LIMIT 1
        """, (request.speech_id,))
        row = cursor.fetchone()
        
        if not row:
            return _error(404, "NOT_FOUND", "No analysis found for this speech")
        
        # Parse the result JSON
        try:
            result_data = json.loads(row[0]) if row[0] else {}
        except:
            result_data = {}
        
        transcript = result_data.get("transcript", "")
        diarization = result_data.get("diarization", [])
        
        # Use diarization segments if available
        if diarization:
            segments = diarization
        # Fallback: create single segment from transcript
        elif transcript:
            segments = [{"text": transcript, "speaker": "Speaker 1", "start": 0, "end": 0}]
    
    if not segments:
        return _error(400, "BAD_REQUEST", "No segments provided. Either provide speech_id or segments array.")
    
    # Run dementia analysis
    try:
        results = analyze_dementia_markers(
            segments=segments,
            speaker_labels=request.speaker_labels,
            repetition_threshold=request.repetition_threshold or 0.85,
            min_segment_gap=request.min_segment_gap or 2,
        )
        
        # Store dementia metrics in the speech record if speech_id provided
        if request.speech_id:
            db = pipeline_runner.get_db()
            conn = db.conn
            cursor = conn.cursor()
            
            # Aggregate metrics from all speakers
            speakers_data = results.get("speakers", {})
            total_words = 0
            total_preps = 0
            total_rep_concepts = 0
            total_rep_questions = 0
            total_aphasic = 0
            max_aphasic_conf = 0
            
            for speaker_id, speaker_data in speakers_data.items():
                metrics = speaker_data.get("metrics", {})
                prep_analysis = speaker_data.get("preposition_analysis", {})
                rep_analysis = speaker_data.get("repetition_analysis", {})
                aphasic_analysis = speaker_data.get("aphasic_analysis", {})
                
                words = prep_analysis.get("word_count", 0)
                total_words += words
                total_preps += prep_analysis.get("preposition_count", 0)
                total_rep_concepts += rep_analysis.get("repeated_concepts_count", 0)
                total_rep_questions += rep_analysis.get("repeated_questions_count", 0)
                total_aphasic += aphasic_analysis.get("event_count", 0)
                max_aphasic_conf = max(max_aphasic_conf, aphasic_analysis.get("confidence_score", 0))
            
            # Calculate overall preposition rate
            prep_rate = (total_preps / total_words * 10) if total_words > 0 else 0
            
            # Build dementia_metrics from aggregated results
            dementia_metrics = {
                "prepositions_per_10_words": round(prep_rate, 2),
                "preposition_count": total_preps,
                "word_count": total_words,
                "repetition_count": total_rep_concepts + total_rep_questions,
                "aphasic_events": total_aphasic,
                "aphasic_confidence": round(max_aphasic_conf, 2),
                "num_speakers": len(speakers_data),
            }
            
            # Update speech record with dementia metrics and set profile to 'dementia'
            cursor.execute("""
                UPDATE speeches 
                SET dementia_metrics = ?, profile = 'dementia'
                WHERE id = ?
            """, (json.dumps(dementia_metrics), request.speech_id))
            conn.commit()
        
        return {
            "success": True,
            "analysis": results,
            "speech_id": request.speech_id,
        }
        
    except Exception as e:
        logger.error(f"Dementia analysis error: {e}")
        return _error(500, "ANALYSIS_ERROR", str(e))


@router.get("/dementia/analyze/{speech_id}", tags=["Dementia"])
async def get_dementia_analysis(
    speech_id: int,
    repetition_threshold: float = Query(default=0.85, ge=0.5, le=1.0, description="Similarity threshold for repetitions (0.5-1.0)"),
    min_segment_gap: int = Query(default=2, ge=1, le=10, description="Min segments apart for repetition"),
    auth: dict = Depends(flexible_auth),
):
    """
    Get dementia analysis for a previously recorded conversation.
    The speech must have been analyzed with speaker diarization.
    
    Query params:
    - repetition_threshold: How similar phrases must be (0.5=lenient, 1.0=strict)
    - min_segment_gap: Min speech segments apart to count as repetition (filters immediate corrections)
    """
    return await analyze_dementia_conversation(
        request=DementiaAnalysisRequest(
            speech_id=speech_id,
            repetition_threshold=repetition_threshold,
            min_segment_gap=min_segment_gap,
        ),
        auth=auth
    )


# ---------------------------------------------------------------------------
# Speaker Fingerprinting (Voice Identification)
# ---------------------------------------------------------------------------

@router.get("/speakers", tags=["Speakers"])
async def list_speakers(
    auth: dict = Depends(flexible_auth),
):
    """
    List all speakers for the current user.
    Returns both registered voice profiles AND unique speakers detected in recordings.
    """
    user_id = auth.get("user_id")
    if not user_id:
        # Return empty list instead of error if not authenticated
        return {"speakers": [], "count": 0}
    
    db = pipeline_runner.get_db()
    
    # Get registered speaker profiles
    registered_profiles = db.get_speaker_embeddings(user_id)
    registered_names = {p.get("name") for p in registered_profiles}
    
    # Get unique speakers from recordings (from diarization)
    conn = db.conn
    cursor = conn.cursor()
    
    # Query unique speakers from analysis segments
    cursor.execute("""
        SELECT DISTINCT json_extract(seg.value, '$.speaker') as speaker_name
        FROM speeches s
        JOIN analyses a ON a.speech_id = s.id,
        json_each(a.result, '$.segments') as seg
        WHERE s.user_id = ?
        AND json_extract(seg.value, '$.speaker') IS NOT NULL
        AND json_extract(seg.value, '$.speaker') != ''
    """, (user_id,))
    
    recording_speakers = set()
    for row in cursor.fetchall():
        if row[0]:
            recording_speakers.add(row[0])
    
    # Build combined speaker list
    speakers = []
    
    # Add registered profiles first
    for p in registered_profiles:
        speakers.append({
            "name": p.get("name"),
            "registered": True,
            "embedding_count": p.get("embedding_count", 1),
        })
    
    # Add speakers from recordings that aren't registered
    for name in sorted(recording_speakers):
        if name not in registered_names:
            speakers.append({
                "name": name,
                "registered": False,
            })
    
    return {
        "speakers": speakers,
        "count": len(speakers),
    }


@router.delete("/speakers/{speaker_name}", tags=["Speakers"])
async def delete_speaker(
    speaker_name: str,
    auth: dict = Depends(flexible_auth),
):
    """
    Delete a speaker's voice profile.
    This will stop automatic identification for that speaker.
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "AUTH_REQUIRED", "User authentication required for speaker profiles")
    
    db = pipeline_runner.get_db()
    deleted = db.delete_speaker_embedding(user_id, speaker_name)
    
    if not deleted:
        return _error(404, "NOT_FOUND", f"Speaker '{speaker_name}' not found")
    
    return {"deleted": speaker_name}


@router.patch("/speakers/{speaker_name}", tags=["Speakers"])
async def rename_speaker(
    speaker_name: str,
    new_name: str = Form(...),
    auth: dict = Depends(flexible_auth),
):
    """
    Rename a speaker's voice profile.
    Useful for changing auto-generated names like "First Speaker" to actual names.
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "AUTH_REQUIRED", "User authentication required for speaker profiles")
    
    db = pipeline_runner.get_db()
    
    # Check if speaker exists
    profiles = db.get_speaker_embeddings(user_id)
    speaker = next((p for p in profiles if p['speaker_name'] == speaker_name), None)
    
    if not speaker:
        return _error(404, "NOT_FOUND", f"Speaker '{speaker_name}' not found")
    
    # Check if new name already exists
    if any(p['speaker_name'] == new_name for p in profiles):
        return _error(409, "CONFLICT", f"Speaker '{new_name}' already exists")
    
    # Rename in database
    db.conn.execute(
        "UPDATE speaker_embeddings SET speaker_name = ? WHERE user_id = ? AND speaker_name = ?",
        (new_name, user_id, speaker_name)
    )
    db.conn.commit()
    
    return {"old_name": speaker_name, "new_name": new_name}


@router.post("/speakers/register", tags=["Speakers"])
async def register_speaker(
    audio: UploadFile = File(...),
    speaker_name: str = Query(..., description="Name for this speaker"),
    auth: dict = Depends(flexible_auth),
):
    """
    Register a speaker's voice from an audio sample.
    Future recordings will be automatically identified if the voice matches.
    
    For best results, use a clear recording of 10+ seconds with minimal background noise.
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "AUTH_REQUIRED", "User authentication required for speaker profiles")
    
    # Save upload temporarily
    tmp_path, ext, result_or_error = await _validate_and_save_upload(audio)
    if tmp_path is None:
        return result_or_error
    
    try:
        from speaker_fingerprint import register_speaker
        db = pipeline_runner.get_db()
        
        success, message = register_speaker(db, user_id, speaker_name, tmp_path)
        
        if success:
            return {"success": True, "message": message, "speaker_name": speaker_name}
        else:
            return _error(422, "FINGERPRINT_FAILED", message)
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/speakers/identify", tags=["Speakers"])
async def identify_speaker(
    audio: UploadFile = File(...),
    threshold: float = Query(0.80, ge=0.5, le=1.0, description="Match confidence threshold"),
    auth: dict = Depends(flexible_auth),
):
    """
    Identify the speaker in an audio sample by comparing against registered voice profiles.
    Returns the matching speaker name and confidence score, or null if no match.
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "AUTH_REQUIRED", "User authentication required for speaker identification")
    
    # Save upload temporarily
    tmp_path, ext, result_or_error = await _validate_and_save_upload(audio)
    if tmp_path is None:
        return result_or_error
    
    try:
        from speaker_fingerprint import extract_embedding, identify_speaker as find_speaker
        db = pipeline_runner.get_db()
        
        embedding = extract_embedding(tmp_path)
        if not embedding:
            return _error(422, "FINGERPRINT_FAILED", "Could not extract voice embedding from audio")
        
        match = find_speaker(db, user_id, embedding, threshold=threshold)
        
        return {
            "identified": match is not None,
            "speaker": match.get("speaker_name") if match else None,
            "confidence": match.get("similarity") if match else None,
            "threshold": threshold,
        }
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/speakers/{speaker_name}/detail", tags=["Speakers"])
async def get_speaker_detail(
    speaker_name: str,
    auth: dict = Depends(flexible_auth),
):
    """
    Get detailed information about a speaker, including their speeches and projects.
    Works for both registered speakers (in speaker_embeddings) and diarized speakers.
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "AUTH_REQUIRED", "User authentication required for speaker profiles")
    
    db = pipeline_runner.get_db()
    
    # Check if speaker exists as registered profile
    profiles = db.get_speaker_embeddings(user_id)
    speaker = next((p for p in profiles if p['speaker_name'] == speaker_name), None)
    is_registered = speaker is not None
    
    # If not registered, check if speaker exists in diarization segments
    if not speaker:
        check_query = """
            SELECT COUNT(*) FROM speeches s
            JOIN analyses a ON a.speech_id = s.id,
            json_each(a.result, '$.segments') as seg
            WHERE s.user_id = ?
            AND json_extract(seg.value, '$.speaker') = ?
        """
        count = db.conn.execute(check_query, (user_id, speaker_name)).fetchone()[0]
        if count == 0:
            return _error(404, "NOT_FOUND", f"Speaker '{speaker_name}' not found")
    
    # Get speeches by this speaker (from speaker field OR diarization segments)
    speeches_query = """
        SELECT DISTINCT s.id, s.title, s.speaker, s.created_at, s.duration_sec, s.profile, s.project_id,
               (SELECT 1 FROM audio WHERE speech_id = s.id LIMIT 1) as has_audio, t.text
        FROM speeches s
        LEFT JOIN transcriptions t ON t.speech_id = s.id
        LEFT JOIN analyses a ON a.speech_id = s.id
        LEFT JOIN json_each(a.result, '$.segments') as seg ON 1=1
        WHERE s.user_id = ? AND (
            s.speaker = ? OR
            json_extract(seg.value, '$.speaker') = ?
        )
        ORDER BY s.created_at DESC
        LIMIT 50
    """
    speeches = db.conn.execute(speeches_query, (user_id, speaker_name, speaker_name)).fetchall()
    
    # Get projects this speaker appears in
    projects_query = """
        SELECT DISTINCT p.id, p.name, p.description, p.created_at
        FROM projects p
        JOIN speeches s ON s.project_id = p.id
        LEFT JOIN analyses a ON a.speech_id = s.id
        LEFT JOIN json_each(a.result, '$.segments') as seg ON 1=1
        WHERE p.user_id = ? AND (
            s.speaker = ? OR
            json_extract(seg.value, '$.speaker') = ?
        )
        ORDER BY p.created_at DESC
    """
    projects = db.conn.execute(projects_query, (user_id, speaker_name, speaker_name)).fetchall()
    
    return {
        "speaker": {
            "name": speaker['speaker_name'] if speaker else speaker_name,
            "registered": is_registered,
            "embedding_count": speaker['embedding_count'] if speaker else 0,
            "created_at": speaker['created_at'] if speaker else None,
        },
        "speeches": [
            {
                "id": s["id"],
                "title": s["title"],
                "speaker": s["speaker"],
                "created_at": s["created_at"],
                "duration_sec": s["duration_sec"],
                "profile": s["profile"],
                "project_id": s["project_id"],
                "has_audio": s["has_audio"] is not None,
                "transcript_preview": s["text"][:200] if s["text"] else None,
            }
            for s in speeches
        ],
        "projects": [
            {
                "id": p["id"],
                "name": p["name"],
                "description": p["description"],
                "created_at": p["created_at"],
            }
            for p in projects
        ],
        "speech_count": len(speeches),
        "project_count": len(projects),
    }


@router.post("/speakers/merge", tags=["Speakers"])
async def merge_speakers(
    source_name: str = Form(..., description="Speaker profile to merge FROM (will be deleted)"),
    target_name: str = Form(..., description="Speaker profile to merge INTO (will be kept)"),
    auth: dict = Depends(flexible_auth),
):
    """
    Merge two speaker profiles into one.
    All voice embeddings from the source speaker will be merged into the target speaker.
    The source speaker profile will be deleted, and all speeches attributed to source will be updated.
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "AUTH_REQUIRED", "User authentication required for speaker profiles")
    
    if source_name == target_name:
        return _error(400, "INVALID_REQUEST", "Cannot merge a speaker with itself")
    
    db = pipeline_runner.get_db()
    
    # Verify both speakers exist
    profiles = db.get_speaker_embeddings(user_id)
    source = next((p for p in profiles if p['speaker_name'] == source_name), None)
    target = next((p for p in profiles if p['speaker_name'] == target_name), None)
    
    if not source:
        return _error(404, "NOT_FOUND", f"Source speaker '{source_name}' not found")
    if not target:
        return _error(404, "NOT_FOUND", f"Target speaker '{target_name}' not found")
    
    # Merge embeddings: update source embeddings to target name
    db.conn.execute(
        "UPDATE speaker_embeddings SET speaker_name = ? WHERE user_id = ? AND speaker_name = ?",
        (target_name, user_id, source_name)
    )
    
    # Update speeches: change speaker field from source to target
    updated_speeches = db.conn.execute(
        "UPDATE speeches SET speaker = ? WHERE user_id = ? AND speaker = ?",
        (target_name, user_id, source_name)
    ).rowcount
    
    db.conn.commit()
    
    return {
        "merged": True,
        "source": source_name,
        "target": target_name,
        "embeddings_merged": source['embedding_count'],
        "speeches_updated": updated_speeches,
    }


# ---------------------------------------------------------------------------
# Project Group Analysis (Multi-Party Meeting Intelligence)
# ---------------------------------------------------------------------------

PROJECT_ANALYSIS_PROMPT = """You are an AI assistant analyzing a series of meeting recordings from a project.

Below are transcripts from multiple sessions, with speaker diarization. Each session has a date and identified speakers.

{transcripts}

Based on ALL the content above, provide a comprehensive analysis including:

1. **Summary**: A concise summary of what was discussed across all sessions (2-3 paragraphs)

2. **Key Topics**: List the main topics/themes discussed

3. **Calendar of Events**: Extract any dates, deadlines, or scheduled events mentioned. Format as:
   - [DATE or TIMEFRAME]: [EVENT/DEADLINE] (mentioned in Session X)

4. **Action Items**: List any tasks, todos, or commitments made by speakers

5. **Conflict Analysis**: Identify any inconsistencies or conflicts between what was said in different sessions. For example:
   - Conflicting deadlines mentioned
   - Different versions of the same information
   - Contradictory statements by the same or different speakers
   
   If no conflicts found, state "No conflicts detected."

6. **Speaker Participation**: Brief summary of each speaker's contributions across sessions

Be specific and cite which session/speaker when referencing information."""


@router.get("/projects/{project_id}/analysis", tags=["Projects"])
async def get_project_analysis(
    project_id: int,
    auth: dict = Depends(flexible_auth),
):
    """
    Get AI-powered analysis of all recordings in a project.
    
    For projects using the "group" profile, this provides:
    - Combined transcript summary
    - Calendar of mentioned events/deadlines
    - Conflict detection between sessions
    - Speaker participation analysis
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "AUTH_REQUIRED", "User authentication required")
    
    db = pipeline_runner.get_db()
    
    # Verify project ownership
    project = db.conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?",
        (project_id, user_id)
    ).fetchone()
    
    if not project:
        return _error(404, "NOT_FOUND", "Project not found")
    
    # Get all speeches in the project with transcriptions
    speeches = db.conn.execute("""
        SELECT s.id, s.title, s.speaker, s.created_at, t.text, t.segments
        FROM speeches s
        LEFT JOIN transcriptions t ON t.speech_id = s.id
        WHERE s.project_id = ? AND s.user_id = ?
        ORDER BY s.created_at ASC
    """, (project_id, user_id)).fetchall()
    
    if not speeches:
        return {"project_id": project_id, "message": "No recordings in this project yet"}
    
    # Build combined transcript document
    transcript_parts = []
    for i, speech in enumerate(speeches, 1):
        speech_dict = dict(speech)
        date_str = speech_dict.get("created_at", "Unknown date")[:10]
        speaker = speech_dict.get("speaker", "Unknown Speaker")
        title = speech_dict.get("title", f"Session {i}")
        transcript = speech_dict.get("text", "")
        
        # Parse segments for speaker diarization if available
        segments_json = speech_dict.get("segments")
        if segments_json:
            import json
            try:
                segments = json.loads(segments_json) if isinstance(segments_json, str) else segments_json
                # Build diarized transcript
                diarized_lines = []
                for seg in segments:
                    seg_speaker = seg.get("speaker", speaker)
                    seg_text = seg.get("text", "").strip()
                    if seg_text:
                        diarized_lines.append(f"  [{seg_speaker}]: {seg_text}")
                if diarized_lines:
                    transcript = "\n".join(diarized_lines)
            except:
                pass
        
        transcript_parts.append(f"""
=== SESSION {i}: {title} ===
Date: {date_str}
Primary Speaker: {speaker}

{transcript}
""")
    
    combined_transcripts = "\n".join(transcript_parts)
    
    # Call Ollama for analysis
    prompt = PROJECT_ANALYSIS_PROMPT.format(transcripts=combined_transcripts)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False,
                }
            )
            response.raise_for_status()
            result = response.json()
            analysis = result.get("response", "").strip()
            
            return {
                "project_id": project_id,
                "project_name": dict(project).get("name"),
                "session_count": len(speeches),
                "analysis": analysis,
                "sessions": [
                    {
                        "id": dict(s)["id"],
                        "title": dict(s)["title"],
                        "speaker": dict(s)["speaker"],
                        "date": dict(s)["created_at"][:10] if dict(s).get("created_at") else None,
                    }
                    for s in speeches
                ],
            }
    except httpx.TimeoutException:
        return _error(504, "TIMEOUT", "Analysis took too long")
    except Exception as e:
        logger.error(f"Project analysis error: {e}")
        return _error(500, "ANALYSIS_ERROR", str(e))


@router.post("/projects/{project_id}/query", tags=["Projects"])
async def query_project(
    project_id: int,
    question: str = Body(..., embed=True, description="Question about the project recordings"),
    auth: dict = Depends(flexible_auth),
):
    """
    Ask questions about the content of all recordings in a project.
    
    The AI has access to all transcripts and can answer questions about:
    - What was discussed
    - What specific people said
    - Dates and deadlines mentioned
    - Decisions made
    - Any topic covered in the recordings
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "AUTH_REQUIRED", "User authentication required")
    
    db = pipeline_runner.get_db()
    
    # Verify project ownership
    project = db.conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?",
        (project_id, user_id)
    ).fetchone()
    
    if not project:
        return _error(404, "NOT_FOUND", "Project not found")
    
    # Get all transcripts
    speeches = db.conn.execute("""
        SELECT s.id, s.title, s.speaker, s.created_at, t.text
        FROM speeches s
        LEFT JOIN transcriptions t ON t.speech_id = s.id
        WHERE s.project_id = ? AND s.user_id = ?
        ORDER BY s.created_at ASC
    """, (project_id, user_id)).fetchall()
    
    if not speeches:
        return {"answer": "This project has no recordings yet."}
    
    # Build context
    context_parts = []
    for i, speech in enumerate(speeches, 1):
        s = dict(speech)
        context_parts.append(f"[Session {i}: {s.get('title', 'Untitled')} - {s.get('created_at', '')[:10]}]\n{s.get('text', '')}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are an AI assistant with access to transcripts from a project's meeting recordings.

PROJECT TRANSCRIPTS:
{context}

USER QUESTION: {question}

Please answer the question based on the transcript content above. Be specific and cite which session when referencing information. If the information isn't in the transcripts, say so."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False,
                }
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
            
            return {
                "project_id": project_id,
                "question": question,
                "answer": answer,
                "sessions_searched": len(speeches),
            }
    except httpx.TimeoutException:
        return _error(504, "TIMEOUT", "Query took too long")
    except Exception as e:
        logger.error(f"Project query error: {e}")
        return _error(500, "QUERY_ERROR", str(e))


# ---------------------------------------------------------------------------
# Error Reporting — creates Trello cards for app errors
# ---------------------------------------------------------------------------

@router.post("/errors/report", tags=["Errors"])
async def report_error(
    error_type: str = Form(...),
    error_message: str = Form(...),
    stack_trace: Optional[str] = Form(None),
    screen: Optional[str] = Form(None),
    app_version: Optional[str] = Form(None),
    device_info: Optional[str] = Form(None),
    auth: dict = Depends(flexible_auth),
):
    """
    Report an app error. Creates a Trello card for tracking.
    """
    import httpx
    
    user_id = auth.get("user_id")
    user_email = None
    if user_id:
        try:
            from .user_auth import get_user_by_id
            user = get_user_by_id(user_id)
            user_email = user.get("email") if user else None
        except:
            pass
    
    # Trello API credentials
    TRELLO_KEY = "94dbc1af84f0abce181fd9433f69d727"
    TRELLO_TOKEN = "ATTA5e7ac0ff9877573186a0a8979f4b5525c0f3c68efdf77514ad640ac653de267aAF589D6A"
    BACKLOG_LIST_ID = "6980f79bb30707574404f57a"
    
    # Build card description
    desc_parts = [
        f"**Error Type:** {error_type}",
        f"**Message:** {error_message}",
        f"**Screen:** {screen or 'Unknown'}",
        f"**App Version:** {app_version or 'Unknown'}",
        f"**Device:** {device_info or 'Unknown'}",
        f"**User:** {user_email or 'Anonymous'}",
        f"**Reported:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
    ]
    if stack_trace:
        desc_parts.append(f"\n**Stack Trace:**\n```\n{stack_trace[:2000]}\n```")
    
    description = "\n".join(desc_parts)
    
    # Create Trello card
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.trello.com/1/cards",
                params={
                    "key": TRELLO_KEY,
                    "token": TRELLO_TOKEN,
                    "idList": BACKLOG_LIST_ID,
                    "name": f"🐛 App Error: {error_type}",
                    "desc": description,
                    "idLabels": "6980f7a7f4f29c2a543dd4b4",  # 'api' label (blue) - closest to bug
                }
            )
            response.raise_for_status()
            card = response.json()
            logger.info(f"Created Trello card for error: {card.get('shortUrl')}")
            return {"reported": True, "card_id": card.get("id"), "card_url": card.get("shortUrl")}
    except Exception as e:
        logger.error(f"Failed to create Trello card for error: {e}")
        # Still return success to client - we logged the error at least
        return {"reported": True, "card_id": None, "error": "Failed to create tracking card"}


# ---------------------------------------------------------------------------
# Metric Exemplars — audio samples demonstrating each grade for each metric
# ---------------------------------------------------------------------------

@router.get("/metrics/exemplars", tags=["Metrics"])
async def get_metric_exemplars():
    """
    Get audio exemplars for all metrics.
    Returns URLs to audio files demonstrating A/B/C/D/F grades for each metric.
    """
    exemplar_path = os.path.join(
        os.path.dirname(__file__), "..", "frontend", "static", "exemplars", "exemplars.json"
    )
    
    try:
        with open(exemplar_path) as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return _error(404, "EXEMPLARS_NOT_FOUND", "Exemplar data not configured")
    except Exception as e:
        logger.error(f"Error loading exemplars: {e}")
        return _error(500, "EXEMPLAR_ERROR", str(e))


@router.get("/metrics/exemplars/{metric_key}", tags=["Metrics"])
async def get_metric_exemplar(metric_key: str):
    """
    Get audio exemplars for a specific metric.
    Returns URLs to audio files demonstrating A/B/C/D/F grades.
    """
    exemplar_path = os.path.join(
        os.path.dirname(__file__), "..", "frontend", "static", "exemplars", "exemplars.json"
    )
    
    try:
        with open(exemplar_path) as f:
            data = json.load(f)
        
        if metric_key not in data.get("metrics", {}):
            return _error(404, "METRIC_NOT_FOUND", f"No exemplars for metric: {metric_key}")
        
        return {"metric": metric_key, **data["metrics"][metric_key]}
    except FileNotFoundError:
        return _error(404, "EXEMPLARS_NOT_FOUND", "Exemplar data not configured")
    except Exception as e:
        logger.error(f"Error loading exemplar for {metric_key}: {e}")
        return _error(500, "EXEMPLAR_ERROR", str(e))


# ---------------------------------------------------------------------------
# AI Summaries — profile-specific AI-generated analysis summaries
# ---------------------------------------------------------------------------

@router.get("/speeches/{speech_id}/summary/{profile}", tags=["Summaries"])
async def get_speech_summary(
    speech_id: int,
    profile: str,
    regenerate: bool = Query(False, description="Force regenerate the summary"),
    auth: dict = Depends(flexible_auth),
):
    """
    Get AI-generated summary for a speech with a specific profile.
    Generates on first request, returns cached version on subsequent requests.
    """
    from ai_summary import get_summary, generate_and_save_summary
    
    db = pipeline_runner.get_db()
    speech = db.get_speech(speech_id)
    if not speech:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found")
    
    # Check ownership
    user_id = auth.get("user_id")
    if user_id and speech.get("user_id") and speech["user_id"] != user_id:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found")
    
    # Valid profiles
    valid_profiles = ["general", "oratory", "pronunciation", "presentation", 
                     "reading_fluency", "clinical"]
    if profile not in valid_profiles:
        return _error(400, "INVALID_PROFILE", f"Profile must be one of: {', '.join(valid_profiles)}")
    
    # Try to get existing summary
    if not regenerate:
        summary = get_summary(speech_id, profile, config.DB_PATH)
        if summary:
            return {
                "speech_id": speech_id,
                "profile": profile,
                "summary": summary,
                "cached": True,
            }
    
    # Generate new summary
    summary = generate_and_save_summary(speech_id, profile, config.DB_PATH)
    if summary:
        return {
            "speech_id": speech_id,
            "profile": profile,
            "summary": summary,
            "cached": False,
        }
    else:
        return _error(500, "SUMMARY_FAILED", "Failed to generate summary")


@router.get("/speeches/{speech_id}/summaries", tags=["Summaries"])
async def get_all_speech_summaries(
    speech_id: int,
    auth: dict = Depends(flexible_auth),
):
    """
    Get all available AI summaries for a speech (one per profile).
    """
    import sqlite3
    
    db = pipeline_runner.get_db()
    speech = db.get_speech(speech_id)
    if not speech:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found")
    
    # Check ownership
    user_id = auth.get("user_id")
    if user_id and speech.get("user_id") and speech["user_id"] != user_id:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found")
    
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT profile, summary, created_at
        FROM speech_summaries
        WHERE speech_id = ?
        ORDER BY profile
    """, (speech_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    summaries = {row["profile"]: {
        "summary": row["summary"],
        "created_at": row["created_at"],
    } for row in rows}
    
    return {
        "speech_id": speech_id,
        "summaries": summaries,
    }


@router.post("/speeches/{speech_id}/summaries/generate", tags=["Summaries"])
async def generate_all_summaries(
    speech_id: int,
    profiles: Optional[List[str]] = Body(None, description="Specific profiles to generate, or all if not specified"),
    auth: dict = Depends(flexible_auth),
):
    """
    Generate AI summaries for a speech for specified profiles (or all profiles).
    This runs in the background and may take 30-60 seconds.
    """
    from ai_summary import generate_and_save_summary, PROFILE_METRICS
    
    db = pipeline_runner.get_db()
    speech = db.get_speech(speech_id)
    if not speech:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found")
    
    # Check ownership
    user_id = auth.get("user_id")
    if user_id and speech.get("user_id") and speech["user_id"] != user_id:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found")
    
    # Determine which profiles to generate
    if profiles:
        target_profiles = [p for p in profiles if p in PROFILE_METRICS]
    else:
        target_profiles = list(PROFILE_METRICS.keys())
    
    # Generate summaries (this is synchronous but fast enough)
    generated = {}
    for profile in target_profiles:
        summary = generate_and_save_summary(speech_id, profile, config.DB_PATH)
        if summary:
            generated[profile] = True
        else:
            generated[profile] = False
    
    return {
        "speech_id": speech_id,
        "generated": generated,
        "message": f"Generated {sum(generated.values())} summaries",
    }



# ---------------------------------------------------------------------------
# Practice Stats — User practice habits and streaks
# ---------------------------------------------------------------------------

@router.get("/practice/stats", tags=["Practice"])
async def get_practice_stats(
    auth: dict = Depends(flexible_auth),
):
    """
    Get user's practice statistics including streaks, activity, and milestones.
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required")
    
    db = pipeline_runner.get_db()
    
    try:
        from datetime import datetime, timedelta
        import json as json_module
        
        # Get all recordings with dates
        cursor = db.conn.execute("""
            SELECT 
                DATE(created_at) as recording_date,
                COUNT(*) as count,
                SUM(duration_sec) as total_duration
            FROM speeches 
            WHERE user_id = ?
            GROUP BY DATE(created_at)
            ORDER BY recording_date DESC
        """, (user_id,))
        daily_records = cursor.fetchall()
        
        # Calculate streaks
        today = datetime.now().date()
        current_streak = 0
        longest_streak = 0
        temp_streak = 0
        last_date = None
        
        recording_dates = set()
        for row in daily_records:
            recording_dates.add(row[0])
        
        # Calculate current streak (consecutive days including today or yesterday)
        check_date = today
        while check_date.isoformat() in recording_dates:
            current_streak += 1
            check_date -= timedelta(days=1)
        
        # If no recording today, check if yesterday starts a streak
        if current_streak == 0:
            check_date = today - timedelta(days=1)
            while check_date.isoformat() in recording_dates:
                current_streak += 1
                check_date -= timedelta(days=1)
        
        # Calculate longest streak
        sorted_dates = sorted(recording_dates)
        for i, date_str in enumerate(sorted_dates):
            date = datetime.fromisoformat(date_str).date()
            if last_date is None or (date - last_date).days == 1:
                temp_streak += 1
            else:
                temp_streak = 1
            longest_streak = max(longest_streak, temp_streak)
            last_date = date
        
        # This week stats
        week_start = today - timedelta(days=today.weekday())
        cursor = db.conn.execute("""
            SELECT COUNT(*), COALESCE(SUM(duration_sec), 0)
            FROM speeches 
            WHERE user_id = ? AND DATE(created_at) >= ?
        """, (user_id, week_start.isoformat()))
        week_row = cursor.fetchone()
        this_week_count = week_row[0]
        this_week_minutes = round(week_row[1] / 60, 1) if week_row[1] else 0
        
        # Last week for comparison
        last_week_start = week_start - timedelta(days=7)
        cursor = db.conn.execute("""
            SELECT COUNT(*) FROM speeches 
            WHERE user_id = ? AND DATE(created_at) >= ? AND DATE(created_at) < ?
        """, (user_id, last_week_start.isoformat(), week_start.isoformat()))
        last_week_count = cursor.fetchone()[0]
        
        # Total stats
        cursor = db.conn.execute("""
            SELECT COUNT(*), COALESCE(SUM(duration_sec), 0), AVG(duration_sec)
            FROM speeches WHERE user_id = ?
        """, (user_id,))
        total_row = cursor.fetchone()
        total_recordings = total_row[0]
        total_minutes = round(total_row[1] / 60, 1) if total_row[1] else 0
        avg_duration = round(total_row[2], 1) if total_row[2] else 0
        
        # Most active day of week
        cursor = db.conn.execute("""
            SELECT strftime('%w', created_at) as dow, COUNT(*) as cnt
            FROM speeches WHERE user_id = ?
            GROUP BY dow ORDER BY cnt DESC LIMIT 1
        """, (user_id,))
        dow_row = cursor.fetchone()
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        most_active_day = day_names[int(dow_row[0])] if dow_row else None
        
        # Activity calendar (last 90 days)
        ninety_days_ago = today - timedelta(days=90)
        cursor = db.conn.execute("""
            SELECT DATE(created_at) as d, COUNT(*) as c
            FROM speeches 
            WHERE user_id = ? AND DATE(created_at) >= ?
            GROUP BY DATE(created_at)
        """, (user_id, ninety_days_ago.isoformat()))
        activity_calendar = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Milestones
        milestones = [
            {"count": 10, "label": "Getting Started", "achieved": total_recordings >= 10},
            {"count": 50, "label": "Committed", "achieved": total_recordings >= 50},
            {"count": 100, "label": "Century", "achieved": total_recordings >= 100},
            {"count": 500, "label": "Dedicated", "achieved": total_recordings >= 500},
            {"count": 1000, "label": "Master", "achieved": total_recordings >= 1000},
        ]
        next_milestone = next((m for m in milestones if not m["achieved"]), None)
        
        return {
            "current_streak": current_streak,
            "longest_streak": longest_streak,
            "this_week": {
                "recordings": this_week_count,
                "minutes": this_week_minutes,
                "vs_last_week": this_week_count - last_week_count,
            },
            "total_recordings": total_recordings,
            "total_minutes": total_minutes,
            "avg_duration_sec": avg_duration,
            "most_active_day": most_active_day,
            "activity_calendar": activity_calendar,
            "milestones": milestones,
            "next_milestone": next_milestone,
        }
        
    except Exception as e:
        logger.exception(f"Failed to get practice stats: {e}")
        return _error(500, "ERROR", str(e))


@router.get("/practice/stats/pronunciation", tags=["Practice"])
async def get_pronunciation_stats(
    auth: dict = Depends(flexible_auth),
    days: int = 30,
):
    """
    Get pronunciation-specific progress statistics.
    Returns clarity trends, problem sounds breakdown, and mastery progress.
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required")
    
    db = pipeline_runner.get_db()
    
    try:
        from datetime import datetime, timedelta
        import json as json_module
        import re
        
        today = datetime.now().date()
        start_date = today - timedelta(days=days)
        
        # Get all pronunciation recordings with analysis data
        cursor = db.conn.execute("""
            SELECT 
                DATE(created_at) as recording_date,
                pronunciation_mean_confidence,
                pronunciation_problem_ratio,
                pronunciation_words_analyzed,
                word_analysis_json
            FROM speeches 
            WHERE user_id = ? 
              AND profile = 'pronunciation'
              AND pronunciation_mean_confidence IS NOT NULL
              AND DATE(created_at) >= ?
            ORDER BY created_at DESC
        """, (user_id, start_date.isoformat()))
        rows = cursor.fetchall()
        
        if not rows:
            return {
                "has_data": False,
                "message": "No pronunciation practice data yet. Start practicing to see your progress!",
                "clarity_trend": [],
                "problem_sounds": {},
                "words_practiced": 0,
                "words_mastered": 0,
                "overall_clarity": None,
                "sessions_count": 0,
            }
        
        # Calculate daily clarity averages for trend
        daily_clarity = {}
        total_words = 0
        total_mastered = 0  # confidence >= 0.85
        all_word_analyses = []
        
        for row in rows:
            date_str = row[0]
            clarity = row[1]
            words_count = row[3] or 0
            word_json = row[4]
            
            # Aggregate daily clarity
            if date_str not in daily_clarity:
                daily_clarity[date_str] = []
            daily_clarity[date_str].append(clarity)
            
            total_words += words_count
            
            # Parse word analysis for sound patterns
            if word_json:
                try:
                    words = json_module.loads(word_json)
                    all_word_analyses.extend(words)
                    # Count mastered words (high confidence)
                    for w in words:
                        conf = w.get('confidence', 0) or w.get('pronunciation_score', 0)
                        if conf >= 0.85:
                            total_mastered += 1
                except:
                    pass
        
        # Build clarity trend (daily averages)
        clarity_trend = []
        for date_str in sorted(daily_clarity.keys()):
            values = daily_clarity[date_str]
            avg = sum(values) / len(values) if values else 0
            clarity_trend.append({
                "date": date_str,
                "clarity": round(avg * 100, 1),
                "sessions": len(values),
            })
        
        # Analyze problem sounds from word data
        sound_patterns = {
            'TH': r'th',
            'R': r'r',
            'L': r'l',
            'S/SH': r's[h]?',
            'CH': r'ch',
            'V/W': r'[vw]',
            'NG': r'ng',
            'Clusters': r'[bcdfghjklmnpqrstvwxyz]{3,}',
        }
        
        sound_stats = {sound: {'total': 0, 'problems': 0} for sound in sound_patterns}
        
        for word_data in all_word_analyses:
            word = (word_data.get('word') or '').lower()
            is_problem = word_data.get('is_problem', False)
            conf = word_data.get('confidence', 0) or word_data.get('pronunciation_score', 0)
            
            # Check which sounds this word contains
            for sound, pattern in sound_patterns.items():
                if re.search(pattern, word, re.IGNORECASE):
                    sound_stats[sound]['total'] += 1
                    if is_problem or conf < 0.7:
                        sound_stats[sound]['problems'] += 1
        
        # Calculate mastery percentage for each sound
        problem_sounds = {}
        for sound, stats in sound_stats.items():
            if stats['total'] >= 5:  # Only include if enough samples
                mastery = 1 - (stats['problems'] / stats['total']) if stats['total'] > 0 else 0
                problem_sounds[sound] = {
                    'mastery': round(mastery * 100, 1),
                    'total_words': stats['total'],
                    'problem_words': stats['problems'],
                }
        
        # Sort by mastery (lowest first = needs most work)
        problem_sounds = dict(sorted(problem_sounds.items(), key=lambda x: x[1]['mastery']))
        
        # Calculate overall stats
        overall_clarity = sum(r[1] for r in rows) / len(rows) if rows else 0
        
        # Identify most improved (compare first half vs second half of period)
        most_improved = None
        if len(clarity_trend) >= 4:
            mid = len(clarity_trend) // 2
            first_half_avg = sum(d['clarity'] for d in clarity_trend[:mid]) / mid
            second_half_avg = sum(d['clarity'] for d in clarity_trend[mid:]) / (len(clarity_trend) - mid)
            improvement = second_half_avg - first_half_avg
            if improvement > 2:  # At least 2% improvement
                most_improved = {
                    'from': round(first_half_avg, 1),
                    'to': round(second_half_avg, 1),
                    'change': round(improvement, 1),
                }
        
        # Find sounds that need most work (lowest mastery with enough samples)
        needs_work = []
        for sound, stats in list(problem_sounds.items())[:3]:  # Top 3 problem areas
            if stats['mastery'] < 80:
                needs_work.append({
                    'sound': sound,
                    'mastery': stats['mastery'],
                    'tip': _get_sound_tip(sound),
                })
        
        return {
            "has_data": True,
            "clarity_trend": clarity_trend,
            "problem_sounds": problem_sounds,
            "words_practiced": total_words,
            "words_mastered": total_mastered,
            "overall_clarity": round(overall_clarity * 100, 1),
            "sessions_count": len(rows),
            "most_improved": most_improved,
            "needs_work": needs_work,
            "period_days": days,
        }
        
    except Exception as e:
        logger.exception(f"Failed to get pronunciation stats: {e}")
        return _error(500, "ERROR", str(e))


def _get_sound_tip(sound: str) -> str:
    """Get a practice tip for a specific sound."""
    tips = {
        'TH': "Place tongue between teeth, blow air gently. Practice: think, this, that",
        'R': "Curl tongue back without touching roof. Practice: red, right, car",
        'L': "Touch tongue tip behind upper teeth. Practice: light, all, feel",
        'S/SH': "S = teeth together, SH = lips rounded. Practice: see vs she",
        'CH': "Start with 't' then release into 'sh'. Practice: church, cheese",
        'V/W': "V = teeth on lip, W = rounded lips. Practice: vine vs wine",
        'NG': "Back of tongue touches soft palate. Practice: sing, ring, long",
        'Clusters': "Slow down and hit each consonant. Practice: strengths, twelfths",
    }
    return tips.get(sound, "Practice slowly and deliberately.")


@router.get("/practice/stats/oratory", tags=["Practice"])
async def get_oratory_stats(
    auth: dict = Depends(flexible_auth),
    days: int = 30,
):
    """
    Get oratory-specific progress statistics.
    Returns persuasion trends, metric averages, device usage, and great speech comparison.
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required")
    
    db = pipeline_runner.get_db()
    
    try:
        from datetime import datetime, timedelta
        import json as json_module
        
        today = datetime.now().date()
        start_date = today - timedelta(days=days)
        
        # Get all oratory recordings with analysis data
        cursor = db.conn.execute("""
            SELECT 
                s.id,
                DATE(s.created_at) as recording_date,
                s.duration_sec,
                s.cached_score,
                a.words_per_minute,
                a.pitch_variation,
                a.pitch_range_semitones,
                a.hnr_mean,
                a.repetition_score,
                a.discourse_connectedness,
                a.analysis_json
            FROM speeches s
            LEFT JOIN analyses a ON s.id = a.speech_id
            WHERE s.user_id = ? 
              AND s.profile = 'oratory'
              AND DATE(s.created_at) >= ?
            ORDER BY s.created_at DESC
        """, (user_id, start_date.isoformat()))
        rows = cursor.fetchall()
        
        if not rows:
            return {
                "has_data": False,
                "message": "No oratory recordings yet. Start practicing to see your progress!",
                "persuasion_trend": [],
                "metric_averages": {},
                "device_stats": {},
                "speech_count": 0,
            }
        
        # Calculate daily persuasion score averages for trend
        daily_scores = {}
        total_minutes = 0
        
        # Metric accumulators
        metric_sums = {
            'pitch_variation': 0,
            'repetition_score': 0,
            'pitch_range': 0,
            'wpm': 0,
            'hnr': 0,
            'discourse': 0,
        }
        metric_counts = {k: 0 for k in metric_sums}
        
        # Device stats
        device_totals = {
            'anaphora': 0,
            'epistrophe': 0,
            'repetition': 0,
        }
        
        for row in rows:
            date_str = row[1]
            duration = row[2] or 0
            score = row[3] or 0
            wpm = row[4]
            pitch_var = row[5]
            pitch_range = row[6]
            hnr = row[7]
            rep_score = row[8]
            discourse = row[9]
            analysis_json = row[10]
            
            total_minutes += duration / 60
            
            # Aggregate daily scores
            if date_str not in daily_scores:
                daily_scores[date_str] = []
            daily_scores[date_str].append(score)
            
            # Accumulate metrics
            if wpm: 
                metric_sums['wpm'] += wpm
                metric_counts['wpm'] += 1
            if pitch_var:
                metric_sums['pitch_variation'] += pitch_var
                metric_counts['pitch_variation'] += 1
            if pitch_range:
                metric_sums['pitch_range'] += pitch_range
                metric_counts['pitch_range'] += 1
            if hnr:
                metric_sums['hnr'] += hnr
                metric_counts['hnr'] += 1
            if rep_score:
                metric_sums['repetition_score'] += rep_score
                metric_counts['repetition_score'] += 1
            if discourse:
                metric_sums['discourse'] += discourse
                metric_counts['discourse'] += 1
            
            # Extract device usage from analysis JSON
            if analysis_json:
                try:
                    analysis = json_module.loads(analysis_json)
                    rhetorical = analysis.get('advanced_text_analysis', {}).get('rhetorical', {})
                    if rhetorical:
                        anaphora = rhetorical.get('anaphora_examples', [])
                        epistrophe = rhetorical.get('epistrophe_examples', [])
                        device_totals['anaphora'] += len(anaphora) if isinstance(anaphora, list) else 0
                        device_totals['epistrophe'] += len(epistrophe) if isinstance(epistrophe, list) else 0
                        device_totals['repetition'] += 1 if rhetorical.get('uses_repetition') else 0
                except:
                    pass
        
        # Build persuasion trend (daily averages)
        persuasion_trend = []
        for date_str in sorted(daily_scores.keys()):
            values = daily_scores[date_str]
            avg = sum(values) / len(values) if values else 0
            persuasion_trend.append({
                "date": date_str,
                "score": round(avg, 1),
                "count": len(values),
            })
        
        # Calculate metric averages
        metric_averages = {}
        for metric, total in metric_sums.items():
            count = metric_counts[metric]
            if count > 0:
                metric_averages[metric] = round(total / count, 1)
        
        # Remove zero device counts
        device_stats = {k: v for k, v in device_totals.items() if v > 0}
        
        # Calculate overall averages
        all_scores = [s for scores in daily_scores.values() for s in scores]
        avg_persuasion = sum(all_scores) / len(all_scores) if all_scores else 0
        
        # Great speech comparison
        # Compare user's WPM to famous speakers
        user_wpm = metric_averages.get('wpm', 120)
        great_speeches = {
            "MLK - I Have a Dream": 107,
            "JFK - Inaugural": 127,
            "Churchill - We Shall Fight": 128,
            "Lincoln - Gettysburg": 112,
            "FDR - Day of Infamy": 118,
        }
        
        closest_orator = None
        min_diff = float('inf')
        for name, wpm in great_speeches.items():
            diff = abs(user_wpm - wpm)
            if diff < min_diff:
                min_diff = diff
                closest_orator = name
        
        # Similarity based on WPM match (simplified)
        similarity = max(0, 100 - min_diff * 3)
        
        return {
            "has_data": True,
            "persuasion_trend": persuasion_trend,
            "metric_averages": metric_averages,
            "device_stats": device_stats,
            "avg_persuasion_score": round(avg_persuasion, 1),
            "speech_count": len(rows),
            "total_minutes": round(total_minutes, 1),
            "great_speech_comparison": {
                "closest": closest_orator,
                "similarity": round(similarity, 0),
            },
            "period_days": days,
        }
        
    except Exception as e:
        logger.exception(f"Failed to get oratory stats: {e}")
        return _error(500, "ERROR", str(e))


@router.post("/reading/analyze", tags=["Reading"])
async def analyze_reading_practice(
    speech_id: int = Body(..., description="Speech ID to analyze"),
    reference_text: str = Body(..., description="The text the user was supposed to read"),
    auth: dict = Depends(flexible_auth),
):
    """
    Analyze a reading practice recording against the reference text.
    
    Returns:
    - Word-by-word accuracy (correct, substituted, inserted, deleted)
    - Fluency score and metrics
    - Problem words identification
    - Hesitation points
    - Focus areas for improvement
    """
    from reading_analysis import analyze_reading
    
    user_id = auth.get("user_id")
    key_data = auth.get("key_data")
    
    # Allow either user auth OR API key
    if not user_id and not key_data:
        return _error(401, "UNAUTHORIZED", "Authentication required")
    
    db = pipeline_runner.get_db()
    
    # Get speech and verify ownership
    speech = db.get_speech(speech_id)
    if not speech:
        return _error(404, "NOT_FOUND", f"Speech {speech_id} not found")
    
    # Check ownership (API keys can access any speech)
    if user_id and speech.get("user_id") != user_id:
        return _error(403, "FORBIDDEN", "Not authorized to access this speech")
    
    # Get transcription
    cursor = db.conn.execute(
        "SELECT text, segments FROM transcriptions WHERE speech_id = ?",
        (speech_id,)
    )
    row = cursor.fetchone()
    
    if not row:
        return _error(404, "NOT_FOUND", f"No transcription found for speech {speech_id}")
    
    transcription = row[0] or ""
    segments = []
    if row[1]:
        import json as json_module
        try:
            segments = json_module.loads(row[1])
        except:
            pass
    
    duration = speech.get("duration_sec", 0)
    
    # Get pitch data if available
    pitch_data = None
    cursor = db.conn.execute(
        "SELECT pitch_mean_hz, pitch_std_hz, pitch_range_hz FROM analyses WHERE speech_id = ?",
        (speech_id,)
    )
    audio_row = cursor.fetchone()
    if audio_row and audio_row[0] is not None:
        pitch_data = {
            'pitch_mean': audio_row[0],
            'pitch_std': audio_row[1],
            'pitch_range': audio_row[2],
        }
    
    # Run reading analysis
    try:
        result = analyze_reading(
            reference_text=reference_text,
            transcription=transcription,
            duration_seconds=duration,
            segments=segments,
            pitch_data=pitch_data,
        )
        
        return {
            "speech_id": speech_id,
            "reference_text": reference_text,
            "transcription": transcription,
            "duration_sec": duration,
            **result,
        }
    except Exception as e:
        logger.exception(f"Reading analysis failed: {e}")
        return _error(500, "ANALYSIS_ERROR", str(e))


@router.get("/reading/stats", tags=["Reading"])
async def get_reading_stats(
    auth: dict = Depends(flexible_auth),
    days: int = 30,
):
    """
    Get reading practice statistics for the user.
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required")
    
    db = pipeline_runner.get_db()
    
    try:
        from datetime import datetime, timedelta
        
        today = datetime.now().date()
        start_date = today - timedelta(days=days)
        
        # Get reading_fluency recordings
        cursor = db.conn.execute("""
            SELECT 
                COUNT(*) as count,
                SUM(duration_sec) as total_duration,
                AVG(cached_score) as avg_score
            FROM speeches 
            WHERE user_id = ? 
              AND profile = 'reading_fluency'
              AND DATE(created_at) >= ?
        """, (user_id, start_date.isoformat()))
        row = cursor.fetchone()
        
        return {
            "period_days": days,
            "recording_count": row[0] or 0,
            "total_minutes": round((row[1] or 0) / 60, 1),
            "avg_score": round(row[2] or 0, 1) if row[2] else None,
        }
    except Exception as e:
        logger.exception(f"Failed to get reading stats: {e}")
        return _error(500, "ERROR", str(e))


@router.get("/projects/{project_id}/score-progress", tags=["Projects"])
async def get_project_score_progress(
    project_id: int,
    auth: dict = Depends(flexible_auth),
):
    """
    Get score progress over time for a project.
    """
    user_id = auth.get("user_id")
    if not user_id:
        return _error(401, "UNAUTHORIZED", "User authentication required")
    
    db = pipeline_runner.get_db()
    
    # Verify project ownership
    cursor = db.conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?",
        (project_id, user_id)
    )
    project = cursor.fetchone()
    if not project:
        return _error(404, "NOT_FOUND", "Project not found")
    
    try:
        # Get scores over time
        cursor = db.conn.execute("""
            SELECT 
                s.id,
                s.title,
                s.cached_score,
                s.created_at,
                s.duration_sec
            FROM speeches s
            WHERE s.project_id = ? AND s.user_id = ?
            ORDER BY s.created_at ASC
        """, (project_id, user_id))
        
        recordings = []
        scores = []
        for row in cursor.fetchall():
            score = row[2]
            recordings.append({
                "id": row[0],
                "title": row[1],
                "score": score,
                "created_at": row[3],
                "duration_sec": row[4],
            })
            if score is not None:
                scores.append(score)
        
        # Calculate trend
        trend = "stable"
        rate_per_week = 0
        
        if len(scores) >= 2:
            # Simple linear regression
            n = len(scores)
            x = list(range(n))
            x_mean = sum(x) / n
            y_mean = sum(scores) / n
            
            numerator = sum((x[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            if denominator > 0:
                slope = numerator / denominator
                
                if slope > 0.5:
                    trend = "improving"
                elif slope < -0.5:
                    trend = "declining"
                
                # Estimate rate per week (assuming ~2 recordings per week)
                rate_per_week = round(slope * 2, 1)
        
        # Best recording
        best_recording = None
        if recordings:
            scored = [r for r in recordings if r["score"] is not None]
            if scored:
                best_recording = max(scored, key=lambda r: r["score"])
        
        # Average score
        avg_score = round(sum(scores) / len(scores), 1) if scores else None
        
        # Recent vs older comparison
        recent_avg = None
        improvement = None
        if len(scores) >= 6:
            recent = scores[-3:]
            older = scores[:3]
            recent_avg = round(sum(recent) / len(recent), 1)
            older_avg = round(sum(older) / len(older), 1)
            improvement = round(recent_avg - older_avg, 1)
        
        return {
            "project_id": project_id,
            "recordings": recordings,
            "total_count": len(recordings),
            "scored_count": len(scores),
            "avg_score": avg_score,
            "trend": trend,
            "rate_per_week": rate_per_week,
            "best_recording": best_recording,
            "recent_avg": recent_avg,
            "improvement": improvement,
        }
        
    except Exception as e:
        logger.exception(f"Failed to get project score progress: {e}")
        return _error(500, "ERROR", str(e))


# ========== Dementia Mode Endpoints ==========

@router.get("/dementia/stats", tags=["Dementia"])
async def get_dementia_stats(user: dict = Depends(flexible_auth)):
    """Get aggregated statistics for dementia assessments with daily best values and trends."""
    from datetime import datetime, timedelta
    from collections import defaultdict
    
    try:
        user_id = user.get("user_id") if user else None
        db = pipeline_runner.get_db()
        conn = db.conn
        cursor = conn.cursor()
        
        # Get all dementia recordings with dates
        if user_id:
            cursor.execute("""
                SELECT dementia_metrics, DATE(created_at) as day, created_at
                FROM speeches 
                WHERE user_id = ? AND profile = 'dementia' AND dementia_metrics IS NOT NULL
                ORDER BY created_at ASC
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT dementia_metrics, DATE(created_at) as day, created_at
                FROM speeches 
                WHERE profile = 'dementia' AND dementia_metrics IS NOT NULL
                ORDER BY created_at ASC
            """)
        
        rows = cursor.fetchall()
        
        if not rows:
            return {
                "total_assessments": 0,
                "daily_best": [],
                "trends": {"prepositions": None, "repetitions": None, "aphasic": None},
                "trend_summary": "No data yet",
            }
        
        # Group by day and get BEST values (highest preposition rate = better)
        daily_metrics = defaultdict(list)
        for row in rows:
            try:
                metrics = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                day = row[1]
                daily_metrics[day].append(metrics)
            except:
                continue
        
        # Calculate daily best values
        # For prepositions: higher is better (more normal speech)
        # For repetitions/aphasic: lower is better (fewer issues)
        daily_best = []
        for day in sorted(daily_metrics.keys()):
            day_data = daily_metrics[day]
            best_prep = max(m.get('prepositions_per_10_words', 0) for m in day_data)
            best_rep = min(m.get('repetition_count', 0) for m in day_data)  # Lower is better
            best_aph = min(m.get('aphasic_events', 0) for m in day_data)  # Lower is better
            
            daily_best.append({
                "date": day,
                "prepositions_per_10_words": round(best_prep, 2),
                "repetition_count": best_rep,
                "aphasic_events": best_aph,
            })
        
        # Calculate trends
        def calculate_trend(values, higher_is_better=True):
            if len(values) < 2:
                return {"change": 0, "direction": "stable", "percent": 0}
            
            # Compare last 7 days vs previous 7 days (or available data)
            half = len(values) // 2
            if half < 1:
                return {"change": 0, "direction": "stable", "percent": 0}
            
            recent = values[-half:] if half > 0 else values
            older = values[:half] if half > 0 else values
            
            recent_avg = sum(recent) / len(recent) if recent else 0
            older_avg = sum(older) / len(older) if older else 0
            
            if older_avg == 0:
                return {"change": 0, "direction": "stable", "percent": 0}
            
            change = recent_avg - older_avg
            percent = round((change / older_avg) * 100, 1)
            
            if higher_is_better:
                direction = "improving" if change > 0.1 else ("declining" if change < -0.1 else "stable")
            else:
                direction = "improving" if change < -0.1 else ("declining" if change > 0.1 else "stable")
            
            return {
                "change": round(change, 2),
                "direction": direction,
                "percent": percent,
                "recent_avg": round(recent_avg, 2),
                "older_avg": round(older_avg, 2),
            }
        
        prep_values = [d['prepositions_per_10_words'] for d in daily_best]
        rep_values = [d['repetition_count'] for d in daily_best]
        aph_values = [d['aphasic_events'] for d in daily_best]
        
        trends = {
            "prepositions": calculate_trend(prep_values, higher_is_better=True),
            "repetitions": calculate_trend(rep_values, higher_is_better=False),
            "aphasic": calculate_trend(aph_values, higher_is_better=False),
        }
        
        # Generate human-readable trend summary
        summaries = []
        if trends["prepositions"]["direction"] == "improving":
            summaries.append(f"Preposition usage improving {abs(trends['prepositions']['percent'])}%")
        elif trends["prepositions"]["direction"] == "declining":
            summaries.append(f"Preposition usage decreased {abs(trends['prepositions']['percent'])}%")
        
        if trends["repetitions"]["direction"] == "improving":
            summaries.append(f"Repetitions reduced {abs(trends['repetitions']['percent'])}%")
        elif trends["repetitions"]["direction"] == "declining":
            summaries.append(f"Repetitions increased {abs(trends['repetitions']['percent'])}%")
            
        if trends["aphasic"]["direction"] == "improving":
            summaries.append(f"Word-finding improved {abs(trends['aphasic']['percent'])}%")
        elif trends["aphasic"]["direction"] == "declining":
            summaries.append(f"Word-finding difficulties increased {abs(trends['aphasic']['percent'])}%")
        
        trend_summary = ". ".join(summaries) if summaries else "Metrics stable"
        
        return {
            "total_assessments": len(rows),
            "days_tracked": len(daily_best),
            "daily_best": daily_best,  # Time series for graphing
            "trends": trends,
            "trend_summary": trend_summary,
            "latest": daily_best[-1] if daily_best else None,
        }
        
    except Exception as e:
        logger.exception(f"Failed to get dementia stats: {e}")
        return _error(500, "ERROR", str(e))


@router.get("/dementia/metrics/aggregated", tags=["Dementia"])
async def get_dementia_metrics_aggregated(user: dict = Depends(flexible_auth)):
    """
    Get dementia metrics aggregated by time period for dashboard display.
    
    Returns today's average plus the last 5 periods (with data) for each granularity:
    - day: last 5 days with recordings
    - week: last 5 weeks with recordings  
    - month: last 5 months with recordings
    - year: last 5 years with recordings
    
    Each metric (repetitions, aphasia, prepositions) is returned separately
    so the UI can display them as individual cards.
    """
    from datetime import datetime, timedelta
    from collections import defaultdict
    
    try:
        user_id = user.get("user_id") if user else None
        db = pipeline_runner.get_db()
        conn = db.conn
        cursor = conn.cursor()
        
        # Get all dementia recordings
        if user_id:
            cursor.execute("""
                SELECT dementia_metrics, created_at
                FROM speeches 
                WHERE user_id = ? AND profile = 'dementia' AND dementia_metrics IS NOT NULL
                ORDER BY created_at DESC
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT dementia_metrics, created_at
                FROM speeches 
                WHERE profile = 'dementia' AND dementia_metrics IS NOT NULL
                ORDER BY created_at DESC
            """)
        
        rows = cursor.fetchall()
        
        if not rows:
            return {
                "has_data": False,
                "today": None,
                "metrics": {
                    "repetitions": {"today": None, "history": {"day": [], "week": [], "month": [], "year": []}},
                    "aphasia": {"today": None, "history": {"day": [], "week": [], "month": [], "year": []}},
                    "prepositions": {"today": None, "history": {"day": [], "week": [], "month": [], "year": []}},
                }
            }
        
        # Parse all metrics with timestamps
        all_data = []
        for row in rows:
            try:
                metrics = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                ts = datetime.fromisoformat(row[1].replace('Z', '+00:00')) if isinstance(row[1], str) else row[1]
                all_data.append({
                    "ts": ts,
                    "repetitions": metrics.get('repetition_count', 0),
                    "aphasia": metrics.get('aphasic_events', 0),
                    "prepositions": metrics.get('prepositions_per_10_words', 0),
                })
            except:
                continue
        
        if not all_data:
            return {"has_data": False, "today": None, "metrics": None}
        
        # Use Eastern timezone for "today" calculation (user's timezone)
        from zoneinfo import ZoneInfo
        eastern = ZoneInfo('America/New_York')
        now = datetime.now(eastern)
        today_str = now.strftime('%Y-%m-%d')
        
        # Convert all timestamps to Eastern for grouping
        for d in all_data:
            if d['ts'].tzinfo is None:
                # Assume UTC if no timezone
                d['ts'] = d['ts'].replace(tzinfo=ZoneInfo('UTC'))
            d['ts'] = d['ts'].astimezone(eastern)
        
        # Group by different time periods
        def group_by_period(data, period_fn):
            """Group data by period function and return averages."""
            grouped = defaultdict(list)
            for d in data:
                period_key = period_fn(d['ts'])
                grouped[period_key].append(d)
            return grouped
        
        def avg_metrics(items):
            """Calculate average for each metric from a list of items."""
            if not items:
                return None
            return {
                "repetitions": round(sum(i['repetitions'] for i in items) / len(items), 2),
                "aphasia": round(sum(i['aphasia'] for i in items) / len(items), 2),
                "prepositions": round(sum(i['prepositions'] for i in items) / len(items), 2),
                "count": len(items),
            }
        
        # Period functions
        def day_key(ts): return ts.strftime('%Y-%m-%d')
        def week_key(ts): return ts.strftime('%Y-W%W')  # ISO week
        def month_key(ts): return ts.strftime('%Y-%m')
        def year_key(ts): return ts.strftime('%Y')
        
        # Group data
        by_day = group_by_period(all_data, day_key)
        by_week = group_by_period(all_data, week_key)
        by_month = group_by_period(all_data, month_key)
        by_year = group_by_period(all_data, year_key)
        
        # Get today's average
        today_data = by_day.get(today_str, [])
        today_avg = avg_metrics(today_data) if today_data else None
        
        # Build sorted period lists (last 5 with data, most recent first for storage, but we'll reverse for display)
        def get_last_n_periods(grouped, n=5):
            """Get last N periods with data, sorted chronologically."""
            sorted_keys = sorted(grouped.keys(), reverse=True)[:n]
            result = []
            for key in reversed(sorted_keys):  # Reverse to chronological order
                avg = avg_metrics(grouped[key])
                if avg:
                    result.append({"period": key, **avg})
            return result
        
        day_history = get_last_n_periods(by_day, 5)
        week_history = get_last_n_periods(by_week, 5)
        month_history = get_last_n_periods(by_month, 5)
        year_history = get_last_n_periods(by_year, 5)
        
        # Build per-metric response (what the UI cards need)
        def extract_metric_history(history_list, metric_key):
            """Extract single metric from aggregated history."""
            return [{"period": h["period"], "value": h[metric_key], "count": h["count"]} for h in history_list]
        
        metrics = {
            "repetitions": {
                "today": today_avg["repetitions"] if today_avg else None,
                "label": "Repetitions",
                "unit": "count",
                "lower_is_better": True,
                "history": {
                    "day": extract_metric_history(day_history, "repetitions"),
                    "week": extract_metric_history(week_history, "repetitions"),
                    "month": extract_metric_history(month_history, "repetitions"),
                    "year": extract_metric_history(year_history, "repetitions"),
                }
            },
            "aphasia": {
                "today": today_avg["aphasia"] if today_avg else None,
                "label": "Word-Finding",
                "unit": "events",
                "lower_is_better": True,
                "history": {
                    "day": extract_metric_history(day_history, "aphasia"),
                    "week": extract_metric_history(week_history, "aphasia"),
                    "month": extract_metric_history(month_history, "aphasia"),
                    "year": extract_metric_history(year_history, "aphasia"),
                }
            },
            "prepositions": {
                "today": today_avg["prepositions"] if today_avg else None,
                "label": "Prepositions",
                "unit": "/10 words",
                "lower_is_better": False,  # Higher is better for prepositions
                "history": {
                    "day": extract_metric_history(day_history, "prepositions"),
                    "week": extract_metric_history(week_history, "prepositions"),
                    "month": extract_metric_history(month_history, "prepositions"),
                    "year": extract_metric_history(year_history, "prepositions"),
                }
            },
        }
        
        return {
            "has_data": True,
            "today": today_avg,
            "total_recordings": len(all_data),
            "metrics": metrics,
        }
        
    except Exception as e:
        logger.exception(f"Failed to get aggregated dementia metrics: {e}")
        return _error(500, "ERROR", str(e))


@router.get("/dementia/recordings", tags=["Dementia"])
async def get_dementia_recordings(
    limit: int = Query(100, ge=1, le=500),
    user: dict = Depends(flexible_auth)
):
    """Get all dementia assessment recordings."""
    try:
        user_id = user.get("user_id") if user else None
        db = pipeline_runner.get_db()
        conn = db.conn
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute("""
                SELECT s.id, s.title, s.created_at, s.dementia_metrics, a.result
                FROM speeches s
                LEFT JOIN analyses a ON a.speech_id = s.id
                WHERE s.user_id = ? AND s.profile = 'dementia'
                ORDER BY s.created_at DESC
                LIMIT ?
            """, (user_id, limit))
        else:
            cursor.execute("""
                SELECT s.id, s.title, s.created_at, s.dementia_metrics, a.result
                FROM speeches s
                LEFT JOIN analyses a ON a.speech_id = s.id
                WHERE s.profile = 'dementia'
                ORDER BY s.created_at DESC
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        recordings = []
        
        for row in rows:
            dementia_metrics = None
            if row[3]:
                try:
                    dementia_metrics = json.loads(row[3]) if isinstance(row[3], str) else row[3]
                except:
                    pass
            
            # Extract transcript from analysis result
            transcript = ""
            if row[4]:
                try:
                    result_data = json.loads(row[4]) if isinstance(row[4], str) else row[4]
                    transcript = result_data.get("transcript", "")
                except:
                    pass
            
            recordings.append({
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "dementia_metrics": dementia_metrics,
                "transcript": transcript[:200] + "..." if transcript and len(transcript) > 200 else transcript,
            })
        
        return {"recordings": recordings}
        
    except Exception as e:
        logger.exception(f"Failed to get dementia recordings: {e}")
        return _error(500, "ERROR", str(e))


@router.get("/dementia/recordings/{speech_id}/flagged-segments", tags=["Dementia"])
async def get_dementia_flagged_segments(speech_id: int, user: dict = Depends(flexible_auth)):
    """
    Get flagged segments for caregiver review.
    
    Returns segments that triggered aphasic or repetition flags, with:
    - Transcript text
    - Audio start/end times for playback
    - Detection stats (what caused the flag)
    - Suggested categories for annotation
    """
    from .dementia_analysis import detect_aphasic_events
    
    try:
        user_id = user.get("user_id") if user else None
        db = pipeline_runner.get_db()
        conn = db.conn
        cursor = conn.cursor()
        
        # Get speech and verify ownership
        if user_id:
            cursor.execute("""
                SELECT s.id, s.title, t.segments, t.text
                FROM speeches s
                JOIN transcriptions t ON t.speech_id = s.id
                WHERE s.id = ? AND s.user_id = ? AND s.profile = 'dementia'
            """, (speech_id, user_id))
        else:
            cursor.execute("""
                SELECT s.id, s.title, t.segments, t.text
                FROM speeches s
                JOIN transcriptions t ON t.speech_id = s.id
                WHERE s.id = ? AND s.profile = 'dementia'
            """, (speech_id,))
        
        row = cursor.fetchone()
        if not row:
            return _error(404, "NOT_FOUND", "Recording not found")
        
        speech_id = row[0]
        title = row[1]
        segments_json = row[2]
        full_transcript = row[3]
        
        segments = json.loads(segments_json) if segments_json else []
        
        flagged_segments = []
        
        for i, seg in enumerate(segments):
            seg_text = seg.get('text', '')
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            
            # Run aphasic detection on this segment
            aphasic_result = detect_aphasic_events(seg_text)
            events = aphasic_result.get('events', [])
            confidence = aphasic_result.get('confidence_score', 0)
            
            # Flag segment if it has events or high confidence
            if events or confidence > 0.3:
                # Categorize the flags
                flag_reasons = []
                for event in events:
                    flag_reasons.append({
                        'type': event.get('type'),
                        'marker': event.get('marker', event.get('phrase', '')),
                        'confidence': event.get('confidence', 0),
                    })
                
                flagged_segments.append({
                    'segment_index': i,
                    'text': seg_text,
                    'start_time': seg_start,
                    'end_time': seg_end,
                    'duration': seg_end - seg_start,
                    'flags': flag_reasons,
                    'overall_confidence': confidence,
                    'event_count': len(events),
                    # Suggested categories for reviewer
                    'suggested_categories': list(set(e.get('type') for e in events)),
                })
        
        # Get audio URL for playback
        audio_url = f"/v1/audio/{speech_id}"
        
        return {
            "speech_id": speech_id,
            "title": title,
            "audio_url": audio_url,
            "total_segments": len(segments),
            "flagged_count": len(flagged_segments),
            "flagged_segments": flagged_segments,
            "categories": [
                {"id": "hesitation", "label": "Hesitation (um, uh)", "color": "#FFC107"},
                {"id": "word_search", "label": "Word-finding difficulty", "color": "#FF5722"},
                {"id": "circumlocution", "label": "Circumlocution (talking around)", "color": "#9C27B0"},
                {"id": "repetition", "label": "Repetition", "color": "#2196F3"},
                {"id": "incomplete", "label": "Incomplete thought", "color": "#607D8B"},
                {"id": "false_positive", "label": "Not an issue (false positive)", "color": "#4CAF50"},
            ],
        }
        
    except Exception as e:
        logger.exception(f"Failed to get flagged segments: {e}")
        return _error(500, "ERROR", str(e))


@router.post("/dementia/recordings/{speech_id}/annotations", tags=["Dementia"])
async def save_dementia_annotations(
    speech_id: int,
    annotations: List[Dict] = Body(..., description="List of segment annotations"),
    user: dict = Depends(flexible_auth)
):
    """
    Save caregiver annotations for flagged segments.
    
    Each annotation should have:
    - segment_index: int
    - categories: list of category IDs the reviewer confirmed
    - notes: optional string
    - is_false_positive: bool
    """
    try:
        user_id = user.get("user_id") if user else None
        db = pipeline_runner.get_db()
        conn = db.conn
        cursor = conn.cursor()
        
        # Verify speech exists and user has access
        if user_id:
            cursor.execute(
                "SELECT id FROM speeches WHERE id = ? AND user_id = ?",
                (speech_id, user_id)
            )
        else:
            cursor.execute("SELECT id FROM speeches WHERE id = ?", (speech_id,))
        
        if not cursor.fetchone():
            return _error(404, "NOT_FOUND", "Recording not found")
        
        # Create annotations table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dementia_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speech_id INTEGER NOT NULL,
                annotator_user_id INTEGER,
                segment_index INTEGER NOT NULL,
                categories TEXT,
                notes TEXT,
                is_false_positive BOOLEAN DEFAULT 0,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                FOREIGN KEY (speech_id) REFERENCES speeches(id)
            )
        """)
        
        # Save annotations
        saved_count = 0
        for ann in annotations:
            cursor.execute("""
                INSERT INTO dementia_annotations 
                (speech_id, annotator_user_id, segment_index, categories, notes, is_false_positive)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                speech_id,
                user_id,
                ann.get('segment_index'),
                json.dumps(ann.get('categories', [])),
                ann.get('notes'),
                ann.get('is_false_positive', False),
            ))
            saved_count += 1
        
        conn.commit()
        
        return {
            "success": True,
            "saved_count": saved_count,
            "message": f"Saved {saved_count} annotations. Thank you for helping improve detection!"
        }
        
    except Exception as e:
        logger.exception(f"Failed to save annotations: {e}")
        return _error(500, "ERROR", str(e))


@router.get("/dementia/recordings/{speech_id}", tags=["Dementia"])
async def get_dementia_recording(speech_id: int, user: dict = Depends(flexible_auth)):
    """Get a single dementia assessment recording with full details and transcript highlights."""
    from .dementia_analysis import detect_aphasic_events, detect_repetitions
    
    try:
        user_id = user.get("user_id") if user else None
        db = pipeline_runner.get_db()
        conn = db.conn
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute("""
                SELECT s.id, s.title, s.created_at, s.dementia_metrics, a.result
                FROM speeches s
                LEFT JOIN analyses a ON a.speech_id = s.id
                WHERE s.id = ? AND s.user_id = ? AND s.profile = 'dementia'
            """, (speech_id, user_id))
        else:
            cursor.execute("""
                SELECT s.id, s.title, s.created_at, s.dementia_metrics, a.result
                FROM speeches s
                LEFT JOIN analyses a ON a.speech_id = s.id
                WHERE s.id = ? AND s.profile = 'dementia'
            """, (speech_id,))
        
        row = cursor.fetchone()
        
        if not row:
            return _error(404, "NOT_FOUND", "Recording not found")
        
        dementia_metrics = None
        if row[3]:
            try:
                dementia_metrics = json.loads(row[3]) if isinstance(row[3], str) else row[3]
            except:
                pass
        
        # Extract transcript and diarization from analysis result
        transcript = ""
        diarization = []
        if row[4]:
            try:
                result_data = json.loads(row[4]) if isinstance(row[4], str) else row[4]
                transcript = result_data.get("transcript", "")
                diarization = result_data.get("diarization", [])
            except:
                pass
        
        # Re-run aphasic event detection to get highlights
        transcript_highlights = []
        if transcript:
            aphasic_result = detect_aphasic_events(transcript)
            transcript_highlights = aphasic_result.get('highlights', [])
        
        # Extract speakers from diarization
        speakers = []
        if diarization:
            try:
                speakers = list(set(seg.get('speaker', '') for seg in diarization if seg.get('speaker')))
            except:
                pass
        
        return {
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "dementia_metrics": dementia_metrics,
            "transcript": transcript,
            "transcript_highlights": transcript_highlights,  # For highlighting affected parts
            "speakers": speakers,
        }
        
    except Exception as e:
        logger.exception(f"Failed to get dementia recording: {e}")
        return _error(500, "ERROR", str(e))
