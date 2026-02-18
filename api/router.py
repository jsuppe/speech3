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
    preset: Optional[str] = Query(None, description="Named preset: quick, text_only, audio_only, full, clinical, coaching"),
    quality_gate: Optional[str] = Query(None, description="Quality gate mode: warn (default), block, skip"),
    language: Optional[str] = Query(None, description="Language code for Whisper (e.g. 'en', 'es'). Default: auto-detect."),
    speaker: Optional[str] = Query(None, description="Speaker name (used for diarization when single speaker detected)"),
    title: Optional[str] = Query(None, description="Title for the speech"),
    profile: Optional[str] = Query("general", description="Scoring profile: general, oratory, reading_fluency, presentation, clinical"),
    auth: dict = Depends(flexible_auth),
):
    """
    Analyze an uploaded audio file through the speech pipeline.

    Default: synchronous (blocks until complete, returns full result).
    With mode=async: returns immediately with a job_id for polling.

    Use `modules` or `preset` to select which analysis modules run.
    If both are provided, `preset` takes precedence.
    """
    key_data = auth.get("key_data")
    key_id = key_data["key_id"] if key_data else "user"
    user_id = auth.get("user_id")
    
    # Generate request ID for tracing
    request_id = str(uuid.uuid4())[:8]
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

        # Record audio minutes in rate limiter and usage tracker
        audio_minutes = duration / 60.0
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
            )
        except pipeline_runner.QualityGateError as e:
            return _error(422, "QUALITY_GATE_BLOCKED", str(e))
        except Exception as e:
            logger.exception("Pipeline error")
            return _error(500, "PIPELINE_ERROR", f"Analysis failed: {str(e)[:500]}")

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
        )
        if speech_id is not None:
            result["speech_id"] = speech_id

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
    preset: Optional[str] = Query(None, description="Named preset"),
    quality_gate: Optional[str] = Query(None, description="Quality gate mode: warn, block, skip"),
    language: Optional[str] = Query(None, description="Language code for Whisper"),
    speaker: Optional[str] = Query(None, description="Speaker name (used for diarization)"),
    title: Optional[str] = Query(None, description="Title for the speech"),
    profile: Optional[str] = Query("general", description="Scoring profile: general, oratory, reading_fluency, presentation, clinical"),
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
    key_id = key_data["key_id"] if key_data else "user"
    user_id = auth.get("user_id")

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
    profile: Optional[str] = Body(None, description="Scoring profile (general, oratory, reading_fluency, presentation, clinical)"),
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

COACH_SYSTEM_PROMPT = """You are a professional speech coach AI assistant. Your role is to help users improve their public speaking skills based on their speech analysis data.

Be encouraging but honest. Give specific, actionable advice. Keep responses concise (2-3 paragraphs max unless asked for more detail).

When discussing metrics:
- WPM (words per minute): 120-150 is ideal for most contexts
- HNR (Harmonics-to-Noise Ratio): Higher is better, indicates voice clarity
- Jitter/Shimmer: Lower is better, high values may indicate vocal strain
- Vocal Fry: Common in casual speech, reduce for formal presentations
- Pitch Variation: More variation = more engaging delivery

Focus on practical tips the user can apply in their next speech."""


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
    
    if context:
        prompt_parts.append(f"\n--- Speech Analysis Context ---\n{context}\n---")
    
    # Add history if provided
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
