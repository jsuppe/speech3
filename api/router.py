"""API route handlers for SpeechScore."""

import os
import time
import logging
import tempfile

from fastapi import APIRouter, UploadFile, File, Query, Request, Depends, Response
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any

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
from .rate_limiter import rate_limiter, TIER_LIMITS
from .usage import usage_tracker

logger = logging.getLogger("speechscore.api")

router = APIRouter(prefix="/v1")


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
    audio: UploadFile = File(...),
    mode: Optional[str] = Query(None, description="Set to 'async' for async processing"),
    webhook_url: Optional[str] = Query(None, description="URL to POST results to when complete"),
    modules: Optional[str] = Query(None, description="Comma-separated module names (e.g. 'transcription,audio'). Use 'all' for everything."),
    preset: Optional[str] = Query(None, description="Named preset: quick, text_only, audio_only, full, clinical, coaching"),
    quality_gate: Optional[str] = Query(None, description="Quality gate mode: warn (default), block, skip"),
    language: Optional[str] = Query(None, description="Language code for Whisper (e.g. 'en', 'es'). Default: auto-detect."),
    key_data: dict = Depends(auth_required),
):
    """
    Analyze an uploaded audio file through the speech pipeline.

    Default: synchronous (blocks until complete, returns full result).
    With mode=async: returns immediately with a job_id for polling.

    Use `modules` or `preset` to select which analysis modules run.
    If both are provided, `preset` takes precedence.
    """
    key_id = key_data["key_id"]

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

        # Check duration
        duration = pipeline_runner.get_audio_duration(tmp_path)
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

        # Convert to WAV
        try:
            tmp_wav = pipeline_runner.convert_to_wav(tmp_path)
        except RuntimeError as e:
            return _error(422, "CONVERSION_FAILED", str(e))

        # Run pipeline with module selection
        try:
            result = pipeline_runner.run_pipeline(
                tmp_wav,
                modules=resolved_modules,
                language=lang,
                quality_gate=qg,
                preset=resolved_preset,
                modules_requested=modules_requested,
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
    key_data: dict = Depends(auth_required),
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
    key_data: dict = Depends(auth_required),
):
    """List speeches stored in the database."""
    db = pipeline_runner.get_db()
    speeches = db.list_speeches(category=category, product=product, limit=limit, offset=offset)
    total = db.count_speeches()
    return {"speeches": speeches, "total": total, "limit": limit, "offset": offset}


@router.get("/speeches/{speech_id}", tags=["Database"])
async def get_speech(speech_id: int, key_data: dict = Depends(auth_required)):
    """Get a speech record with its latest analysis metrics."""
    db = pipeline_runner.get_db()
    speech = db.get_speech(speech_id)
    if not speech:
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


@router.get("/speeches/{speech_id}/analysis", tags=["Database"])
async def get_speech_analysis(speech_id: int, key_data: dict = Depends(auth_required)):
    """Get the full analysis JSON for a speech."""
    db = pipeline_runner.get_db()
    result = db.get_full_analysis(speech_id)
    if not result:
        return _error(404, "NOT_FOUND", f"No analysis found for speech {speech_id}.")
    return result


@router.get("/speeches/{speech_id}/audio", tags=["Database"])
async def get_speech_audio(speech_id: int, key_data: dict = Depends(auth_required)):
    """Download the stored audio (Opus) for a speech."""
    db = pipeline_runner.get_db()
    audio = db.get_audio(speech_id)
    if not audio:
        return _error(404, "NOT_FOUND", f"No audio stored for speech {speech_id}.")

    fmt = audio["format"]
    mime = {"opus": "audio/opus", "mp3": "audio/mpeg", "wav": "audio/wav"}.get(fmt, "application/octet-stream")
    filename = audio.get("original_filename", f"speech_{speech_id}.{fmt}")
    if not filename.endswith(f".{fmt}"):
        filename = f"{os.path.splitext(filename)[0]}.{fmt}"

    return Response(
        content=audio["data"],
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/speeches/{speech_id}/spectrogram", tags=["Database"])
async def get_speech_spectrogram(
    speech_id: int,
    type: str = Query("combined", description="Spectrogram type: combined, spectrogram, mel, loudness"),
    key_data: dict = Depends(auth_required),
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
async def db_stats(key_data: dict = Depends(auth_required)):
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
