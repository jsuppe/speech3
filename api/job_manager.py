"""
Job queue manager for async speech analysis.

Uses in-memory job store + ThreadPoolExecutor(max_workers=1) for GPU serialization.
No external dependencies (no Redis/Celery).
"""

import os
import time
import uuid
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import httpx

from . import config
from . import pipeline_runner

logger = logging.getLogger("speechscore.jobs")

# ---------------------------------------------------------------------------
# Job states
# ---------------------------------------------------------------------------
STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"


# ---------------------------------------------------------------------------
# Job data structure
# ---------------------------------------------------------------------------
class Job:
    __slots__ = (
        "job_id", "status", "filename", "upload_path", "wav_path",
        "created_at", "started_at", "completed_at",
        "progress", "result", "error", "webhook_url",
        "modules", "preset", "quality_gate", "language", "modules_requested",
        "key_id",
    )

    def __init__(self, job_id: str, filename: str, upload_path: str,
                 webhook_url: Optional[str] = None,
                 modules: Optional[set] = None,
                 preset: Optional[str] = None,
                 quality_gate: str = "warn",
                 language: Optional[str] = None,
                 modules_requested: Optional[List[str]] = None,
                 key_id: Optional[str] = None):
        self.job_id = job_id
        self.status = STATUS_QUEUED
        self.filename = filename
        self.upload_path = upload_path
        self.wav_path: Optional[str] = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.progress: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.webhook_url = webhook_url
        self.modules = modules
        self.preset = preset
        self.quality_gate = quality_gate
        self.language = language
        self.modules_requested = modules_requested
        self.key_id = key_id

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "filename": self.filename,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
        }

    def to_summary(self) -> dict:
        duration = None
        if self.started_at and self.completed_at:
            try:
                t0 = datetime.fromisoformat(self.started_at)
                t1 = datetime.fromisoformat(self.completed_at)
                duration = round((t1 - t0).total_seconds(), 2)
            except Exception:
                pass
        return {
            "job_id": self.job_id,
            "status": self.status,
            "filename": self.filename,
            "created_at": self.created_at,
            "duration_sec": duration,
        }


# ---------------------------------------------------------------------------
# JobManager — singleton
# ---------------------------------------------------------------------------
class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        self._cleanup_task: Optional[asyncio.Task] = None
        # Track recent processing times for ETA estimates
        self._recent_durations: List[float] = []

    # -- Lifecycle --

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start background cleanup task."""
        os.makedirs(config.ASYNC_UPLOAD_DIR, exist_ok=True)
        self._cleanup_task = loop.create_task(self._cleanup_loop())
        logger.info("Job manager started (max_queue=%d, expiry=%dh)",
                     config.MAX_QUEUE_DEPTH, config.JOB_EXPIRATION_HOURS)

    def shutdown(self):
        """Shutdown executor and cancel cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        self._executor.shutdown(wait=False)
        # Clean up any remaining temp files
        self._cleanup_all_files()
        logger.info("Job manager shut down.")

    # -- Job creation --

    def create_job(self, filename: str, upload_path: str,
                   webhook_url: Optional[str] = None,
                   modules: Optional[set] = None,
                   preset: Optional[str] = None,
                   quality_gate: str = "warn",
                   language: Optional[str] = None,
                   modules_requested: Optional[List[str]] = None,
                   key_id: Optional[str] = None) -> Job:
        """Create a new job and submit it for execution. Returns the Job."""
        with self._lock:
            queued_count = sum(
                1 for j in self._jobs.values()
                if j.status in (STATUS_QUEUED, STATUS_PROCESSING)
            )
            if queued_count >= config.MAX_QUEUE_DEPTH:
                raise QueueFullError(
                    f"Queue is full ({queued_count}/{config.MAX_QUEUE_DEPTH}). "
                    "Try again later."
                )

            job_id = str(uuid.uuid4())
            job = Job(job_id, filename, upload_path, webhook_url,
                      modules=modules, preset=preset,
                      quality_gate=quality_gate, language=language,
                      modules_requested=modules_requested,
                      key_id=key_id)
            self._jobs[job_id] = job

        logger.info("Job %s created for '%s' (queue depth: %d)",
                     job_id, filename, queued_count + 1)

        # Submit to thread pool
        self._executor.submit(self._execute_job, job_id)
        return job

    # -- Job queries --

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, status_filter: Optional[str] = None,
                  limit: int = 50) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())

        # Filter
        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def get_queue_position(self, job_id: str) -> Optional[int]:
        """Get 1-based queue position for a queued job."""
        with self._lock:
            queued = [
                j for j in self._jobs.values()
                if j.status == STATUS_QUEUED
            ]
        queued.sort(key=lambda j: j.created_at)
        for i, j in enumerate(queued):
            if j.job_id == job_id:
                return i + 1
        return None

    def get_queue_stats(self) -> dict:
        """Get queue statistics."""
        with self._lock:
            jobs = list(self._jobs.values())

        counts = {STATUS_QUEUED: 0, STATUS_PROCESSING: 0,
                  STATUS_COMPLETE: 0, STATUS_FAILED: 0}
        processing_job_id = None
        processing_filename = None
        for j in jobs:
            counts[j.status] = counts.get(j.status, 0) + 1
            if j.status == STATUS_PROCESSING:
                processing_job_id = j.job_id
                processing_filename = j.filename

        active = counts[STATUS_QUEUED] + counts[STATUS_PROCESSING]

        # Estimate wait time based on recent durations
        estimated_wait = None
        if self._recent_durations and active > 0:
            avg = sum(self._recent_durations) / len(self._recent_durations)
            estimated_wait = round(avg * active, 1)

        return {
            "queue_depth": active,
            "max_queue_depth": config.MAX_QUEUE_DEPTH,
            "currently_processing": processing_job_id,
            "currently_processing_filename": processing_filename,
            "estimated_wait_sec": estimated_wait,
            "jobs_queued": counts[STATUS_QUEUED],
            "jobs_processing": counts[STATUS_PROCESSING],
            "jobs_complete": counts[STATUS_COMPLETE],
            "jobs_failed": counts[STATUS_FAILED],
        }

    # -- Internal execution --

    def _execute_job(self, job_id: str):
        """Run pipeline for a job. Runs in thread pool."""
        job = self.get_job(job_id)
        if not job:
            logger.error("Job %s not found for execution", job_id)
            return

        # Mark processing
        with self._lock:
            job.status = STATUS_PROCESSING
            job.started_at = datetime.now(timezone.utc).isoformat()
            job.progress = "processing:converting"

        t0 = time.time()
        wav_path = None

        try:
            # Convert to WAV
            logger.info("Job %s: converting to WAV", job_id)
            wav_path = pipeline_runner.convert_to_wav(job.upload_path)
            job.wav_path = wav_path

            # Check duration
            duration = pipeline_runner.get_audio_duration(wav_path)
            if duration <= 0:
                raise RuntimeError("Could not read audio file. It may be corrupt.")
            if duration > config.MAX_DURATION_SEC:
                raise RuntimeError(
                    f"Audio duration {duration:.0f}s exceeds {config.MAX_DURATION_SEC}s limit."
                )

            # Track audio minutes for the key that submitted the job
            if job.key_id:
                audio_minutes = duration / 60.0
                try:
                    from .rate_limiter import rate_limiter
                    from .usage import usage_tracker
                    rate_limiter.record_audio_minutes(job.key_id, audio_minutes)
                    usage_tracker.record_audio_minutes(job.key_id, audio_minutes)
                except Exception as e:
                    logger.warning("Failed to record audio minutes for job %s: %s", job_id, e)

            # Progress callback
            def on_progress(stage: str):
                with self._lock:
                    job.progress = f"processing:{stage}"

            # Run pipeline with module selection
            logger.info("Job %s: running pipeline (modules=%s, preset=%s, qg=%s, lang=%s)",
                        job_id, job.modules_requested, job.preset, job.quality_gate, job.language)
            result = pipeline_runner.run_pipeline(
                wav_path,
                progress_callback=on_progress,
                modules=job.modules,
                language=job.language,
                quality_gate=job.quality_gate,
                preset=job.preset,
                modules_requested=job.modules_requested,
            )

            # Override filename
            result["audio_file"] = job.filename

            elapsed = time.time() - t0
            result["processing"]["processing_time_s"] = round(elapsed, 2)
            result["processing"]["timing"]["total_sec"] = round(elapsed, 2)

            # Persist to SQLite before cleaning up temp files
            speech_id = pipeline_runner.persist_result(
                original_audio_path=job.upload_path,
                result=result,
                filename=job.filename,
                preset=job.preset,
            )
            if speech_id is not None:
                result["speech_id"] = speech_id

            # Mark complete
            with self._lock:
                job.status = STATUS_COMPLETE
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.progress = None
                job.result = result

            # Track duration for ETA estimates
            self._recent_durations.append(elapsed)
            if len(self._recent_durations) > 20:
                self._recent_durations = self._recent_durations[-20:]

            logger.info("Job %s complete (%.1fs, speech_id=%s)", job_id, elapsed, speech_id)

        except pipeline_runner.QualityGateError as e:
            elapsed = time.time() - t0
            logger.warning("Job %s blocked by quality gate after %.1fs: %s", job_id, elapsed, e)
            with self._lock:
                job.status = STATUS_FAILED
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.progress = None
                job.error = f"QUALITY_GATE_BLOCKED: {str(e)[:450]}"

        except Exception as e:
            elapsed = time.time() - t0
            logger.exception("Job %s failed after %.1fs", job_id, elapsed)
            with self._lock:
                job.status = STATUS_FAILED
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.progress = None
                job.error = str(e)[:500]

        finally:
            # Cleanup temp files
            for path in [job.upload_path, wav_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            # Fire webhook if configured
            if job.webhook_url:
                self._fire_webhook(job)

    def _fire_webhook(self, job: Job):
        """POST job result to webhook URL (fire-and-forget)."""
        try:
            payload = job.to_dict()
            logger.info("Job %s: sending webhook to %s", job.job_id, job.webhook_url)
            with httpx.Client(timeout=config.WEBHOOK_TIMEOUT_SEC) as client:
                resp = client.post(job.webhook_url, json=payload)
                logger.info("Job %s: webhook delivered (status=%d)",
                           job.job_id, resp.status_code)
        except Exception as e:
            logger.warning("Job %s: webhook delivery failed: %s",
                          job.job_id, str(e)[:200])

    # -- Cleanup --

    async def _cleanup_loop(self):
        """Periodically remove expired jobs."""
        while True:
            try:
                await asyncio.sleep(config.JOB_CLEANUP_INTERVAL_SEC)
                self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in job cleanup loop")

    def _cleanup_expired(self):
        """Remove jobs older than JOB_EXPIRATION_HOURS."""
        expiry_seconds = config.JOB_EXPIRATION_HOURS * 3600
        now = datetime.now(timezone.utc)
        to_remove = []

        with self._lock:
            for job_id, job in self._jobs.items():
                try:
                    created = datetime.fromisoformat(job.created_at)
                    age = (now - created).total_seconds()
                    if age > expiry_seconds and job.status in (STATUS_COMPLETE, STATUS_FAILED):
                        to_remove.append(job_id)
                except Exception:
                    pass

            for job_id in to_remove:
                del self._jobs[job_id]

        if to_remove:
            logger.info("Cleaned up %d expired jobs", len(to_remove))

    def _cleanup_all_files(self):
        """Remove all temp upload files (shutdown cleanup)."""
        try:
            if os.path.exists(config.ASYNC_UPLOAD_DIR):
                for f in os.listdir(config.ASYNC_UPLOAD_DIR):
                    try:
                        os.remove(os.path.join(config.ASYNC_UPLOAD_DIR, f))
                    except OSError:
                        pass
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class QueueFullError(Exception):
    pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
manager = JobManager()
