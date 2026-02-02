"""SpeechScore API client."""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Optional, List, BinaryIO, Union

import httpx

from .models import AnalysisResult, Job
from .exceptions import (
    SpeechScoreError,
    AuthError,
    RateLimitError,
    ValidationError,
    QueueFullError,
)

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120.0  # seconds


class SpeechScore:
    """SpeechScore API client.

    Args:
        api_key: API key for authentication.
        base_url: API base URL (default: http://localhost:8000).
        timeout: Request timeout in seconds (default: 120).

    Example::

        client = SpeechScore(api_key="sk-YOUR-KEY")
        result = client.analyze("speech.mp3")
        print(result.transcript)
        print(result.wpm)
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("SPEECHSCORE_API_KEY", "")
        self.base_url = (base_url or os.environ.get("SPEECHSCORE_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._auth_headers(),
        )

    def _auth_headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _handle_error(self, response: httpx.Response) -> None:
        """Raise appropriate exception for error responses."""
        if response.status_code < 400:
            return

        try:
            body = response.json()
            error = body.get("error", {})
            if isinstance(error, dict):
                code = error.get("code", "UNKNOWN")
                message = error.get("message", response.text)
            else:
                code = "UNKNOWN"
                message = str(error)
        except (json.JSONDecodeError, KeyError):
            code = "UNKNOWN"
            message = response.text

        if response.status_code == 401:
            raise AuthError(message, code=code, status_code=401)
        elif response.status_code == 429:
            if code == "QUEUE_FULL":
                raise QueueFullError(message, code=code, status_code=429)
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message,
                code=code,
                status_code=429,
                retry_after=float(retry_after) if retry_after else None,
            )
        elif response.status_code in (400, 413, 422):
            raise ValidationError(message, code=code, status_code=response.status_code)
        else:
            raise SpeechScoreError(message, code=code, status_code=response.status_code)

    # -------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------

    def health(self) -> dict:
        """Check API health.

        Returns:
            dict with status, gpu_available, whisper_model_loaded, uptime_sec.
        """
        resp = self._client.get("/v1/health")
        self._handle_error(resp)
        return resp.json()

    # -------------------------------------------------------------------
    # Analyze (sync)
    # -------------------------------------------------------------------

    def analyze(
        self,
        audio: Union[str, Path, BinaryIO],
        *,
        modules: str = None,
        preset: str = None,
        quality_gate: str = None,
        language: str = None,
        filename: str = None,
    ) -> AnalysisResult:
        """Analyze an audio file synchronously.

        Args:
            audio: File path (str/Path) or file-like object.
            modules: Comma-separated module names (e.g. "transcription,audio").
            preset: Named preset (overrides modules): quick, text_only, audio_only, full, clinical, coaching.
            quality_gate: "warn" (default), "block", or "skip".
            language: Language code for Whisper (e.g. "en").
            filename: Override filename (used when audio is a file-like object).

        Returns:
            AnalysisResult with all requested metrics.
        """
        files, params = self._prepare_request(
            audio, modules=modules, preset=preset,
            quality_gate=quality_gate, language=language,
            filename=filename,
        )

        resp = self._client.post("/v1/analyze", files=files, params=params)
        self._handle_error(resp)
        return AnalysisResult(resp.json())

    # -------------------------------------------------------------------
    # Analyze (async)
    # -------------------------------------------------------------------

    def analyze_async(
        self,
        audio: Union[str, Path, BinaryIO],
        *,
        modules: str = None,
        preset: str = None,
        quality_gate: str = None,
        language: str = None,
        webhook_url: str = None,
        filename: str = None,
    ) -> Job:
        """Submit an audio file for async analysis.

        Returns immediately with a Job object. Poll with job.refresh()
        or use client.wait_for_job().

        Args:
            audio: File path or file-like object.
            modules: Comma-separated module names.
            preset: Named preset.
            quality_gate: "warn", "block", or "skip".
            language: Language code.
            webhook_url: URL to receive results when complete.
            filename: Override filename.

        Returns:
            Job with job_id and initial status.
        """
        files, params = self._prepare_request(
            audio, modules=modules, preset=preset,
            quality_gate=quality_gate, language=language,
            filename=filename,
        )
        params["mode"] = "async"
        if webhook_url:
            params["webhook_url"] = webhook_url

        resp = self._client.post("/v1/analyze", files=files, params=params)
        self._handle_error(resp)
        return Job(resp.json())

    def get_job(self, job_id: str) -> Job:
        """Get current status of an async job.

        Args:
            job_id: The job ID returned from analyze_async().

        Returns:
            Job with current status and result (if complete).
        """
        resp = self._client.get(f"/v1/jobs/{job_id}")
        self._handle_error(resp)
        return Job(resp.json())

    def list_jobs(self, status: str = None) -> List[Job]:
        """List recent jobs.

        Args:
            status: Filter by status (queued/processing/complete/failed).

        Returns:
            List of Job objects.
        """
        params = {}
        if status:
            params["status"] = status
        resp = self._client.get("/v1/jobs", params=params)
        self._handle_error(resp)
        data = resp.json()
        return [Job(j) for j in data.get("jobs", [])]

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> Job:
        """Poll an async job until it completes or fails.

        Args:
            job_id: Job ID to poll.
            poll_interval: Seconds between polls (default 2).
            timeout: Max seconds to wait (default 300).

        Returns:
            Completed Job with result.

        Raises:
            SpeechScoreError: If job fails.
            TimeoutError: If timeout exceeded.
        """
        start = time.time()
        while True:
            job = self.get_job(job_id)
            if job.is_complete:
                return job
            if job.is_failed:
                raise SpeechScoreError(
                    f"Job failed: {job.error}",
                    code="JOB_FAILED",
                )
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout}s"
                )
            time.sleep(poll_interval)

    # -------------------------------------------------------------------
    # Queue
    # -------------------------------------------------------------------

    def queue_status(self) -> dict:
        """Get current queue status.

        Returns:
            dict with queue_depth, currently_processing, estimated_wait_sec, etc.
        """
        resp = self._client.get("/v1/queue")
        self._handle_error(resp)
        return resp.json()

    # -------------------------------------------------------------------
    # Usage
    # -------------------------------------------------------------------

    def usage(self) -> dict:
        """Get usage stats for the current API key.

        Returns:
            dict with current_period and all_time usage.
        """
        resp = self._client.get("/v1/usage")
        self._handle_error(resp)
        return resp.json()

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _prepare_request(
        self,
        audio: Union[str, Path, BinaryIO],
        **params,
    ) -> tuple:
        """Prepare file upload and query params."""
        # Build params dict (skip None values)
        query = {k: v for k, v in params.items() if v is not None and k != "filename"}
        filename = params.get("filename")

        # Handle file input
        if isinstance(audio, (str, Path)):
            path = Path(audio)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")
            f = open(path, "rb")
            fname = filename or path.name
            files = {"audio": (fname, f, "application/octet-stream")}
        else:
            fname = filename or "audio.wav"
            files = {"audio": (fname, audio, "application/octet-stream")}

        return files, query

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f'<SpeechScore base_url="{self.base_url}">'
