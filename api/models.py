"""Pydantic request/response models for SpeechScore API."""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    whisper_model_loaded: bool
    whisper_model: Optional[str] = None
    uptime_sec: float


class ProcessingMetadata(BaseModel):
    duration_s: float
    processing_time_s: float
    modules_requested: List[str]
    modules_run: List[str]
    preset: Optional[str] = None
    quality_gate: str = "warn"
    language_requested: Optional[str] = None
    audio_quality: Optional[Dict[str, Any]] = None
    timing: Dict[str, float]


class AnalysisResponse(BaseModel):
    """
    Flexible response — only includes sections for modules that were run.
    All analysis sections are Optional so omitted modules don't appear.
    """
    audio_file: str
    audio_duration_sec: float
    audio_quality: Optional[Dict[str, Any]] = None
    language: str
    language_probability: float
    transcript: str
    segments: List[Dict[str, Any]]
    speech3_text_analysis: Optional[Dict[str, Any]] = None
    advanced_text_analysis: Optional[Dict[str, Any]] = None
    rhetorical_analysis: Optional[Dict[str, Any]] = None
    sentiment: Optional[Dict[str, Any]] = None
    audio_analysis: Optional[Dict[str, Any]] = None
    advanced_audio_analysis: Optional[Dict[str, Any]] = None
    processing: ProcessingMetadata


# ---------------------------------------------------------------------------
# Job / Queue models (Milestone 2)
# ---------------------------------------------------------------------------

class JobCreatedResponse(BaseModel):
    job_id: str
    status: str
    created_at: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    filename: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[str] = None
    queue_position: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobSummary(BaseModel):
    job_id: str
    status: str
    filename: Optional[str] = None
    created_at: str
    duration_sec: Optional[float] = None


class JobListResponse(BaseModel):
    jobs: List[JobSummary]
    total: int


class QueueStatusResponse(BaseModel):
    queue_depth: int
    max_queue_depth: int
    currently_processing: Optional[str] = None
    currently_processing_filename: Optional[str] = None
    estimated_wait_sec: Optional[float] = None
    jobs_queued: int
    jobs_processing: int
    jobs_complete: int
    jobs_failed: int


# ---------------------------------------------------------------------------
# Auth / Admin / Usage models (Milestone 4)
# ---------------------------------------------------------------------------

class CreateKeyRequest(BaseModel):
    name: str
    tier: str  # free, pro, enterprise, admin


class CreateKeyResponse(BaseModel):
    key_id: str
    name: str
    tier: str
    api_key: str  # plaintext, returned ONCE
    message: str = "Store this key securely. It will not be shown again."


class KeyInfo(BaseModel):
    key_id: str
    name: str
    tier: str
    created_at: str
    enabled: bool
    usage: Optional[Dict[str, Any]] = None


class KeyListResponse(BaseModel):
    keys: List[KeyInfo]
    total: int


class UpdateKeyRequest(BaseModel):
    name: Optional[str] = None
    tier: Optional[str] = None
    enabled: Optional[bool] = None


class UsagePeriod(BaseModel):
    requests_today: int
    audio_minutes_today: float
    requests_limit_per_minute: Any  # int or "unlimited"
    audio_minutes_limit_per_day: Any  # float or "unlimited"


class UsageAllTime(BaseModel):
    total_requests: int
    total_audio_minutes: float


class UsageResponse(BaseModel):
    key_id: str
    key_name: str
    tier: str
    current_period: UsagePeriod
    all_time: UsageAllTime


class AdminUsageEntry(BaseModel):
    key_id: str
    key_name: str
    tier: str
    enabled: bool
    total_requests: int
    total_audio_minutes: float
    last_used: Optional[str] = None


class AdminUsageResponse(BaseModel):
    keys: List[AdminUsageEntry]
    total: int
