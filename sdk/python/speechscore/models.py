"""Response models for the SpeechScore SDK.

These are lightweight wrappers around the API JSON responses,
providing dot-access and convenience methods.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional


class _DotDict:
    """Dict wrapper with dot-access for nested data."""

    def __init__(self, data: dict):
        self._data = data or {}

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            return super().__getattribute__(key)
        val = self._data.get(key)
        if isinstance(val, dict):
            return _DotDict(val)
        if isinstance(val, list):
            return [_DotDict(v) if isinstance(v, dict) else v for v in val]
        return val

    def __repr__(self):
        return repr(self._data)

    def __bool__(self):
        return bool(self._data)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def to_dict(self) -> dict:
        return self._data


class AnalysisResult:
    """Result from POST /v1/analyze (sync mode).

    Attributes:
        transcript (str): Full transcription text.
        segments (list): Timestamped segments.
        audio_duration_sec (float): Audio duration in seconds.
        language (str): Detected language code.
        audio_quality (_DotDict): SNR, confidence, issues.
        text (_DotDict): Core text analysis (lexical diversity, complexity, etc.).
        advanced_text (_DotDict): Readability, discourse, entities, perspective.
        rhetorical (_DotDict): Anaphora, repetition score, rhythm.
        sentiment (_DotDict): Overall label/score, per-segment, emotional arc.
        audio (_DotDict): Pitch, intensity, voice quality, pauses, speaking rate.
        advanced_audio (_DotDict): Formants, vocal fry, pitch contour, emphasis.
        processing (_DotDict): Timing, modules run, quality gate.
        raw (dict): Complete raw JSON response.
    """

    def __init__(self, data: dict):
        self.raw = data
        self.audio_file = data.get("audio_file", "")
        self.audio_duration_sec = data.get("audio_duration_sec", 0.0)
        self.language = data.get("language", "")
        self.language_probability = data.get("language_probability", 0.0)
        self.transcript = data.get("transcript", "")
        self.segments = data.get("segments", [])

        # Analysis sections (dot-accessible)
        self.audio_quality = _DotDict(data.get("audio_quality", {}))
        self.text = _DotDict(data.get("speech3_text_analysis", {}).get("metrics", {}))
        self.text_key_terms = data.get("speech3_text_analysis", {}).get("key_terms", [])
        self.advanced_text = _DotDict(data.get("advanced_text_analysis", {}))
        self.rhetorical = _DotDict(data.get("rhetorical_analysis", {}))
        self.sentiment = _DotDict(data.get("sentiment", {}))
        self.audio = _DotDict(data.get("audio_analysis", {}))
        self.advanced_audio = _DotDict(data.get("advanced_audio_analysis", {}))
        self.processing = _DotDict(data.get("processing", {}))

    # --- Convenience properties ---

    @property
    def wpm(self) -> Optional[float]:
        """Overall words per minute."""
        return self.audio.speaking_rate.overall_wpm if self.audio.speaking_rate else None

    @property
    def mean_pitch_hz(self) -> Optional[float]:
        """Mean fundamental frequency (F0) in Hz."""
        return self.audio.pitch.mean_hz if self.audio.pitch else None

    @property
    def hnr_db(self) -> Optional[float]:
        """Harmonics-to-noise ratio in dB."""
        return self.audio.voice_quality.harmonics_to_noise_db if self.audio.voice_quality else None

    @property
    def vocal_fry_pct(self) -> Optional[float]:
        """Vocal fry ratio as percentage (0-100)."""
        vf = self.advanced_audio.vocal_fry
        if vf and vf.vocal_fry_ratio is not None:
            return round(vf.vocal_fry_ratio * 100, 1)
        return None

    @property
    def overall_sentiment(self) -> Optional[str]:
        """Overall sentiment label (POSITIVE/NEGATIVE/NEUTRAL)."""
        return self.sentiment.overall.label if self.sentiment.overall else None

    @property
    def readability_grade(self) -> Optional[float]:
        """Flesch-Kincaid grade level."""
        return self.advanced_text.readability.flesch_kincaid_grade if self.advanced_text.readability else None

    @property
    def snr_db(self) -> Optional[float]:
        """Signal-to-noise ratio in dB."""
        return self.audio_quality.snr_db

    @property
    def confidence(self) -> Optional[str]:
        """Audio confidence level (HIGH/MEDIUM/LOW/UNUSABLE)."""
        return self.audio_quality.confidence

    def summary(self) -> dict:
        """Return a compact summary of key metrics."""
        return {
            "transcript_length": len(self.transcript),
            "wpm": self.wpm,
            "mean_pitch_hz": self.mean_pitch_hz,
            "hnr_db": self.hnr_db,
            "vocal_fry_pct": self.vocal_fry_pct,
            "sentiment": self.overall_sentiment,
            "readability_grade": self.readability_grade,
            "lexical_diversity": self.text.lexical_diversity,
            "snr_db": self.snr_db,
            "confidence": self.confidence,
            "processing_time_s": self.processing.processing_time_s,
        }

    def __repr__(self):
        return (
            f"<AnalysisResult "
            f'file="{self.audio_file}" '
            f"duration={self.audio_duration_sec}s "
            f"wpm={self.wpm} "
            f"sentiment={self.overall_sentiment}>"
        )


class Job:
    """Represents an async analysis job.

    Attributes:
        job_id (str): Unique job identifier.
        status (str): queued | processing | complete | failed.
        filename (str): Original uploaded filename.
        created_at (str): ISO timestamp.
        result (AnalysisResult): Populated when status is 'complete'.
        error (str): Error message if status is 'failed'.
        raw (dict): Complete raw JSON response.
    """

    def __init__(self, data: dict):
        self.raw = data
        self.job_id = data.get("job_id", "")
        self.status = data.get("status", "")
        self.filename = data.get("filename")
        self.created_at = data.get("created_at", "")
        self.started_at = data.get("started_at")
        self.completed_at = data.get("completed_at")
        self.progress = data.get("progress")
        self.queue_position = data.get("queue_position")
        self.error = data.get("error")

        # Wrap result in AnalysisResult if present
        raw_result = data.get("result")
        self.result = AnalysisResult(raw_result) if raw_result else None

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @property
    def is_pending(self) -> bool:
        return self.status in ("queued", "processing")

    def __repr__(self):
        return f'<Job id="{self.job_id}" status="{self.status}">'


class AudioAnalysis(_DotDict):
    """Audio analysis section (pitch, intensity, voice quality, speaking rate)."""
    pass


class TextAnalysis(_DotDict):
    """Text analysis section (lexical diversity, complexity, fluency)."""
    pass


class Sentiment(_DotDict):
    """Sentiment section (overall, per-segment, emotional arc)."""
    pass
