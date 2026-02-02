"""SpeechScore Python SDK — comprehensive speech analysis as a service."""

from .client import SpeechScore
from .models import AnalysisResult, AudioAnalysis, TextAnalysis, Sentiment, Job
from .exceptions import SpeechScoreError, AuthError, RateLimitError, ValidationError

__version__ = "0.1.0"
__all__ = [
    "SpeechScore",
    "AnalysisResult",
    "AudioAnalysis",
    "TextAnalysis",
    "Sentiment",
    "Job",
    "SpeechScoreError",
    "AuthError",
    "RateLimitError",
    "ValidationError",
]
