"""SpeechScore SDK exceptions."""


class SpeechScoreError(Exception):
    """Base exception for SpeechScore API errors."""

    def __init__(self, message: str, code: str = None, status_code: int = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)

    def __str__(self):
        parts = []
        if self.code:
            parts.append(f"[{self.code}]")
        parts.append(self.message)
        if self.status_code:
            parts.append(f"(HTTP {self.status_code})")
        return " ".join(parts)


class AuthError(SpeechScoreError):
    """Authentication failed — invalid or missing API key."""
    pass


class RateLimitError(SpeechScoreError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: float = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ValidationError(SpeechScoreError):
    """Request validation failed (bad file, too large, etc.)."""
    pass


class QueueFullError(SpeechScoreError):
    """Async job queue is at capacity."""
    pass
