"""
In-memory rate limiter for SpeechScore API.

Uses sliding window for requests/minute and daily counters for audio-minutes.
No external dependencies (no Redis needed).
"""

import time
import threading
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("speechscore.ratelimit")

# ---------------------------------------------------------------------------
# Rate limit tiers
# ---------------------------------------------------------------------------
TIER_LIMITS = {
    "free": {
        "requests_per_minute": 5,
        "audio_minutes_per_day": 10.0,
    },
    "pro": {
        "requests_per_minute": 30,
        "audio_minutes_per_day": 500.0,
    },
    "enterprise": {
        "requests_per_minute": 100,
        "audio_minutes_per_day": float("inf"),  # unlimited
    },
    "admin": {
        "requests_per_minute": float("inf"),  # unlimited
        "audio_minutes_per_day": float("inf"),  # unlimited
    },
}


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        # Sliding window: key_id -> list of timestamps
        self._request_windows: Dict[str, list] = defaultdict(list)
        # Daily audio minutes: (key_id, date_str) -> float
        self._daily_audio: Dict[Tuple[str, str], float] = defaultdict(float)
        # Daily request count: (key_id, date_str) -> int
        self._daily_requests: Dict[Tuple[str, str], int] = defaultdict(int)

    def _today_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _clean_window(self, key_id: str, now: float, window_sec: float = 60.0):
        """Remove timestamps older than window_sec from the sliding window."""
        cutoff = now - window_sec
        self._request_windows[key_id] = [
            t for t in self._request_windows[key_id] if t > cutoff
        ]

    def check_rate_limit(self, key_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if a request is allowed under rate limits.

        Returns None if allowed, or a dict with error info if rate limited:
          {"retry_after_sec": float, "reason": str}
        """
        tier = key_data.get("tier", "free")
        key_id = key_data.get("key_id", "unknown")
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

        rpm_limit = limits["requests_per_minute"]
        daily_audio_limit = limits["audio_minutes_per_day"]

        now = time.time()
        today = self._today_utc()

        with self._lock:
            # Check requests per minute (sliding window)
            if rpm_limit != float("inf"):
                self._clean_window(key_id, now)
                current_rpm = len(self._request_windows[key_id])

                if current_rpm >= rpm_limit:
                    # Find when the oldest request in window expires
                    oldest = min(self._request_windows[key_id])
                    retry_after = 60.0 - (now - oldest)
                    retry_after = max(1.0, round(retry_after, 1))
                    return {
                        "retry_after_sec": retry_after,
                        "reason": f"Rate limit exceeded: {int(rpm_limit)} requests/minute for {tier} tier.",
                    }

            # Check daily audio minutes
            if daily_audio_limit != float("inf"):
                daily_key = (key_id, today)
                current_audio = self._daily_audio[daily_key]
                if current_audio >= daily_audio_limit:
                    # Calculate seconds until midnight UTC
                    now_utc = datetime.now(timezone.utc)
                    midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                    midnight_next = midnight.replace(day=midnight.day + 1) if midnight < now_utc else midnight
                    # Handle month boundaries
                    import calendar
                    from datetime import timedelta
                    midnight_next = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                    retry_after = (midnight_next - now_utc).total_seconds()
                    retry_after = max(1.0, round(retry_after, 0))
                    return {
                        "retry_after_sec": retry_after,
                        "reason": f"Daily audio limit exceeded: {daily_audio_limit} minutes/day for {tier} tier.",
                    }

        return None  # Allowed

    def record_request(self, key_id: str):
        """Record a request in the sliding window and daily counter."""
        now = time.time()
        today = self._today_utc()

        with self._lock:
            self._request_windows[key_id].append(now)
            self._daily_requests[(key_id, today)] = self._daily_requests.get((key_id, today), 0) + 1

    def record_audio_minutes(self, key_id: str, audio_minutes: float):
        """Record audio minutes consumed."""
        today = self._today_utc()
        with self._lock:
            self._daily_audio[(key_id, today)] += audio_minutes

    def get_rate_limit_headers(self, key_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate rate limit headers for the response.
        Returns dict of header name -> value.
        """
        tier = key_data.get("tier", "free")
        key_id = key_data.get("key_id", "unknown")
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

        rpm_limit = limits["requests_per_minute"]
        now = time.time()

        headers = {}

        if rpm_limit == float("inf"):
            headers["X-RateLimit-Limit"] = "unlimited"
            headers["X-RateLimit-Remaining"] = "unlimited"
            headers["X-RateLimit-Reset"] = "0"
        else:
            with self._lock:
                self._clean_window(key_id, now)
                current = len(self._request_windows[key_id])

            remaining = max(0, int(rpm_limit) - current)
            # Reset = seconds until the window clears
            reset = 60  # window is always 60s from now

            headers["X-RateLimit-Limit"] = str(int(rpm_limit))
            headers["X-RateLimit-Remaining"] = str(remaining)
            headers["X-RateLimit-Reset"] = str(reset)

        return headers

    def get_daily_stats(self, key_id: str) -> Dict[str, Any]:
        """Get daily request/audio stats for a key."""
        today = self._today_utc()
        with self._lock:
            requests_today = self._daily_requests.get((key_id, today), 0)
            audio_today = round(self._daily_audio.get((key_id, today), 0.0), 2)
        return {
            "requests_today": requests_today,
            "audio_minutes_today": audio_today,
        }

    def cleanup_old_data(self):
        """Remove data from previous days to prevent memory growth."""
        today = self._today_utc()
        with self._lock:
            # Clean daily counters
            old_keys = [k for k in self._daily_audio if k[1] != today]
            for k in old_keys:
                del self._daily_audio[k]

            old_keys = [k for k in self._daily_requests if k[1] != today]
            for k in old_keys:
                del self._daily_requests[k]

            # Clean old sliding window entries
            now = time.time()
            cutoff = now - 120  # keep 2 minutes of data
            for key_id in list(self._request_windows.keys()):
                self._request_windows[key_id] = [
                    t for t in self._request_windows[key_id] if t > cutoff
                ]
                if not self._request_windows[key_id]:
                    del self._request_windows[key_id]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
rate_limiter = RateLimiter()
