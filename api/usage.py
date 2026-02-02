"""
Usage tracking for SpeechScore API.

Tracks per-key: total requests, total audio minutes, last used timestamp.
Persists to usage.json periodically (every 5 minutes) to avoid I/O on every request.
"""

import json
import os
import time
import asyncio
import threading
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger("speechscore.usage")

USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usage.json")
FLUSH_INTERVAL_SEC = 300  # 5 minutes


class UsageTracker:
    def __init__(self):
        self._lock = threading.Lock()
        # key_id -> {total_requests, total_audio_minutes, last_used}
        self._data: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._flush_task: Optional[asyncio.Task] = None

    def load(self):
        """Load usage data from disk."""
        if not os.path.exists(USAGE_FILE):
            logger.info("No usage.json found, starting fresh.")
            return
        try:
            with open(USAGE_FILE, "r") as f:
                data = json.load(f)
            with self._lock:
                self._data = data.get("usage", {})
            logger.info("Loaded usage data for %d keys", len(self._data))
        except Exception as e:
            logger.error("Failed to load usage.json: %s", e)

    def save(self):
        """Persist usage data to disk."""
        with self._lock:
            if not self._dirty:
                return
            data = {"usage": dict(self._data), "saved_at": datetime.now(timezone.utc).isoformat()}
            self._dirty = False

        try:
            tmp_path = USAGE_FILE + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, USAGE_FILE)
            logger.debug("Usage data flushed to disk (%d keys)", len(data["usage"]))
        except Exception as e:
            logger.error("Failed to save usage.json: %s", e)

    def start_flush_loop(self, loop: asyncio.AbstractEventLoop):
        """Start the periodic flush task."""
        self._flush_task = loop.create_task(self._flush_loop())
        logger.info("Usage flush loop started (interval=%ds)", FLUSH_INTERVAL_SEC)

    def shutdown(self):
        """Cancel flush task and do a final save."""
        if self._flush_task:
            self._flush_task.cancel()
        self.save()
        logger.info("Usage tracker shut down (final flush done).")

    async def _flush_loop(self):
        """Periodically flush usage data to disk."""
        while True:
            try:
                await asyncio.sleep(FLUSH_INTERVAL_SEC)
                self.save()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in usage flush loop")

    def record_request(self, key_id: str):
        """Record a request for a key."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if key_id not in self._data:
                self._data[key_id] = {
                    "total_requests": 0,
                    "total_audio_minutes": 0.0,
                    "last_used": None,
                }
            self._data[key_id]["total_requests"] += 1
            self._data[key_id]["last_used"] = now
            self._dirty = True

    def record_audio_minutes(self, key_id: str, minutes: float):
        """Record audio minutes processed for a key."""
        with self._lock:
            if key_id not in self._data:
                self._data[key_id] = {
                    "total_requests": 0,
                    "total_audio_minutes": 0.0,
                    "last_used": None,
                }
            self._data[key_id]["total_audio_minutes"] += minutes
            self._dirty = True

    def get_usage(self, key_id: str) -> Dict[str, Any]:
        """Get all-time usage for a key."""
        with self._lock:
            entry = self._data.get(key_id, {})
            return {
                "total_requests": entry.get("total_requests", 0),
                "total_audio_minutes": round(entry.get("total_audio_minutes", 0.0), 2),
                "last_used": entry.get("last_used"),
            }

    def get_all_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get usage for all keys."""
        with self._lock:
            result = {}
            for key_id, entry in self._data.items():
                result[key_id] = {
                    "total_requests": entry.get("total_requests", 0),
                    "total_audio_minutes": round(entry.get("total_audio_minutes", 0.0), 2),
                    "last_used": entry.get("last_used"),
                }
            return result


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
usage_tracker = UsageTracker()
