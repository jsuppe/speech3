"""
Persistent usage limits for SpeakFit.

Tracks monthly recordings and audio minutes per user in the database.
Enforces tier-based limits and provides usage statistics.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("speakfit.usage")

# ---------------------------------------------------------------------------
# Tier Definitions
# ---------------------------------------------------------------------------
TIER_LIMITS = {
    "free": {
        "recordings_per_month": -1,  # unlimited recordings
        "audio_minutes_per_month": 30.0,  # 30 min analysis/month
        "features": ["basic"],
        "price": 0,
    },
    "pro": {
        "recordings_per_month": 100,
        "audio_minutes_per_month": 300.0,  # 5 hours
        "features": ["basic", "pronunciation", "diarization", "dementia", "projects"],
        "price": 9.99,
    },
    "power": {
        "recordings_per_month": 500,
        "audio_minutes_per_month": 1500.0,  # 25 hours
        "features": ["basic", "pronunciation", "diarization", "dementia", "projects", "priority"],
        "price": 29.99,
    },
    "admin": {
        "recordings_per_month": -1,  # unlimited
        "audio_minutes_per_month": -1,  # unlimited
        "features": ["all"],
        "price": 0,
    },
}


def get_current_month() -> str:
    """Get current month as YYYY-MM string."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


def get_month_reset_date() -> str:
    """Get the first day of next month as ISO date."""
    now = datetime.now(timezone.utc)
    if now.month == 12:
        next_month = now.replace(year=now.year + 1, month=1, day=1)
    else:
        next_month = now.replace(month=now.month + 1, day=1)
    return next_month.strftime("%Y-%m-%d")


class UsageLimiter:
    """Database-backed usage limiter for persistent tracking."""
    
    def __init__(self, db_conn):
        self.conn = db_conn
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Add usage columns to users table if they don't exist."""
        cursor = self.conn.cursor()
        
        # Check if columns exist
        cursor.execute("PRAGMA table_info(users)")
        columns = {row[1] for row in cursor.fetchall()}
        
        if "tier" not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN tier TEXT DEFAULT 'free'")
            logger.info("Added 'tier' column to users table")
        
        if "monthly_recordings" not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN monthly_recordings INTEGER DEFAULT 0")
            logger.info("Added 'monthly_recordings' column to users table")
        
        if "monthly_audio_minutes" not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN monthly_audio_minutes REAL DEFAULT 0.0")
            logger.info("Added 'monthly_audio_minutes' column to users table")
        
        if "usage_month" not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN usage_month TEXT")
            logger.info("Added 'usage_month' column to users table")
        
        if "eula_accepted_at" not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN eula_accepted_at TEXT")
            logger.info("Added 'eula_accepted_at' column to users table")
        
        self.conn.commit()
    
    def get_user_usage(self, user_id: int) -> Dict[str, Any]:
        """Get current usage and limits for a user."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT tier, monthly_recordings, monthly_audio_minutes, usage_month
            FROM users WHERE id = ?
        """, (user_id,))
        row = cursor.fetchone()
        
        if not row:
            return {"error": "User not found"}
        
        tier, recordings, audio_mins, usage_month = row
        tier = tier or "free"
        recordings = recordings or 0
        audio_mins = audio_mins or 0.0
        
        # Reset if new month
        current_month = get_current_month()
        if usage_month != current_month:
            recordings = 0
            audio_mins = 0.0
            cursor.execute("""
                UPDATE users 
                SET monthly_recordings = 0, monthly_audio_minutes = 0.0, usage_month = ?
                WHERE id = ?
            """, (current_month, user_id))
            self.conn.commit()
        
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
        rec_limit = limits["recordings_per_month"]
        audio_limit = limits["audio_minutes_per_month"]
        
        return {
            "tier": tier,
            "recordings_used": recordings,
            "recordings_limit": rec_limit if rec_limit != -1 else "unlimited",
            "recordings_remaining": max(0, rec_limit - recordings) if rec_limit != -1 else "unlimited",
            "audio_minutes_used": round(audio_mins, 2),
            "audio_minutes_limit": audio_limit if audio_limit != -1 else "unlimited",
            "audio_minutes_remaining": round(max(0, audio_limit - audio_mins), 2) if audio_limit != -1 else "unlimited",
            "resets": get_month_reset_date(),
            "features": limits["features"],
            "usage_month": current_month,
        }
    
    def check_can_record(self, user_id: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if user can make a new recording.
        
        Returns: (allowed, error_info)
        - If allowed: (True, None)
        - If blocked: (False, {"reason": str, "limit": int, "used": int, "resets": str})
        """
        usage = self.get_user_usage(user_id)
        if "error" in usage:
            return True, None  # Allow if user not found (shouldn't happen)
        
        tier = usage["tier"]
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
        rec_limit = limits["recordings_per_month"]
        
        # Unlimited
        if rec_limit == -1:
            return True, None
        
        recordings = usage["recordings_used"]
        if recordings >= rec_limit:
            return False, {
                "reason": f"Monthly recording limit reached ({rec_limit} recordings/month on {tier} tier)",
                "limit": rec_limit,
                "used": recordings,
                "resets": usage["resets"],
                "upgrade_url": "https://speakfit.app/upgrade",
            }
        
        return True, None
    
    def check_audio_minutes(self, user_id: int, audio_duration_minutes: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if user has enough audio minutes remaining.
        
        Returns: (allowed, error_info)
        """
        usage = self.get_user_usage(user_id)
        if "error" in usage:
            return True, None
        
        tier = usage["tier"]
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
        audio_limit = limits["audio_minutes_per_month"]
        
        # Unlimited
        if audio_limit == -1:
            return True, None
        
        audio_used = usage["audio_minutes_used"]
        if audio_used + audio_duration_minutes > audio_limit:
            return False, {
                "reason": f"Monthly audio limit would be exceeded ({audio_limit} minutes/month on {tier} tier)",
                "limit": audio_limit,
                "used": round(audio_used, 2),
                "requested": round(audio_duration_minutes, 2),
                "remaining": round(max(0, audio_limit - audio_used), 2),
                "resets": usage["resets"],
                "upgrade_url": "https://speakfit.app/upgrade",
            }
        
        return True, None
    
    def record_usage(self, user_id: int, audio_minutes: float = 0.0):
        """Record a recording and its audio duration."""
        cursor = self.conn.cursor()
        current_month = get_current_month()
        
        # First check if we need to reset (new month)
        cursor.execute("SELECT usage_month FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        
        if row and row[0] != current_month:
            # New month - reset counters
            cursor.execute("""
                UPDATE users 
                SET monthly_recordings = 1, 
                    monthly_audio_minutes = ?,
                    usage_month = ?
                WHERE id = ?
            """, (audio_minutes, current_month, user_id))
        else:
            # Same month - increment
            cursor.execute("""
                UPDATE users 
                SET monthly_recordings = COALESCE(monthly_recordings, 0) + 1,
                    monthly_audio_minutes = COALESCE(monthly_audio_minutes, 0.0) + ?,
                    usage_month = ?
                WHERE id = ?
            """, (audio_minutes, current_month, user_id))
        
        self.conn.commit()
        logger.info(f"Recorded usage for user {user_id}: +1 recording, +{audio_minutes:.2f} minutes")
    
    def set_user_tier(self, user_id: int, tier: str) -> bool:
        """Set a user's tier (admin function)."""
        if tier not in TIER_LIMITS:
            return False
        
        cursor = self.conn.cursor()
        cursor.execute("UPDATE users SET tier = ? WHERE id = ?", (tier, user_id))
        self.conn.commit()
        logger.info(f"Set user {user_id} tier to {tier}")
        return True
    
    def check_feature_access(self, user_id: int, feature: str) -> bool:
        """Check if user's tier has access to a feature."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT tier FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        
        tier = row[0] if row else "free"
        tier = tier or "free"
        
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
        features = limits["features"]
        
        return "all" in features or feature in features
    
    def record_eula_acceptance(self, user_id: int):
        """Record that user accepted the EULA."""
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "UPDATE users SET eula_accepted_at = ? WHERE id = ?",
            (now, user_id)
        )
        self.conn.commit()


# Module-level instance (initialized lazily)
_usage_limiter: Optional[UsageLimiter] = None


def get_usage_limiter(db_conn) -> UsageLimiter:
    """Get or create the usage limiter, always using fresh connection."""
    global _usage_limiter
    if _usage_limiter is None:
        _usage_limiter = UsageLimiter(db_conn)
    else:
        # Always update to use the current connection (prevents stale connection errors)
        _usage_limiter.conn = db_conn
    return _usage_limiter
