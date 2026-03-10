"""
Supabase PostgreSQL client for SpeakFit API.

This module handles all user-related database operations via Supabase,
while research/benchmark data stays in local SQLite.
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Any, Iterator, List, Dict
from functools import lru_cache

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values, Json

logger = logging.getLogger(__name__)

# Connection string from environment
SUPABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres.fkxuqyvcvxklzrxjmzsa:Y4ZLP97tHSTQn7Jz@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
)


def get_connection():
    """Get a new Supabase connection."""
    return psycopg2.connect(SUPABASE_URL, cursor_factory=RealDictCursor)


@contextmanager
def get_db() -> Iterator[Any]:
    """Get Supabase connection as context manager."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ============================================================================
# User Operations
# ============================================================================

def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get user by ID."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_user_by_provider(provider: str, provider_id: str) -> Optional[Dict]:
    """Get user by provider and provider_id."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM users WHERE provider = %s AND provider_id = %s",
            (provider, provider_id)
        )
        row = cur.fetchone()
        return dict(row) if row else None


def create_user(provider: str, provider_id: str, email: str, name: str, 
                avatar_url: str = None) -> Dict:
    """Create a new user and return it."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO users (provider, provider_id, email, name, avatar_url)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
        """, (provider, provider_id, email, name, avatar_url))
        row = cur.fetchone()
        return dict(row)


def update_user(user_id: int, **fields) -> Optional[Dict]:
    """Update user fields."""
    if not fields:
        return get_user_by_id(user_id)
    
    # Handle special fields
    extra_updates = []
    if 'app_version' in fields or 'patch_number' in fields:
        extra_updates.append("last_seen = NOW()")
    
    set_clause = ", ".join(f"{k} = %s" for k in fields.keys())
    if extra_updates:
        set_clause += ", " + ", ".join(extra_updates)
    set_clause += ", last_login = NOW()"
    
    values = list(fields.values()) + [user_id]
    
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            UPDATE users SET {set_clause}
            WHERE id = %s
            RETURNING *
        """, values)
        row = cur.fetchone()
        return dict(row) if row else None


def update_user_usage(user_id: int, minutes: float = 0, recordings: int = 0):
    """Update user's monthly usage counters."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users 
            SET monthly_audio_minutes = monthly_audio_minutes + %s,
                monthly_recordings = monthly_recordings + %s,
                usage_month = to_char(NOW(), 'YYYY-MM')
            WHERE id = %s
        """, (minutes, recordings, user_id))


# ============================================================================
# Speech Operations
# ============================================================================

def get_speech_by_id(speech_id: int, user_id: int = None) -> Optional[Dict]:
    """Get speech by ID, optionally filtered by user."""
    with get_db() as conn:
        cur = conn.cursor()
        if user_id:
            cur.execute(
                "SELECT * FROM speeches WHERE id = %s AND user_id = %s",
                (speech_id, user_id)
            )
        else:
            cur.execute("SELECT * FROM speeches WHERE id = %s", (speech_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_speeches_for_user(user_id: int, limit: int = 100, offset: int = 0) -> List[Dict]:
    """Get speeches for a user."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM speeches 
            WHERE user_id = %s 
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (user_id, limit, offset))
        return [dict(row) for row in cur.fetchall()]


def create_speech(user_id: int, title: str, profile: str = 'general', 
                  **fields) -> Dict:
    """Create a new speech."""
    all_fields = {'user_id': user_id, 'title': title, 'profile': profile, **fields}
    columns = list(all_fields.keys())
    values = list(all_fields.values())
    
    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)
    
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            INSERT INTO speeches ({col_names})
            VALUES ({placeholders})
            RETURNING *
        """, values)
        row = cur.fetchone()
        return dict(row)


def update_speech(speech_id: int, user_id: int = None, **fields) -> Optional[Dict]:
    """Update speech fields."""
    if not fields:
        return get_speech_by_id(speech_id, user_id)
    
    set_clause = ", ".join(f"{k} = %s" for k in fields.keys())
    values = list(fields.values()) + [speech_id]
    
    query = f"UPDATE speeches SET {set_clause}, updated_at = NOW() WHERE id = %s"
    if user_id:
        query += " AND user_id = %s"
        values.append(user_id)
    query += " RETURNING *"
    
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(query, values)
        row = cur.fetchone()
        return dict(row) if row else None


def delete_speech(speech_id: int, user_id: int = None) -> bool:
    """Delete a speech."""
    with get_db() as conn:
        cur = conn.cursor()
        if user_id:
            cur.execute(
                "DELETE FROM speeches WHERE id = %s AND user_id = %s",
                (speech_id, user_id)
            )
        else:
            cur.execute("DELETE FROM speeches WHERE id = %s", (speech_id,))
        return cur.rowcount > 0


# ============================================================================
# Analysis Operations
# ============================================================================

def get_analysis_for_speech(speech_id: int) -> Optional[Dict]:
    """Get the latest analysis for a speech."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM analyses 
            WHERE speech_id = %s 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (speech_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def create_analysis(speech_id: int, result: dict, preset: str = 'full',
                    wpm: float = None, pitch_mean_hz: float = None,
                    fluency_score: float = None) -> Dict:
    """Create a new analysis."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO analyses (speech_id, preset, result, wpm, pitch_mean_hz, fluency_score)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
        """, (speech_id, preset, Json(result), wpm, pitch_mean_hz, fluency_score))
        row = cur.fetchone()
        return dict(row)


# ============================================================================
# Transcription Operations
# ============================================================================

def get_transcription_for_speech(speech_id: int) -> Optional[Dict]:
    """Get transcription for a speech."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM transcriptions 
            WHERE speech_id = %s 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (speech_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def create_transcription(speech_id: int, text: str, segments: list = None,
                         word_count: int = None, language: str = 'en',
                         model: str = 'large-v3') -> Dict:
    """Create a new transcription."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO transcriptions (speech_id, text, segments, word_count, language, model)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
        """, (speech_id, text, Json(segments) if segments else None, word_count, language, model))
        row = cur.fetchone()
        return dict(row)


# ============================================================================
# Audio Operations
# ============================================================================

def get_audio_for_speech(speech_id: int) -> Optional[Dict]:
    """Get audio metadata for a speech."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM audio WHERE speech_id = %s", (speech_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def create_audio(speech_id: int, format: str = 'opus', r2_key: str = None,
                 **fields) -> Dict:
    """Create audio metadata entry."""
    all_fields = {'speech_id': speech_id, 'format': format, 'r2_key': r2_key, **fields}
    columns = list(all_fields.keys())
    values = list(all_fields.values())
    
    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)
    
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            INSERT INTO audio ({col_names})
            VALUES ({placeholders})
            RETURNING *
        """, values)
        row = cur.fetchone()
        return dict(row)


def update_audio_r2_key(audio_id: int, r2_key: str) -> bool:
    """Update audio R2 key."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE audio SET r2_key = %s WHERE id = %s",
            (r2_key, audio_id)
        )
        return cur.rowcount > 0


# ============================================================================
# Coach Messages
# ============================================================================

def get_coach_messages(user_id: int, speech_id: int = None, limit: int = 50) -> List[Dict]:
    """Get coach messages for a user."""
    with get_db() as conn:
        cur = conn.cursor()
        if speech_id:
            cur.execute("""
                SELECT * FROM coach_messages 
                WHERE user_id = %s AND speech_id = %s
                ORDER BY created_at ASC
                LIMIT %s
            """, (user_id, speech_id, limit))
        else:
            cur.execute("""
                SELECT * FROM coach_messages 
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (user_id, limit))
        return [dict(row) for row in cur.fetchall()]


def create_coach_message(user_id: int, role: str, content: str, 
                         speech_id: int = None, model: str = None) -> Dict:
    """Create a coach message."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO coach_messages (user_id, speech_id, role, content, model)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
        """, (user_id, speech_id, role, content, model))
        row = cur.fetchone()
        return dict(row)


# ============================================================================
# Projects
# ============================================================================

def get_projects_for_user(user_id: int) -> List[Dict]:
    """Get projects for a user."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM projects 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """, (user_id,))
        return [dict(row) for row in cur.fetchall()]


def create_project(user_id: int, name: str, **fields) -> Dict:
    """Create a new project."""
    all_fields = {'user_id': user_id, 'name': name, **fields}
    columns = list(all_fields.keys())
    values = list(all_fields.values())
    
    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)
    
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            INSERT INTO projects ({col_names})
            VALUES ({placeholders})
            RETURNING *
        """, values)
        row = cur.fetchone()
        return dict(row)


# ============================================================================
# Health Check
# ============================================================================

def check_connection() -> bool:
    """Check if Supabase connection is working."""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            return True
    except Exception as e:
        logger.error(f"Supabase connection failed: {e}")
        return False


def get_stats() -> Dict:
    """Get database statistics."""
    with get_db() as conn:
        cur = conn.cursor()
        stats = {}
        
        for table in ['users', 'speeches', 'analyses', 'transcriptions', 'audio']:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cur.fetchone()['count']
        
        return stats
