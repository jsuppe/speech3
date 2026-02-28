"""
SpeechScore SQLite Database Module

Stores all speech analysis data persistently:
- Speech metadata (title, speaker, source, products)
- Audio files (as Opus blobs)
- Full analysis results (JSON + indexed key metrics)
- Transcriptions (text + segments)
- Spectrograms (PNG blobs)
"""

import sqlite3
import json
import subprocess
import os
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Union, List


DB_PATH = Path(__file__).parent / "speechscore.db"

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS speeches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    speaker TEXT,
    year INTEGER,
    source_url TEXT,
    source_file TEXT,
    duration_sec REAL,
    language TEXT DEFAULT 'en',
    category TEXT,
    products TEXT,           -- JSON array: ["SpeechScore", "ReadAloud"]
    tags TEXT,               -- JSON array: ["clinical", "children"]
    notes TEXT,
    description TEXT,
    audio_hash TEXT,         -- SHA-256 of original audio for dedup
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS audio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speech_id INTEGER NOT NULL REFERENCES speeches(id) ON DELETE CASCADE,
    format TEXT NOT NULL DEFAULT 'opus',
    data BLOB NOT NULL,
    original_filename TEXT,
    sample_rate INTEGER,
    channels INTEGER DEFAULT 1,
    bitrate_kbps INTEGER,
    duration_sec REAL,
    size_bytes INTEGER,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speech_id INTEGER NOT NULL REFERENCES speeches(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    segments TEXT,           -- JSON array: [{start, end, text}, ...]
    word_count INTEGER,
    language TEXT,
    model TEXT DEFAULT 'large-v3',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speech_id INTEGER NOT NULL REFERENCES speeches(id) ON DELETE CASCADE,
    preset TEXT,
    modules TEXT,            -- JSON array of modules used
    result TEXT NOT NULL,    -- Full JSON result from pipeline

    -- Indexed key metrics for fast queries
    wpm REAL,
    articulation_rate REAL,
    pitch_mean_hz REAL,
    pitch_std_hz REAL,
    pitch_range_hz REAL,
    jitter_pct REAL,
    shimmer_pct REAL,
    hnr_db REAL,
    vocal_fry_ratio REAL,
    uptalk_ratio REAL,
    rate_variability_cv REAL,
    f1_mean_hz REAL,
    f2_mean_hz REAL,
    vowel_space_index REAL,
    articulation_clarity REAL,

    lexical_diversity REAL,
    syntactic_complexity REAL,
    fluency_score REAL,
    connectedness REAL,
    mlu REAL,
    fk_grade_level REAL,
    flesch_reading_ease REAL,

    repetition_score REAL,
    anaphora_count INTEGER,
    rhythm_regularity REAL,

    sentiment TEXT,          -- positive/negative/neutral
    sentiment_positive REAL,
    sentiment_negative REAL,
    sentiment_neutral REAL,

    quality_level TEXT,      -- HIGH/MEDIUM/LOW/UNUSABLE
    snr_db REAL,

    processing_time_sec REAL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS spectrograms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speech_id INTEGER NOT NULL REFERENCES speeches(id) ON DELETE CASCADE,
    type TEXT NOT NULL,      -- 'combined', 'spectrogram', 'mel', 'loudness'
    format TEXT DEFAULT 'png',
    data BLOB NOT NULL,
    width INTEGER,
    height INTEGER,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS coach_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    speech_id INTEGER REFERENCES speeches(id) ON DELETE SET NULL,
    role TEXT NOT NULL,        -- 'user' or 'assistant'
    content TEXT NOT NULL,
    model TEXT,                -- which model responded
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_coach_messages_user_id ON coach_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_coach_messages_speech_id ON coach_messages(speech_id);

CREATE TABLE IF NOT EXISTS speaker_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    speaker_name TEXT NOT NULL,
    embedding BLOB NOT NULL,          -- 256-dim float32 vector (Resemblyzer)
    speech_id INTEGER REFERENCES speeches(id) ON DELETE SET NULL,  -- reference recording
    embedding_count INTEGER DEFAULT 1, -- how many recordings contributed to this embedding
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_user_id ON speaker_embeddings(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_speaker_embeddings_user_speaker ON speaker_embeddings(user_id, speaker_name);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_speeches_speaker ON speeches(speaker);
CREATE INDEX IF NOT EXISTS idx_speeches_category ON speeches(category);
CREATE INDEX IF NOT EXISTS idx_speeches_audio_hash ON speeches(audio_hash);
CREATE INDEX IF NOT EXISTS idx_analyses_speech_id ON analyses(speech_id);
CREATE INDEX IF NOT EXISTS idx_analyses_quality ON analyses(quality_level);
CREATE INDEX IF NOT EXISTS idx_analyses_wpm ON analyses(wpm);
CREATE INDEX IF NOT EXISTS idx_audio_speech_id ON audio(speech_id);
CREATE INDEX IF NOT EXISTS idx_transcriptions_speech_id ON transcriptions(speech_id);
CREATE INDEX IF NOT EXISTS idx_spectrograms_speech_id ON spectrograms(speech_id);
"""


class SpeechDB:
    """SQLite database for SpeechScore analysis data."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._conn = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        cur = self.conn.executescript(SCHEMA)
        # Track schema version
        existing = self.conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if not existing:
            self.conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
            )
        self.conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Audio conversion ──────────────────────────────────────────────

    @staticmethod
    def encode_opus(audio_path: str, bitrate_kbps: int = 48) -> bytes:
        """Convert audio file to Opus and return bytes."""
        with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", audio_path,
                    "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k",
                    "-vn", "-ar", "48000", "-ac", "1",
                    tmp_path,
                ],
                capture_output=True, check=True,
            )
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def get_audio_hash(audio_path: str) -> str:
        """SHA-256 hash of audio file for deduplication."""
        h = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def get_audio_duration(audio_path: str) -> Optional[float]:
        """Get audio duration in seconds via ffprobe."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", audio_path],
                capture_output=True, text=True, check=True,
            )
            return float(result.stdout.strip())
        except Exception:
            return None

    # ── Speech CRUD ───────────────────────────────────────────────────

    def add_speech(
        self,
        title: str,
        speaker: str = None,
        year: int = None,
        source_url: str = None,
        source_file: str = None,
        duration_sec: float = None,
        language: str = "en",
        category: str = None,
        products: list = None,
        tags: list = None,
        notes: str = None,
        description: str = None,
        audio_hash: str = None,
        user_id: int = None,
        profile: str = "general",
    ) -> int:
        """Insert a new speech record. Returns speech_id."""
        cur = self.conn.execute(
            """INSERT INTO speeches
               (title, speaker, year, source_url, source_file, duration_sec,
                language, category, products, tags, notes, description, audio_hash, user_id, profile)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                title, speaker, year, source_url, source_file, duration_sec,
                language, category,
                json.dumps(products) if products else None,
                json.dumps(tags) if tags else None,
                notes, description, audio_hash, user_id, profile,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_speech(self, speech_id: int) -> Optional[dict]:
        """Get speech by ID."""
        row = self.conn.execute(
            "SELECT * FROM speeches WHERE id = ?", (speech_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["products"] = json.loads(d["products"]) if d["products"] else []
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
            return d
        return None

    def update_speech(
        self,
        speech_id: int,
        title: str = None,
        speaker: str = None,
        category: str = None,
        tags: list = None,
        notes: str = None,
        profile: str = None,
        project_id: int = None,
        user_id: int = None,
        _unset_project: bool = False,
    ) -> Optional[dict]:
        """
        Update speech metadata. Only updates fields that are provided (not None).
        If user_id is provided, ensures the speech belongs to that user.
        Use project_id=0 or _unset_project=True to unassign from project.
        Returns updated speech or None if not found/unauthorized.
        """
        # Check ownership if user_id provided
        if user_id is not None:
            existing = self.conn.execute(
                "SELECT user_id FROM speeches WHERE id = ?", (speech_id,)
            ).fetchone()
            if not existing or existing["user_id"] != user_id:
                return None
        
        # Build dynamic UPDATE query
        updates = []
        params = []
        
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if speaker is not None:
            updates.append("speaker = ?")
            params.append(speaker)
        if category is not None:
            updates.append("category = ?")
            params.append(category)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        if profile is not None:
            updates.append("profile = ?")
            params.append(profile)
        if project_id is not None or _unset_project:
            updates.append("project_id = ?")
            # project_id of 0 or None means unassign
            params.append(project_id if project_id and project_id > 0 else None)
        
        if not updates:
            # Nothing to update, just return the existing speech
            return self.get_speech(speech_id)
        
        # Always update updated_at
        updates.append("updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')")
        
        query = f"UPDATE speeches SET {', '.join(updates)} WHERE id = ?"
        params.append(speech_id)
        
        self.conn.execute(query, params)
        self.conn.commit()
        
        return self.get_speech(speech_id)

    def delete_speech(self, speech_id: int, user_id: int = None) -> bool:
        """
        Delete a speech and all associated data (audio, transcription, analysis, spectrogram).
        If user_id is provided, ensures the speech belongs to that user.
        Returns True if deleted, False if not found or unauthorized.
        """
        # Check ownership if user_id provided
        if user_id is not None:
            existing = self.conn.execute(
                "SELECT user_id FROM speeches WHERE id = ?", (speech_id,)
            ).fetchone()
            if not existing or existing["user_id"] != user_id:
                return False
        
        # Check if speech exists
        exists = self.conn.execute(
            "SELECT id FROM speeches WHERE id = ?", (speech_id,)
        ).fetchone()
        if not exists:
            return False
        
        # Delete from all related tables (CASCADE should handle this, but be explicit)
        self.conn.execute("DELETE FROM spectrograms WHERE speech_id = ?", (speech_id,))
        self.conn.execute("DELETE FROM analyses WHERE speech_id = ?", (speech_id,))
        self.conn.execute("DELETE FROM transcriptions WHERE speech_id = ?", (speech_id,))
        self.conn.execute("DELETE FROM audio WHERE speech_id = ?", (speech_id,))
        self.conn.execute("DELETE FROM speeches WHERE id = ?", (speech_id,))
        self.conn.commit()
        
        return True

    def find_speech_by_hash(self, audio_hash: str) -> Optional[dict]:
        """Find speech by audio hash (dedup)."""
        row = self.conn.execute(
            "SELECT * FROM speeches WHERE audio_hash = ?", (audio_hash,)
        ).fetchone()
        if row:
            d = dict(row)
            d["products"] = json.loads(d["products"]) if d["products"] else []
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
            return d
        return None

    def list_speeches(
        self,
        category: str = None,
        product: str = None,
        limit: int = 100,
        offset: int = 0,
        user_id: int = None,
        include_unowned: bool = False,
        unassigned: bool = False,
    ) -> list:
        """List speeches with optional filters. If user_id is set, only return that user's speeches.
        If unassigned=True, only return speeches not assigned to any project."""
        # Explicitly select columns, excluding heavy cached_score_json (3.5KB avg)
        # Join transcriptions to get word_count for WPM calculation
        query = """SELECT s.id, s.title, s.speaker, s.year, s.source_url, s.source_file, s.duration_sec,
                   s.language, s.category, s.products, s.tags, s.notes, s.description, s.audio_hash,
                   s.created_at, s.updated_at, s.user_id, s.profile, s.project_id,
                   s.cached_score, s.cached_score_profile, t.word_count
                   FROM speeches s
                   LEFT JOIN transcriptions t ON t.speech_id = s.id
                   WHERE 1=1"""
        params = []
        if user_id is not None:
            query += " AND s.user_id = ?"
            params.append(user_id)
        elif not include_unowned:
            # If no user_id filter and not explicitly including unowned, exclude NULL user_id
            query += " AND s.user_id IS NOT NULL"
        if unassigned:
            query += " AND s.project_id IS NULL"
        if category:
            query += " AND s.category = ?"
            params.append(category)
        if product:
            query += " AND s.products LIKE ?"
            params.append(f'%"{product}"%')
        query += " ORDER BY s.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["products"] = json.loads(d["products"]) if d["products"] else []
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
            results.append(d)
        return results

    def count_speeches(self, user_id: int = None, include_unowned: bool = False, unassigned: bool = False) -> int:
        """Total speech count, optionally filtered by user. If unassigned=True, only count unassigned."""
        query = "SELECT COUNT(*) FROM speeches WHERE 1=1"
        params = []
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        elif not include_unowned:
            query += " AND user_id IS NOT NULL"
        if unassigned:
            query += " AND project_id IS NULL"
        return self.conn.execute(query, params).fetchone()[0]

    # ── Projects ─────────────────────────────────────────────────────

    def create_project(
        self, user_id: int, name: str, description: str = None, profile: str = "general", type: str = "standard",
        reminder_enabled: bool = False, reminder_frequency: str = "daily", reminder_days: list = None, reminder_time: str = "09:00"
    ) -> int:
        """Create a new project. Returns project id. Type can be 'standard' or 'group_meeting'."""
        days_json = json.dumps(reminder_days or [])
        cur = self.conn.execute(
            """INSERT INTO projects (user_id, name, description, profile, type, 
               reminder_enabled, reminder_frequency, reminder_days, reminder_time) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, name, description, profile, type, reminder_enabled, reminder_frequency, days_json, reminder_time)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_project(self, project_id: int) -> Optional[dict]:
        """Get a project by id."""
        row = self.conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        # Parse reminder_days JSON
        if d.get("reminder_days"):
            try:
                d["reminder_days"] = json.loads(d["reminder_days"])
            except:
                d["reminder_days"] = []
        else:
            d["reminder_days"] = []
        return d

    def list_projects(self, user_id: int) -> list:
        """List all projects for a user."""
        rows = self.conn.execute(
            "SELECT * FROM projects WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            # Parse reminder_days JSON
            if d.get("reminder_days"):
                try:
                    d["reminder_days"] = json.loads(d["reminder_days"])
                except:
                    d["reminder_days"] = []
            else:
                d["reminder_days"] = []
            results.append(d)
        return results

    def update_project(
        self, project_id: int, name: str = None, description: str = None, profile: str = None, type: str = None,
        reminder_enabled: bool = None, reminder_frequency: str = None, reminder_days: list = None, reminder_time: str = None
    ) -> bool:
        """Update a project. Returns True if updated."""
        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if profile is not None:
            updates.append("profile = ?")
            params.append(profile)
        if type is not None:
            updates.append("type = ?")
            params.append(type)
        if reminder_enabled is not None:
            updates.append("reminder_enabled = ?")
            params.append(reminder_enabled)
        if reminder_frequency is not None:
            updates.append("reminder_frequency = ?")
            params.append(reminder_frequency)
        if reminder_days is not None:
            updates.append("reminder_days = ?")
            params.append(json.dumps(reminder_days))
        if reminder_time is not None:
            updates.append("reminder_time = ?")
            params.append(reminder_time)
        if not updates:
            return False
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(project_id)
        self.conn.execute(f"UPDATE projects SET {', '.join(updates)} WHERE id = ?", params)
        self.conn.commit()
        return True

    def delete_project(self, project_id: int, delete_speeches: bool = True) -> int:
        """Delete a project. Returns number of speeches deleted if delete_speeches=True."""
        deleted_count = 0
        if delete_speeches:
            # Delete associated speeches
            cur = self.conn.execute("DELETE FROM speeches WHERE project_id = ?", (project_id,))
            deleted_count = cur.rowcount
        else:
            # Just unlink speeches from project
            self.conn.execute("UPDATE speeches SET project_id = NULL WHERE project_id = ?", (project_id,))
        self.conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self.conn.commit()
        return deleted_count

    def assign_speech_to_project(self, speech_id: int, project_id: int) -> bool:
        """Assign a speech to a project. Returns True if updated."""
        cur = self.conn.execute(
            "UPDATE speeches SET project_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (project_id, speech_id)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def get_project_speeches(self, project_id: int) -> list:
        """Get all speeches in a project."""
        rows = self.conn.execute(
            "SELECT * FROM speeches WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,)
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["products"] = json.loads(d["products"]) if d["products"] else []
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
            results.append(d)
        return results

    def count_project_speeches(self, project_id: int) -> int:
        """Count speeches in a project."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM speeches WHERE project_id = ?", (project_id,)
        ).fetchone()[0]

    def cache_speech_score(self, speech_id: int, score: float, profile: str) -> None:
        """Cache the composite score for a speech to avoid recalculation."""
        self.conn.execute(
            "UPDATE speeches SET cached_score = ?, cached_score_profile = ? WHERE id = ?",
            (score, profile, speech_id)
        )
        self.conn.commit()

    def cache_speech_score_full(self, speech_id: int, score_json: str, score: float, profile: str) -> None:
        """Cache the full score JSON including summary for instant loads."""
        self.conn.execute(
            "UPDATE speeches SET cached_score_json = ?, cached_score = ?, cached_score_profile = ? WHERE id = ?",
            (score_json, score, profile, speech_id)
        )
        self.conn.commit()

    def invalidate_score_cache(self, speech_id: int) -> None:
        """Clear cached score when analysis is updated."""
        self.conn.execute(
            "UPDATE speeches SET cached_score = NULL, cached_score_profile = NULL WHERE id = ?",
            (speech_id,)
        )
        self.conn.commit()

    def get_cache(self, key: str) -> Optional[dict]:
        """Get a cached value by key."""
        row = self.conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set_cache(self, key: str, value: dict) -> None:
        """Set a cached value."""
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))",
            (key, json.dumps(value))
        )
        self.conn.commit()

    def delete_cache(self, key: str) -> None:
        """Delete a cached value."""
        self.conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        self.conn.commit()

    def clear_cache_prefix(self, prefix: str) -> int:
        """Clear all cache entries with a given prefix. Returns count deleted."""
        cur = self.conn.execute("DELETE FROM cache WHERE key LIKE ?", (f"{prefix}%",))
        self.conn.commit()
        return cur.rowcount

    def get_meeting_data(self, project_id: int) -> dict:
        """Get meeting data for a group_meeting project.
        Returns participants, chronological transcripts, and basic stats.
        Uses diarization results when available.
        """
        import json
        
        # Get all speeches with their transcripts and analysis in chronological order
        rows = self.conn.execute(
            """SELECT s.id, s.speaker, s.created_at, s.duration_sec, 
                      t.text as transcript, a.result as analysis_result
               FROM speeches s
               LEFT JOIN transcriptions t ON s.id = t.speech_id
               LEFT JOIN analyses a ON s.id = a.speech_id
               WHERE s.project_id = ? 
               ORDER BY s.created_at ASC""",
            (project_id,)
        ).fetchall()
        
        # Collect all speakers from diarization results
        all_speakers = set()
        transcripts = []
        
        for row in rows:
            # Try to get speakers from diarization
            diarization_speakers = []
            if row["analysis_result"]:
                try:
                    result = json.loads(row["analysis_result"])
                    diarization = result.get("diarization", {})
                    speakers_dict = diarization.get("speakers", {})
                    # Get speaker names from diarization (excluding UNKNOWN)
                    for spk_name in speakers_dict.keys():
                        if spk_name != "UNKNOWN":
                            diarization_speakers.append(spk_name)
                            all_speakers.add(spk_name)
                    # If only UNKNOWN speaker, use the speech's speaker field or generic name
                    if not diarization_speakers and "UNKNOWN" in speakers_dict:
                        num_unknown = diarization.get("num_speakers", 1)
                        if row["speaker"]:
                            diarization_speakers.append(row["speaker"])
                            all_speakers.add(row["speaker"])
                        else:
                            for i in range(num_unknown):
                                spk = f"Speaker {len(all_speakers) + 1}"
                                diarization_speakers.append(spk)
                                all_speakers.add(spk)
                except (json.JSONDecodeError, KeyError):
                    pass
            
            # Fallback if no diarization
            if not diarization_speakers:
                if row["speaker"]:
                    diarization_speakers = [row["speaker"]]
                    all_speakers.add(row["speaker"])
                else:
                    spk = f"Speaker {len(all_speakers) + 1}"
                    diarization_speakers = [spk]
                    all_speakers.add(spk)
            
            transcripts.append({
                "id": row["id"],
                "speaker": diarization_speakers[0] if len(diarization_speakers) == 1 else ", ".join(diarization_speakers),
                "transcript": row["transcript"] or "",
                "timestamp": row["created_at"],
                "duration_sec": row["duration_sec"],
                "speakers_detected": len(diarization_speakers),
            })
        
        # Calculate total duration
        total_duration = sum(row["duration_sec"] or 0 for row in rows)
        
        # Build per-participant stats
        participant_stats = {}
        for spk in all_speakers:
            participant_stats[spk] = {
                "name": spk,
                "word_count": 0,
                "recording_count": 0,
                "total_duration_sec": 0,
                "transcripts": [],
            }
        
        for t in transcripts:
            speaker = t["speaker"]
            if speaker in participant_stats:
                words = len(t["transcript"].split()) if t["transcript"] else 0
                participant_stats[speaker]["word_count"] += words
                participant_stats[speaker]["recording_count"] += 1
                participant_stats[speaker]["total_duration_sec"] += t["duration_sec"] or 0
                participant_stats[speaker]["transcripts"].append(t["transcript"])
        
        # Build participant summaries (text snippets for each speaker)
        participant_summaries = []
        for spk, stats in sorted(participant_stats.items()):
            # Combine all their transcripts
            combined_text = " ".join(stats["transcripts"])
            # Get first 500 chars as preview
            preview = combined_text[:500] + "..." if len(combined_text) > 500 else combined_text
            
            participant_summaries.append({
                "name": spk,
                "word_count": stats["word_count"],
                "recording_count": stats["recording_count"],
                "total_duration_sec": stats["total_duration_sec"],
                "text_preview": preview,
                "full_text": combined_text,
            })
        
        return {
            "participants": sorted(all_speakers),
            "participant_count": len(all_speakers),
            "recording_count": len(rows),
            "total_duration_sec": total_duration,
            "transcripts": transcripts,
            "participant_summaries": participant_summaries,
        }

    # ── Audio storage ─────────────────────────────────────────────────

    def store_audio(
        self,
        speech_id: int,
        audio_path: str,
        encode_to_opus: bool = True,
        bitrate_kbps: int = 48,
    ) -> int:
        """Store audio file (converted to Opus by default). Returns audio row id."""
        if encode_to_opus:
            data = self.encode_opus(audio_path, bitrate_kbps)
            fmt = "opus"
        else:
            with open(audio_path, "rb") as f:
                data = f.read()
            fmt = Path(audio_path).suffix.lstrip(".")

        duration = self.get_audio_duration(audio_path)
        cur = self.conn.execute(
            """INSERT INTO audio
               (speech_id, format, data, original_filename, sample_rate,
                channels, bitrate_kbps, duration_sec, size_bytes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                speech_id, fmt, data, Path(audio_path).name,
                48000 if encode_to_opus else None,
                1, bitrate_kbps if encode_to_opus else None,
                duration, len(data),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_audio(self, speech_id: int) -> Optional[dict]:
        """Get audio blob for a speech."""
        row = self.conn.execute(
            "SELECT * FROM audio WHERE speech_id = ? ORDER BY id DESC LIMIT 1",
            (speech_id,),
        ).fetchone()
        return dict(row) if row else None

    def export_audio(self, speech_id: int, output_path: str) -> bool:
        """Export stored audio to a file."""
        audio = self.get_audio(speech_id)
        if not audio:
            return False
        with open(output_path, "wb") as f:
            f.write(audio["data"])
        return True

    # ── Transcription storage ─────────────────────────────────────────

    def store_transcription(
        self,
        speech_id: int,
        text: str,
        segments: list = None,
        word_count: int = None,
        language: str = None,
        model: str = "large-v3",
    ) -> int:
        """Store transcription. Returns row id."""
        if word_count is None:
            word_count = len(text.split())
        cur = self.conn.execute(
            """INSERT INTO transcriptions
               (speech_id, text, segments, word_count, language, model)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                speech_id, text,
                json.dumps(segments) if segments else None,
                word_count, language, model,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_transcription(self, speech_id: int) -> Optional[dict]:
        """Get latest transcription for a speech."""
        row = self.conn.execute(
            "SELECT * FROM transcriptions WHERE speech_id = ? ORDER BY id DESC LIMIT 1",
            (speech_id,),
        ).fetchone()
        if row:
            d = dict(row)
            d["segments"] = json.loads(d["segments"]) if d["segments"] else []
            return d
        return None

    # ── Analysis storage ──────────────────────────────────────────────

    def store_analysis(
        self,
        speech_id: int,
        result: dict,
        preset: str = None,
        modules: list = None,
    ) -> int:
        """
        Store full analysis result. Extracts key metrics for indexed columns.
        Returns row id.
        """
        # Extract key metrics from the result JSON
        # Handle both API-style and pipeline-style field names
        aa = result.get("audio_analysis", {})
        sr = aa.get("speaking_rate", {})
        vq = aa.get("voice_quality", {})
        pitch = aa.get("pitch", {})
        adv = result.get("advanced_audio_analysis", {})
        fry = adv.get("vocal_fry", {})
        txt = result.get("speech3_text_analysis", {})
        metrics = txt.get("metrics", {})
        adv_txt = result.get("advanced_text_analysis", {})
        readability = adv_txt.get("readability", {})
        rhet = result.get("rhetorical_analysis", {})
        rep = rhet.get("repetition", {})
        sent = result.get("sentiment", {})
        overall_sent = sent.get("overall", {})
        aq = result.get("audio_quality", {})
        timing = result.get("timing", result.get("processing", {}).get("timing", {}))

        # Calculate total processing time
        proc_time = None
        if timing:
            proc_time = sum(
                v for v in timing.values()
                if isinstance(v, (int, float))
            )

        cur = self.conn.execute(
            """INSERT INTO analyses
               (speech_id, preset, modules, result,
                wpm, articulation_rate,
                pitch_mean_hz, pitch_std_hz, pitch_range_hz,
                jitter_pct, shimmer_pct, hnr_db,
                vocal_fry_ratio, uptalk_ratio, rate_variability_cv,
                f1_mean_hz, f2_mean_hz, vowel_space_index, articulation_clarity,
                lexical_diversity, syntactic_complexity, fluency_score,
                connectedness, mlu, fk_grade_level, flesch_reading_ease,
                repetition_score, anaphora_count, rhythm_regularity,
                sentiment, sentiment_positive, sentiment_negative, sentiment_neutral,
                quality_level, snr_db, processing_time_sec)
               VALUES (?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?,
                       ?, ?, ?, ?,
                       ?, ?, ?)""",
            (
                speech_id, preset,
                json.dumps(modules) if modules else None,
                json.dumps(result),
                # Audio metrics (handle both field name variants)
                _f(sr.get("words_per_minute") or sr.get("overall_wpm")),
                _f(sr.get("articulation_rate") or sr.get("articulation_wpm")),
                _f(pitch.get("mean_f0") or pitch.get("mean_hz")),
                _f(pitch.get("std_f0") or pitch.get("std_hz")),
                _f(pitch.get("range_f0") or pitch.get("range_hz")),
                _f(vq.get("jitter_percent")),
                _f(vq.get("shimmer_percent")),
                _f(vq.get("hnr_db") or vq.get("harmonics_to_noise_db")),
                _f(fry.get("vocal_fry_ratio")),
                _f(adv.get("uptalk", {}).get("uptalk_ratio")),
                _f(adv.get("rate_variability", {}).get("cv")),
                _f(adv.get("formants", {}).get("f1_mean")),
                _f(adv.get("formants", {}).get("f2_mean")),
                _f(adv.get("formants", {}).get("vowel_space_index")),
                _f(adv.get("formants", {}).get("articulation_clarity")),
                # Text metrics
                _f(metrics.get("lexical_diversity")),
                _f(metrics.get("syntactic_complexity")),
                _f(metrics.get("fluency_score")),
                _f(metrics.get("connectedness")),
                _f(metrics.get("mlu")),
                _f(readability.get("flesch_kincaid_grade")),
                _f(readability.get("flesch_reading_ease")),
                # Rhetorical
                _f(rep.get("repetition_score")),
                _i(rhet.get("anaphora", {}).get("count")),
                _f(rhet.get("rhythm", {}).get("regularity")),
                # Sentiment
                overall_sent.get("label") if isinstance(overall_sent, dict) else str(overall_sent) if overall_sent else None,
                _f(overall_sent.get("positive")) if isinstance(overall_sent, dict) else None,
                _f(overall_sent.get("negative")) if isinstance(overall_sent, dict) else None,
                _f(overall_sent.get("neutral")) if isinstance(overall_sent, dict) else None,
                # Quality (handle both field name variants)
                aq.get("quality_level") or aq.get("confidence"),
                _f(aq.get("snr_db")),
                proc_time,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_analysis(self, speech_id: int) -> Optional[dict]:
        """Get latest analysis for a speech."""
        row = self.conn.execute(
            "SELECT * FROM analyses WHERE speech_id = ? ORDER BY id DESC LIMIT 1",
            (speech_id,),
        ).fetchone()
        if row:
            d = dict(row)
            d["result"] = json.loads(d["result"]) if d["result"] else {}
            d["modules"] = json.loads(d["modules"]) if d["modules"] else []
            return d
        return None

    def get_full_analysis(self, speech_id: int) -> Optional[dict]:
        """Get the full JSON result for a speech."""
        row = self.conn.execute(
            "SELECT result FROM analyses WHERE speech_id = ? ORDER BY id DESC LIMIT 1",
            (speech_id,),
        ).fetchone()
        if row:
            return json.loads(row["result"])
        return None

    # ── Spectrogram storage ───────────────────────────────────────────

    def store_spectrogram(
        self,
        speech_id: int,
        image_data: bytes,
        spec_type: str = "combined",
        fmt: str = "png",
        width: int = None,
        height: int = None,
    ) -> int:
        """Store spectrogram image. Returns row id."""
        cur = self.conn.execute(
            """INSERT INTO spectrograms
               (speech_id, type, format, data, width, height)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (speech_id, spec_type, fmt, image_data, width, height),
        )
        self.conn.commit()
        return cur.lastrowid

    def store_spectrogram_from_file(
        self,
        speech_id: int,
        image_path: str,
        spec_type: str = "combined",
    ) -> int:
        """Store spectrogram from image file."""
        with open(image_path, "rb") as f:
            data = f.read()
        fmt = Path(image_path).suffix.lstrip(".")
        return self.store_spectrogram(speech_id, data, spec_type, fmt)

    def get_spectrogram(self, speech_id: int, spec_type: str = "combined") -> Optional[dict]:
        """Get spectrogram for a speech."""
        row = self.conn.execute(
            "SELECT * FROM spectrograms WHERE speech_id = ? AND type = ? ORDER BY id DESC LIMIT 1",
            (speech_id, spec_type),
        ).fetchone()
        return dict(row) if row else None

    # ── Convenience: full ingest ──────────────────────────────────────

    def ingest(
        self,
        audio_path: str,
        title: str,
        api_result: dict,
        speaker: str = None,
        year: int = None,
        source_url: str = None,
        category: str = None,
        products: list = None,
        tags: list = None,
        description: str = None,
        notes: str = None,
        preset: str = "full",
        spectrogram_path: str = None,
        skip_audio: bool = False,
    ) -> int:
        """
        Full ingest: speech + audio + transcription + analysis + spectrogram.
        Returns speech_id.
        """
        audio_hash = self.get_audio_hash(audio_path)
        duration = self.get_audio_duration(audio_path)

        # Check for duplicate
        existing = self.find_speech_by_hash(audio_hash)
        if existing:
            print(f"  ⚠ Duplicate audio detected (speech_id={existing['id']}): {existing['title']}")
            return existing["id"]

        # 1. Create speech record
        speech_id = self.add_speech(
            title=title, speaker=speaker, year=year,
            source_url=source_url, source_file=Path(audio_path).name,
            duration_sec=duration, category=category,
            products=products or ["SpeechScore"],
            tags=tags, notes=notes, description=description,
            audio_hash=audio_hash,
        )

        # 2. Store audio as Opus
        if not skip_audio:
            self.store_audio(speech_id, audio_path)

        # 3. Store transcription
        tx = api_result.get("transcription", {})
        if tx.get("text"):
            self.store_transcription(
                speech_id,
                text=tx["text"],
                segments=tx.get("segments"),
                language=tx.get("language"),
            )

        # 4. Store analysis
        self.store_analysis(speech_id, api_result, preset=preset)

        # 5. Store spectrogram if provided
        if spectrogram_path and os.path.exists(spectrogram_path):
            self.store_spectrogram_from_file(speech_id, spectrogram_path)

        return speech_id

    # ── Query helpers ─────────────────────────────────────────────────

    def compare_speeches(self, speech_ids: list) -> list:
        """Get key metrics for comparison across speeches."""
        placeholders = ",".join("?" * len(speech_ids))
        rows = self.conn.execute(
            f"""SELECT s.id, s.title, s.speaker, s.category,
                       a.wpm, a.pitch_mean_hz, a.pitch_std_hz,
                       a.jitter_pct, a.shimmer_pct, a.hnr_db,
                       a.vocal_fry_ratio, a.lexical_diversity,
                       a.fk_grade_level, a.repetition_score,
                       a.sentiment, a.quality_level
                FROM speeches s
                JOIN analyses a ON a.speech_id = s.id
                WHERE s.id IN ({placeholders})
                ORDER BY s.id""",
            speech_ids,
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Get DB statistics."""
        speeches = self.count_speeches()
        analyses = self.conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        audio_count = self.conn.execute("SELECT COUNT(*) FROM audio").fetchone()[0]
        # Skip slow SUM(size_bytes) query — estimate from DB file size instead
        # audio_size = self.conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM audio").fetchone()[0]
        spectrograms = self.conn.execute("SELECT COUNT(*) FROM spectrograms").fetchone()[0]
        
        # Duration stats
        duration_stats = self.conn.execute(
            "SELECT COALESCE(SUM(duration_sec), 0), COALESCE(AVG(duration_sec), 0) FROM speeches WHERE duration_sec > 0"
        ).fetchone()
        total_seconds = duration_stats[0] or 0
        avg_duration = duration_stats[1] or 0

        # Product distribution
        products = {}
        for row in self.conn.execute("SELECT products FROM speeches WHERE products IS NOT NULL"):
            for p in json.loads(row[0]):
                products[p] = products.get(p, 0) + 1

        # Estimate audio size from DB file size (audio is ~50% of DB)
        db_size_mb = round(os.path.getsize(self.db_path) / (1024 * 1024), 2) if self.db_path.exists() else 0
        audio_size_mb = round(db_size_mb * 0.5, 2)  # Approximate
        
        return {
            "speeches": speeches,
            "analyses": analyses,
            "audio_files": audio_count,
            "audio_size_mb": audio_size_mb,
            "spectrograms": spectrograms,
            "total_minutes": round(total_seconds / 60, 1),
            "avg_duration": round(avg_duration, 1),
            "products": products,
            "db_size_mb": db_size_mb,
        }


    # ── Coach Messages ─────────────────────────────────────────────────────

    def save_coach_message(
        self,
        user_id: int,
        role: str,
        content: str,
        speech_id: Optional[int] = None,
        model: Optional[str] = None,
        project_id: Optional[int] = None,
    ) -> int:
        """Save a coach chat message. Returns message ID."""
        cur = self.conn.execute(
            """
            INSERT INTO coach_messages (user_id, speech_id, role, content, model, project_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, speech_id, role, content, model, project_id),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_coach_messages(
        self,
        user_id: int,
        limit: int = 100,
        speech_id: Optional[int] = None,
    ) -> List[dict]:
        """Get coach messages for a user, optionally filtered by speech."""
        if speech_id:
            rows = self.conn.execute(
                """
                SELECT cm.*, s.title as speech_title
                FROM coach_messages cm
                LEFT JOIN speeches s ON s.id = cm.speech_id
                WHERE cm.user_id = ? AND cm.speech_id = ?
                ORDER BY cm.created_at DESC
                LIMIT ?
                """,
                (user_id, speech_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT cm.*, s.title as speech_title
                FROM coach_messages cm
                LEFT JOIN speeches s ON s.id = cm.speech_id
                WHERE cm.user_id = ?
                ORDER BY cm.created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_user_coach_stats(self, user_id: int) -> dict:
        """Get coach usage stats for a user."""
        row = self.conn.execute(
            """
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT speech_id) as speeches_discussed,
                MIN(created_at) as first_message,
                MAX(created_at) as last_message
            FROM coach_messages
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        return dict(row) if row else {}

    # ── Speaker Embeddings (Voice Fingerprinting) ─────────────────────

    def store_speaker_embedding(
        self,
        user_id: int,
        speaker_name: str,
        embedding: bytes,
        speech_id: Optional[int] = None,
    ) -> int:
        """
        Store or update a speaker's voice embedding.
        
        If an embedding already exists for this user+speaker, we average them
        to create a more robust fingerprint over time.
        
        Args:
            user_id: User who owns this speaker profile
            speaker_name: Display name for the speaker
            embedding: 256-dim float32 numpy array as bytes
            speech_id: Reference recording (optional)
        
        Returns:
            Embedding ID
        """
        import numpy as np
        
        # Check if embedding already exists for this user+speaker
        existing = self.conn.execute(
            "SELECT id, embedding, embedding_count FROM speaker_embeddings WHERE user_id = ? AND speaker_name = ?",
            (user_id, speaker_name),
        ).fetchone()
        
        if existing:
            # Average the embeddings for more robust fingerprinting
            old_embedding = np.frombuffer(existing["embedding"], dtype=np.float32)
            new_embedding = np.frombuffer(embedding, dtype=np.float32)
            count = existing["embedding_count"]
            
            # Weighted average: more weight to accumulated embedding
            averaged = (old_embedding * count + new_embedding) / (count + 1)
            averaged = averaged / np.linalg.norm(averaged)  # Re-normalize
            
            self.conn.execute(
                """
                UPDATE speaker_embeddings 
                SET embedding = ?, embedding_count = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                WHERE id = ?
                """,
                (averaged.tobytes(), count + 1, existing["id"]),
            )
            self.conn.commit()
            return existing["id"]
        else:
            # Create new embedding
            cur = self.conn.execute(
                """
                INSERT INTO speaker_embeddings (user_id, speaker_name, embedding, speech_id, embedding_count)
                VALUES (?, ?, ?, ?, 1)
                """,
                (user_id, speaker_name, embedding, speech_id),
            )
            self.conn.commit()
            return cur.lastrowid

    def find_matching_speaker(
        self,
        user_id: int,
        embedding: bytes,
        threshold: float = 0.80,
    ) -> Optional[dict]:
        """
        Find a matching speaker by voice embedding similarity.
        
        Args:
            user_id: User to search within
            embedding: Query embedding (256-dim float32 bytes)
            threshold: Cosine similarity threshold (0.80 = quite confident match)
        
        Returns:
            Dict with speaker_name and similarity, or None if no match
        """
        import numpy as np
        
        query_embed = np.frombuffer(embedding, dtype=np.float32)
        
        rows = self.conn.execute(
            "SELECT id, speaker_name, embedding FROM speaker_embeddings WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        
        if not rows:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for row in rows:
            stored_embed = np.frombuffer(row["embedding"], dtype=np.float32)
            # Cosine similarity
            similarity = float(np.dot(query_embed, stored_embed))
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = {
                    "speaker_name": row["speaker_name"],
                    "similarity": round(similarity, 3),
                    "embedding_id": row["id"],
                }
        
        return best_match

    def get_speaker_embeddings(self, user_id: int) -> List[dict]:
        """Get all speaker embeddings for a user (without the raw embedding data)."""
        rows = self.conn.execute(
            """
            SELECT id, speaker_name, speech_id, embedding_count, created_at, updated_at
            FROM speaker_embeddings
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_speaker_embedding(self, user_id: int, speaker_name: str) -> bool:
        """Delete a speaker embedding."""
        cur = self.conn.execute(
            "DELETE FROM speaker_embeddings WHERE user_id = ? AND speaker_name = ?",
            (user_id, speaker_name),
        )
        self.conn.commit()
        return cur.rowcount > 0


def _f(v) -> Optional[float]:
    """Safely cast to float."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v) -> Optional[int]:
    """Safely cast to int."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    db = SpeechDB()

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        s = db.stats()
        print(f"Speeches:     {s['speeches']}")
        print(f"Analyses:     {s['analyses']}")
        print(f"Audio files:  {s['audio_files']} ({s['audio_size_mb']} MB)")
        print(f"Spectrograms: {s['spectrograms']}")
        print(f"DB size:      {s['db_size_mb']} MB")
        print(f"Products:     {s['products']}")
    elif len(sys.argv) > 1 and sys.argv[1] == "list":
        for s in db.list_speeches():
            print(f"  [{s['id']:3d}] {s['title']} — {s['speaker'] or '?'} ({s['products']})")
    else:
        print("Usage: python speech_db.py [stats|list]")
        print(f"DB path: {db.db_path}")

    db.close()
