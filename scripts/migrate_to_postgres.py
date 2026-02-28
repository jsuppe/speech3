#!/usr/bin/env python3
"""
SQLite to PostgreSQL Migration Script for SpeakFit

Usage:
    python migrate_to_postgres.py --postgres-url "postgresql://user:pass@host:5432/speakfit"

Options:
    --skip-audio     Skip audio BLOB migration (migrate later to object storage)
    --batch-size     Rows per batch (default: 1000)
    --dry-run        Show what would be migrated without doing it
    --users-only     Only migrate user data (skip batch imports)
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("Install psycopg2: pip install psycopg2-binary")
    sys.exit(1)

SQLITE_PATH = Path("/home/melchior/speech3/speechscore.db")

# Tables in dependency order (parents before children)
TABLES = [
    "users",
    "projects", 
    "speeches",
    "audio",
    "transcriptions",
    "analyses",
    "spectrograms",
    "coach_messages",
    "speaker_embeddings",
    "cache",
    "speech_summaries",
    "user_streaks",
    "achievements",
    "user_achievements",
    "activity_log",
    "oratory_cache",
    "content_analysis_cache",
    "user_surveys",
    "dementia_annotations",
]

# Postgres schema (converted from SQLite)
POSTGRES_SCHEMA = '''
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Users
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    email TEXT,
    name TEXT,
    avatar_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    app_version TEXT,
    patch_number INTEGER,
    last_seen TIMESTAMPTZ,
    preferred_accent TEXT DEFAULT 'en-US',
    tier TEXT DEFAULT 'free',
    monthly_recordings INTEGER DEFAULT 0,
    monthly_audio_minutes REAL DEFAULT 0.0,
    usage_month TEXT,
    eula_accepted_at TIMESTAMPTZ,
    tts_voice TEXT DEFAULT 'rachel',
    enabled BOOLEAN DEFAULT TRUE,
    disabled BOOLEAN DEFAULT FALSE,
    UNIQUE(provider, provider_id)
);

-- Projects
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    name TEXT NOT NULL,
    description TEXT,
    profile TEXT DEFAULT 'general',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    type TEXT DEFAULT 'standard',
    reminder_enabled BOOLEAN DEFAULT FALSE,
    reminder_frequency TEXT DEFAULT 'daily',
    reminder_days JSONB DEFAULT '[]',
    reminder_time TEXT DEFAULT '09:00'
);

-- Speeches
CREATE TABLE IF NOT EXISTS speeches (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    speaker TEXT,
    year INTEGER,
    source_url TEXT,
    source_file TEXT,
    duration_sec REAL,
    language TEXT DEFAULT 'en',
    category TEXT,
    products JSONB,
    tags JSONB,
    notes TEXT,
    description TEXT,
    audio_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id INTEGER REFERENCES users(id),
    profile TEXT DEFAULT 'general',
    project_id INTEGER REFERENCES projects(id),
    cached_score REAL,
    cached_score_profile TEXT,
    cached_score_json JSONB,
    pronunciation_mean_confidence REAL,
    pronunciation_std_confidence REAL,
    pronunciation_problem_ratio REAL,
    pronunciation_words_analyzed INTEGER,
    pronunciation_analyzed_at TIMESTAMPTZ,
    pronunciation_features_json JSONB,
    word_analysis_json JSONB,
    phoneme_analysis_at TIMESTAMPTZ,
    grammar_analysis_json JSONB,
    dementia_metrics JSONB,
    is_reference BOOLEAN DEFAULT FALSE,
    reference_text TEXT
);

-- Audio (BLOBs stored as BYTEA or move to object storage)
CREATE TABLE IF NOT EXISTS audio (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER NOT NULL REFERENCES speeches(id) ON DELETE CASCADE,
    format TEXT NOT NULL DEFAULT 'opus',
    data BYTEA NOT NULL,
    original_filename TEXT,
    sample_rate INTEGER,
    channels INTEGER DEFAULT 1,
    bitrate_kbps INTEGER,
    duration_sec REAL,
    size_bytes INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Transcriptions
CREATE TABLE IF NOT EXISTS transcriptions (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER NOT NULL REFERENCES speeches(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    segments JSONB,
    word_count INTEGER,
    language TEXT,
    model TEXT DEFAULT 'large-v3',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Analyses
CREATE TABLE IF NOT EXISTS analyses (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER NOT NULL REFERENCES speeches(id) ON DELETE CASCADE,
    preset TEXT,
    modules JSONB,
    result JSONB NOT NULL,
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
    sentiment TEXT,
    sentiment_positive REAL,
    sentiment_negative REAL,
    sentiment_neutral REAL,
    quality_level TEXT,
    snr_db REAL,
    processing_time_sec REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Spectrograms
CREATE TABLE IF NOT EXISTS spectrograms (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER NOT NULL REFERENCES speeches(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    format TEXT DEFAULT 'png',
    data BYTEA NOT NULL,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Coach messages
CREATE TABLE IF NOT EXISTS coach_messages (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    speech_id INTEGER REFERENCES speeches(id) ON DELETE SET NULL,
    project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    read_at TIMESTAMPTZ
);

-- Speaker embeddings
CREATE TABLE IF NOT EXISTS speaker_embeddings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    speaker_name TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    speech_id INTEGER REFERENCES speeches(id) ON DELETE SET NULL,
    embedding_count INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, speaker_name)
);

-- Cache
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Speech summaries
CREATE TABLE IF NOT EXISTS speech_summaries (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER NOT NULL REFERENCES speeches(id) ON DELETE CASCADE,
    profile TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(speech_id, profile)
);

-- User streaks (gamification)
CREATE TABLE IF NOT EXISTS user_streaks (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    current_streak INTEGER DEFAULT 0,
    longest_streak INTEGER DEFAULT 0,
    last_activity_date DATE,
    total_xp INTEGER DEFAULT 0,
    daily_xp INTEGER DEFAULT 0,
    daily_xp_date DATE,
    streak_freezes INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Achievements
CREATE TABLE IF NOT EXISTS achievements (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    icon TEXT,
    category TEXT,
    xp_reward INTEGER DEFAULT 0,
    requirement_type TEXT,
    requirement_value INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User achievements
CREATE TABLE IF NOT EXISTS user_achievements (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    achievement_id TEXT NOT NULL REFERENCES achievements(id),
    earned_at TIMESTAMPTZ DEFAULT NOW(),
    notified_at TIMESTAMPTZ,
    UNIQUE(user_id, achievement_id)
);

-- Activity log
CREATE TABLE IF NOT EXISTS activity_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    activity_type TEXT NOT NULL,
    xp_earned INTEGER DEFAULT 0,
    metadata_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Oratory emotion cache
CREATE TABLE IF NOT EXISTS oratory_cache (
    speech_id INTEGER PRIMARY KEY REFERENCES speeches(id),
    emotion_data JSONB NOT NULL,
    segment_duration REAL NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Content analysis cache
CREATE TABLE IF NOT EXISTS content_analysis_cache (
    speech_id INTEGER PRIMARY KEY REFERENCES speeches(id),
    analysis_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User surveys
CREATE TABLE IF NOT EXISTS user_surveys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id),
    primary_goal TEXT NOT NULL,
    skill_level TEXT NOT NULL,
    focus_areas TEXT NOT NULL,
    context TEXT NOT NULL,
    time_commitment TEXT NOT NULL,
    native_language TEXT,
    completed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Dementia annotations
CREATE TABLE IF NOT EXISTS dementia_annotations (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER NOT NULL REFERENCES speeches(id),
    annotator_user_id INTEGER,
    segment_index INTEGER NOT NULL,
    categories JSONB,
    notes TEXT,
    is_false_positive BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_speeches_user_id ON speeches(user_id);
CREATE INDEX IF NOT EXISTS idx_speeches_created_at ON speeches(created_at);
CREATE INDEX IF NOT EXISTS idx_speeches_speaker ON speeches(speaker);
CREATE INDEX IF NOT EXISTS idx_speeches_category ON speeches(category);
CREATE INDEX IF NOT EXISTS idx_speeches_audio_hash ON speeches(audio_hash);
CREATE INDEX IF NOT EXISTS idx_analyses_speech_id ON analyses(speech_id);
CREATE INDEX IF NOT EXISTS idx_analyses_quality ON analyses(quality_level);
CREATE INDEX IF NOT EXISTS idx_analyses_wpm ON analyses(wpm);
CREATE INDEX IF NOT EXISTS idx_audio_speech_id ON audio(speech_id);
CREATE INDEX IF NOT EXISTS idx_transcriptions_speech_id ON transcriptions(speech_id);
CREATE INDEX IF NOT EXISTS idx_spectrograms_speech_id ON spectrograms(speech_id);
CREATE INDEX IF NOT EXISTS idx_coach_messages_user_id ON coach_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_coach_messages_speech_id ON coach_messages(speech_id);
CREATE INDEX IF NOT EXISTS idx_coach_messages_project_id ON coach_messages(project_id);
CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_user_id ON speaker_embeddings(user_id);
'''


def parse_json_column(value):
    """Parse JSON string from SQLite to Python object."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def convert_timestamp(value):
    """Convert SQLite timestamp to Postgres-compatible format."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    # Handle various SQLite timestamp formats
    for fmt in [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return value


def convert_bool(value):
    """Convert SQLite integer to Postgres boolean."""
    if value is None:
        return None
    return bool(value)


def migrate_table(sqlite_conn, pg_conn, table_name, batch_size=1000, skip_audio=False, users_only=False):
    """Migrate a single table from SQLite to Postgres."""
    if skip_audio and table_name == "audio":
        print(f"  Skipping {table_name} (--skip-audio)")
        return 0
    
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    # Get column info
    sqlite_cur.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in sqlite_cur.fetchall()]
    
    if not columns:
        print(f"  Skipping {table_name} (no columns)")
        return 0
    
    # Count rows
    if users_only and table_name == "speeches":
        sqlite_cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE user_id IS NOT NULL")
    else:
        sqlite_cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = sqlite_cur.fetchone()[0]
    
    if total_rows == 0:
        print(f"  Skipping {table_name} (empty)")
        return 0
    
    print(f"  Migrating {table_name}: {total_rows:,} rows...")
    
    # Build query
    col_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    
    # For tables with SERIAL id, we need to handle sequence
    has_serial_id = "id" in columns
    
    # Process in batches
    migrated = 0
    offset = 0
    
    while offset < total_rows:
        if users_only and table_name == "speeches":
            sqlite_cur.execute(
                f"SELECT {col_list} FROM {table_name} WHERE user_id IS NOT NULL LIMIT ? OFFSET ?",
                (batch_size, offset)
            )
        else:
            sqlite_cur.execute(
                f"SELECT {col_list} FROM {table_name} LIMIT ? OFFSET ?",
                (batch_size, offset)
            )
        
        rows = sqlite_cur.fetchall()
        if not rows:
            break
        
        # Convert data types
        converted_rows = []
        for row in rows:
            converted = []
            for i, (col, val) in enumerate(zip(columns, row)):
                # JSON columns
                if col in ("products", "tags", "segments", "modules", "result", 
                          "cached_score_json", "pronunciation_features_json",
                          "word_analysis_json", "grammar_analysis_json", 
                          "dementia_metrics", "reminder_days", "emotion_data",
                          "analysis_data", "categories", "metadata_json"):
                    converted.append(json.dumps(parse_json_column(val)) if val else None)
                # Timestamp columns
                elif col.endswith("_at") or col in ("last_login", "last_seen", "last_activity_date", "daily_xp_date"):
                    converted.append(convert_timestamp(val))
                # Boolean columns
                elif col in ("enabled", "disabled", "is_reference", "reminder_enabled", "is_false_positive"):
                    converted.append(convert_bool(val))
                # BLOB columns (audio data, embeddings)
                elif col in ("data", "embedding") and isinstance(val, bytes):
                    converted.append(psycopg2.Binary(val))
                else:
                    converted.append(val)
            converted_rows.append(tuple(converted))
        
        # Insert with ON CONFLICT DO NOTHING for safety
        try:
            insert_sql = f"""
                INSERT INTO {table_name} ({col_list})
                VALUES %s
                ON CONFLICT DO NOTHING
            """
            execute_values(pg_cur, insert_sql, converted_rows)
            pg_conn.commit()
            migrated += len(rows)
        except Exception as e:
            print(f"    Error at offset {offset}: {e}")
            pg_conn.rollback()
            raise
        
        offset += batch_size
        if offset % 10000 == 0:
            print(f"    Progress: {migrated:,} / {total_rows:,}")
    
    # Reset sequence if table has serial id
    if has_serial_id:
        pg_cur.execute(f"SELECT setval(pg_get_serial_sequence('{table_name}', 'id'), COALESCE(MAX(id), 1)) FROM {table_name}")
        pg_conn.commit()
    
    return migrated


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite to PostgreSQL")
    parser.add_argument("--postgres-url", required=True, help="PostgreSQL connection URL")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio BLOB migration")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per batch")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--users-only", action="store_true", help="Only migrate user data")
    parser.add_argument("--schema-only", action="store_true", help="Only create schema")
    args = parser.parse_args()
    
    print(f"SpeakFit SQLite → PostgreSQL Migration")
    print(f"=" * 50)
    print(f"SQLite: {SQLITE_PATH}")
    print(f"Postgres: {args.postgres_url.split('@')[1] if '@' in args.postgres_url else args.postgres_url}")
    print(f"Options: skip_audio={args.skip_audio}, users_only={args.users_only}")
    print()
    
    if args.dry_run:
        print("DRY RUN - No changes will be made")
        print()
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row
    
    # Connect to Postgres
    pg_conn = psycopg2.connect(args.postgres_url)
    
    try:
        # Create schema
        print("Creating PostgreSQL schema...")
        pg_cur = pg_conn.cursor()
        if not args.dry_run:
            pg_cur.execute(POSTGRES_SCHEMA)
            pg_conn.commit()
        print("  Schema created.")
        print()
        
        if args.schema_only:
            print("Schema-only mode, exiting.")
            return
        
        # Migrate tables
        print("Migrating tables...")
        start_time = time.time()
        total_migrated = 0
        
        for table in TABLES:
            if args.dry_run:
                sqlite_cur = sqlite_conn.cursor()
                sqlite_cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = sqlite_cur.fetchone()[0]
                print(f"  Would migrate {table}: {count:,} rows")
            else:
                count = migrate_table(
                    sqlite_conn, pg_conn, table,
                    batch_size=args.batch_size,
                    skip_audio=args.skip_audio,
                    users_only=args.users_only
                )
                total_migrated += count
        
        elapsed = time.time() - start_time
        print()
        print(f"Migration complete!")
        print(f"  Total rows: {total_migrated:,}")
        print(f"  Time: {elapsed:.1f}s")
        
    finally:
        sqlite_conn.close()
        pg_conn.close()


if __name__ == "__main__":
    main()
