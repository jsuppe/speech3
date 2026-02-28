#!/usr/bin/env python3
"""
Migrate ONLY user data to PostgreSQL (cloud-ready).
Skips batch imports (LibriSpeech, LJSpeech, etc.)

This is what goes to the cloud:
- users (5)
- projects  
- speeches (user_id IS NOT NULL only)
- audio (for user speeches only)
- transcriptions (for user speeches only)
- analyses (for user speeches only)
- coach_messages
- speaker_embeddings
- user_streaks, achievements, etc.

What stays local:
- 114K batch imported speeches (no user_id)
- Reference speeches
- Training/benchmark data
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import execute_values, RealDictCursor
except ImportError:
    print("Install psycopg2: pip install psycopg2-binary")
    sys.exit(1)

SQLITE_PATH = Path("/home/melchior/speech3/speechscore.db")


def get_schema():
    """Return PostgreSQL schema for user data tables."""
    return '''
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
    tts_voice TEXT DEFAULT 'elli',
    enabled BOOLEAN DEFAULT TRUE,
    UNIQUE(provider, provider_id)
);

-- Projects
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    profile TEXT DEFAULT 'general',
    type TEXT DEFAULT 'standard',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Speeches (user recordings only)
CREATE TABLE IF NOT EXISTS speeches (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    speaker TEXT,
    profile TEXT DEFAULT 'general',
    duration_sec REAL,
    language TEXT DEFAULT 'en',
    category TEXT,
    tags JSONB,
    notes TEXT,
    reference_text TEXT,
    cached_score REAL,
    cached_score_profile TEXT,
    cached_score_json JSONB,
    pronunciation_mean_confidence REAL,
    pronunciation_problem_ratio REAL,
    word_analysis_json JSONB,
    grammar_analysis_json JSONB,
    dementia_metrics JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Audio blobs (small for user data, ~14MB total)
CREATE TABLE IF NOT EXISTS audio (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER NOT NULL REFERENCES speeches(id) ON DELETE CASCADE,
    format TEXT NOT NULL DEFAULT 'opus',
    data BYTEA NOT NULL,
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
    result JSONB NOT NULL,
    wpm REAL,
    pitch_mean_hz REAL,
    pitch_std_hz REAL,
    jitter_pct REAL,
    shimmer_pct REAL,
    hnr_db REAL,
    lexical_diversity REAL,
    fluency_score REAL,
    fk_grade_level REAL,
    sentiment TEXT,
    quality_level TEXT,
    snr_db REAL,
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

-- Speaker voice profiles
CREATE TABLE IF NOT EXISTS speaker_embeddings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    speaker_name TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    embedding_count INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, speaker_name)
);

-- Gamification
CREATE TABLE IF NOT EXISTS user_streaks (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    current_streak INTEGER DEFAULT 0,
    longest_streak INTEGER DEFAULT 0,
    last_activity_date DATE,
    total_xp INTEGER DEFAULT 0,
    streak_freezes INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS achievements (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    icon TEXT,
    xp_reward INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_achievements (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    achievement_id TEXT NOT NULL REFERENCES achievements(id),
    earned_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, achievement_id)
);

-- User preferences/surveys
CREATE TABLE IF NOT EXISTS user_surveys (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    primary_goal TEXT,
    skill_level TEXT,
    focus_areas TEXT,
    context TEXT,
    time_commitment TEXT,
    native_language TEXT,
    completed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Caregiver annotations
CREATE TABLE IF NOT EXISTS dementia_annotations (
    id SERIAL PRIMARY KEY,
    speech_id INTEGER NOT NULL REFERENCES speeches(id) ON DELETE CASCADE,
    annotator_user_id INTEGER REFERENCES users(id),
    segment_index INTEGER NOT NULL,
    categories JSONB,
    notes TEXT,
    is_false_positive BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_speeches_user_id ON speeches(user_id);
CREATE INDEX IF NOT EXISTS idx_speeches_created_at ON speeches(created_at);
CREATE INDEX IF NOT EXISTS idx_speeches_profile ON speeches(profile);
CREATE INDEX IF NOT EXISTS idx_audio_speech_id ON audio(speech_id);
CREATE INDEX IF NOT EXISTS idx_transcriptions_speech_id ON transcriptions(speech_id);
CREATE INDEX IF NOT EXISTS idx_analyses_speech_id ON analyses(speech_id);
CREATE INDEX IF NOT EXISTS idx_coach_messages_user_id ON coach_messages(user_id);
'''


def migrate_users(sqlite_cur, pg_cur):
    """Migrate users table."""
    sqlite_cur.execute("SELECT * FROM users")
    rows = sqlite_cur.fetchall()
    
    for row in rows:
        pg_cur.execute("""
            INSERT INTO users (id, provider, provider_id, email, name, avatar_url,
                created_at, last_login, app_version, patch_number, preferred_accent,
                tier, monthly_recordings, monthly_audio_minutes, usage_month, tts_voice, enabled)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            row['id'], row['provider'], row['provider_id'], row['email'], row['name'],
            row['avatar_url'], row['created_at'], row['last_login'], row['app_version'],
            row['patch_number'], row['preferred_accent'], row['tier'],
            row['monthly_recordings'], row['monthly_audio_minutes'], row['usage_month'],
            row['tts_voice'], bool(row['enabled']) if row['enabled'] is not None else True
        ))
    
    # Reset sequence
    pg_cur.execute("SELECT setval('users_id_seq', COALESCE(MAX(id), 1)) FROM users")
    return len(rows)


def migrate_projects(sqlite_cur, pg_cur):
    """Migrate projects table."""
    sqlite_cur.execute("SELECT * FROM projects WHERE user_id IS NOT NULL")
    rows = sqlite_cur.fetchall()
    
    for row in rows:
        pg_cur.execute("""
            INSERT INTO projects (id, user_id, name, description, profile, type, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            row['id'], row['user_id'], row['name'], row['description'],
            row['profile'], row['type'], row['created_at']
        ))
    
    pg_cur.execute("SELECT setval('projects_id_seq', COALESCE(MAX(id), 1)) FROM projects")
    return len(rows)


def migrate_speeches(sqlite_cur, pg_cur):
    """Migrate user speeches only."""
    sqlite_cur.execute("""
        SELECT * FROM speeches 
        WHERE user_id IS NOT NULL
        ORDER BY id
    """)
    rows = sqlite_cur.fetchall()
    
    for row in rows:
        pg_cur.execute("""
            INSERT INTO speeches (id, user_id, project_id, title, speaker, profile,
                duration_sec, language, category, tags, notes, reference_text,
                cached_score, cached_score_profile, cached_score_json,
                pronunciation_mean_confidence, pronunciation_problem_ratio,
                word_analysis_json, grammar_analysis_json, dementia_metrics,
                created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            row['id'], row['user_id'], row['project_id'], row['title'], row['speaker'],
            row['profile'], row['duration_sec'], row['language'], row['category'],
            row['tags'], row['notes'], row['reference_text'],
            row['cached_score'], row['cached_score_profile'], row['cached_score_json'],
            row['pronunciation_mean_confidence'], row['pronunciation_problem_ratio'],
            row['word_analysis_json'], row['grammar_analysis_json'], row['dementia_metrics'],
            row['created_at'], row['updated_at']
        ))
    
    pg_cur.execute("SELECT setval('speeches_id_seq', COALESCE(MAX(id), 1)) FROM speeches")
    return len(rows)


def migrate_audio(sqlite_cur, pg_cur, speech_ids):
    """Migrate audio for user speeches."""
    if not speech_ids:
        return 0
    
    placeholders = ",".join(["?"] * len(speech_ids))
    sqlite_cur.execute(f"""
        SELECT * FROM audio 
        WHERE speech_id IN ({placeholders})
    """, speech_ids)
    rows = sqlite_cur.fetchall()
    
    for row in rows:
        pg_cur.execute("""
            INSERT INTO audio (id, speech_id, format, data, duration_sec, size_bytes, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            row['id'], row['speech_id'], row['format'],
            psycopg2.Binary(row['data']) if row['data'] else None,
            row['duration_sec'], row['size_bytes'], row['created_at']
        ))
    
    pg_cur.execute("SELECT setval('audio_id_seq', COALESCE(MAX(id), 1)) FROM audio")
    return len(rows)


def migrate_transcriptions(sqlite_cur, pg_cur, speech_ids):
    """Migrate transcriptions for user speeches."""
    if not speech_ids:
        return 0
    
    placeholders = ",".join(["?"] * len(speech_ids))
    sqlite_cur.execute(f"""
        SELECT * FROM transcriptions 
        WHERE speech_id IN ({placeholders})
    """, speech_ids)
    rows = sqlite_cur.fetchall()
    
    for row in rows:
        pg_cur.execute("""
            INSERT INTO transcriptions (id, speech_id, text, segments, word_count, language, model, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            row['id'], row['speech_id'], row['text'], row['segments'],
            row['word_count'], row['language'], row['model'], row['created_at']
        ))
    
    pg_cur.execute("SELECT setval('transcriptions_id_seq', COALESCE(MAX(id), 1)) FROM transcriptions")
    return len(rows)


def migrate_analyses(sqlite_cur, pg_cur, speech_ids):
    """Migrate analyses for user speeches."""
    if not speech_ids:
        return 0
    
    placeholders = ",".join(["?"] * len(speech_ids))
    sqlite_cur.execute(f"""
        SELECT * FROM analyses 
        WHERE speech_id IN ({placeholders})
    """, speech_ids)
    rows = sqlite_cur.fetchall()
    
    for row in rows:
        pg_cur.execute("""
            INSERT INTO analyses (id, speech_id, preset, result, wpm, pitch_mean_hz, pitch_std_hz,
                jitter_pct, shimmer_pct, hnr_db, lexical_diversity, fluency_score,
                fk_grade_level, sentiment, quality_level, snr_db, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            row['id'], row['speech_id'], row['preset'], row['result'],
            row['wpm'], row['pitch_mean_hz'], row['pitch_std_hz'],
            row['jitter_pct'], row['shimmer_pct'], row['hnr_db'],
            row['lexical_diversity'], row['fluency_score'], row['fk_grade_level'],
            row['sentiment'], row['quality_level'], row['snr_db'], row['created_at']
        ))
    
    pg_cur.execute("SELECT setval('analyses_id_seq', COALESCE(MAX(id), 1)) FROM analyses")
    return len(rows)


def migrate_coach_messages(sqlite_cur, pg_cur):
    """Migrate coach messages."""
    sqlite_cur.execute("SELECT * FROM coach_messages")
    rows = sqlite_cur.fetchall()
    
    for row in rows:
        pg_cur.execute("""
            INSERT INTO coach_messages (id, user_id, speech_id, project_id, role, content, model, created_at, read_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            row['id'], row['user_id'], row['speech_id'], row['project_id'],
            row['role'], row['content'], row['model'], row['created_at'], row['read_at']
        ))
    
    pg_cur.execute("SELECT setval('coach_messages_id_seq', COALESCE(MAX(id), 1)) FROM coach_messages")
    return len(rows)


def migrate_speaker_embeddings(sqlite_cur, pg_cur):
    """Migrate speaker voice profiles."""
    sqlite_cur.execute("SELECT * FROM speaker_embeddings")
    rows = sqlite_cur.fetchall()
    
    for row in rows:
        pg_cur.execute("""
            INSERT INTO speaker_embeddings (id, user_id, speaker_name, embedding, embedding_count, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, speaker_name) DO NOTHING
        """, (
            row['id'], row['user_id'], row['speaker_name'],
            psycopg2.Binary(row['embedding']) if row['embedding'] else None,
            row['embedding_count'], row['created_at']
        ))
    
    pg_cur.execute("SELECT setval('speaker_embeddings_id_seq', COALESCE(MAX(id), 1)) FROM speaker_embeddings")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Migrate user data to PostgreSQL")
    parser.add_argument("--postgres-url", required=True, help="PostgreSQL connection URL")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SpeakFit User Data Migration (SQLite → PostgreSQL)")
    print("=" * 60)
    print()
    
    # Connect
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cur = sqlite_conn.cursor()
    
    pg_conn = psycopg2.connect(args.postgres_url)
    pg_cur = pg_conn.cursor()
    
    try:
        # Create schema
        print("Creating schema...")
        if not args.dry_run:
            pg_cur.execute(get_schema())
            pg_conn.commit()
        print("  ✓ Schema created")
        print()
        
        # Migrate tables
        print("Migrating user data...")
        start = time.time()
        
        # Users
        count = migrate_users(sqlite_cur, pg_cur)
        print(f"  ✓ users: {count} rows")
        
        # Projects
        count = migrate_projects(sqlite_cur, pg_cur)
        print(f"  ✓ projects: {count} rows")
        
        # Speeches (get IDs for related tables)
        count = migrate_speeches(sqlite_cur, pg_cur)
        print(f"  ✓ speeches: {count} rows")
        
        # Get user speech IDs
        sqlite_cur.execute("SELECT id FROM speeches WHERE user_id IS NOT NULL")
        speech_ids = [row['id'] for row in sqlite_cur.fetchall()]
        
        # Audio
        count = migrate_audio(sqlite_cur, pg_cur, speech_ids)
        print(f"  ✓ audio: {count} rows")
        
        # Transcriptions
        count = migrate_transcriptions(sqlite_cur, pg_cur, speech_ids)
        print(f"  ✓ transcriptions: {count} rows")
        
        # Analyses
        count = migrate_analyses(sqlite_cur, pg_cur, speech_ids)
        print(f"  ✓ analyses: {count} rows")
        
        # Coach messages
        count = migrate_coach_messages(sqlite_cur, pg_cur)
        print(f"  ✓ coach_messages: {count} rows")
        
        # Speaker embeddings
        count = migrate_speaker_embeddings(sqlite_cur, pg_cur)
        print(f"  ✓ speaker_embeddings: {count} rows")
        
        if not args.dry_run:
            pg_conn.commit()
        
        elapsed = time.time() - start
        print()
        print(f"✅ Migration complete in {elapsed:.1f}s")
        print()
        
        # Verify
        print("Verification:")
        pg_cur.execute("SELECT COUNT(*) FROM users")
        print(f"  Users: {pg_cur.fetchone()[0]}")
        pg_cur.execute("SELECT COUNT(*) FROM speeches")
        print(f"  Speeches: {pg_cur.fetchone()[0]}")
        pg_cur.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM audio")
        size_bytes = pg_cur.fetchone()[0]
        print(f"  Audio size: {size_bytes / 1024 / 1024:.1f} MB")
        
    finally:
        sqlite_conn.close()
        pg_conn.close()


if __name__ == "__main__":
    main()
