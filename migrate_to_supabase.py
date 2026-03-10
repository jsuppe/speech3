#!/usr/bin/env python3
"""
Migrate user data from SQLite to Supabase PostgreSQL.

Only migrates user-related data (user_id > 0), leaves research data in SQLite.
"""

import sqlite3
import psycopg2
from psycopg2.extras import execute_values, Json
from datetime import datetime
import json

# Connection strings
SQLITE_PATH = "/home/melchior/speech3/speechscore.db"
SUPABASE_URL = "postgresql://postgres.fkxuqyvcvxklzrxjmzsa:Y4ZLP97tHSTQn7Jz@aws-1-us-east-1.pooler.supabase.com:6543/postgres"

def create_schema(pg_conn):
    """Create PostgreSQL schema."""
    cur = pg_conn.cursor()
    
    # Drop existing tables (fresh start)
    cur.execute("""
        DROP TABLE IF EXISTS dementia_annotations CASCADE;
        DROP TABLE IF EXISTS coach_messages CASCADE;
        DROP TABLE IF EXISTS speaker_embeddings CASCADE;
        DROP TABLE IF EXISTS user_achievements CASCADE;
        DROP TABLE IF EXISTS user_streaks CASCADE;
        DROP TABLE IF EXISTS user_surveys CASCADE;
        DROP TABLE IF EXISTS activity_log CASCADE;
        DROP TABLE IF EXISTS audio CASCADE;
        DROP TABLE IF EXISTS transcriptions CASCADE;
        DROP TABLE IF EXISTS analyses CASCADE;
        DROP TABLE IF EXISTS speeches CASCADE;
        DROP TABLE IF EXISTS projects CASCADE;
        DROP TABLE IF EXISTS achievements CASCADE;
        DROP TABLE IF EXISTS beta_signups CASCADE;
        DROP TABLE IF EXISTS users CASCADE;
    """)
    
    # Users table (matches SQLite schema)
    cur.execute("""
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            provider TEXT NOT NULL DEFAULT 'google',
            provider_id TEXT,
            email TEXT,
            name TEXT,
            avatar_url TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_login TIMESTAMPTZ,
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
    """)
    
    # Achievements table
    cur.execute("""
        CREATE TABLE achievements (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            icon TEXT,
            category TEXT,
            xp_reward INTEGER DEFAULT 0,
            requirement_type TEXT,
            requirement_value INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    
    # Projects table
    cur.execute("""
        CREATE TABLE projects (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            description TEXT,
            profile TEXT DEFAULT 'general',
            type TEXT DEFAULT 'standard',
            reminder_enabled BOOLEAN DEFAULT FALSE,
            reminder_frequency TEXT DEFAULT 'daily',
            reminder_days TEXT DEFAULT '[]',
            reminder_time TEXT DEFAULT '09:00',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    
    # Speeches table (comprehensive)
    cur.execute("""
        CREATE TABLE speeches (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL,
            title TEXT NOT NULL,
            speaker TEXT,
            year INTEGER,
            source_url TEXT,
            source_file TEXT,
            duration_sec REAL,
            language TEXT DEFAULT 'en',
            category TEXT,
            products TEXT,
            tags TEXT,
            notes TEXT,
            description TEXT,
            audio_hash TEXT,
            profile TEXT DEFAULT 'general',
            cached_score REAL,
            cached_score_profile TEXT,
            cached_score_json TEXT,
            is_reference BOOLEAN DEFAULT FALSE,
            reference_text TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX idx_speeches_user_id ON speeches(user_id);
        CREATE INDEX idx_speeches_created_at ON speeches(created_at);
    """)
    
    # Analyses table (store full result as JSONB)
    cur.execute("""
        CREATE TABLE analyses (
            id SERIAL PRIMARY KEY,
            speech_id INTEGER REFERENCES speeches(id) ON DELETE CASCADE,
            preset TEXT,
            modules TEXT,
            result JSONB NOT NULL,
            wpm REAL,
            pitch_mean_hz REAL,
            fluency_score REAL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX idx_analyses_speech_id ON analyses(speech_id);
    """)
    
    # Transcriptions table
    cur.execute("""
        CREATE TABLE transcriptions (
            id SERIAL PRIMARY KEY,
            speech_id INTEGER REFERENCES speeches(id) ON DELETE CASCADE,
            text TEXT NOT NULL,
            segments JSONB,
            word_count INTEGER,
            language TEXT,
            model TEXT DEFAULT 'large-v3',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX idx_transcriptions_speech_id ON transcriptions(speech_id);
    """)
    
    # Audio metadata table (actual audio on R2)
    cur.execute("""
        CREATE TABLE audio (
            id SERIAL PRIMARY KEY,
            speech_id INTEGER REFERENCES speeches(id) ON DELETE CASCADE,
            format TEXT DEFAULT 'opus',
            r2_key TEXT,
            original_filename TEXT,
            sample_rate INTEGER,
            channels INTEGER DEFAULT 1,
            bitrate_kbps INTEGER,
            duration_sec REAL,
            size_bytes INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX idx_audio_speech_id ON audio(speech_id);
        CREATE INDEX idx_audio_r2_key ON audio(r2_key);
    """)
    
    # Coach messages table
    cur.execute("""
        CREATE TABLE coach_messages (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            speech_id INTEGER REFERENCES speeches(id) ON DELETE SET NULL,
            project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT,
            read_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX idx_coach_messages_user_id ON coach_messages(user_id);
        CREATE INDEX idx_coach_messages_speech_id ON coach_messages(speech_id);
    """)
    
    # Speaker embeddings table
    cur.execute("""
        CREATE TABLE speaker_embeddings (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            speaker_name TEXT NOT NULL,
            embedding BYTEA NOT NULL,
            speech_id INTEGER REFERENCES speeches(id) ON DELETE SET NULL,
            embedding_count INTEGER DEFAULT 1,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(user_id, speaker_name)
        );
        CREATE INDEX idx_speaker_embeddings_user_id ON speaker_embeddings(user_id);
    """)
    
    # User achievements table
    cur.execute("""
        CREATE TABLE user_achievements (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            achievement_id TEXT REFERENCES achievements(id) ON DELETE CASCADE,
            earned_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(user_id, achievement_id)
        );
    """)
    
    # User streaks table
    cur.execute("""
        CREATE TABLE user_streaks (
            user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            current_streak INTEGER DEFAULT 0,
            longest_streak INTEGER DEFAULT 0,
            last_activity_date TEXT,
            total_xp INTEGER DEFAULT 0,
            daily_xp INTEGER DEFAULT 0,
            daily_xp_date TEXT,
            streak_freezes INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    
    # User surveys table
    cur.execute("""
        CREATE TABLE user_surveys (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            survey_type TEXT NOT NULL,
            responses JSONB,
            completed_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    
    # Activity log table
    cur.execute("""
        CREATE TABLE activity_log (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            action TEXT NOT NULL,
            details JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX idx_activity_log_user_id ON activity_log(user_id);
    """)
    
    # Beta signups table
    cur.execute("""
        CREATE TABLE beta_signups (
            id SERIAL PRIMARY KEY,
            app TEXT NOT NULL,
            email TEXT NOT NULL,
            name TEXT,
            use_cases TEXT,
            experience TEXT,
            biggest_challenge TEXT,
            features TEXT,
            other_details TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(app, email)
        );
    """)
    
    # Dementia annotations table
    cur.execute("""
        CREATE TABLE dementia_annotations (
            id SERIAL PRIMARY KEY,
            speech_id INTEGER REFERENCES speeches(id) ON DELETE CASCADE,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            segment_index INTEGER,
            categories TEXT[],
            notes TEXT,
            is_false_positive BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    
    pg_conn.commit()
    print("✅ Schema created")


def reset_sequence(pg_conn, table):
    """Reset a table's sequence to max id."""
    cur = pg_conn.cursor()
    cur.execute(f"SELECT setval('{table}_id_seq', COALESCE((SELECT MAX(id) FROM {table}), 0) + 1, false)")
    pg_conn.commit()


def migrate_users(sqlite_conn, pg_conn):
    """Migrate users table."""
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    sqlite_cur.execute("""
        SELECT id, provider, provider_id, email, name, avatar_url, created_at, last_login,
               app_version, patch_number, last_seen, preferred_accent, tier, 
               monthly_recordings, monthly_audio_minutes, usage_month, eula_accepted_at,
               tts_voice, enabled, disabled 
        FROM users
    """)
    rows = sqlite_cur.fetchall()
    
    # Transform enabled/disabled from int to bool
    transformed = []
    for row in rows:
        row = list(row)
        row[18] = bool(row[18]) if row[18] is not None else True   # enabled
        row[19] = bool(row[19]) if row[19] is not None else False  # disabled
        transformed.append(tuple(row))
    
    if transformed:
        execute_values(pg_cur, """
            INSERT INTO users (id, provider, provider_id, email, name, avatar_url, created_at, last_login,
                   app_version, patch_number, last_seen, preferred_accent, tier, 
                   monthly_recordings, monthly_audio_minutes, usage_month, eula_accepted_at,
                   tts_voice, enabled, disabled) VALUES %s
        """, transformed)
        pg_conn.commit()
        reset_sequence(pg_conn, "users")
        print(f"  users: {len(transformed)} rows")
    else:
        print("  users: 0 rows")


def migrate_achievements(sqlite_conn, pg_conn):
    """Migrate achievements table."""
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    sqlite_cur.execute("SELECT id, name, description, icon, category, xp_reward, requirement_type, requirement_value, created_at FROM achievements")
    rows = sqlite_cur.fetchall()
    
    if rows:
        execute_values(pg_cur, """
            INSERT INTO achievements (id, name, description, icon, category, xp_reward, requirement_type, requirement_value, created_at) VALUES %s
        """, rows)
        pg_conn.commit()
        print(f"  achievements: {len(rows)} rows")
    else:
        print("  achievements: 0 rows")


def migrate_projects(sqlite_conn, pg_conn):
    """Migrate projects table."""
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    sqlite_cur.execute("""
        SELECT id, user_id, name, description, profile, type, reminder_enabled, 
               reminder_frequency, reminder_days, reminder_time, created_at, updated_at 
        FROM projects WHERE user_id > 0
    """)
    rows = sqlite_cur.fetchall()
    
    # Transform reminder_enabled from int to bool
    transformed = []
    for row in rows:
        row = list(row)
        row[6] = bool(row[6]) if row[6] is not None else False  # reminder_enabled
        transformed.append(tuple(row))
    
    if transformed:
        execute_values(pg_cur, """
            INSERT INTO projects (id, user_id, name, description, profile, type, reminder_enabled, 
                   reminder_frequency, reminder_days, reminder_time, created_at, updated_at) VALUES %s
        """, transformed)
        pg_conn.commit()
        reset_sequence(pg_conn, "projects")
        print(f"  projects: {len(transformed)} rows")
    else:
        print("  projects: 0 rows")


def migrate_speeches(sqlite_conn, pg_conn):
    """Migrate speeches table (user data only)."""
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    sqlite_cur.execute("""
        SELECT id, user_id, project_id, title, speaker, year, source_url, source_file,
               duration_sec, language, category, products, tags, notes, description,
               audio_hash, profile, cached_score, cached_score_profile, cached_score_json,
               is_reference, reference_text, created_at, updated_at
        FROM speeches WHERE user_id > 0
    """)
    rows = sqlite_cur.fetchall()
    
    # Transform is_reference from int to bool
    transformed = []
    for row in rows:
        row = list(row)
        row[20] = bool(row[20]) if row[20] is not None else False  # is_reference
        transformed.append(tuple(row))
    
    if transformed:
        execute_values(pg_cur, """
            INSERT INTO speeches (id, user_id, project_id, title, speaker, year, source_url, source_file,
                   duration_sec, language, category, products, tags, notes, description,
                   audio_hash, profile, cached_score, cached_score_profile, cached_score_json,
                   is_reference, reference_text, created_at, updated_at) VALUES %s
        """, transformed)
        pg_conn.commit()
        reset_sequence(pg_conn, "speeches")
        print(f"  speeches: {len(transformed)} rows")
    else:
        print("  speeches: 0 rows")


def sanitize_json(data):
    """Replace NaN, Infinity with null for valid JSON."""
    import re
    if isinstance(data, str):
        # Replace NaN, Infinity, -Infinity with null
        data = re.sub(r'\bNaN\b', 'null', data)
        data = re.sub(r'\bInfinity\b', 'null', data)
        data = re.sub(r'-Infinity\b', 'null', data)
    return data


def migrate_analyses(sqlite_conn, pg_conn):
    """Migrate analyses table (for user speeches)."""
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    sqlite_cur.execute("""
        SELECT a.id, a.speech_id, a.preset, a.modules, a.result, a.wpm, a.pitch_mean_hz, a.fluency_score, a.created_at
        FROM analyses a 
        JOIN speeches s ON a.speech_id = s.id 
        WHERE s.user_id > 0
    """)
    rows = sqlite_cur.fetchall()
    
    # Transform result to JSONB
    transformed = []
    for row in rows:
        row = list(row)
        if row[4]:  # result column
            try:
                sanitized = sanitize_json(row[4]) if isinstance(row[4], str) else row[4]
                row[4] = Json(json.loads(sanitized)) if isinstance(sanitized, str) else Json(sanitized)
            except Exception as e:
                print(f"    Warning: Could not parse result for analysis {row[0]}: {e}")
                row[4] = Json({})
        else:
            row[4] = Json({})
        transformed.append(tuple(row))
    
    if transformed:
        execute_values(pg_cur, """
            INSERT INTO analyses (id, speech_id, preset, modules, result, wpm, pitch_mean_hz, fluency_score, created_at) VALUES %s
        """, transformed)
        pg_conn.commit()
        reset_sequence(pg_conn, "analyses")
        print(f"  analyses: {len(transformed)} rows")
    else:
        print("  analyses: 0 rows")


def migrate_transcriptions(sqlite_conn, pg_conn):
    """Migrate transcriptions table (for user speeches)."""
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    sqlite_cur.execute("""
        SELECT t.id, t.speech_id, t.text, t.segments, t.word_count, t.language, t.model, t.created_at
        FROM transcriptions t 
        JOIN speeches s ON t.speech_id = s.id 
        WHERE s.user_id > 0
    """)
    rows = sqlite_cur.fetchall()
    
    # Transform segments to JSONB
    transformed = []
    for row in rows:
        row = list(row)
        if row[3]:  # segments column
            try:
                row[3] = Json(json.loads(row[3])) if isinstance(row[3], str) else Json(row[3])
            except:
                row[3] = Json([])
        else:
            row[3] = Json([])
        transformed.append(tuple(row))
    
    if transformed:
        execute_values(pg_cur, """
            INSERT INTO transcriptions (id, speech_id, text, segments, word_count, language, model, created_at) VALUES %s
        """, transformed)
        pg_conn.commit()
        reset_sequence(pg_conn, "transcriptions")
        print(f"  transcriptions: {len(transformed)} rows")
    else:
        print("  transcriptions: 0 rows")


def migrate_audio(sqlite_conn, pg_conn):
    """Migrate audio metadata (for user speeches)."""
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    sqlite_cur.execute("""
        SELECT au.id, au.speech_id, au.format, au.r2_key, au.original_filename, 
               au.sample_rate, au.channels, au.bitrate_kbps, au.duration_sec, au.size_bytes, au.created_at
        FROM audio au 
        JOIN speeches s ON au.speech_id = s.id 
        WHERE s.user_id > 0
    """)
    rows = sqlite_cur.fetchall()
    
    if rows:
        execute_values(pg_cur, """
            INSERT INTO audio (id, speech_id, format, r2_key, original_filename, 
                   sample_rate, channels, bitrate_kbps, duration_sec, size_bytes, created_at) VALUES %s
        """, rows)
        pg_conn.commit()
        reset_sequence(pg_conn, "audio")
        print(f"  audio: {len(rows)} rows")
    else:
        print("  audio: 0 rows")


def migrate_coach_messages(sqlite_conn, pg_conn):
    """Migrate coach messages."""
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    sqlite_cur.execute("SELECT id, user_id, speech_id, project_id, role, content, model, read_at, created_at FROM coach_messages")
    rows = sqlite_cur.fetchall()
    
    if rows:
        execute_values(pg_cur, """
            INSERT INTO coach_messages (id, user_id, speech_id, project_id, role, content, model, read_at, created_at) VALUES %s
        """, rows)
        pg_conn.commit()
        reset_sequence(pg_conn, "coach_messages")
        print(f"  coach_messages: {len(rows)} rows")
    else:
        print("  coach_messages: 0 rows")


def migrate_speaker_embeddings(sqlite_conn, pg_conn):
    """Migrate speaker embeddings."""
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    sqlite_cur.execute("SELECT id, user_id, speaker_name, embedding, speech_id, embedding_count, created_at, updated_at FROM speaker_embeddings")
    rows = sqlite_cur.fetchall()
    
    if rows:
        # Embedding should already be bytes from SQLite BLOB
        execute_values(pg_cur, """
            INSERT INTO speaker_embeddings (id, user_id, speaker_name, embedding, speech_id, embedding_count, created_at, updated_at) VALUES %s
        """, rows)
        pg_conn.commit()
        reset_sequence(pg_conn, "speaker_embeddings")
        print(f"  speaker_embeddings: {len(rows)} rows")
    else:
        print("  speaker_embeddings: 0 rows")


def migrate_user_achievements(sqlite_conn, pg_conn):
    """Migrate user achievements."""
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    sqlite_cur.execute("SELECT id, user_id, achievement_id, earned_at FROM user_achievements")
    rows = sqlite_cur.fetchall()
    
    if rows:
        execute_values(pg_cur, """
            INSERT INTO user_achievements (id, user_id, achievement_id, earned_at) VALUES %s
        """, rows)
        pg_conn.commit()
        reset_sequence(pg_conn, "user_achievements")
        print(f"  user_achievements: {len(rows)} rows")
    else:
        print("  user_achievements: 0 rows")


def migrate_user_streaks(sqlite_conn, pg_conn):
    """Migrate user streaks."""
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    sqlite_cur.execute("""
        SELECT user_id, current_streak, longest_streak, last_activity_date, 
               total_xp, daily_xp, daily_xp_date, streak_freezes, created_at, updated_at 
        FROM user_streaks
    """)
    rows = sqlite_cur.fetchall()
    
    if rows:
        execute_values(pg_cur, """
            INSERT INTO user_streaks (user_id, current_streak, longest_streak, last_activity_date, 
                   total_xp, daily_xp, daily_xp_date, streak_freezes, created_at, updated_at) VALUES %s
        """, rows)
        pg_conn.commit()
        print(f"  user_streaks: {len(rows)} rows")
    else:
        print("  user_streaks: 0 rows")


def migrate_beta_signups(sqlite_conn, pg_conn):
    """Migrate beta signups."""
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    sqlite_cur.execute("""
        SELECT id, app, email, name, use_cases, experience, biggest_challenge, 
               features, other_details, created_at 
        FROM beta_signups
    """)
    rows = sqlite_cur.fetchall()
    
    if rows:
        execute_values(pg_cur, """
            INSERT INTO beta_signups (id, app, email, name, use_cases, experience, biggest_challenge, 
                   features, other_details, created_at) VALUES %s
        """, rows)
        pg_conn.commit()
        reset_sequence(pg_conn, "beta_signups")
        print(f"  beta_signups: {len(rows)} rows")
    else:
        print("  beta_signups: 0 rows")


def verify_migration(pg_conn):
    """Verify migration counts."""
    print("\n✅ Verification:")
    cur = pg_conn.cursor()
    
    tables = ['users', 'speeches', 'analyses', 'transcriptions', 'audio', 
              'coach_messages', 'projects', 'achievements', 'speaker_embeddings']
    
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table}: {count}")


def main():
    print("🚀 SpeakFit Migration: SQLite → Supabase")
    print(f"  Source: {SQLITE_PATH}")
    print(f"  Target: Supabase PostgreSQL")
    
    # Connect
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    pg_conn = psycopg2.connect(SUPABASE_URL)
    
    # Create schema
    print("\n📝 Creating schema...")
    create_schema(pg_conn)
    
    # Migrate data
    print("\n📦 Migrating data...")
    migrate_users(sqlite_conn, pg_conn)
    migrate_achievements(sqlite_conn, pg_conn)
    migrate_projects(sqlite_conn, pg_conn)
    migrate_speeches(sqlite_conn, pg_conn)
    migrate_analyses(sqlite_conn, pg_conn)
    migrate_transcriptions(sqlite_conn, pg_conn)
    migrate_audio(sqlite_conn, pg_conn)
    migrate_coach_messages(sqlite_conn, pg_conn)
    migrate_speaker_embeddings(sqlite_conn, pg_conn)
    migrate_user_achievements(sqlite_conn, pg_conn)
    migrate_user_streaks(sqlite_conn, pg_conn)
    migrate_beta_signups(sqlite_conn, pg_conn)
    
    # Verify
    verify_migration(pg_conn)
    
    # Cleanup
    sqlite_conn.close()
    pg_conn.close()
    
    print("\n🎉 Migration complete!")


if __name__ == "__main__":
    main()
