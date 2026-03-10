#!/usr/bin/env python3
"""
R2 Backup & Recovery Script

Syncs audio between Cloudflare R2 and SQLite database.

Commands:
  backup   - Download R2 files to SQLite (for files missing BLOB)
  restore  - Upload SQLite BLOBs to R2 (for disaster recovery)
  status   - Show sync status
"""

import sqlite3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from api.r2_storage import get_r2_storage

DB_PATH = Path(__file__).parent / "speechscore.db"


def get_status(db: sqlite3.Connection) -> dict:
    """Get backup status."""
    cursor = db.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN r2_key IS NOT NULL THEN 1 ELSE 0 END) as in_r2,
            SUM(CASE WHEN data IS NOT NULL THEN 1 ELSE 0 END) as has_blob,
            SUM(CASE WHEN r2_key IS NOT NULL AND data IS NOT NULL THEN 1 ELSE 0 END) as fully_backed_up,
            SUM(CASE WHEN r2_key IS NOT NULL AND data IS NULL THEN 1 ELSE 0 END) as r2_only,
            SUM(CASE WHEN r2_key IS NULL AND data IS NOT NULL THEN 1 ELSE 0 END) as sqlite_only
        FROM audio
    """)
    row = cursor.fetchone()
    return {
        "total": row[0],
        "in_r2": row[1],
        "has_blob": row[2],
        "fully_backed_up": row[3],
        "r2_only": row[4],
        "sqlite_only": row[5],
    }


def backup_r2_to_sqlite(db: sqlite3.Connection, r2, limit: int = 100, dry_run: bool = False) -> dict:
    """Download R2 files to SQLite for entries missing BLOB."""
    cursor = db.execute("""
        SELECT id, speech_id, r2_key, format
        FROM audio
        WHERE r2_key IS NOT NULL AND data IS NULL
        ORDER BY id
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    stats = {"downloaded": 0, "failed": 0, "bytes": 0}
    
    for row in rows:
        audio_id, speech_id, r2_key, fmt = row
        print(f"  [{audio_id}] {r2_key}", end=" ")
        
        if dry_run:
            print("→ [DRY RUN]")
            stats["downloaded"] += 1
            continue
        
        try:
            data = r2.download(r2_key)
            if data:
                db.execute("UPDATE audio SET data = ? WHERE id = ?", (data, audio_id))
                db.commit()
                stats["downloaded"] += 1
                stats["bytes"] += len(data)
                print(f"→ ✓ {len(data)/1024:.1f} KB")
            else:
                print("→ ✗ Empty response")
                stats["failed"] += 1
        except Exception as e:
            print(f"→ ✗ {e}")
            stats["failed"] += 1
    
    return stats


def restore_sqlite_to_r2(db: sqlite3.Connection, r2, limit: int = 100, dry_run: bool = False) -> dict:
    """Upload SQLite BLOBs to R2 for entries missing r2_key."""
    cursor = db.execute("""
        SELECT a.id, a.speech_id, a.format, a.data, a.original_filename, s.user_id
        FROM audio a
        JOIN speeches s ON a.speech_id = s.id
        WHERE a.r2_key IS NULL AND a.data IS NOT NULL
        ORDER BY a.id
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    stats = {"uploaded": 0, "failed": 0, "bytes": 0}
    
    for row in rows:
        audio_id, speech_id, fmt, data, filename, user_id = row
        print(f"  [{audio_id}] Speech {speech_id} ({fmt}, {len(data)/1024:.1f} KB)", end=" ")
        
        if dry_run:
            print("→ [DRY RUN]")
            stats["uploaded"] += 1
            continue
        
        try:
            content_type = {
                "opus": "audio/opus",
                "mp3": "audio/mpeg",
                "wav": "audio/wav",
            }.get(fmt, "application/octet-stream")
            
            result = r2.upload(
                data=data,
                user_id=user_id or 0,
                speech_id=speech_id,
                filename=filename or f"speech_{speech_id}.{fmt}",
                format=fmt,
                content_type=content_type,
            )
            
            db.execute("UPDATE audio SET r2_key = ? WHERE id = ?", (result["key"], audio_id))
            db.commit()
            
            stats["uploaded"] += 1
            stats["bytes"] += len(data)
            print(f"→ ✓ {result['key']}")
        except Exception as e:
            print(f"→ ✗ {e}")
            stats["failed"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="R2 Backup & Recovery")
    parser.add_argument("command", choices=["backup", "restore", "status"], help="Command to run")
    parser.add_argument("--limit", type=int, default=100, help="Max files to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()
    
    db = sqlite3.connect(DB_PATH)
    
    if args.command == "status":
        status = get_status(db)
        print("=== R2 Backup Status ===")
        print(f"Total audio files:    {status['total']:,}")
        print(f"In R2:                {status['in_r2']:,}")
        print(f"In SQLite (BLOB):     {status['has_blob']:,}")
        print(f"Fully backed up:      {status['fully_backed_up']:,} (both R2 + SQLite)")
        print(f"R2 only:              {status['r2_only']:,} (needs backup)")
        print(f"SQLite only:          {status['sqlite_only']:,} (not yet in R2)")
        db.close()
        return
    
    r2 = get_r2_storage()
    if not r2:
        print("ERROR: R2 storage not configured")
        sys.exit(1)
    
    if args.command == "backup":
        print(f"=== Backing up R2 → SQLite (limit: {args.limit}) ===")
        stats = backup_r2_to_sqlite(db, r2, args.limit, args.dry_run)
        print(f"\nDownloaded: {stats['downloaded']}, Failed: {stats['failed']}, Bytes: {stats['bytes']/1024/1024:.1f} MB")
    
    elif args.command == "restore":
        print(f"=== Restoring SQLite → R2 (limit: {args.limit}) ===")
        stats = restore_sqlite_to_r2(db, r2, args.limit, args.dry_run)
        print(f"\nUploaded: {stats['uploaded']}, Failed: {stats['failed']}, Bytes: {stats['bytes']/1024/1024:.1f} MB")
    
    db.close()


if __name__ == "__main__":
    main()
