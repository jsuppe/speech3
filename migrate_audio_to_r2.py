#!/usr/bin/env python3
"""
Migrate audio BLOBs from SQLite to Cloudflare R2.

This script:
1. Reads audio entries that don't have an r2_key
2. Uploads the audio data to R2
3. Updates the database with the R2 key
4. Optionally clears the BLOB data to save space

Run with --dry-run first to preview what would be migrated.
"""

import sqlite3
import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from api.r2_storage import get_r2_storage

DB_PATH = Path(__file__).parent / "speechscore.db"


def get_pending_audio(db: sqlite3.Connection, limit: int = 100) -> list:
    """Get audio entries without r2_key."""
    cursor = db.execute("""
        SELECT a.id, a.speech_id, a.format, a.data, a.original_filename,
               a.sample_rate, a.channels, a.bitrate_kbps, a.duration_sec, a.size_bytes,
               s.user_id
        FROM audio a
        JOIN speeches s ON a.speech_id = s.id
        WHERE a.r2_key IS NULL AND a.data IS NOT NULL
        ORDER BY a.id
        LIMIT ?
    """, (limit,))
    return cursor.fetchall()


def migrate_batch(db: sqlite3.Connection, r2, batch: list, clear_blobs: bool = False, dry_run: bool = False) -> dict:
    """Migrate a batch of audio entries to R2."""
    stats = {"migrated": 0, "failed": 0, "bytes_uploaded": 0}
    
    for row in batch:
        audio_id, speech_id, fmt, data, filename, sample_rate, channels, bitrate, duration, size_bytes, user_id = row
        
        if not data:
            continue
        
        content_type = {
            "opus": "audio/opus",
            "mp3": "audio/mpeg", 
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "webm": "audio/webm",
        }.get(fmt, "application/octet-stream")
        
        print(f"  [{audio_id}] Speech {speech_id} ({fmt}, {len(data)/1024:.1f} KB)", end=" ")
        
        if dry_run:
            print("→ [DRY RUN] would upload")
            stats["migrated"] += 1
            stats["bytes_uploaded"] += len(data)
            continue
        
        try:
            result = r2.upload(
                data=data,
                user_id=user_id or 0,
                speech_id=speech_id,
                filename=filename or f"speech_{speech_id}.{fmt}",
                format=fmt,
                content_type=content_type,
            )
            
            # Update database with R2 key
            db.execute(
                "UPDATE audio SET r2_key = ? WHERE id = ?",
                (result["key"], audio_id)
            )
            
            # Optionally clear the BLOB data
            if clear_blobs:
                db.execute("UPDATE audio SET data = NULL WHERE id = ?", (audio_id,))
            
            db.commit()
            
            print(f"→ ✓ {result['key']}")
            stats["migrated"] += 1
            stats["bytes_uploaded"] += len(data)
            
        except Exception as e:
            print(f"→ ✗ ERROR: {e}")
            stats["failed"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Migrate audio to Cloudflare R2")
    parser.add_argument("--limit", type=int, default=100, help="Max entries per batch")
    parser.add_argument("--batch-count", type=int, default=1, help="Number of batches to run")
    parser.add_argument("--clear-blobs", action="store_true", help="Clear BLOB data after upload (saves space)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't actually upload")
    args = parser.parse_args()
    
    print(f"=== Audio Migration to R2 ===")
    print(f"Database: {DB_PATH}")
    print(f"Batch size: {args.limit}, Batches: {args.batch_count}")
    print(f"Clear BLOBs: {args.clear_blobs}, Dry run: {args.dry_run}")
    print()
    
    db = sqlite3.connect(DB_PATH)
    
    # Count pending
    cursor = db.execute("SELECT COUNT(*) FROM audio WHERE r2_key IS NULL AND data IS NOT NULL")
    pending = cursor.fetchone()[0]
    print(f"Pending audio entries: {pending:,}")
    
    if pending == 0:
        print("Nothing to migrate!")
        return
    
    r2 = get_r2_storage() if not args.dry_run else None
    
    total_stats = {"migrated": 0, "failed": 0, "bytes_uploaded": 0}
    
    for batch_num in range(args.batch_count):
        print(f"\n--- Batch {batch_num + 1}/{args.batch_count} ---")
        batch = get_pending_audio(db, args.limit)
        
        if not batch:
            print("No more entries to migrate")
            break
        
        stats = migrate_batch(db, r2, batch, args.clear_blobs, args.dry_run)
        
        for key in total_stats:
            total_stats[key] += stats[key]
        
        # Re-check remaining
        cursor = db.execute("SELECT COUNT(*) FROM audio WHERE r2_key IS NULL AND data IS NOT NULL")
        remaining = cursor.fetchone()[0]
        print(f"Remaining: {remaining:,}")
    
    print(f"\n=== Summary ===")
    print(f"Migrated: {total_stats['migrated']:,}")
    print(f"Failed: {total_stats['failed']:,}")
    print(f"Bytes uploaded: {total_stats['bytes_uploaded']/1024/1024:.1f} MB")
    
    db.close()


if __name__ == "__main__":
    main()
