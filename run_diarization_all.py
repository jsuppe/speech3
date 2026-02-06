#!/usr/bin/env python3
"""
Run speaker diarization on all speeches in the database.
Updates the segments in both analyses and transcriptions tables.
"""

import os
import sys
import json
import time
import tempfile
import sqlite3
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add speech3 to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from diarization import run_diarization, get_speaker_stats


def get_db_connection():
    """Get database connection."""
    db_path = os.path.join(SCRIPT_DIR, 'speechscore.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def process_speech(conn, speech_id: int) -> dict:
    """Run diarization on a single speech."""
    cur = conn.cursor()
    
    # Get audio data
    cur.execute('SELECT data, format FROM audio WHERE speech_id = ?', (speech_id,))
    audio_row = cur.fetchone()
    if not audio_row:
        return {'status': 'error', 'message': 'No audio found'}
    
    # Get current segments from analyses
    cur.execute('SELECT result FROM analyses WHERE speech_id = ?', (speech_id,))
    analysis_row = cur.fetchone()
    if not analysis_row:
        return {'status': 'error', 'message': 'No analysis found'}
    
    result = json.loads(analysis_row['result'])
    segments = result.get('segments', [])
    
    if not segments:
        return {'status': 'skipped', 'message': 'No segments'}
    
    # Check if already diarized
    if any(seg.get('speaker') for seg in segments):
        return {'status': 'skipped', 'message': 'Already diarized'}
    
    # Save audio to temp file
    audio_data = audio_row['data']
    audio_format = audio_row['format']
    
    with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as f:
        f.write(audio_data)
        audio_path = f.name
    
    try:
        # Run diarization
        updated_segments = run_diarization(audio_path, segments)
        stats = get_speaker_stats(updated_segments)
        
        # Update analyses table
        result['segments'] = updated_segments
        result['diarization'] = stats
        cur.execute('UPDATE analyses SET result = ? WHERE speech_id = ?', 
                   (json.dumps(result), speech_id))
        
        # Update transcriptions table
        cur.execute('UPDATE transcriptions SET segments = ? WHERE speech_id = ?',
                   (json.dumps(updated_segments), speech_id))
        
        conn.commit()
        
        return {
            'status': 'success',
            'num_speakers': stats['num_speakers'],
            'segments': len(updated_segments)
        }
    
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def main():
    """Process all speeches."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get all speech IDs with audio
    cur.execute('''
        SELECT DISTINCT a.speech_id, s.title, s.speaker
        FROM audio a
        JOIN speeches s ON s.id = a.speech_id
        JOIN analyses an ON an.speech_id = a.speech_id
        ORDER BY a.speech_id
    ''')
    speeches = cur.fetchall()
    
    logger.info(f"Found {len(speeches)} speeches to process")
    
    success = 0
    skipped = 0
    errors = 0
    
    for i, speech in enumerate(speeches):
        speech_id = speech['speech_id']
        title = speech['title'] or f"Speech {speech_id}"
        speaker = speech['speaker'] or "Unknown"
        
        logger.info(f"[{i+1}/{len(speeches)}] Processing: {title} by {speaker}")
        
        try:
            t0 = time.time()
            result = process_speech(conn, speech_id)
            elapsed = time.time() - t0
            
            if result['status'] == 'success':
                logger.info(f"  ✓ Found {result['num_speakers']} speakers in {elapsed:.1f}s")
                success += 1
            elif result['status'] == 'skipped':
                logger.info(f"  → Skipped: {result['message']}")
                skipped += 1
            else:
                logger.warning(f"  ✗ Error: {result['message']}")
                errors += 1
                
        except Exception as e:
            logger.error(f"  ✗ Exception: {e}")
            errors += 1
    
    conn.close()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Diarization complete!")
    logger.info(f"  Success: {success}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Errors: {errors}")


if __name__ == '__main__':
    main()
