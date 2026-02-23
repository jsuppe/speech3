#!/usr/bin/env python3
"""
Batch pronunciation analysis for cron job.
Runs during overnight window (1am-7am ET), same as batch_import.
Processes recordings that have audio but no pronunciation analysis.

Usage: python batch_pronunciation_cron.py [--limit N] [--time-limit MINUTES]
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

DB_PATH = Path(__file__).parent / 'speechscore.db'
AUDIO_BASE = Path(__file__).parent / 'audio'
STATE_FILE = Path(__file__).parent / 'pronunciation_batch_state.json'
LOG_FILE = Path('/tmp/batch_pronunciation.log')

# Default limits
DEFAULT_LIMIT = 100  # Recordings per run
DEFAULT_TIME_LIMIT = 60  # Minutes per run


def log(msg):
    """Log to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_state():
    """Load processing state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_id": 0, "total_processed": 0, "last_run": None}


def save_state(state):
    """Save processing state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def compute_features(word_analysis: list) -> dict:
    """Compute pronunciation features from word analysis list."""
    if not word_analysis:
        return {}
    
    confidences = [w['confidence'] for w in word_analysis]
    durations = [(w['end_time'] - w['start_time']) * 1000 for w in word_analysis]
    
    threshold = 0.6
    problem_count = sum(1 for c in confidences if c < threshold)
    duration_seconds = word_analysis[-1]['end_time'] if word_analysis else 0
    
    return {
        'total_words': len(word_analysis),
        'problem_word_count': problem_count,
        'problem_word_ratio': problem_count / len(word_analysis) if word_analysis else 0,
        'mean_word_confidence': float(np.mean(confidences)),
        'std_word_confidence': float(np.std(confidences)),
        'min_word_confidence': float(np.min(confidences)),
        'max_word_confidence': float(np.max(confidences)),
        'median_word_confidence': float(np.median(confidences)),
        'pct_words_below_50': sum(1 for c in confidences if c < 0.5) / len(confidences),
        'pct_words_below_60': sum(1 for c in confidences if c < 0.6) / len(confidences),
        'pct_words_below_70': sum(1 for c in confidences if c < 0.7) / len(confidences),
        'words_per_minute': len(word_analysis) / (duration_seconds / 60) if duration_seconds > 0 else 0,
        'mean_word_duration_ms': float(np.mean(durations)),
    }


def get_unprocessed_recordings(limit: int, last_id: int = 0):
    """Get recordings needing pronunciation analysis."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, source_file, title
        FROM speeches 
        WHERE source_file IS NOT NULL 
          AND source_file != ''
          AND pronunciation_analyzed_at IS NULL
          AND id > ?
        ORDER BY id
        LIMIT ?
    """, (last_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def find_audio_path(source_file: str) -> Path:
    """Find the audio file."""
    if not source_file:
        return None
    
    candidates = [
        AUDIO_BASE / source_file,
        Path(source_file),
        Path(__file__).parent / source_file,
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def process_recording(rec_id: int, audio_path: Path, analyzer) -> dict:
    """Analyze a single recording."""
    try:
        # Run analysis
        analysis = analyzer.analyze(str(audio_path))
        
        # Convert to feature format
        word_analysis = [
            {
                'word': w.word,
                'confidence': w.whisper_confidence,
                'start_time': w.start,
                'end_time': w.end,
            }
            for w in analysis.words
        ]
        
        features = compute_features(word_analysis)
        
        return {
            'id': rec_id,
            'success': True,
            'features': features
        }
    except Exception as e:
        return {
            'id': rec_id,
            'success': False,
            'error': str(e)
        }


def save_result(result: dict):
    """Save analysis result to database."""
    if not result['success']:
        return
    
    features = result['features']
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE speeches SET
            pronunciation_mean_confidence = ?,
            pronunciation_std_confidence = ?,
            pronunciation_problem_ratio = ?,
            pronunciation_words_analyzed = ?,
            pronunciation_analyzed_at = ?,
            pronunciation_features_json = ?
        WHERE id = ?
    """, (
        features.get('mean_word_confidence'),
        features.get('std_word_confidence'),
        features.get('problem_word_ratio'),
        features.get('total_words'),
        datetime.utcnow().isoformat(),
        json.dumps(features),
        result['id']
    ))
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Batch pronunciation analysis')
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, 
                        help=f'Max recordings to process (default: {DEFAULT_LIMIT})')
    parser.add_argument('--time-limit', type=int, default=DEFAULT_TIME_LIMIT,
                        help=f'Max minutes to run (default: {DEFAULT_TIME_LIMIT})')
    parser.add_argument('--reset', action='store_true',
                        help='Reset state and start from beginning')
    args = parser.parse_args()
    
    log("="*60)
    log("BATCH PRONUNCIATION ANALYSIS - CRON JOB")
    log("="*60)
    
    # Load or reset state
    if args.reset:
        state = {"last_id": 0, "total_processed": 0, "last_run": None}
        log("State reset")
    else:
        state = load_state()
        log(f"Resuming from ID {state['last_id']}, total processed: {state['total_processed']}")
    
    # Get recordings to process
    recordings = get_unprocessed_recordings(args.limit, state['last_id'])
    
    if not recordings:
        log("No recordings to process")
        return
    
    log(f"Found {len(recordings)} recordings to process")
    log(f"Time limit: {args.time_limit} minutes")
    
    # Load analyzer
    log("Loading pronunciation analyzer...")
    from pronunciation import PronunciationAnalyzer
    analyzer = PronunciationAnalyzer()
    log("Analyzer ready")
    
    start_time = time.time()
    time_limit_sec = args.time_limit * 60
    processed = 0
    errors = 0
    
    for rec in recordings:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > time_limit_sec:
            log(f"Time limit reached ({args.time_limit} min)")
            break
        
        rec_id = rec['id']
        title = rec.get('title', 'Unknown')[:40]
        
        # Find audio
        audio_path = find_audio_path(rec['source_file'])
        if not audio_path:
            log(f"  [{rec_id}] {title} - SKIP (no audio)")
            continue
        
        # Process
        result = process_recording(rec_id, audio_path, analyzer)
        
        if result['success']:
            save_result(result)
            conf = result['features'].get('mean_word_confidence', 0)
            log(f"  [{rec_id}] {title} - OK ({conf:.2%})")
            processed += 1
            state['last_id'] = rec_id
            state['total_processed'] += 1
        else:
            log(f"  [{rec_id}] {title} - ERROR: {result.get('error', 'unknown')}")
            errors += 1
        
        # Save state periodically
        if processed % 10 == 0:
            save_state(state)
    
    # Final state save
    state['last_run'] = datetime.utcnow().isoformat()
    save_state(state)
    
    elapsed = (time.time() - start_time) / 60
    log("="*60)
    log(f"COMPLETE: {processed} processed, {errors} errors in {elapsed:.1f} min")
    log(f"Total processed all-time: {state['total_processed']}")
    log("="*60)


if __name__ == "__main__":
    main()
