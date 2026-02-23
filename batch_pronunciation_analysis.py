#!/usr/bin/env python3
"""
Batch pronunciation analysis for all recordings in the database.

Uses wav2vec2 to analyze pronunciation quality for each recording.
Stores results in pronunciation_* columns.

Run with: python batch_pronunciation_analysis.py [--limit N] [--workers N]
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pronunciation import PronunciationAnalyzer
import numpy as np


def compute_features(word_analysis: list) -> dict:
    """Compute pronunciation features from word analysis list."""
    if not word_analysis:
        return {}
    
    confidences = [w['confidence'] for w in word_analysis]
    durations = [(w['end_time'] - w['start_time']) * 1000 for w in word_analysis]
    
    # Problem threshold
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
        'pct_words_below_80': sum(1 for c in confidences if c < 0.8) / len(confidences),
        'pct_words_below_90': sum(1 for c in confidences if c < 0.9) / len(confidences),
        'words_per_minute': len(word_analysis) / (duration_seconds / 60) if duration_seconds > 0 else 0,
        'mean_word_duration_ms': float(np.mean(durations)),
        'std_word_duration_ms': float(np.std(durations)),
    }


DB_PATH = Path(__file__).parent / 'speechscore.db'
AUDIO_BASE = Path(__file__).parent / 'audio'


def get_unprocessed_recordings(limit: int = None) -> list:
    """Get recordings that haven't been pronunciation-analyzed yet."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
        SELECT id, source_file, title
        FROM speeches 
        WHERE source_file IS NOT NULL 
          AND source_file != ''
          AND pronunciation_analyzed_at IS NULL
        ORDER BY id
    """
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def analyze_recording(recording: dict, analyzer: PronunciationAnalyzer = None) -> dict:
    """Analyze a single recording for pronunciation features."""
    rec_id = recording['id']
    source_file = recording['source_file']
    
    # Find audio file
    audio_path = None
    if source_file:
        # Try various locations
        candidates = [
            AUDIO_BASE / source_file,
            Path(source_file),
            Path(__file__).parent / source_file,
        ]
        for candidate in candidates:
            if candidate.exists():
                audio_path = candidate
                break
    
    if not audio_path or not audio_path.exists():
        return {
            'id': rec_id,
            'success': False,
            'error': f'Audio file not found: {source_file}'
        }
    
    try:
        # Initialize analyzer if not provided
        if analyzer is None:
            analyzer = PronunciationAnalyzer()
        
        # Analyze pronunciation - returns PronunciationAnalysis object
        analysis = analyzer.analyze(str(audio_path))
        
        # Convert WordPronunciation objects to dict format for compute_features
        word_analysis = [
            {
                'word': w.word,
                'confidence': w.whisper_confidence,
                'start_time': w.start,
                'end_time': w.end,
                'pronunciation_score': w.pronunciation_score,
            }
            for w in analysis.words
        ]
        
        # Extract features
        features = compute_features(word_analysis)
        
        return {
            'id': rec_id,
            'success': True,
            'mean_confidence': features.get('mean_word_confidence'),
            'std_confidence': features.get('std_word_confidence'),
            'problem_ratio': features.get('problem_word_ratio'),
            'words_analyzed': features.get('total_words'),
            'features_json': json.dumps(features),
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
        result['mean_confidence'],
        result['std_confidence'],
        result['problem_ratio'],
        result['words_analyzed'],
        datetime.utcnow().isoformat(),
        result['features_json'],
        result['id']
    ))
    
    conn.commit()
    conn.close()


def run_batch(limit: int = None, workers: int = 1):
    """Run batch pronunciation analysis."""
    print("=" * 70)
    print("BATCH PRONUNCIATION ANALYSIS")
    print("=" * 70)
    
    # Get unprocessed recordings
    recordings = get_unprocessed_recordings(limit)
    total = len(recordings)
    
    if total == 0:
        print("No unprocessed recordings found!")
        return
    
    print(f"\nRecordings to process: {total:,}")
    print(f"Workers: {workers}")
    print(f"Estimated time: ~{(total * 5) / workers / 60:.1f} minutes")
    print("\n" + "-" * 70)
    
    # Initialize analyzer (single instance for efficiency)
    print("Loading wav2vec2 model...")
    analyzer = PronunciationAnalyzer()
    print("Model loaded!\n")
    
    # Process recordings
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    for i, recording in enumerate(recordings, 1):
        rec_id = recording['id']
        title = recording['title'][:40] if recording['title'] else f"Recording {rec_id}"
        
        result = analyze_recording(recording, analyzer)
        
        if result['success']:
            save_result(result)
            success_count += 1
            status = f"✓ {result['mean_confidence']:.3f}"
        else:
            error_count += 1
            status = f"✗ {result.get('error', 'Unknown error')[:30]}"
        
        # Progress
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total - i) / rate if rate > 0 else 0
        
        print(f"[{i:,}/{total:,}] {title:<42} {status}")
        
        if i % 100 == 0:
            print(f"    Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m | Success: {success_count} | Errors: {error_count}")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print(f"Processed: {total:,}")
    print(f"Success: {success_count:,}")
    print(f"Errors: {error_count:,}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Rate: {total/elapsed:.2f} recordings/second")


def main():
    parser = argparse.ArgumentParser(description='Batch pronunciation analysis')
    parser.add_argument('--limit', type=int, help='Limit number of recordings to process')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    args = parser.parse_args()
    
    run_batch(limit=args.limit, workers=args.workers)


if __name__ == '__main__':
    main()
