#!/usr/bin/env python3
"""
Regenerate exemplars.json with speeches that properly represent each grade level
for each metric, using the current scoring algorithm.

Focuses on curated speeches with known speakers, validates audio file existence.
"""

import json
import os
import sys
import sqlite3

# Add parent dir to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from scoring import METRIC_DEFS, _percentile_rank, _score_metric, _letter_grade

DB_PATH = os.path.join(parent_dir, "speechscore.db")
EXEMPLARS_DIR = os.path.join(parent_dir, "frontend", "static", "exemplars")

# Metrics we want exemplars for
EXEMPLAR_METRICS = [
    "wpm", "pitch_variation", "voice_quality", "jitter", "shimmer",
    "vocal_fry", "lexical_diversity", "snr", "pitch_range"
]

# Grade bands (simplified - we use base grades A/B/C/D/F)
GRADE_BANDS = {
    "A": (68.5, 100),
    "B": (60, 68.5),
    "C": (52, 60),
    "D": (41, 52),
    "F": (0, 41),
}

# Audio files that actually exist in the exemplars directory
AVAILABLE_AUDIO = set()


def load_available_audio():
    """Scan the exemplars directory for available audio files."""
    global AVAILABLE_AUDIO
    if os.path.exists(EXEMPLARS_DIR):
        for f in os.listdir(EXEMPLARS_DIR):
            if f.endswith(('.mp3', '.wav', '.m4a')):
                AVAILABLE_AUDIO.add(f)
    print(f"  Found {len(AVAILABLE_AUDIO)} audio files in exemplars directory")


def get_distributions():
    """Load metric distributions from full database for percentile calculation."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Get distributions from ALL speeches for accurate percentiles
    print("  Loading full distributions...")
    dist_rows = conn.execute("""
        SELECT a.*
        FROM analyses a
        INNER JOIN (
            SELECT speech_id, MAX(id) as max_id
            FROM analyses GROUP BY speech_id
        ) latest ON a.id = latest.max_id
        WHERE a.wpm IS NOT NULL
    """).fetchall()
    
    distributions = {}
    for mdef in METRIC_DEFS.values():
        col = mdef["column"]
        vals = [r[col] for r in dist_rows if r[col] is not None]
        distributions[col] = sorted([float(v) for v in vals])
    
    print(f"  Built distributions from {len(dist_rows)} speeches")
    
    # Get curated speeches with proper metadata for exemplar selection
    # Filter to speeches whose audio files exist in the exemplars directory
    print("  Loading curated speeches with available audio...")
    curated_rows = conn.execute("""
        SELECT a.*, s.title, s.speaker, s.category, s.source_file, s.id as speech_id
        FROM analyses a
        INNER JOIN (
            SELECT speech_id, MAX(id) as max_id
            FROM analyses GROUP BY speech_id
        ) latest ON a.id = latest.max_id
        INNER JOIN speeches s ON a.speech_id = s.id
        WHERE a.wpm IS NOT NULL
        AND a.wpm > 10  -- Filter out bad data (0 or extremely low WPM)
        AND s.source_file IS NOT NULL
        AND s.speaker IS NOT NULL
        AND s.speaker != ''
        AND s.speaker != 'Unknown'
        AND s.speaker != 'Unknown Speaker'
        AND s.source_file NOT LIKE '%recording-%'  -- Exclude user recordings
        AND s.source_file NOT LIKE '%.webm'  -- Exclude user uploads
    """).fetchall()
    
    # Filter to only speeches with available audio
    curated = []
    for row in curated_rows:
        r = dict(row)
        source_file = r.get('source_file', '')
        if source_file:
            # Handle different path formats
            filename = os.path.basename(source_file)
            # Also try removing test_speeches/ prefix
            if filename in AVAILABLE_AUDIO:
                r['audio_filename'] = filename
                curated.append(r)
            elif source_file.startswith('test_speeches/'):
                filename = source_file.replace('test_speeches/', '')
                if filename in AVAILABLE_AUDIO:
                    r['audio_filename'] = filename
                    curated.append(r)
    
    print(f"  Found {len(curated)} speeches with available exemplar audio")
    
    conn.close()
    return curated, distributions


def score_value(value, distribution, mdef):
    """Score a single metric value."""
    if value is None:
        return None, None
    
    value = float(value)
    direction = mdef["direction"]
    
    pct = _percentile_rank(value, distribution)
    score = _score_metric(
        value, distribution, direction,
        optimal=mdef.get("optimal"),
        optimal_range=mdef.get("optimal_range"),
    )
    
    if score is not None:
        score = round(min(100.0, max(0.0, score)), 1)
    
    return score, pct


def find_best_exemplars(rows, distributions, metric_key):
    """Find the best exemplar for each grade band for a metric."""
    mdef = METRIC_DEFS[metric_key]
    col = mdef["column"]
    dist = distributions.get(col, [])
    
    if not dist:
        return {}
    
    # Score all speeches for this metric
    scored = []
    for r in rows:
        value = r.get(col)
        if value is None:
            continue
        
        # Additional validation for specific metrics
        if metric_key == "wpm" and (value < 20 or value > 300):
            continue  # Skip unrealistic WPM values
        if metric_key == "voice_quality" and value < 1:
            continue  # Skip invalid HNR
        
        score, pct = score_value(value, dist, mdef)
        if score is None:
            continue
        
        # Prioritize famous speakers
        speaker = r.get('speaker') or ''
        title = r.get('title') or ''
        
        # Famous historical speakers get highest priority
        famous = ['Martin Luther King', 'John F. Kennedy', 'Franklin D. Roosevelt',
                  'Winston Churchill', 'Barack Obama', 'Ronald Reagan', 'Lou Gehrig',
                  'Stokely Carmichael', 'Mario Cuomo', 'Jesse Jackson', 'Carl Sagan',
                  'Steve Jobs', 'King George', 'LibriVox']
        is_famous = any(f.lower() in speaker.lower() for f in famous)
        
        scored.append({
            'speech_id': r['speech_id'],
            'title': title or 'Unknown',
            'speaker': speaker or 'Unknown',
            'category': r.get('category', ''),
            'audio_filename': r.get('audio_filename'),
            'value': round(float(value), 2),
            'score': score,
            'percentile': round(pct, 1),
            'priority': 3 if is_famous else 1
        })
    
    # Find best exemplar for each grade band
    exemplars = {}
    for grade, (min_score, max_score) in GRADE_BANDS.items():
        candidates = [s for s in scored if min_score <= s['score'] < max_score]
        if not candidates:
            continue
        
        # Sort by: priority (famous speakers), then by how close to middle of grade band
        target_score = (min_score + max_score) / 2
        candidates.sort(key=lambda x: (-x['priority'], abs(x['score'] - target_score)))
        
        best = candidates[0]
        exemplars[grade] = {
            'value': best['value'],
            'speaker': best['speaker'],
            'title': best['title'],
            'score': best['score'],
            'percentile': best['percentile'],
            'speech_id': best['speech_id'],
            'audio_filename': best['audio_filename'],
        }
    
    return exemplars


def main():
    print("=== Regenerating Metric Exemplars ===\n")
    
    load_available_audio()
    rows, distributions = get_distributions()
    
    results = {"metrics": {}}
    
    for metric_key in EXEMPLAR_METRICS:
        mdef = METRIC_DEFS.get(metric_key)
        if not mdef:
            print(f"Skipping {metric_key}: not defined")
            continue
        
        print(f"\n{metric_key} ({mdef['label']}):")
        exemplars = find_best_exemplars(rows, distributions, metric_key)
        
        if not exemplars:
            print(f"  No exemplars found")
            continue
        
        results["metrics"][metric_key] = {
            "label": mdef["label"],
            "direction": mdef["direction"],
            "unit": mdef["unit"],
            "grades": {}
        }
        
        for grade in ["A", "B", "C", "D", "F"]:
            if grade in exemplars:
                ex = exemplars[grade]
                audio_url = f"/static/exemplars/{ex['audio_filename']}"
                
                print(f"  {grade}: {ex['value']:.2f} {mdef['unit']} - {ex['speaker'][:35]} (score: {ex['score']:.1f})")
                
                results["metrics"][metric_key]["grades"][grade] = {
                    "value": ex['value'],
                    "speaker": ex['speaker'],
                    "title": ex['title'],
                    "audio_url": audio_url,
                }
            else:
                print(f"  {grade}: No candidate found")
    
    # Save results
    output_path = os.path.join(EXEMPLARS_DIR, "exemplars_new.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved to {output_path}")
    
    # Print coverage summary
    print("\n=== COVERAGE SUMMARY ===")
    for metric_key, mdata in results["metrics"].items():
        grades_found = list(mdata["grades"].keys())
        missing = [g for g in ["A", "B", "C", "D", "F"] if g not in grades_found]
        status = "✓" if not missing else f"⚠ missing: {', '.join(missing)}"
        print(f"  {metric_key}: {', '.join(grades_found)} {status}")


if __name__ == "__main__":
    main()
