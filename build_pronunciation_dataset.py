"""
Build Pronunciation Training Dataset
Extracts pronunciation features from recordings and pairs them with existing oratory scores.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

from pronunciation_features import PronunciationFeatureExtractor, RecordingFeatures


def get_scored_recordings(db_path: str = "speechscore.db") -> List[Dict]:
    """Get all recordings that have oratory scores."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT id, title, source_file, cached_score, cached_score_json
        FROM speeches 
        WHERE cached_score IS NOT NULL 
        AND source_file IS NOT NULL
    ''')
    
    recordings = []
    for row in c.fetchall():
        score_data = json.loads(row['cached_score_json']) if row['cached_score_json'] else {}
        recordings.append({
            'id': row['id'],
            'title': row['title'],
            'audio_path': row['source_file'],
            'oratory_score': row['cached_score'],
            'composite_score': score_data.get('composite_score'),
            'category_scores': score_data.get('category_scores', {}),
        })
    
    conn.close()
    return recordings


def build_dataset(
    db_path: str = "speechscore.db",
    output_path: str = "pronunciation_training_data.json",
    output_csv: str = "pronunciation_training_data.csv"
) -> pd.DataFrame:
    """
    Build the pronunciation training dataset.
    
    Returns a DataFrame with:
    - All pronunciation features from wav2vec2
    - oratory_score as the training label
    """
    recordings = get_scored_recordings(db_path)
    print(f"Found {len(recordings)} scored recordings")
    
    extractor = PronunciationFeatureExtractor()
    
    results = []
    for i, rec in enumerate(recordings):
        audio_path = rec['audio_path']
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"  [{i+1}/{len(recordings)}] SKIP - File not found: {audio_path}")
            continue
        
        print(f"  [{i+1}/{len(recordings)}] Processing: {rec['title'][:40]}...")
        
        try:
            features = extractor.extract_features(
                audio_path=audio_path,
                recording_id=str(rec['id']),
                human_score=rec['oratory_score']  # Use oratory score as label
            )
            
            # Add extra metadata
            feature_dict = features.to_dict()
            feature_dict['title'] = rec['title']
            feature_dict['oratory_score'] = rec['oratory_score']
            feature_dict['category_scores'] = rec['category_scores']
            
            results.append(feature_dict)
            print(f"    → oratory_score: {rec['oratory_score']:.1f}, " +
                  f"mean_confidence: {features.mean_word_confidence:.3f}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save outputs
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} recordings to {output_path}")
    
    # Save CSV (without nested dicts)
    csv_df = df.drop(columns=['category_scores'], errors='ignore')
    csv_df.to_csv(output_csv, index=False)
    print(f"Saved CSV to {output_csv}")
    
    return df


def analyze_correlations(df: pd.DataFrame):
    """Analyze which pronunciation features correlate with oratory score."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS: Pronunciation Features vs Oratory Score")
    print("="*60)
    
    # Numeric columns only
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Calculate correlations with oratory_score
    correlations = {}
    for col in numeric_cols:
        if col not in ['oratory_score', 'recording_id']:
            corr = df['oratory_score'].corr(df[col])
            if not pd.isna(corr):
                correlations[col] = corr
    
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTop features correlated with oratory_score:")
    print("-" * 50)
    for feat, corr in sorted_corrs[:10]:
        direction = "↑" if corr > 0 else "↓"
        print(f"  {feat:35} {direction} {corr:+.3f}")
    
    return correlations


if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "speechscore.db"
    
    print("Building pronunciation training dataset...")
    print("=" * 60)
    
    df = build_dataset(db_path)
    
    if len(df) > 5:
        analyze_correlations(df)
    else:
        print("\nNeed more samples for correlation analysis.")
    
    print("\n✅ Dataset ready for model training!")
    print(f"   Samples: {len(df)}")
    print(f"   Features: {len(df.columns) - 3}")  # minus id, title, category_scores
