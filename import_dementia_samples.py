#!/usr/bin/env python3
"""Import dementia/aphasia samples and analyze them for test accounts."""

import os
import sys
import json
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from speech_db import SpeechDB
from api.pipeline_runner import run_pipeline, persist_result, get_db, preload_models

SAMPLES_DIR = Path("/home/melchior/speech3/dementia_samples")

# Sample metadata
SAMPLES = [
    {"file": "broca_aphasia.mp3", "title": "Broca's Aphasia Example", "category": "aphasia"},
    {"file": "wernicke_aphasia.mp3", "title": "Wernicke's Aphasia (Fluent)", "category": "aphasia"},
    {"file": "conduction_aphasia.mp3", "title": "Conduction Aphasia Example", "category": "aphasia"},
    {"file": "global_aphasia.mp3", "title": "Global Aphasia Example", "category": "aphasia"},
    {"file": "expressive_aphasia_sarah.mp3", "title": "Expressive Aphasia - Sarah Scott", "category": "aphasia"},
    {"file": "dementia_kids_interview.mp3", "title": "Dementia Patient Interview", "category": "dementia"},
    {"file": "dementia_speech_patterns.mp3", "title": "Dementia Speech Patterns", "category": "dementia"},
    {"file": "alzheimers_patient_10yr.mp3", "title": "Alzheimer's Patient 10-Year Follow-up", "category": "dementia"},
]

# Target accounts
TARGET_USERS = [4, 5]  # Eric Arezzo, Lorraine Boogich


def main():
    print("Loading models...")
    preload_models()
    print("Models loaded.")
    
    db = get_db()
    
    imported_count = 0
    
    for sample in SAMPLES:
        audio_path = SAMPLES_DIR / sample["file"]
        
        if not audio_path.exists():
            print(f"[SKIP] File not found: {audio_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {sample['title']}")
        print(f"{'='*60}")
        
        # Import for each target user
        for user_id in TARGET_USERS:
            print(f"\n  User ID: {user_id}")
            
            # Check if already imported
            existing = db.conn.execute(
                "SELECT id FROM speeches WHERE title = ? AND user_id = ?",
                (sample["title"], user_id)
            ).fetchone()
            
            if existing:
                print(f"  [SKIP] Already exists: speech_id={existing[0]}")
                continue
            
            try:
                # Run analysis pipeline
                print(f"  Running analysis...")
                result = run_pipeline(
                    audio_path=str(audio_path),
                    preset="full",
                    language="en",
                )
                
                # Persist the result (note: persist_result gets db internally)
                speech_id = persist_result(
                    original_audio_path=str(audio_path),
                    result=result,
                    title=sample["title"],
                    category=sample["category"],
                    profile="dementia",
                    user_id=user_id,
                )
                
                if speech_id:
                    print(f"  [OK] Imported: speech_id={speech_id}")
                    imported_count += 1
                else:
                    print(f"  [ERROR] No speech_id returned")
                    
            except Exception as e:
                import traceback
                print(f"  [ERROR] {e}")
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Imported {imported_count} recordings")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
