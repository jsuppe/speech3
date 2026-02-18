#!/usr/bin/env python3
"""Batch import audio files into SpeechScore database."""

import os
import sys
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from speech_db import SpeechDB

def analyze_file(audio_path: str, db_path: str, category: str = "reading") -> dict:
    """Analyze a single audio file and add to database."""
    try:
        # Use the API endpoint for analysis
        import requests
        
        with open(audio_path, 'rb') as f:
            files = {'audio': (os.path.basename(audio_path), f)}
            data = {
                'preset': 'full',
                'language': 'en',
                'category': category,
            }
            resp = requests.post(
                'http://localhost:8000/v1/analyze',
                files=files,
                data=data,
                params={'api_key': 'sk-5b7fd66025ead6b8731ef73b2c970f26'},
                timeout=120
            )
            
        if resp.status_code == 200:
            return {"success": True, "path": audio_path, "data": resp.json()}
        else:
            return {"success": False, "path": audio_path, "error": resp.text}
            
    except Exception as e:
        return {"success": False, "path": audio_path, "error": str(e)}


def process_directory(input_dir: str, category: str, limit: int = None, workers: int = 4):
    """Process all audio files in a directory."""
    
    # Find all audio files
    audio_files = []
    for ext in ['*.flac', '*.wav', '*.mp3', '*.ogg']:
        audio_files.extend(Path(input_dir).rglob(ext))
    
    if limit:
        audio_files = audio_files[:limit]
    
    print(f"Found {len(audio_files)} audio files to process")
    
    success_count = 0
    error_count = 0
    
    db_path = str(Path(__file__).parent / "speechscore.db")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(analyze_file, str(f), db_path, category): f 
            for f in audio_files
        }
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result["success"]:
                success_count += 1
                print(f"[{i+1}/{len(audio_files)}] ✓ {Path(result['path']).name}")
            else:
                error_count += 1
                print(f"[{i+1}/{len(audio_files)}] ✗ {Path(result['path']).name}: {result['error'][:50]}")
            
            # Progress update every 100 files
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{len(audio_files)} ({success_count} success, {error_count} errors)")
    
    print(f"\nComplete! {success_count} succeeded, {error_count} failed")
    return success_count, error_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch import audio files")
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("--category", default="reading", help="Category for speeches")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.category, args.limit, args.workers)
