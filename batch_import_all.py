#!/usr/bin/env python3
"""
Batch import all LibriSpeech and LJSpeech datasets to SpeechScore.
Robust version with retry logic, progress tracking, and resume capability.
"""

import os
import sys
import json
import time
import random
import requests
from pathlib import Path
from datetime import datetime

API_URL = "http://localhost:8000/v1/analyze"
API_KEY = "sk-5b7fd66025ead6b8731ef73b2c970f26"
DATASETS_DIR = Path("/home/melchior/speech3/datasets")
STATE_FILE = Path("/home/melchior/speech3/import_state.json")
LOG_FILE = Path("/tmp/batch_import_all.log")

# Dataset configs
DATASETS = [
    {"name": "dev-clean", "path": "LibriSpeech/dev-clean", "category": "reading", "ext": ".flac"},
    {"name": "test-clean", "path": "LibriSpeech/test-clean", "category": "reading", "ext": ".flac"},
    {"name": "train-clean-100", "path": "LibriSpeech/train-clean-100", "category": "reading", "ext": ".flac"},
    {"name": "train-clean-360", "path": "LibriSpeech/train-clean-360", "category": "reading", "ext": ".flac"},
    {"name": "train-other-500", "path": "LibriSpeech/train-other-500", "category": "reading", "ext": ".flac"},
    {"name": "LJSpeech", "path": "LJSpeech-1.1/wavs", "category": "reading", "ext": ".wav"},
]

MAX_RETRIES = 3
RETRY_DELAY = 5
BATCH_SIZE = 50  # Log progress every N files
IMPORT_LIMIT = 10  # Max files to import per run (0 = unlimited)


def log(msg):
    """Log to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_state():
    """Load processing state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"processed": {}, "stats": {"success": 0, "failed": 0, "skipped": 0}}


def save_state(state):
    """Save processing state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_all_files(dataset):
    """Get all audio files for a dataset."""
    dataset_path = DATASETS_DIR / dataset["path"]
    if not dataset_path.exists():
        log(f"Dataset path not found: {dataset_path}")
        return []
    
    files = list(dataset_path.rglob(f"*{dataset['ext']}"))
    return sorted(files)


def analyze_file(filepath, category, speaker=None, retries=0):
    """Send file to API for analysis with retry logic."""
    try:
        with open(filepath, "rb") as f:
            mime_type = "audio/flac" if filepath.suffix == ".flac" else "audio/wav"
            files = {"audio": (filepath.name, f, mime_type)}
            data = {
                "category": category,
                "profile": "reading_fluency",
            }
            if speaker:
                data["speaker"] = speaker
            
            response = requests.post(
                f"{API_URL}?api_key={API_KEY}",
                files=files,
                data=data,
                timeout=120  # 2 minute timeout for processing
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"
                
    except requests.exceptions.Timeout:
        if retries < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return analyze_file(filepath, category, speaker, retries + 1)
        return False, "Timeout after retries"
    except requests.exceptions.ConnectionError as e:
        if retries < MAX_RETRIES:
            log(f"Connection error, retry {retries + 1}/{MAX_RETRIES}...")
            time.sleep(RETRY_DELAY * (retries + 1))  # Exponential backoff
            return analyze_file(filepath, category, speaker, retries + 1)
        return False, f"Connection error: {e}"
    except Exception as e:
        return False, str(e)


def extract_speaker_from_path(filepath, dataset_name):
    """Extract speaker ID from filepath for LibriSpeech format."""
    if dataset_name.startswith("LJ"):
        return "LJSpeech"
    # LibriSpeech format: speaker_id/chapter_id/file.flac
    parts = filepath.parts
    for i, part in enumerate(parts):
        if part in ["dev-clean", "test-clean", "train-clean-100", "train-clean-360", "train-other-500"]:
            if i + 1 < len(parts):
                return f"Speaker-{parts[i+1]}"
    return None


def process_dataset(dataset, state, remaining_limit):
    """Process files in a dataset up to remaining_limit. Returns how many were processed."""
    dataset_name = dataset["name"]
    log(f"\n{'='*60}")
    log(f"Processing dataset: {dataset_name}")
    log(f"{'='*60}")
    
    files = get_all_files(dataset)
    if not files:
        log(f"No files found for {dataset_name}")
        return 0
    
    log(f"Found {len(files)} files")
    
    # Track processed files for this dataset
    if dataset_name not in state["processed"]:
        state["processed"][dataset_name] = []
    
    processed_set = set(state["processed"][dataset_name])
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    imported_this_run = 0
    
    for i, filepath in enumerate(files, 1):
        # Check limit
        if IMPORT_LIMIT > 0 and imported_this_run >= remaining_limit:
            log(f"Reached import limit ({IMPORT_LIMIT}), stopping.")
            break
            
        file_key = str(filepath)
        
        # Skip already processed
        if file_key in processed_set:
            skip_count += 1
            continue
        
        speaker = extract_speaker_from_path(filepath, dataset_name)
        success, result = analyze_file(filepath, dataset["category"], speaker)
        
        if success:
            success_count += 1
            state["stats"]["success"] += 1
        else:
            fail_count += 1
            state["stats"]["failed"] += 1
            if "Connection" in str(result) or "Timeout" in str(result):
                log(f"[{i}/{len(files)}] ✗ {filepath.name}: {result}")
        
        # Mark as processed (even failures, to not retry endlessly)
        state["processed"][dataset_name].append(file_key)
        imported_this_run += 1
        
        # Log progress periodically
        if imported_this_run % BATCH_SIZE == 0:
            log(f"Progress: {imported_this_run} imported ({success_count} success, {fail_count} fail)")
            save_state(state)
        
        # Small delay to not overwhelm API
        time.sleep(0.1)
    
    log(f"Dataset {dataset_name}: {success_count} success, {fail_count} fail, {skip_count} skipped")
    save_state(state)
    return imported_this_run


def main():
    log("\n" + "="*60)
    log(f"BATCH IMPORT (limit: {IMPORT_LIMIT} per run)")
    log("="*60)
    
    state = load_state()
    log(f"Loaded state: {state['stats']['success']} success, {state['stats']['failed']} failed so far")
    
    # Check API health
    try:
        r = requests.get(f"http://localhost:8000/v1/speeches?api_key={API_KEY}", 
                         timeout=5)
        if r.status_code != 200:
            log(f"API health check failed: {r.status_code}")
            sys.exit(1)
        log("API is healthy")
    except Exception as e:
        log(f"Cannot connect to API: {e}")
        sys.exit(1)
    
    # Process each dataset, respecting the limit across all
    remaining = IMPORT_LIMIT if IMPORT_LIMIT > 0 else float('inf')
    total_imported = 0
    
    for dataset in DATASETS:
        if IMPORT_LIMIT > 0 and remaining <= 0:
            break
        try:
            imported = process_dataset(dataset, state, remaining)
            total_imported += imported
            remaining -= imported
        except KeyboardInterrupt:
            log("Interrupted! Saving state...")
            save_state(state)
            sys.exit(0)
        except Exception as e:
            log(f"Error processing {dataset['name']}: {e}")
            save_state(state)
    
    log("\n" + "="*60)
    log(f"RUN COMPLETE: imported {total_imported} this run")
    log(f"Total all-time: {state['stats']['success']} success, {state['stats']['failed']} failed")
    log("="*60)


if __name__ == "__main__":
    main()
