#!/usr/bin/env python3
"""
Round 7: Download and process new speeches for SpeechScore database.
Targets: ReadAloud, OratoScore, PitchPerfect diversity.
Runs pipeline on each, persists to SQLite. GC between speeches.
"""

import os
import sys
import time
import json
import gc
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from speech_db import SpeechDB

SPEECHES = [
    # --- ReadAloud (children's content) ---
    {
        "url": "https://archive.org/download/english_fairy_tales_joy_librivox/english_fairy_tales_01.mp3",
        "title": "English Fairy Tales - Tom Tit Tot",
        "speaker": "Joy Chan",
        "category": "childrens_reading",
        "products": ["SpeechScore", "ReadAloud"],
        "clip_start": 0, "clip_duration": 180,
    },
    {
        "url": "https://archive.org/download/andersensfairy_1307_librivox/andersensfairytales_01_anderson.mp3",
        "title": "Andersen's Fairy Tales - The Emperor's New Clothes",
        "speaker": "LibriVox Volunteer",
        "category": "childrens_reading",
        "products": ["SpeechScore", "ReadAloud"],
        "clip_start": 0, "clip_duration": 180,
    },
    # --- OratoScore (diverse speakers) ---
    {
        "url": "https://archive.org/download/Malcolm_X/Malcolm_X_House_Negro_and_Field_Negro.mp3",
        "title": "House Negro and Field Negro",
        "speaker": "Malcolm X",
        "category": "civil_rights",
        "products": ["SpeechScore", "OratoScore"],
        "clip_start": 0, "clip_duration": 180,
    },
    {
        "url": "https://archive.org/download/Malcolm_X/Malcolm_X_On_Black_Nationalism.mp3",
        "title": "On Black Nationalism",
        "speaker": "Malcolm X",
        "category": "civil_rights",
        "products": ["SpeechScore", "OratoScore"],
        "clip_start": 0, "clip_duration": 180,
    },
    {
        "url": "https://archive.org/download/BarackObamaBarackObamaAcceptanceSpeech/obama_inauguration.mp3",
        "title": "2009 Presidential Inauguration",
        "speaker": "Barack Obama",
        "category": "political",
        "products": ["SpeechScore", "OratoScore", "PitchPerfect"],
        "clip_start": 0, "clip_duration": 180,
    },
    # --- PitchPerfect (presentations/lectures) ---
    {
        "url": "https://archive.org/download/Malcolm_X/Malcolm_X_1964_Oxford_Debate.mp3",
        "title": "Oxford Debate 1964",
        "speaker": "Malcolm X",
        "category": "academic_debate",
        "products": ["SpeechScore", "PitchPerfect"],
        "clip_start": 60, "clip_duration": 180,  # skip intro
    },
]


def download_and_clip(url, clip_start, clip_duration, output_path):
    """Download audio from URL and clip to specified duration."""
    # Download to temp
    tmp = output_path + ".download"
    print(f"    Downloading: {url[:80]}...")
    result = subprocess.run(
        ["curl", "-fsSL", "-o", tmp, url],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"    ERROR downloading: {result.stderr[:200]}")
        return False
    
    # Clip with ffmpeg
    print(f"    Clipping: {clip_start}s -> {clip_start + clip_duration}s")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp, "-ss", str(clip_start), "-t", str(clip_duration),
         "-acodec", "copy", output_path],
        capture_output=True, text=True, timeout=60
    )
    os.unlink(tmp)
    
    if result.returncode != 0:
        print(f"    ERROR clipping: {result.stderr[:200]}")
        return False
    
    return os.path.isfile(output_path)


def main():
    db = SpeechDB()
    test_dir = Path(__file__).parent / "test_speeches"
    test_dir.mkdir(exist_ok=True)
    
    # Load pipeline
    print("Loading pipeline models...")
    t0 = time.time()
    from api.pipeline_runner import preload_models, run_pipeline, convert_to_wav
    preload_models()
    print(f"Models loaded in {time.time() - t0:.1f}s\n")
    
    total = len(SPEECHES)
    success = 0
    
    for i, speech in enumerate(SPEECHES):
        print(f"\n[{i+1}/{total}] {speech['title']} ({speech['speaker']})")
        
        # Download and clip
        safe_name = speech['title'].lower().replace(' ', '_').replace("'", "")[:50]
        clip_path = str(test_dir / f"{safe_name}_clip.mp3")
        
        if os.path.isfile(clip_path):
            print(f"    Already downloaded: {clip_path}")
        else:
            ok = download_and_clip(
                speech["url"], 
                speech["clip_start"], 
                speech["clip_duration"],
                clip_path
            )
            if not ok:
                print(f"    FAILED — skipping")
                continue
        
        # Convert to WAV for pipeline
        wav_path = convert_to_wav(clip_path)
        
        # Run full pipeline
        print(f"    Running pipeline...")
        t1 = time.time()
        try:
            result = run_pipeline(
                wav_path,
                modules=None,
                language="en",
                quality_gate="warn",
                preset="full",
                modules_requested=["all"],
            )
            elapsed = time.time() - t1
            print(f"    Pipeline done in {elapsed:.1f}s")
        except Exception as e:
            print(f"    Pipeline FAILED: {e}")
            if os.path.isfile(wav_path) and wav_path != clip_path:
                os.unlink(wav_path)
            continue
        
        # Store in DB
        try:
            # Read raw audio for storage
            with open(clip_path, "rb") as f:
                audio_data = f.read()
            
            # Store speech
            speech_id = db.store_speech(
                title=speech["title"],
                speaker=speech["speaker"],
                category=speech["category"],
                audio_data=audio_data,
                audio_format="mp3",
                source_url=speech["url"],
                products=speech["products"],
            )
            
            # Store analysis
            db.store_analysis(speech_id, result, preset="full")
            
            # Store transcription
            transcript = result.get("transcript", "")
            segments = result.get("segments", [])
            if transcript:
                db.store_transcription(speech_id, transcript, segments=segments, engine="whisper-large-v3")
            
            print(f"    Stored as speech #{speech_id}")
            success += 1
        except Exception as e:
            print(f"    DB storage FAILED: {e}")
        
        # Clean up wav
        if os.path.isfile(wav_path) and wav_path != clip_path:
            os.unlink(wav_path)
        
        # GC between speeches
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
    
    print(f"\n{'='*60}")
    print(f"Done: {success}/{total} speeches added successfully")
    print(f"Total speeches in DB: {len(db.list_speeches(limit=500))}")


if __name__ == "__main__":
    main()
