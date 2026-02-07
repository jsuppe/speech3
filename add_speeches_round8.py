#!/usr/bin/env python3
"""
Round 8: Expand database to 50+ speeches.
Focus on: presentation, debate, casual, motivational.
"""

import os
import sys
import time
import gc
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from speech_db import SpeechDB

SPEECHES = [
    # --- Presentations / TED-style ---
    {
        "url": "https://archive.org/download/SteveJobsStanfordCommencement2005/Steve_Jobs_Stanford_Commencement_2005.mp3",
        "title": "Stanford Commencement 2005 - Stay Hungry Stay Foolish",
        "speaker": "Steve Jobs",
        "category": "presentation",
        "products": ["SpeechScore", "PitchPerfect"],
        "clip_start": 60, "clip_duration": 180,
    },
    {
        "url": "https://archive.org/download/RandyPauschLectureAchievingYourChildhoodDreams/Randy-Pausch-Lecture.mp3",
        "title": "The Last Lecture - Achieving Your Childhood Dreams",
        "speaker": "Randy Pausch",
        "category": "presentation",
        "products": ["SpeechScore", "PitchPerfect"],
        "clip_start": 120, "clip_duration": 180,
    },
    # --- Debates ---
    {
        "url": "https://archive.org/download/Kennedy-Nixon-FirstDebate/Kennedy-Nixon-Debate-1.mp3",
        "title": "Kennedy-Nixon First Presidential Debate 1960",
        "speaker": "John F. Kennedy",
        "category": "debate",
        "products": ["SpeechScore", "OratoScore"],
        "clip_start": 300, "clip_duration": 180,  # Skip intro
    },
    {
        "url": "https://archive.org/download/Reagan-Carter-Debate-1980/Reagan-Carter-Debate.mp3",
        "title": "Reagan-Carter Presidential Debate 1980",
        "speaker": "Ronald Reagan",
        "category": "debate",
        "products": ["SpeechScore", "OratoScore"],
        "clip_start": 300, "clip_duration": 180,
    },
    # --- Motivational ---
    {
        "url": "https://archive.org/download/MotivationalSpeakers/art_williams_just_do_it.mp3",
        "title": "Just Do It Speech",
        "speaker": "Art Williams",
        "category": "motivational",
        "products": ["SpeechScore", "PitchPerfect"],
        "clip_start": 30, "clip_duration": 180,
    },
    {
        "url": "https://archive.org/download/CharlieChaplinTheGreatDictator/Charlie_Chaplin_Final_Speech.mp3",
        "title": "The Great Dictator - Final Speech",
        "speaker": "Charlie Chaplin",
        "category": "motivational",
        "products": ["SpeechScore", "OratoScore"],
        "clip_start": 0, "clip_duration": 180,
    },
    # --- More Readings (diverse voices) ---
    {
        "url": "https://archive.org/download/grimms_fairy_tales_librivox/grimms_fairy_tales_01_grimm.mp3",
        "title": "Grimm's Fairy Tales - The Frog Prince",
        "speaker": "LibriVox Volunteer",
        "category": "reading",
        "products": ["SpeechScore", "ReadAloud"],
        "clip_start": 30, "clip_duration": 180,
    },
    {
        "url": "https://archive.org/download/aesops_fables_vol1_librivox/aesopsfables_01_aesop.mp3",
        "title": "Aesop's Fables - The Fox and the Grapes",
        "speaker": "LibriVox Volunteer",
        "category": "reading",
        "products": ["SpeechScore", "ReadAloud"],
        "clip_start": 0, "clip_duration": 120,
    },
    # --- Political (diverse) ---
    {
        "url": "https://archive.org/download/EleanorRoosevelt/Eleanor_Roosevelt_UN_Human_Rights.mp3",
        "title": "United Nations Human Rights Declaration",
        "speaker": "Eleanor Roosevelt",
        "category": "political",
        "products": ["SpeechScore", "OratoScore"],
        "clip_start": 0, "clip_duration": 180,
    },
    {
        "url": "https://archive.org/download/BarbaraJordan1976DNCKeynote/Barbara_Jordan_1976_DNC.mp3",
        "title": "1976 DNC Keynote Address",
        "speaker": "Barbara Jordan",
        "category": "political",
        "products": ["SpeechScore", "OratoScore"],
        "clip_start": 60, "clip_duration": 180,
    },
]


def download_and_clip(url, clip_start, clip_duration, output_path):
    """Download audio from URL and clip to specified duration."""
    tmp = output_path + ".download"
    print(f"    Downloading: {url[:80]}...")
    
    result = subprocess.run(
        ["curl", "-fsSL", "-o", tmp, url],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"    ERROR downloading: {result.stderr[:200]}")
        return False
    
    print(f"    Clipping: {clip_start}s -> {clip_start + clip_duration}s")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp, "-ss", str(clip_start), "-t", str(clip_duration),
         "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", output_path],
        capture_output=True, text=True, timeout=60
    )
    
    if os.path.isfile(tmp):
        os.unlink(tmp)
    
    if result.returncode != 0:
        print(f"    ERROR clipping: {result.stderr[:200]}")
        return False
    
    return os.path.isfile(output_path)


def main():
    db = SpeechDB()
    test_dir = Path(__file__).parent / "test_speeches"
    test_dir.mkdir(exist_ok=True)
    
    print("Loading pipeline models...")
    t0 = time.time()
    from api.pipeline_runner import preload_models, run_pipeline, convert_to_wav
    preload_models()
    print(f"Models loaded in {time.time() - t0:.1f}s\n")
    
    total = len(SPEECHES)
    success = 0
    
    for i, speech in enumerate(SPEECHES):
        print(f"\n[{i+1}/{total}] {speech['title']} ({speech['speaker']})")
        
        safe_name = speech['title'].lower().replace(' ', '_').replace("'", "").replace("-", "_")[:50]
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
        
        wav_path = convert_to_wav(clip_path)
        
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
        
        try:
            with open(clip_path, "rb") as f:
                audio_data = f.read()
            
            speech_id = db.store_speech(
                title=speech["title"],
                speaker=speech["speaker"],
                category=speech["category"],
                audio_data=audio_data,
                audio_format="mp3",
                source_url=speech["url"],
                products=speech["products"],
            )
            
            db.store_analysis(speech_id, result, preset="full")
            
            transcript = result.get("transcript", "")
            segments = result.get("segments", [])
            if transcript:
                db.store_transcription(speech_id, transcript, segments=segments, engine="whisper-large-v3")
            
            print(f"    Stored as speech #{speech_id}")
            success += 1
        except Exception as e:
            print(f"    DB storage FAILED: {e}")
        
        if os.path.isfile(wav_path) and wav_path != clip_path:
            os.unlink(wav_path)
        
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
