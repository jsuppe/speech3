#!/usr/bin/env python3
"""Fix speech #39 — add missing transcription and spectrogram."""
import os, sys, json, subprocess, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from speech_db import SpeechDB
db = SpeechDB()

speech = db.get_speech(39)
print(f"Speech #39: {speech['title']}")

# Get analysis to extract transcript
full = db.get_full_analysis(39)
if full:
    transcript = full.get("transcript", "")
    segments = full.get("segments", [])
    if transcript:
        db.store_transcription(39, transcript, segments=segments, model="whisper-large-v3")
        print(f"  Transcription stored ({len(transcript)} chars)")
    else:
        print("  No transcript in analysis result")
else:
    print("  No analysis found")

# Generate spectrogram
clip_path = "test_speeches/english_fairy_tales_-_tom_tit_tot_clip.mp3"
if os.path.isfile(clip_path) and shutil.which("songsee"):
    fd, png_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    r = subprocess.run(["songsee", clip_path, "-o", png_path],
                       capture_output=True, text=True, timeout=60)
    if r.returncode == 0 and os.path.exists(png_path):
        with open(png_path, "rb") as f:
            png_data = f.read()
        if png_data:
            db.store_spectrogram(39, png_data, spec_type="combined", fmt="png")
            print(f"  Spectrogram stored ({len(png_data)} bytes)")
    if os.path.exists(png_path):
        os.unlink(png_path)
else:
    print(f"  Cannot generate spectrogram (file={os.path.isfile(clip_path)}, songsee={shutil.which('songsee')})")

print("Done")
