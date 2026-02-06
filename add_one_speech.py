#!/usr/bin/env python3
"""
Process and add a single speech to the SpeechScore database.
Designed to be run as a separate process per speech to avoid OOM.
Usage: python add_one_speech.py <clip_path> <title> <speaker> <category> <products_json>
"""

import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    if len(sys.argv) < 6:
        print("Usage: python add_one_speech.py <clip_path> <title> <speaker> <category> <products_json>")
        sys.exit(1)
    
    clip_path = sys.argv[1]
    title = sys.argv[2]
    speaker = sys.argv[3]
    category = sys.argv[4]
    products = json.loads(sys.argv[5])
    
    if not os.path.isfile(clip_path):
        print(f"ERROR: File not found: {clip_path}")
        sys.exit(1)
    
    # Suppress noisy logs
    import logging
    logging.disable(logging.WARNING)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    from speech_db import SpeechDB
    db = SpeechDB()
    
    # Load pipeline
    print(f"Processing: {title}")
    t0 = time.time()
    from api.pipeline_runner import preload_models, run_pipeline, convert_to_wav
    preload_models()
    print(f"  Models loaded in {time.time() - t0:.1f}s")
    
    # Convert to WAV
    wav_path = convert_to_wav(clip_path)
    
    # Run full pipeline
    print(f"  Running pipeline...")
    t1 = time.time()
    result = run_pipeline(
        wav_path,
        modules=None,
        language="en",
        quality_gate="warn",
        preset="full",
        modules_requested=["all"],
    )
    elapsed = time.time() - t1
    print(f"  Pipeline done in {elapsed:.1f}s")
    
    # Store in DB
    speech_id = db.add_speech(
        title=title,
        speaker=speaker,
        category=category,
        products=products,
    )
    
    # Store audio (encodes to Opus automatically)
    db.store_audio(speech_id, clip_path)
    
    db.store_analysis(speech_id, result, preset="full")
    
    transcript = result.get("transcript", "")
    segments = result.get("segments", [])
    if transcript:
        db.store_transcription(speech_id, transcript, segments=segments, model="whisper-large-v3")
    
    # Generate and store spectrogram
    try:
        import shutil, subprocess, tempfile
        if shutil.which("songsee"):
            fd, png_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            r = subprocess.run(["songsee", clip_path, "-o", png_path],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and os.path.exists(png_path):
                with open(png_path, "rb") as f:
                    png_data = f.read()
                if png_data:
                    db.store_spectrogram(speech_id, png_data, spec_type="combined", fmt="png")
                    print(f"  Spectrogram stored ({len(png_data)} bytes)")
            if os.path.exists(png_path):
                os.unlink(png_path)
    except Exception as e:
        print(f"  Warning: spectrogram failed: {e}")
    
    # Clean up wav
    if os.path.isfile(wav_path) and wav_path != clip_path:
        os.unlink(wav_path)
    
    print(f"  SUCCESS: speech #{speech_id} — {title}")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
