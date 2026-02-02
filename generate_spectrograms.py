#!/usr/bin/env python3
"""
Generate spectrograms for all speeches in the database using songsee.
Stores results as PNG blobs in the spectrograms table.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from speech_db import SpeechDB


def generate_spectrogram(audio_data: bytes, audio_format: str = "opus") -> bytes | None:
    """Generate a spectrogram PNG from audio bytes using songsee."""
    with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as audio_tmp:
        audio_tmp.write(audio_data)
        audio_path = audio_tmp.name

    png_path = audio_path + ".png"
    try:
        result = subprocess.run(
            ["songsee", audio_path, "-o", png_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"    songsee error: {result.stderr[:200]}")
            return None
        if not os.path.exists(png_path):
            return None
        with open(png_path, "rb") as f:
            return f.read()
    except subprocess.TimeoutExpired:
        print("    songsee timed out")
        return None
    except Exception as e:
        print(f"    songsee exception: {e}")
        return None
    finally:
        for p in [audio_path, png_path]:
            if os.path.exists(p):
                os.unlink(p)


def main():
    db = SpeechDB()
    speeches = db.list_speeches(limit=500)
    
    total = len(speeches)
    generated = 0
    skipped = 0
    failed = 0

    print(f"Generating spectrograms for {total} speeches...\n")

    for i, speech in enumerate(sorted(speeches, key=lambda s: s["id"])):
        sid = speech["id"]
        title = speech["title"]

        # Check if spectrogram already exists
        existing = db.get_spectrogram(sid, spec_type="combined")
        if existing:
            print(f"  [{i+1}/{total}] #{sid} {title} — already has spectrogram, skipping")
            skipped += 1
            continue

        # Get audio
        audio = db.get_audio(sid)
        if not audio:
            print(f"  [{i+1}/{total}] #{sid} {title} — no audio, skipping")
            skipped += 1
            continue

        print(f"  [{i+1}/{total}] #{sid} {title}...", end=" ", flush=True)

        png_data = generate_spectrogram(audio["data"], audio["format"])
        if png_data:
            db.store_spectrogram(sid, png_data, spec_type="combined", fmt="png")
            size_kb = len(png_data) / 1024
            print(f"✓ ({size_kb:.0f} KB)")
            generated += 1
        else:
            print("✗ failed")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done! Generated: {generated}, Skipped: {skipped}, Failed: {failed}")
    
    stats = db.stats()
    print(f"Spectrograms in DB: {stats['spectrograms']}")
    print(f"DB size: {stats['db_size_mb']} MB")
    db.close()


if __name__ == "__main__":
    main()
