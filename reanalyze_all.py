#!/usr/bin/env python3
"""
Re-analyze all speeches in the DB through the full pipeline.
Exports audio from DB → runs pipeline → stores fresh analysis.
Replaces migrated/partial data with complete, consistent metrics.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from speech_db import SpeechDB


def main():
    db = SpeechDB()
    speeches = db.list_speeches(limit=500)
    speeches.sort(key=lambda s: s["id"])

    # Lazy-import pipeline (loads models)
    print("Loading pipeline models...")
    t0 = time.time()
    sys.path.insert(0, str(Path(__file__).parent))
    from api.pipeline_runner import preload_models, run_pipeline, convert_to_wav
    preload_models()
    print(f"Models loaded in {time.time() - t0:.1f}s\n")

    total = len(speeches)
    analyzed = 0
    skipped = 0
    failed = 0

    for i, speech in enumerate(speeches):
        sid = speech["id"]
        title = speech["title"]

        # Get audio from DB
        audio = db.get_audio(sid)
        if not audio:
            print(f"  [{i+1}/{total}] #{sid} {title} — no audio, skipping")
            skipped += 1
            continue

        print(f"  [{i+1}/{total}] #{sid} {title}...", end=" ", flush=True)

        # Export audio to temp file
        suffix = f".{audio['format']}"
        fd, audio_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        wav_path = None

        try:
            with open(audio_path, "wb") as f:
                f.write(audio["data"])

            # Convert to WAV for pipeline
            wav_path = convert_to_wav(audio_path)

            # Run full pipeline
            t1 = time.time()
            result = run_pipeline(
                wav_path,
                modules=None,  # all modules
                language="en",
                quality_gate="warn",
                preset="full",
                modules_requested=["all"],
            )
            elapsed = time.time() - t1

            # Store fresh analysis
            db.store_analysis(sid, result, preset="full")

            # Update transcription if we got a better one
            transcript = result.get("transcript", "")
            segments = result.get("segments", [])
            if transcript:
                db.store_transcription(
                    sid,
                    text=transcript,
                    segments=segments,
                    language=result.get("language"),
                )

            wpm = result.get("audio_analysis", {}).get("speaking_rate", {}).get("overall_wpm", "?")
            pitch = result.get("audio_analysis", {}).get("pitch", {}).get("mean_hz", "?")
            quality = result.get("audio_quality", {}).get("confidence", "?")

            print(f"✓ ({elapsed:.1f}s) WPM={wpm} Pitch={pitch} Q={quality}")
            analyzed += 1

        except Exception as e:
            print(f"✗ {e}")
            failed += 1

        finally:
            for p in [audio_path, wav_path]:
                if p and os.path.exists(p):
                    os.unlink(p)

    print(f"\n{'='*60}")
    print(f"Re-analysis complete!")
    print(f"  Analyzed: {analyzed}")
    print(f"  Skipped:  {skipped}")
    print(f"  Failed:   {failed}")

    # Verify completeness
    print("\nVerifying metric completeness...")
    incomplete = 0
    for s in db.list_speeches(limit=500):
        a = db.get_analysis(s["id"])
        if a:
            missing = [k for k in ["wpm", "pitch_mean_hz", "hnr_db", "quality_level", "snr_db"]
                       if a.get(k) is None]
            if missing:
                incomplete += 1
    print(f"  Speeches with missing core metrics: {incomplete}")

    db.close()


if __name__ == "__main__":
    main()
