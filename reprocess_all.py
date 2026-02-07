#!/usr/bin/env python3
"""Reprocess all speeches with word-level timestamps."""
import sys, os, time, tempfile, subprocess, gc, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speech_db import SpeechDB

def main():
    db = SpeechDB()
    
    # Get all speeches with audio
    rows = db.conn.execute("""
        SELECT s.id, s.title FROM speeches s
        JOIN audio a ON a.speech_id = s.id
        ORDER BY s.id
    """).fetchall()
    
    total = len(rows)
    print(f"Found {total} speeches to reprocess\n")
    
    print("Loading pipeline models...")
    t0 = time.time()
    from api.pipeline_runner import preload_models, run_pipeline
    preload_models()
    print(f"Models loaded in {time.time()-t0:.1f}s\n")
    
    success = 0
    failed = 0
    
    for idx, row in enumerate(rows, 1):
        sid = row["id"]
        title = row["title"]
        
        # Get audio
        audio_row = db.conn.execute("SELECT data, format FROM audio WHERE speech_id = ?", (sid,)).fetchone()
        if not audio_row:
            print(f"  [{idx}/{total}] #{sid} {title} — no audio, skipping")
            failed += 1
            continue
        
        print(f"  [{idx}/{total}] #{sid} {title[:50]}...", end=" ", flush=True)
        
        # Export audio to temp file
        fmt = audio_row["format"] or "opus"
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
            f.write(audio_row["data"])
            audio_path = f.name
        
        # Convert to WAV for processing
        wav_path = audio_path.replace(f".{fmt}", ".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"✗ ffmpeg error")
            os.unlink(audio_path)
            failed += 1
            continue
        
        os.unlink(audio_path)
        
        t1 = time.time()
        try:
            result = run_pipeline(wav_path)
        except Exception as e:
            print(f"✗ pipeline error: {e}")
            if os.path.exists(wav_path):
                os.unlink(wav_path)
            failed += 1
            continue
        
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        
        # Update analysis (replace existing)
        db.conn.execute("DELETE FROM analyses WHERE speech_id = ? AND preset = 'full'", (sid,))
        db.store_analysis(sid, result, preset="full")
        
        # Update transcription with word timestamps
        transcript = result.get("transcript", "")
        segments = result.get("segments", [])
        if transcript:
            db.conn.execute("DELETE FROM transcriptions WHERE speech_id = ?", (sid,))
            db.conn.execute("""
                INSERT INTO transcriptions (speech_id, text, segments, word_count, language, model)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                sid,
                transcript,
                __import__('json').dumps(segments),
                len(transcript.split()),
                result.get("language", "en"),
                "faster-whisper-large-v3"
            ))
            db.conn.commit()
        
        elapsed = time.time() - t1
        word_count = sum(len(seg.get("words", [])) for seg in segments)
        print(f"✓ ({elapsed:.1f}s) {word_count} words timestamped")
        success += 1
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n--- Done ---")
    print(f"Success: {success}, Failed: {failed}")
    
    # Verify word timestamps
    sample = db.conn.execute("""
        SELECT segments FROM transcriptions WHERE speech_id = 1
    """).fetchone()
    if sample and sample["segments"]:
        import json
        segs = json.loads(sample["segments"])
        if segs and segs[0].get("words"):
            print(f"\nWord timestamps verified! Sample: {segs[0]['words'][:3]}")
        else:
            print("\nWarning: Word timestamps not found in sample")

if __name__ == "__main__":
    main()
