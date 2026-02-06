#!/usr/bin/env python3
"""Analyze all speeches that are missing analysis."""
import sys, os, time, tempfile, subprocess, gc, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speech_db import SpeechDB

def get_missing_ids(db):
    """Find speech IDs that have audio but no analysis."""
    rows = db.conn.execute("""
        SELECT s.id FROM speeches s
        JOIN audio a ON a.speech_id = s.id
        WHERE s.id NOT IN (SELECT DISTINCT speech_id FROM analyses)
        ORDER BY s.id
    """).fetchall()
    return [r["id"] for r in rows]

def main():
    db = SpeechDB()
    
    missing_ids = get_missing_ids(db)
    if not missing_ids:
        print("All speeches are already analyzed!")
        return
    
    print(f"Found {len(missing_ids)} speeches missing analysis: {missing_ids}\n")
    
    print("Loading pipeline models...")
    t0 = time.time()
    from api.pipeline_runner import preload_models, run_pipeline, convert_to_wav
    preload_models()
    print(f"Models loaded in {time.time()-t0:.1f}s\n")
    
    success = 0
    failed = 0
    
    for idx, sid in enumerate(missing_ids, 1):
        row = db.conn.execute("SELECT id, title FROM speeches WHERE id = ?", (sid,)).fetchone()
        if not row:
            print(f"  [{idx}/{len(missing_ids)}] #{sid} — not found, skipping")
            failed += 1
            continue
        
        title = row["title"]
        
        # Check audio
        audio_row = db.conn.execute("SELECT data FROM audio WHERE speech_id = ?", (sid,)).fetchone()
        if not audio_row:
            print(f"  [{idx}/{len(missing_ids)}] #{sid} {title} — no audio, skipping")
            failed += 1
            continue
        
        print(f"  [{idx}/{len(missing_ids)}] #{sid} {title}...", end=" ", flush=True)
        
        # Export opus to temp, convert to wav
        with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f:
            f.write(audio_row["data"])
            opus_path = f.name
        
        wav_path = opus_path.replace(".opus", ".wav")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", opus_path, "-ar", "16000", "-ac", "1", wav_path],
                         capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"✗ ffmpeg error: {e.stderr.decode()[:200]}")
            os.unlink(opus_path)
            failed += 1
            continue
        
        os.unlink(opus_path)
        
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
        
        # Store analysis
        db.store_analysis(sid, result, preset="full")
        
        # Store transcription if available
        transcript = result.get("transcription", {})
        if transcript and transcript.get("text"):
            db.store_transcription(sid, transcript["text"],
                                   language=transcript.get("language", "en"),
                                   engine="faster-whisper-large-v3")
        
        elapsed = time.time() - t1
        wpm = result.get("audio", {}).get("overall_wpm") or result.get("audio", {}).get("words_per_minute", "?")
        print(f"✓ ({elapsed:.1f}s) WPM={wpm}")
        success += 1
        
        # Force garbage collection between speeches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n--- Done ---")
    print(f"Success: {success}, Failed: {failed}")
    
    # Verify
    remaining = get_missing_ids(db)
    if remaining:
        print(f"Still missing analysis: {remaining}")
    else:
        print("All speeches now have analysis! ✓")

if __name__ == "__main__":
    main()
