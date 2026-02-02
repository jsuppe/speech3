#!/usr/bin/env python3
"""Re-analyze only speeches #32-36 that were missed when the full run got killed."""
import sys, os, time, json, tempfile, subprocess, gc, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speech_db import SpeechDB

REMAINING_IDS = [32, 33, 34, 35, 36]

def main():
    db = SpeechDB()
    
    print("Loading pipeline models...")
    t0 = time.time()
    from api.pipeline_runner import preload_models, run_pipeline, convert_to_wav
    preload_models()
    print(f"Models loaded in {time.time()-t0:.1f}s\n")
    
    for idx, sid in enumerate(REMAINING_IDS, 1):
        row = db.conn.execute("SELECT id, title FROM speeches WHERE id = ?", (sid,)).fetchone()
        if not row:
            print(f"  [{idx}/{len(REMAINING_IDS)}] #{sid} — not found, skipping")
            continue
        
        title = row["title"]
        
        # Check audio
        audio_row = db.conn.execute("SELECT data FROM audio WHERE speech_id = ?", (sid,)).fetchone()
        if not audio_row:
            print(f"  [{idx}/{len(REMAINING_IDS)}] #{sid} {title} — no audio, skipping")
            continue
        
        print(f"  [{idx}/{len(REMAINING_IDS)}] #{sid} {title}...", end=" ", flush=True)
        
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
            continue
        
        os.unlink(opus_path)
        
        t1 = time.time()
        try:
            result = run_pipeline(wav_path)
        except Exception as e:
            print(f"✗ pipeline error: {e}")
            if os.path.exists(wav_path):
                os.unlink(wav_path)
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
        pitch = result.get("audio", {}).get("pitch_mean_hz") or result.get("audio", {}).get("mean_f0", "?")
        quality = result.get("audio_quality", {}).get("quality_level") or result.get("audio_quality", {}).get("confidence", "?")
        print(f"✓ ({elapsed:.1f}s) WPM={wpm} Pitch={pitch} Q={quality}")
        
        # Force garbage collection between speeches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Verify
    print("\n--- Verification ---")
    rows = db.conn.execute("""
        SELECT s.id, s.title,
               (SELECT COUNT(*) FROM analyses a WHERE a.speech_id = s.id AND a.preset = 'full') as full_count,
               (SELECT COUNT(*) FROM audio au WHERE au.speech_id = s.id) as has_audio
        FROM speeches s
        WHERE s.id IN (32,33,34,35,36)
        ORDER BY s.id
    """).fetchall()
    for r in rows:
        status = '✓' if r['full_count'] > 0 else '✗'
        print(f"  {status} #{r['id']} {r['title']} — full analyses: {r['full_count']}")
    
    all_done = all(r['full_count'] > 0 for r in rows if r['has_audio'] > 0)
    print(f"\nAll remaining speeches processed: {'YES' if all_done else 'NO'}")

if __name__ == "__main__":
    main()
