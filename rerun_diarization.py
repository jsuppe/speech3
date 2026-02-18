#!/usr/bin/env python3
"""Re-run diarization on specific speeches and update them in-place."""

import json
import sys
import os
import tempfile

# Add parent to path for imports
sys.path.insert(0, '/home/melchior/speech3')

from speech_db import SpeechDB
from diarization import run_diarization, match_speakers_to_profiles, register_diarized_speakers

def rerun_diarization(speech_ids: list):
    """Re-run diarization on specified speech IDs."""
    db = SpeechDB()
    
    for speech_id in speech_ids:
        print(f"\n{'='*50}")
        print(f"Processing speech {speech_id}...")
        
        # Get current speech data
        speech = db.get_speech(speech_id)
        if not speech:
            print(f"  ERROR: Speech {speech_id} not found")
            continue
            
        print(f"  Title: {speech.get('title', 'Untitled')}")
        print(f"  Current profile: {speech.get('profile', 'unknown')}")
        
        # Get audio
        audio_row = db.conn.execute(
            "SELECT data FROM audio WHERE speech_id = ?", (speech_id,)
        ).fetchone()
        
        if not audio_row:
            print(f"  ERROR: No audio found for speech {speech_id}")
            continue
        
        # Write audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(audio_row[0])
            audio_path = f.name
        
        print(f"  Audio extracted: {os.path.getsize(audio_path)} bytes")
        
        # Get segments from the analyses table
        analysis_row = db.conn.execute(
            "SELECT json_extract(result, '$.segments') FROM analyses WHERE speech_id = ? ORDER BY id DESC LIMIT 1",
            (speech_id,)
        ).fetchone()
        
        if not analysis_row or not analysis_row[0]:
            print(f"  ERROR: No transcript segments found in analyses")
            os.unlink(audio_path)
            continue
            
        segments = json.loads(analysis_row[0])
            
        print(f"  Found {len(segments)} transcript segments")
        
        try:
            # Run diarization
            print("  Running diarization...")
            diarized_segments = run_diarization(audio_path, segments)
            
            if diarized_segments:
                # Count unique speakers
                speakers_found = set(s.get('speaker', 'Unknown') for s in diarized_segments)
                print(f"  SUCCESS: {len(speakers_found)} speakers, {len(diarized_segments)} segments")
                
                user_id = speech.get('user_id', 1)
                
                # Match to stored profiles (db, user_id, audio_path, segments)
                matched_segments, match_info = match_speakers_to_profiles(db, user_id, audio_path, diarized_segments)
                print(f"  Matched {len(match_info.get('matches', []))} speakers to stored profiles")
                
                # Register any new speakers (db, user_id, audio_path, segments)
                reg_result = register_diarized_speakers(db, user_id, audio_path, matched_segments)
                print(f"  Registered {len(reg_result.get('registered', []))} new speaker profiles")
                
                # Build diarization object
                from diarization import get_speaker_stats
                stats = get_speaker_stats(matched_segments)
                diarization_obj = {
                    "segments": matched_segments,
                    "num_speakers": stats['num_speakers'],
                    "speaker_stats": stats['speakers'],
                    "profile_matches": match_info.get('matches', []),
                }
                
                # Update the analysis record with new diarization
                diarization_json = json.dumps(diarization_obj)
                
                # Get current analysis result
                current_result_row = db.conn.execute(
                    "SELECT id, result FROM analyses WHERE speech_id = ? ORDER BY id DESC LIMIT 1",
                    (speech_id,)
                ).fetchone()
                
                if current_result_row:
                    analysis_id, current_result_str = current_result_row
                    current_result = json.loads(current_result_str)
                    current_result['diarization'] = diarization_obj
                    updated_result = json.dumps(current_result)
                    
                    db.conn.execute(
                        "UPDATE analyses SET result = ? WHERE id = ?",
                        (updated_result, analysis_id)
                    )
                
                # Also update the speech profile
                db.conn.execute(
                    "UPDATE speeches SET profile = 'group' WHERE id = ?",
                    (speech_id,)
                )
                db.conn.commit()
                
                print(f"  Updated speech {speech_id} with diarization data")
                
                # Show speakers
                print(f"  Speakers: {', '.join(speakers_found)}")
            else:
                print(f"  WARNING: Diarization returned no segments")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            os.unlink(audio_path)
    
    print(f"\n{'='*50}")
    print("Done!")

if __name__ == "__main__":
    # Speech IDs to process
    speech_ids = [9560, 9561, 9562, 9564]
    rerun_diarization(speech_ids)
