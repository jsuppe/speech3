#!/usr/bin/env python3
"""
Batch analyze all oratory audio files that haven't been processed yet.
Uses the SpeakFit API for analysis.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

AUDIO_DIR = Path("/home/melchior/speech3/audio/oratory")
STATE_FILE = Path("/home/melchior/speech3/oratory_analysis_state.json")
API_URL = "http://localhost:8000/v1/analyze"
API_KEY = "sk-5b7fd66025ead6b8731ef73b2c970f26"

# Speaker name mappings from filename patterns
SPEAKER_MAP = {
    "Martin_Luther_King": ("Martin Luther King Jr.", "civil_rights"),
    "MLK": ("Martin Luther King Jr.", "civil_rights"),
    "John_F_Kennedy": ("John F. Kennedy", "political"),
    "JFK": ("John F. Kennedy", "political"),
    "Franklin_D_Roosevelt": ("Franklin D. Roosevelt", "political"),
    "FDR": ("Franklin D. Roosevelt", "political"),
    "Ronald_Reagan": ("Ronald Reagan", "political"),
    "Richard_Nixon": ("Richard Nixon", "political"),
    "Dwight_Eisenhower": ("Dwight Eisenhower", "political"),
    "Lyndon_B_Johnson": ("Lyndon B. Johnson", "political"),
    "LBJ": ("Lyndon B. Johnson", "political"),
    "Barbara_Jordan": ("Barbara Jordan", "political_keynote"),
    "Malcolm_X": ("Malcolm X", "civil_rights"),
    "Jesse_Jackson": ("Jesse Jackson", "political"),
    "Douglas_MacArthur": ("Douglas MacArthur", "military_address"),
    "Winston_Churchill": ("Winston Churchill", "political_wartime"),
    "Harry_Truman": ("Harry Truman", "political"),
    "Mario_Cuomo": ("Mario Cuomo", "political_keynote"),
    "Ted_Kennedy": ("Ted Kennedy", "political"),
    "Robert_F_Kennedy": ("Robert F. Kennedy", "political"),
    "Hillary_Clinton": ("Hillary Clinton", "advocacy"),
    "Eleanor_Roosevelt": ("Eleanor Roosevelt", "advocacy"),
    "Barbara_Bush": ("Barbara Bush", "commencement"),
    "Ann_Richards": ("Ann Richards", "political_keynote"),
    "Barry_Goldwater": ("Barry Goldwater", "political"),
    "Geraldine_Ferraro": ("Geraldine Ferraro", "political"),
    "George_Marshall": ("George Marshall", "political"),
    "Anita_Hill": ("Anita Hill", "testimony"),
    "William_Faulkner": ("William Faulkner", "literary_address"),
    "Mary_Fisher": ("Mary Fisher", "advocacy"),
    "Stokely_Carmichael": ("Stokely Carmichael", "civil_rights"),
    "Newton_Minow": ("Newton Minow", "regulatory_address"),
    "Elizabeth_Glaser": ("Elizabeth Glaser", "advocacy"),
    "Gerald_Ford": ("Gerald Ford", "political"),
    "Mario_Savio": ("Mario Savio", "protest"),
    "Spiro_Agnew": ("Spiro Agnew", "political"),
}

def get_speaker_info(filename):
    """Extract speaker and category from filename."""
    name = Path(filename).stem
    for pattern, (speaker, category) in SPEAKER_MAP.items():
        if pattern.lower() in name.lower():
            return speaker, category
    # Default
    clean_name = name.replace("_", " ").replace("-", " ")
    return clean_name.split()[0], "oratory"

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"analyzed": [], "failed": [], "stats": {"total": 0, "success": 0, "errors": 0}}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def analyze_file(audio_path):
    """Analyze a single audio file via API."""
    filename = os.path.basename(audio_path)
    speaker, category = get_speaker_info(filename)
    title = Path(filename).stem.replace("_", " ")
    
    try:
        with open(audio_path, 'rb') as f:
            files = {'audio': (filename, f, 'audio/mpeg')}
            data = {
                'title': title,
                'speaker': speaker,
                'category': category,
                'product': 'SpeakFit',
                'profile': 'default'
            }
            
            response = requests.post(
                f"{API_URL}?api_key={API_KEY}",
                files=files,
                data=data,
                timeout=600  # 10 min timeout for long speeches
            )
            
            if response.status_code == 200:
                result = response.json()
                return True, result.get('speech_id', 'unknown')
            else:
                return False, f"{response.status_code}: {response.text[:100]}"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("ORATORY BATCH ANALYZER")
    print(f"Started: {datetime.now()}")
    print("=" * 60)
    
    state = load_state()
    analyzed_files = set(state["analyzed"])
    failed_files = set(state["failed"])
    processed_files = analyzed_files | failed_files
    
    # Get all audio files, sorted by size (smallest first for quick wins)
    audio_files = sorted(AUDIO_DIR.glob("*.mp3"), key=lambda f: f.stat().st_size)
    print(f"Found {len(audio_files)} audio files (sorted by size, smallest first)")
    print(f"Already processed: {len(processed_files)} (analyzed: {len(analyzed_files)}, failed: {len(failed_files)})")
    
    to_analyze = [f for f in audio_files if str(f) not in processed_files]
    print(f"To analyze: {len(to_analyze)}")
    
    for i, audio_path in enumerate(to_analyze):
        filename = audio_path.name
        speaker, category = get_speaker_info(filename)
        
        print(f"\n[{i+1}/{len(to_analyze)}] Analyzing: {filename}")
        print(f"  Speaker: {speaker}, Category: {category}")
        sys.stdout.flush()  # Force output
        
        try:
            success, result = analyze_file(str(audio_path))
            
            if success:
                print(f"  ✓ Success: id={result}")
                state["analyzed"].append(str(audio_path))
                state["stats"]["success"] += 1
            else:
                print(f"  ✗ Failed: {result}")
                state["failed"].append(str(audio_path))
                state["stats"]["errors"] += 1
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            state["failed"].append(str(audio_path))
            state["stats"]["errors"] += 1
        
        state["stats"]["total"] += 1
        save_state(state)
        sys.stdout.flush()
        
        # Brief pause between requests
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total processed: {state['stats']['total']}")
    print(f"Successful: {state['stats']['success']}")
    print(f"Errors: {state['stats']['errors']}")

if __name__ == "__main__":
    main()
