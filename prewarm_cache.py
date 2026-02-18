#!/usr/bin/env python3
"""Pre-warm score cache for all speeches."""
import sqlite3
import requests
import time
import sys

API_KEY = "sk-5b7fd66025ead6b8731ef73b2c970f26"
BASE_URL = "http://localhost:8000/v1"
DB_PATH = "/home/melchior/speech3/speechscore.db"
BATCH_SIZE = 100

def get_uncached_speeches():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id FROM speeches 
        WHERE cached_score_json IS NULL
        ORDER BY id
    ''')
    ids = [r[0] for r in c.fetchall()]
    conn.close()
    return ids

def warm_speech(speech_id):
    try:
        r = requests.get(
            f"{BASE_URL}/speeches/{speech_id}/score",
            params={"api_key": API_KEY},
            timeout=60
        )
        # Rate limit: 1 req/sec to avoid saturating API
        time.sleep(1)
        if r.status_code == 200:
            return True, r.json().get("cached", False)
        return False, r.status_code
    except Exception as e:
        return False, str(e)

def main():
    uncached = get_uncached_speeches()
    total = len(uncached)
    print(f"Starting pre-warm for {total} speeches...")
    sys.stdout.flush()
    
    success = 0
    failed = 0
    start_time = time.time()
    
    for i, speech_id in enumerate(uncached):
        ok, info = warm_speech(speech_id)
        if ok:
            success += 1
        else:
            failed += 1
            print(f"  ✗ {speech_id}: {info}")
        
        # Progress every 100
        if (i + 1) % BATCH_SIZE == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"Progress: {i+1}/{total} ({success} ok, {failed} failed) - {remaining/3600:.1f}h remaining")
            sys.stdout.flush()
    
    elapsed = time.time() - start_time
    print(f"\nDone! {success} cached, {failed} failed in {elapsed/3600:.1f}h")

if __name__ == "__main__":
    main()
