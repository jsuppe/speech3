#!/usr/bin/env python3
"""
Download World's Famous Orations from LibriVox (Archive.org)
10 volumes of classical and modern speeches
"""

import os
import time
import json
import requests
from pathlib import Path

AUDIO_DIR = Path("/home/melchior/speech3/audio/oratory")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# World's Famous Orations collections on Archive.org
LIBRIVOX_COLLECTIONS = [
    ("worlds_famous_orations1_1703_librivox", "Vol1_Greece"),
    ("worlds_famous_orations_2_1710_librivox", "Vol2_Rome"),
    ("worlds_famous_orations3_1804_librivox", "Vol3_GreatBritain1"),
    ("worlds_famous_orations_4_1901_librivox", "Vol4_GreatBritain2"),
    ("worldsfamousorations5_greatbritain3_1908_librivox", "Vol5_GreatBritain3"),
    ("worlds_famous_orations_7_continental_europe_2211_librivox", "Vol7_Europe"),
    ("worldsfamousorations8_2504_librivox", "Vol8_America1"),
    ("worldsfamousorationsvolx_2602_librivox", "Vol10_AmericaCanada"),
]

def get_collection_files(identifier):
    """Get list of MP3 files from an Archive.org collection."""
    url = f"https://archive.org/metadata/{identifier}"
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        mp3s = [f for f in data.get('files', []) 
                if f.get('name', '').endswith('.mp3') and '64kb' in f.get('name', '')]
        return mp3s
    except Exception as e:
        print(f"  Error fetching metadata: {e}")
        return []

def download_file(identifier, filename, prefix):
    """Download a single file."""
    url = f"https://archive.org/download/{identifier}/{filename}"
    # Create safe local filename
    safe_name = f"LibriVox_{prefix}_{filename}"
    local_path = AUDIO_DIR / safe_name
    
    if local_path.exists():
        print(f"    SKIP: {safe_name} (exists)")
        return True
    
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        local_path.write_bytes(response.content)
        print(f"    ✓ {safe_name} ({len(response.content)//1024} KB)")
        return True
    except Exception as e:
        print(f"    ✗ {filename}: {e}")
        return False

def main():
    print("=" * 60)
    print("LIBRIVOX WORLD'S FAMOUS ORATIONS DOWNLOADER")
    print("=" * 60)
    
    total_downloaded = 0
    
    for identifier, prefix in LIBRIVOX_COLLECTIONS:
        print(f"\n[{prefix}] {identifier}")
        files = get_collection_files(identifier)
        print(f"  Found {len(files)} MP3 files")
        
        for f in files:
            if download_file(identifier, f['name'], prefix):
                total_downloaded += 1
            time.sleep(0.2)
    
    print(f"\n✓ Total downloaded: {total_downloaded}")

if __name__ == "__main__":
    main()
