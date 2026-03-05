#!/usr/bin/env python3
"""Download Spanish audiobooks from LibriVox for Reader mode."""

import os
import json
import requests
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

OUTPUT_DIR = Path("/home/melchior/speech3/audiobooks/spanish")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Spanish audiobooks on LibriVox
SPANISH_BOOKS = [
    {"id": "73", "title": "Fábulas de Esopo Vol 1"},
    {"id": "89", "title": "Fábulas de Esopo Vol 2"},
    {"id": "121", "title": "Fábulas de Esopo Vol 3"},
    {"id": "474", "title": "Spanish Poetry Collection"},
    {"id": "567", "title": "Don Quijote 1"},
]

def get_rss_items(book_id):
    """Get audio items from RSS feed."""
    rss_url = f"https://librivox.org/rss/{book_id}"
    print(f"Fetching RSS: {rss_url}")
    
    try:
        resp = requests.get(rss_url, timeout=30)
        resp.raise_for_status()
        
        # Parse RSS XML
        root = ET.fromstring(resp.content)
        items = []
        
        for item in root.findall('.//item'):
            title = item.find('title')
            enclosure = item.find('enclosure')
            
            if title is not None and enclosure is not None:
                items.append({
                    'title': title.text,
                    'url': enclosure.get('url'),
                })
        
        return items
    except Exception as e:
        print(f"Error fetching RSS {book_id}: {e}")
        return []

def download_audio(url, output_path):
    """Download audio file."""
    if output_path.exists():
        print(f"  Already exists: {output_path.name}")
        return True
    
    try:
        print(f"  Downloading: {output_path.name}")
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False

def convert_to_opus(mp3_path, opus_path):
    """Convert MP3 to Opus for smaller size."""
    if opus_path.exists():
        return True
    
    try:
        subprocess.run([
            'ffmpeg', '-i', str(mp3_path),
            '-c:a', 'libopus', '-b:a', '32k',
            '-y', str(opus_path)
        ], capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"  Error converting: {e}")
        return False

def main():
    total_downloaded = 0
    
    for book in SPANISH_BOOKS:
        book_id = book['id']
        book_title = book['title']
        
        print(f"\n{'='*60}")
        print(f"Processing: {book_title}")
        print(f"{'='*60}")
        
        # Create book directory
        book_dir = OUTPUT_DIR / book_title.replace(' ', '_').replace('/', '_')
        book_dir.mkdir(exist_ok=True)
        
        # Get RSS items
        items = get_rss_items(book_id)
        print(f"Found {len(items)} chapters")
        
        # Download each chapter (limit to 10 per book for now)
        for i, item in enumerate(items[:10]):
            title = item['title']
            url = item['url']
            
            # Clean filename
            safe_title = "".join(c if c.isalnum() or c in ' -_' else '_' for c in title)
            mp3_path = book_dir / f"{i+1:02d}_{safe_title[:50]}.mp3"
            opus_path = book_dir / f"{i+1:02d}_{safe_title[:50]}.opus"
            
            if download_audio(url, mp3_path):
                total_downloaded += 1
                # Convert to Opus
                convert_to_opus(mp3_path, opus_path)
    
    print(f"\n{'='*60}")
    print(f"Downloaded {total_downloaded} Spanish audio files")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
