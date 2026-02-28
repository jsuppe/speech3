#!/usr/bin/env python3
"""
Download LibriVox children's audiobooks
https://librivox.org/

Uses LibriVox API to find and download kid-friendly audiobooks.
"""

import os
import json
import time
import requests
from pathlib import Path

BASE_URL = "https://librivox.org/api/feed/audiobooks"
OUTPUT_DIR = Path("/home/melchior/speech3/audiobooks/librivox")

# Kid-friendly search terms
KIDS_SEARCHES = [
    "aesop fables",
    "grimm fairy tales", 
    "andersen fairy tales",
    "peter rabbit",
    "wizard of oz",
    "alice wonderland",
    "jungle book",
    "wind in the willows",
    "just so stories",
    "nursery rhymes",
    "mother goose",
    "pinocchio",
    "black beauty",
    "heidi",
    "little princess",
    "secret garden",
]

def search_librivox(query, limit=5):
    """Search LibriVox API for audiobooks."""
    params = {
        'title': query,
        'format': 'json',
        'limit': limit,
    }
    
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get('books', [])
    except Exception as e:
        print(f"  Error searching '{query}': {e}")
        return []

def get_book_details(book_id):
    """Get detailed book info including chapter URLs."""
    url = f"{BASE_URL}/?id={book_id}&format=json"
    
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        books = data.get('books', [])
        return books[0] if books else None
    except Exception as e:
        print(f"  Error getting book {book_id}: {e}")
        return None

def download_chapter(url, output_path):
    """Download a single chapter MP3."""
    try:
        print(f"    Downloading: {output_path.name}")
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False

def process_book(book):
    """Download a book and create metadata."""
    book_id = book.get('id')
    title = book.get('title', 'Unknown')
    url_librivox = book.get('url_librivox', '')
    
    # Create slug from title
    slug = ''.join(c if c.isalnum() else '-' for c in title.lower())[:50]
    book_dir = OUTPUT_DIR / slug
    
    # Skip if already downloaded
    meta_file = book_dir / "metadata.json"
    if meta_file.exists():
        print(f"  ⏭️  Skipping (exists): {title}")
        return None
    
    print(f"\n📖 Processing: {title}")
    book_dir.mkdir(parents=True, exist_ok=True)
    
    # Get RSS feed URL for chapter list
    rss_url = book.get('url_rss', '')
    
    book_data = {
        'id': book_id,
        'title': title,
        'author': book.get('authors', [{}])[0].get('last_name', 'Unknown'),
        'description': book.get('description', ''),
        'language': book.get('language', 'English'),
        'url_librivox': url_librivox,
        'url_text': book.get('url_text_source', ''),  # Gutenberg link
        'total_time': book.get('totaltime', ''),
        'chapters': []
    }
    
    # Try to get chapters from RSS or sections
    sections = book.get('sections', [])
    
    for i, section in enumerate(sections[:30]):  # Limit chapters
        section_title = section.get('title', f'Chapter {i+1}')
        mp3_url = section.get('listen_url', '')
        
        if mp3_url:
            chapter_file = book_dir / f"{i:03d}_{section_title[:30].replace(' ', '_')}.mp3"
            
            if not chapter_file.exists():
                time.sleep(0.5)  # Rate limit
                download_chapter(mp3_url, chapter_file)
            
            book_data['chapters'].append({
                'index': i,
                'title': section_title,
                'audio_file': chapter_file.name if chapter_file.exists() else None,
                'duration': section.get('playtime', ''),
            })
    
    # Save metadata
    with open(meta_file, 'w') as f:
        json.dump(book_data, f, indent=2)
    
    print(f"  ✅ Downloaded {len(book_data['chapters'])} chapters")
    return book_data

def main():
    """Download LibriVox kids audiobooks."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_books = []
    seen_ids = set()
    
    for search in KIDS_SEARCHES:
        print(f"\n🔍 Searching: {search}")
        books = search_librivox(search, limit=3)
        
        for book in books:
            book_id = book.get('id')
            if book_id and book_id not in seen_ids:
                seen_ids.add(book_id)
                
                # Get full details with sections
                details = get_book_details(book_id)
                if details:
                    result = process_book(details)
                    if result:
                        all_books.append(result)
                
                time.sleep(1)  # Rate limit
    
    # Save master index
    index_file = OUTPUT_DIR / "index.json"
    with open(index_file, 'w') as f:
        json.dump(all_books, f, indent=2)
    
    print(f"\n✅ Done! Downloaded {len(all_books)} books")
    print(f"   Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
