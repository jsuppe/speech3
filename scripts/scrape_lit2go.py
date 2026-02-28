#!/usr/bin/env python3
"""
Scrape Lit2Go for children's audiobooks with text
https://etc.usf.edu/lit2go/

Downloads:
- Audio files (MP3)
- Text transcripts
- Lexile levels for difficulty grading
"""

import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin

BASE_URL = "https://etc.usf.edu/lit2go"
OUTPUT_DIR = Path("/home/melchior/speech3/audiobooks/lit2go")
KIDS_BOOKS = [
    # Pre-K to Grade 2 appropriate
    "/35/aesops-fables/",
    "/66/counting-and-math-rhymes/",
    "/165/buttercup-gold-and-other-stories/",
    "/141/the-blue-fairy-book/",
    "/139/the-brown-fairy-book/",
    "/59/a-childs-garden-of-verses-selected-poems/",
    "/124/the-birds-christmas-carol/",
    "/1/alices-adventures-in-wonderland/",
    "/125/black-beauty/",
    "/29/the-adventures-of-jerry-muskrat/",
]

def get_soup(url):
    """Fetch and parse a URL."""
    print(f"  Fetching: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    time.sleep(0.5)  # Be nice to the server
    return BeautifulSoup(resp.text, 'html.parser')

def get_book_chapters(book_url):
    """Get list of chapters/passages from a book page."""
    soup = get_soup(book_url)
    chapters = []
    
    # Find chapter links
    for link in soup.select('a[href*="/lit2go/"]'):
        href = link.get('href', '')
        # Chapter URLs have format: /lit2go/35/aesops-fables/101/the-ants-and-the-grasshopper/
        if re.search(r'/lit2go/\d+/[^/]+/\d+/', href):
            title = link.get_text(strip=True)
            if title and len(title) > 2:
                chapters.append({
                    'url': urljoin(BASE_URL, href),
                    'title': title
                })
    
    return chapters

def get_chapter_content(chapter_url):
    """Get audio URL, text, and metadata from a chapter page."""
    soup = get_soup(chapter_url)
    content = {
        'url': chapter_url,
        'title': '',
        'text': '',
        'audio_url': None,
        'lexile': None,
        'word_count': 0,
    }
    
    # Title
    title_el = soup.select_one('h1')
    if title_el:
        content['title'] = title_el.get_text(strip=True)
    
    # Audio URL - look for MP3 link
    for link in soup.select('a[href$=".mp3"]'):
        content['audio_url'] = link.get('href')
        break
    
    # Also check for audio player
    audio_el = soup.select_one('audio source[src$=".mp3"]')
    if audio_el and not content['audio_url']:
        content['audio_url'] = audio_el.get('src')
    
    # Text content - usually in a specific div
    text_div = soup.select_one('.passage-text, .chapter-text, article p')
    if text_div:
        content['text'] = text_div.get_text(separator='\n', strip=True)
    else:
        # Try getting all paragraphs
        paragraphs = soup.select('article p, .content p')
        content['text'] = '\n\n'.join(p.get_text(strip=True) for p in paragraphs)
    
    # Lexile level - look for it in metadata
    for text in soup.stripped_strings:
        if 'Lexile' in text:
            match = re.search(r'(\d+)L?', text)
            if match:
                content['lexile'] = int(match.group(1))
                break
    
    content['word_count'] = len(content['text'].split())
    
    return content

def download_audio(audio_url, output_path):
    """Download an audio file."""
    if not audio_url:
        return False
    
    try:
        print(f"  Downloading audio: {audio_url}")
        resp = requests.get(audio_url, timeout=60, stream=True)
        resp.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"  Error downloading audio: {e}")
        return False

def scrape_book(book_path):
    """Scrape a single book."""
    book_url = BASE_URL + book_path
    book_slug = book_path.strip('/').split('/')[-1]
    book_dir = OUTPUT_DIR / book_slug
    book_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📚 Scraping: {book_slug}")
    
    chapters = get_book_chapters(book_url)
    print(f"  Found {len(chapters)} chapters")
    
    book_data = {
        'title': book_slug.replace('-', ' ').title(),
        'source': 'lit2go',
        'url': book_url,
        'chapters': []
    }
    
    for i, chapter in enumerate(chapters[:50]):  # Limit to 50 chapters per book
        print(f"  [{i+1}/{len(chapters)}] {chapter['title'][:50]}...")
        
        try:
            content = get_chapter_content(chapter['url'])
            
            if content['text'] and len(content['text']) > 50:
                chapter_slug = re.sub(r'[^a-z0-9]+', '-', content['title'].lower())[:50]
                
                # Save text
                text_file = book_dir / f"{i:03d}_{chapter_slug}.txt"
                with open(text_file, 'w') as f:
                    f.write(content['text'])
                
                # Download audio if available
                audio_file = None
                if content['audio_url']:
                    audio_file = book_dir / f"{i:03d}_{chapter_slug}.mp3"
                    if not audio_file.exists():
                        download_audio(content['audio_url'], audio_file)
                
                book_data['chapters'].append({
                    'index': i,
                    'title': content['title'],
                    'text_file': str(text_file.relative_to(OUTPUT_DIR)),
                    'audio_file': str(audio_file.relative_to(OUTPUT_DIR)) if audio_file and audio_file.exists() else None,
                    'lexile': content['lexile'],
                    'word_count': content['word_count'],
                })
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save book metadata
    meta_file = book_dir / "metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(book_data, f, indent=2)
    
    print(f"  ✅ Saved {len(book_data['chapters'])} chapters to {book_dir}")
    return book_data

def main():
    """Scrape all kids books."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_books = []
    for book_path in KIDS_BOOKS:
        try:
            book_data = scrape_book(book_path)
            all_books.append(book_data)
        except Exception as e:
            print(f"Error scraping {book_path}: {e}")
    
    # Save master index
    index_file = OUTPUT_DIR / "index.json"
    with open(index_file, 'w') as f:
        json.dump(all_books, f, indent=2)
    
    print(f"\n✅ Done! Scraped {len(all_books)} books")
    print(f"   Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
