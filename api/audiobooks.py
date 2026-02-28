"""
Audiobook API endpoints
Serves audiobook content in Opus format for efficient delivery
"""

import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from .audio_utils import ensure_opus, transcode_to_opus

router = APIRouter(prefix="/v1/audiobooks", tags=["audiobooks"])

# Audiobook data directory
AUDIOBOOKS_DIR = Path("/home/melchior/speech3/audiobooks/lit2go")


@router.get("/")
async def list_audiobooks():
    """List all available audiobooks."""
    index_file = AUDIOBOOKS_DIR / "index.json"
    
    if not index_file.exists():
        return {"books": [], "count": 0}
    
    with open(index_file) as f:
        books = json.load(f)
    
    # Return summary (without full chapter lists)
    summaries = []
    for book in books:
        summaries.append({
            "title": book.get("title", "Unknown"),
            "slug": book.get("url", "").split("/")[-2] if book.get("url") else "",
            "chapter_count": len(book.get("chapters", [])),
            "source": book.get("source", "lit2go"),
        })
    
    return {"books": summaries, "count": len(summaries)}


@router.get("/{book_slug}")
async def get_audiobook(book_slug: str):
    """Get audiobook details with chapter list."""
    book_dir = AUDIOBOOKS_DIR / book_slug
    meta_file = book_dir / "metadata.json"
    
    if not meta_file.exists():
        raise HTTPException(404, f"Audiobook '{book_slug}' not found")
    
    with open(meta_file) as f:
        book = json.load(f)
    
    return book


@router.get("/{book_slug}/chapters/{chapter_index}/audio")
async def get_chapter_audio(
    book_slug: str,
    chapter_index: int,
    format: str = Query("opus", description="Audio format: opus (default) or original"),
):
    """
    Get audio file for a specific chapter.
    Returns Opus format by default for efficient streaming.
    """
    book_dir = AUDIOBOOKS_DIR / book_slug
    meta_file = book_dir / "metadata.json"
    
    if not meta_file.exists():
        raise HTTPException(404, f"Audiobook '{book_slug}' not found")
    
    with open(meta_file) as f:
        book = json.load(f)
    
    chapters = book.get("chapters", [])
    
    # Find chapter by index
    chapter = None
    for ch in chapters:
        if ch.get("index") == chapter_index:
            chapter = ch
            break
    
    if not chapter:
        raise HTTPException(404, f"Chapter {chapter_index} not found")
    
    audio_file = chapter.get("audio_file")
    if not audio_file:
        raise HTTPException(404, "No audio available for this chapter")
    
    # Build full path
    audio_path = book_dir / audio_file
    
    # Check for Opus version first
    opus_path = audio_path.with_suffix(".opus")
    if opus_path.exists():
        return FileResponse(
            opus_path,
            media_type="audio/opus",
            filename=opus_path.name,
        )
    
    if not audio_path.exists():
        raise HTTPException(404, "Audio file not found")
    
    # Transcode to Opus if requested
    if format == "opus":
        opus_file, content_type = ensure_opus(str(audio_path), for_speech=True)
        return FileResponse(
            opus_file,
            media_type=content_type,
            filename=Path(opus_file).name,
        )
    else:
        # Return original format
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename=audio_path.name,
        )


@router.get("/{book_slug}/chapters/{chapter_index}/text")
async def get_chapter_text(book_slug: str, chapter_index: int):
    """Get text content for a specific chapter."""
    book_dir = AUDIOBOOKS_DIR / book_slug
    meta_file = book_dir / "metadata.json"
    
    if not meta_file.exists():
        raise HTTPException(404, f"Audiobook '{book_slug}' not found")
    
    with open(meta_file) as f:
        book = json.load(f)
    
    chapters = book.get("chapters", [])
    
    # Find chapter by index
    chapter = None
    for ch in chapters:
        if ch.get("index") == chapter_index:
            chapter = ch
            break
    
    if not chapter:
        raise HTTPException(404, f"Chapter {chapter_index} not found")
    
    text_file = chapter.get("text_file")
    if not text_file:
        raise HTTPException(404, "No text available for this chapter")
    
    # Build full path (text_file is relative to AUDIOBOOKS_DIR)
    text_path = AUDIOBOOKS_DIR / text_file
    
    if not text_path.exists():
        raise HTTPException(404, "Text file not found")
    
    with open(text_path) as f:
        text = f.read()
    
    return {
        "title": chapter.get("title", ""),
        "text": text,
        "word_count": chapter.get("word_count", len(text.split())),
        "lexile": chapter.get("lexile"),
    }


@router.get("/search")
async def search_audiobooks(
    q: str = Query(..., description="Search query"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty: beginner, intermediate, advanced"),
):
    """Search audiobooks by title or content."""
    index_file = AUDIOBOOKS_DIR / "index.json"
    
    if not index_file.exists():
        return {"results": [], "count": 0}
    
    with open(index_file) as f:
        books = json.load(f)
    
    q_lower = q.lower()
    results = []
    
    for book in books:
        title = book.get("title", "").lower()
        if q_lower in title:
            results.append({
                "title": book.get("title"),
                "slug": book.get("url", "").split("/")[-2] if book.get("url") else "",
                "chapter_count": len(book.get("chapters", [])),
                "match_type": "title",
            })
    
    return {"results": results, "count": len(results), "query": q}
