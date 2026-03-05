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

# Audiobook data directories by language
AUDIOBOOKS_BASE = Path("/home/melchior/speech3/audiobooks")
LANGUAGE_DIRS = {
    "en": AUDIOBOOKS_BASE / "lit2go",
    "es": AUDIOBOOKS_BASE / "spanish",
}

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    # Future: "fr": "French", "de": "German", "pt": "Portuguese"
}


def get_audiobooks_dir(language: str = "en") -> Path:
    """Get audiobooks directory for language."""
    return LANGUAGE_DIRS.get(language, LANGUAGE_DIRS["en"])


@router.get("/languages")
async def list_languages():
    """List supported languages for audiobooks."""
    languages = []
    for code, name in SUPPORTED_LANGUAGES.items():
        dir_path = LANGUAGE_DIRS.get(code)
        has_content = dir_path.exists() and any(dir_path.iterdir()) if dir_path else False
        languages.append({
            "code": code,
            "name": name,
            "has_content": has_content,
        })
    return {"languages": languages}


@router.get("/")
async def list_audiobooks(language: str = Query("en", description="Language code: en, es")):
    """List all available audiobooks for a language."""
    audiobooks_dir = get_audiobooks_dir(language)
    index_file = audiobooks_dir / "index.json"
    
    # For Spanish, generate index from directory structure if no index.json
    if not index_file.exists() and language == "es":
        return await list_spanish_audiobooks()
    
    if not index_file.exists():
        return {"books": [], "count": 0, "language": language}
    
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
            "language": language,
        })
    
    return {"books": summaries, "count": len(summaries), "language": language}


async def list_spanish_audiobooks():
    """List Spanish audiobooks from directory structure."""
    spanish_dir = LANGUAGE_DIRS["es"]
    if not spanish_dir.exists():
        return {"books": [], "count": 0, "language": "es"}
    
    books = []
    for book_dir in sorted(spanish_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        
        # Count audio files
        audio_files = list(book_dir.glob("*.mp3")) + list(book_dir.glob("*.opus"))
        if not audio_files:
            continue
        
        books.append({
            "title": book_dir.name.replace("_", " "),
            "slug": book_dir.name,
            "chapter_count": len(audio_files),
            "source": "librivox",
            "language": "es",
        })
    
    return {"books": books, "count": len(books), "language": "es"}


@router.get("/{book_slug}")
async def get_audiobook(
    book_slug: str,
    language: str = Query("en", description="Language code: en, es"),
):
    """Get audiobook details with chapter list."""
    audiobooks_dir = get_audiobooks_dir(language)
    book_dir = audiobooks_dir / book_slug
    meta_file = book_dir / "metadata.json"
    
    # For Spanish, generate metadata from directory
    if not meta_file.exists() and language == "es":
        return await get_spanish_audiobook(book_slug)
    
    if not meta_file.exists():
        raise HTTPException(404, f"Audiobook '{book_slug}' not found")
    
    with open(meta_file) as f:
        book = json.load(f)
    
    book["language"] = language
    return book


async def get_spanish_audiobook(book_slug: str):
    """Get Spanish audiobook from directory structure."""
    book_dir = LANGUAGE_DIRS["es"] / book_slug
    
    if not book_dir.exists():
        raise HTTPException(404, f"Audiobook '{book_slug}' not found")
    
    # Find all audio files
    audio_files = sorted(list(book_dir.glob("*.opus")) + list(book_dir.glob("*.mp3")))
    
    chapters = []
    for i, audio_file in enumerate(audio_files):
        title = audio_file.stem.split("_", 1)[-1] if "_" in audio_file.stem else audio_file.stem
        chapters.append({
            "index": i + 1,
            "title": title.replace("_", " "),
            "audio_file": audio_file.name,
            "has_audio": True,
        })
    
    return {
        "title": book_slug.replace("_", " "),
        "slug": book_slug,
        "language": "es",
        "source": "librivox",
        "chapters": chapters,
        "chapter_count": len(chapters),
    }


async def get_spanish_chapter_audio(book_slug: str, chapter_index: int, format: str = "opus"):
    """Get Spanish chapter audio from directory."""
    book_dir = LANGUAGE_DIRS["es"] / book_slug
    
    if not book_dir.exists():
        raise HTTPException(404, f"Audiobook '{book_slug}' not found")
    
    # Find all audio files sorted
    audio_files = sorted(list(book_dir.glob("*.opus")) + list(book_dir.glob("*.mp3")))
    
    if chapter_index < 1 or chapter_index > len(audio_files):
        raise HTTPException(404, f"Chapter {chapter_index} not found")
    
    audio_path = audio_files[chapter_index - 1]
    
    # Prefer opus format
    opus_path = audio_path.with_suffix(".opus")
    if opus_path.exists():
        return FileResponse(
            opus_path,
            media_type="audio/opus",
            filename=opus_path.name,
        )
    
    if format == "opus" and audio_path.suffix == ".mp3":
        # Transcode on the fly
        opus_file, content_type = ensure_opus(str(audio_path), for_speech=True)
        return FileResponse(
            opus_file,
            media_type=content_type,
            filename=Path(opus_file).name,
        )
    
    return FileResponse(
        audio_path,
        media_type="audio/mpeg" if audio_path.suffix == ".mp3" else "audio/opus",
        filename=audio_path.name,
    )


@router.get("/{book_slug}/chapters/{chapter_index}/audio")
async def get_chapter_audio(
    book_slug: str,
    chapter_index: int,
    language: str = Query("en", description="Language code: en, es"),
    format: str = Query("opus", description="Audio format: opus (default) or original"),
):
    """
    Get audio file for a specific chapter.
    Returns Opus format by default for efficient streaming.
    """
    audiobooks_dir = get_audiobooks_dir(language)
    book_dir = audiobooks_dir / book_slug
    
    # For Spanish, use directory-based lookup
    if language == "es":
        return await get_spanish_chapter_audio(book_slug, chapter_index, format)
    
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
