"""
Kids Mode API endpoints.

POST /v1/kids/word-help - Get contextual pronunciation help for a word
GET /v1/kids/books - List available children's books
"""

import json
import httpx
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/v1/kids", tags=["kids"])

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# Cache for common word tips (reduces latency)
_word_tip_cache: dict[str, str] = {}


class WordHelpRequest(BaseModel):
    word: str
    sentence: str
    book_title: Optional[str] = None
    book_context: Optional[str] = None  # Brief description of what's happening
    child_age: int = 6


class WordHelpResponse(BaseModel):
    word: str
    tip: str
    phonetic_hint: Optional[str] = None


class BookInfo(BaseModel):
    id: str
    title: str
    author: str
    reading_level: str  # "pre-k", "kindergarten", "grade-1", "grade-2", "grade-3"
    lexile: Optional[int] = None  # Lexile score if known
    theme: str  # "animals", "adventure", "fairy-tale", "everyday", "nature"
    page_count: int
    description: str
    character_helper: Optional[str] = None  # Name of helper character for 3D mode


# Book library metadata (content is in the app)
BOOKS_METADATA = [
    BookInfo(
        id="simple_sentences",
        title="My First Words",
        author="SpeakFit",
        reading_level="pre-k",
        lexile=100,
        theme="everyday",
        page_count=8,
        description="Simple sentences for beginning readers.",
        character_helper="Sunny the Star",
    ),
    BookInfo(
        id="nursery_rhymes",
        title="Nursery Rhymes",
        author="Traditional",
        reading_level="pre-k",
        lexile=200,
        theme="songs",
        page_count=6,
        description="Classic nursery rhymes to read and sing along.",
        character_helper="Melody Mouse",
    ),
    BookInfo(
        id="tortoise_hare",
        title="The Tortoise and the Hare",
        author="Aesop",
        reading_level="kindergarten",
        lexile=400,
        theme="animals",
        page_count=10,
        description="A slow tortoise races a speedy hare. Who will win?",
        character_helper="Terry Tortoise",
    ),
    BookInfo(
        id="lion_mouse",
        title="The Lion and the Mouse",
        author="Aesop",
        reading_level="kindergarten",
        lexile=420,
        theme="animals",
        page_count=10,
        description="A tiny mouse helps a mighty lion. Kindness matters!",
        character_helper="Leo Lion",
    ),
    BookInfo(
        id="adventure_forest",
        title="Adventure in the Forest",
        author="SpeakFit",
        reading_level="grade-1",
        lexile=500,
        theme="adventure",
        page_count=10,
        description="Emma explores the magical forest behind her house.",
        character_helper="Emma Explorer",
    ),
    # New books to add
    BookInfo(
        id="three_little_pigs",
        title="The Three Little Pigs",
        author="Traditional",
        reading_level="kindergarten",
        lexile=380,
        theme="fairy-tale",
        page_count=12,
        description="Three pigs build houses. Will they stop the big bad wolf?",
        character_helper="Penny Pig",
    ),
    BookInfo(
        id="goldilocks",
        title="Goldilocks and the Three Bears",
        author="Traditional",
        reading_level="kindergarten",
        lexile=410,
        theme="fairy-tale",
        page_count=10,
        description="A curious girl finds a house in the woods.",
        character_helper="Baby Bear",
    ),
    BookInfo(
        id="little_red_hen",
        title="The Little Red Hen",
        author="Traditional",
        reading_level="grade-1",
        lexile=470,
        theme="animals",
        page_count=10,
        description="A hen learns that hard work pays off.",
        character_helper="Rosie Hen",
    ),
    BookInfo(
        id="ugly_duckling",
        title="The Ugly Duckling",
        author="Hans Christian Andersen",
        reading_level="grade-1",
        lexile=520,
        theme="animals",
        page_count=12,
        description="A little bird discovers who he really is.",
        character_helper="Swan",
    ),
    BookInfo(
        id="town_mouse",
        title="The Town Mouse and Country Mouse",
        author="Aesop",
        reading_level="grade-2",
        lexile=580,
        theme="animals",
        page_count=10,
        description="Two mice learn that home is where you're happy.",
        character_helper="Country Mouse",
    ),
]


@router.get("/books", response_model=list[BookInfo])
async def list_books(
    reading_level: Optional[str] = None,
    theme: Optional[str] = None,
):
    """List available children's books with optional filters."""
    books = BOOKS_METADATA
    
    if reading_level:
        books = [b for b in books if b.reading_level == reading_level]
    
    if theme:
        books = [b for b in books if b.theme == theme]
    
    return books


@router.post("/word-help", response_model=WordHelpResponse)
async def get_word_help(request: WordHelpRequest):
    """
    Get contextual pronunciation help for a word.
    Uses Ollama to generate kid-friendly tips based on context.
    """
    word = request.word.lower().strip()
    
    # Check cache first
    cache_key = f"{word}:{request.book_title or 'general'}"
    if cache_key in _word_tip_cache:
        return WordHelpResponse(
            word=request.word,
            tip=_word_tip_cache[cache_key],
        )
    
    # Build prompt for Ollama
    context_parts = []
    if request.book_title:
        context_parts.append(f"reading '{request.book_title}'")
    if request.book_context:
        context_parts.append(f"({request.book_context})")
    
    context = " ".join(context_parts) if context_parts else "practicing reading"
    
    prompt = f"""Help a {request.child_age}-year-old say the word "{request.word}" from: "{request.sentence}"

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
"Say it like: [phonetic]. [One fun memory tip]!"

PHONETIC RULES (text-to-speech friendly):
- kuh (k sound), guh (g), tuh (t), duh (d), puh (p), buh (b)
- sss (s), zzz (z), fff (f), vvv (v), shh (sh), thh (th)  
- mmm (m), nnn (n), lll (l), rrr (r)
- ah (short a), eh (short e), ih (short i), oh (short o), uh (short u)
- ay (long a), ee (long e), eye (long i), oh (long o), oo (long u)
- ow (as in "cow"), oy (as in "boy")

EXAMPLES:
- "cat" → "Say it like: kuh-ah-tuh. Think of a cat saying 'meow'!"
- "house" → "Say it like: hah-ow-sss. A house keeps you safe and sound!"
- "slowly" → "Say it like: sss-loh-lee. Like a slow sleepy sloth!"

Now give the tip for "{request.word}":"""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 60,  # Keep responses short
                    }
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail="AI service unavailable")
            
            result = response.json()
            tip = result.get("response", "").strip()
            
            # Clean up the tip
            tip = tip.replace('"', '').replace("'", "'")
            if not tip:
                tip = f"Let's break it down: '{request.word}'. Say it slowly with me!"
            
            # Cache for future use
            _word_tip_cache[cache_key] = tip
            
            return WordHelpResponse(
                word=request.word,
                tip=tip,
            )
            
    except httpx.TimeoutException:
        # Fallback for timeout
        return WordHelpResponse(
            word=request.word,
            tip=f"Let's try '{request.word}' together! Say each part slowly.",
        )
    except Exception as e:
        # Generic fallback
        return WordHelpResponse(
            word=request.word,
            tip=f"Sound it out: '{request.word}'. You've got this!",
        )


@router.delete("/cache")
async def clear_cache():
    """Clear the word tip cache (admin only)."""
    _word_tip_cache.clear()
    return {"status": "ok", "message": "Cache cleared"}
