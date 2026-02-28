"""
TTS API endpoints for SpeakFit
- Free tier: Returns error (app uses device TTS)
- Pro tier: ElevenLabs high-quality voices
"""

import sqlite3
import logging
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from typing import Optional

from .tts_elevenlabs import text_to_speech, VOICES
from .user_auth import get_current_user
from . import config

logger = logging.getLogger("speechscore.tts")

router = APIRouter(prefix="/v1/tts", tags=["tts"])


def get_user_tier(user_id: int) -> str:
    """Get user's subscription tier."""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT tier FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        conn.close()
        tier = row[0] if row else "free"
        logger.info(f"get_user_tier({user_id}) = {tier}")
        return tier
    except Exception as e:
        logger.error(f"get_user_tier({user_id}) error: {e}")
        return "free"


def get_user_voice_preference(user_id: int) -> str:
    """Get user's preferred TTS voice."""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT tts_voice FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row and row[0] else "elli"
    except Exception:
        return "elli"


def set_user_voice_preference(user_id: int, voice: str) -> bool:
    """Set user's preferred TTS voice."""
    if voice.lower() not in VOICES:
        return False
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE users SET tts_voice = ? WHERE id = ?", (voice.lower(), user_id))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


@router.get("/speak")
async def speak(
    text: str = Query(..., description="Text to convert to speech", max_length=500),
    voice: Optional[str] = Query(None, description="Voice name (uses preference if not specified)"),
    user = Depends(get_current_user),
):
    """
    Convert text to speech using ElevenLabs (Pro tier only).
    Free tier users should use device TTS.
    
    Returns Opus audio file.
    """
    logger.info(f"TTS speak request: text_len={len(text)}, voice={voice}, user={user}")
    if not user:
        logger.warning("TTS speak: No user in request")
        raise HTTPException(401, "Authentication required")
    
    tier = get_user_tier(user["id"])
    # Accept pro, premium, or paid as valid paid tiers
    if tier not in ("pro", "premium", "paid"):
        raise HTTPException(
            403, 
            detail={
                "error": "pro_required",
                "message": "Premium TTS requires Pro subscription",
                "tier": tier,
            }
        )
    
    if len(text) > 500:
        raise HTTPException(400, "Text too long (max 500 chars)")
    
    # Use specified voice or user's preference
    use_voice = voice or get_user_voice_preference(user["id"])
    
    audio_path = await text_to_speech(text, voice=use_voice)
    
    if not audio_path:
        raise HTTPException(500, "TTS generation failed")
    
    return FileResponse(
        audio_path,
        media_type="audio/opus",
        filename="speech.opus",
    )


@router.get("/voices")
async def list_voices(user = Depends(get_current_user)):
    """List available TTS voices."""
    user_voice = "elli"
    tier = "free"
    
    if user:
        user_voice = get_user_voice_preference(user["id"])
        tier = get_user_tier(user["id"])
    
    return {
        "voices": [
            {"id": "rachel", "name": "Rachel", "description": "Calm, warm - great for stories"},
            {"id": "bella", "name": "Bella", "description": "Soft, gentle"},
            {"id": "elli", "name": "Elli", "description": "Young, friendly"},
            {"id": "charlotte", "name": "Charlotte", "description": "Warm, engaging"},
            {"id": "adam", "name": "Adam", "description": "Deep, clear"},
            {"id": "josh", "name": "Josh", "description": "Young, energetic"},
            {"id": "sam", "name": "Sam", "description": "Warm, friendly"},
        ],
        "recommended_for_kids": ["rachel", "bella", "elli", "charlotte"],
        "default": "elli",
        "user_preference": user_voice,
        "tier": tier,
        "pro_required": True,
    }


@router.post("/voice")
async def set_voice(
    voice: str = Query(..., description="Voice ID to set as preference"),
    user = Depends(get_current_user),
):
    """Set user's preferred TTS voice."""
    if not user:
        raise HTTPException(401, "Authentication required")
    
    if voice.lower() not in VOICES:
        raise HTTPException(400, f"Invalid voice. Available: {list(VOICES.keys())}")
    
    if set_user_voice_preference(user["id"], voice):
        return {"success": True, "voice": voice.lower()}
    else:
        raise HTTPException(500, "Failed to save preference")


@router.post("/sentence")
async def speak_sentence(
    text: str,
    voice: Optional[str] = None,
    user = Depends(get_current_user),
):
    """
    Generate TTS for a sentence (for Listen & Read mode).
    Pro tier only - returns audio URL.
    """
    if not user:
        raise HTTPException(401, "Authentication required")
    
    tier = get_user_tier(user["id"])
    # Accept pro, premium, or paid as valid paid tiers
    if tier not in ("pro", "premium", "paid"):
        raise HTTPException(
            403,
            detail={
                "error": "pro_required",
                "message": "Premium TTS requires Pro subscription",
            }
        )
    
    if len(text) > 1000:
        raise HTTPException(400, "Text too long (max 1000 chars)")
    
    use_voice = voice or get_user_voice_preference(user["id"])
    audio_path = await text_to_speech(text, voice=use_voice)
    
    if not audio_path:
        raise HTTPException(500, "TTS generation failed")
    
    return {
        "audio_url": f"/v1/tts/audio/{audio_path.name}",
        "text": text,
        "voice": use_voice,
        "format": "opus",
    }


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve cached TTS audio file."""
    from pathlib import Path
    
    cache_dir = Path("/home/melchior/speech3/tts_cache")
    audio_path = cache_dir / filename
    
    if not audio_path.exists() or not audio_path.is_relative_to(cache_dir):
        raise HTTPException(404, "Audio not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/opus",
        filename=filename,
    )


@router.get("/debug")
async def tts_debug(user = Depends(get_current_user)):
    """Debug endpoint to check TTS auth status."""
    if not user:
        return {
            "authenticated": False,
            "user": None,
            "tier": None,
            "message": "No user found in request"
        }
    
    tier = get_user_tier(user["id"])
    voice = get_user_voice_preference(user["id"])
    
    return {
        "authenticated": True,
        "user_id": user["id"],
        "email": user.get("email"),
        "tier": tier,
        "voice_preference": voice,
        "tts_enabled": tier == "pro",
    }


@router.get("/status")
async def tts_status(user = Depends(get_current_user)):
    """Check TTS availability for current user."""
    if not user:
        return {
            "available": False,
            "reason": "not_authenticated",
            "tier": None,
        }
    
    tier = get_user_tier(user["id"])
    
    return {
        "available": tier == "pro",
        "reason": "pro_subscription" if tier == "pro" else "free_tier",
        "tier": tier,
        "fallback": "device_tts",
    }
