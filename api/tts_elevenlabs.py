"""
ElevenLabs TTS integration for SpeakFit
High-quality text-to-speech for kids reading app
"""

import json
import hashlib
from pathlib import Path
from typing import Optional
import httpx

# Config
CONFIG_PATH = Path("/home/melchior/speech3/config/elevenlabs.json")
CACHE_DIR = Path("/home/melchior/speech3/tts_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Load config
def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}

# ElevenLabs voice IDs (pre-defined voices)
VOICES = {
    # Female voices
    "rachel": "21m00Tcm4TlvDq8ikWAM",  # Calm, warm - good for stories
    "domi": "AZnzlk1XvdvUeBnXmlld",    # Strong, confident
    "bella": "EXAVITQu4vr4xnSDxMaL",   # Soft, gentle - great for kids
    "elli": "MF3mGyEYCl7XYWbV9V6O",    # Young, friendly
    "charlotte": "XB0fDUnXU5powFXDhCwa", # Warm, engaging
    
    # Male voices
    "adam": "pNInz6obpgDQGcFmaJgB",    # Deep, clear
    "arnold": "VR6AewLTigWG4xSOukaG",  # Strong narrator
    "josh": "TxGEqnHWrfWFTfGW9XjX",    # Young, energetic
    "sam": "yoZ06aMxZJJ28mfd3POQ",     # Warm, friendly
    
    # Kids-friendly recommended
    "default": "21m00Tcm4TlvDq8ikWAM",  # Rachel - best for kids stories
}


def get_cache_path(text: str, voice: str) -> Path:
    """Generate cache path for TTS audio."""
    key = f"{voice}:{text}"
    hash_val = hashlib.md5(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"tts_{hash_val}.opus"


async def text_to_speech(
    text: str,
    voice: str = "rachel",
    model: str = "eleven_turbo_v2_5",  # Latest model, works on free tier
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    use_cache: bool = True,
) -> Optional[Path]:
    """
    Convert text to speech using ElevenLabs API.
    
    Args:
        text: Text to convert
        voice: Voice name or ID
        model: ElevenLabs model
        stability: Voice stability (0-1)
        similarity_boost: Similarity boost (0-1)
        use_cache: Whether to use cached audio
    
    Returns:
        Path to audio file (Opus format), or None if failed
    """
    config = load_config()
    api_key = config.get("api_key")
    
    if not api_key:
        print("ElevenLabs API key not configured")
        return None
    
    # Get voice ID
    voice_id = VOICES.get(voice.lower(), voice)
    
    # Check cache
    cache_path = get_cache_path(text, voice_id)
    if use_cache and cache_path.exists():
        return cache_path
    
    # API request
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }
    
    data = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
        }
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json=data,
                timeout=30.0,
            )
            
            if response.status_code == 200:
                # Save as MP3 first
                mp3_path = cache_path.with_suffix(".mp3")
                with open(mp3_path, "wb") as f:
                    f.write(response.content)
                
                # Convert to Opus for smaller size
                import subprocess
                result = subprocess.run([
                    "ffmpeg", "-i", str(mp3_path),
                    "-c:a", "libopus", "-b:a", "24k",
                    "-vbr", "on", "-application", "voip",
                    "-y", str(cache_path)
                ], capture_output=True)
                
                if result.returncode == 0:
                    mp3_path.unlink()  # Remove MP3
                    return cache_path
                else:
                    # Fallback to MP3
                    return mp3_path
            else:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"TTS error: {e}")
        return None


def text_to_speech_sync(
    text: str,
    voice: str = "rachel",
    **kwargs
) -> Optional[Path]:
    """Synchronous wrapper for text_to_speech."""
    import asyncio
    return asyncio.run(text_to_speech(text, voice, **kwargs))


async def get_voices() -> list:
    """Get list of available voices from ElevenLabs."""
    config = load_config()
    api_key = config.get("api_key")
    
    if not api_key:
        return list(VOICES.keys())
    
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": api_key}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return [v["name"] for v in data.get("voices", [])]
    except Exception as e:
        print(f"Error fetching voices: {e}")
    
    return list(VOICES.keys())


async def get_usage() -> dict:
    """Get current usage stats from ElevenLabs."""
    config = load_config()
    api_key = config.get("api_key")
    
    if not api_key:
        return {"error": "No API key configured"}
    
    url = "https://api.elevenlabs.io/v1/user/subscription"
    headers = {"xi-api-key": api_key}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return {
                    "character_count": data.get("character_count", 0),
                    "character_limit": data.get("character_limit", 0),
                    "remaining": data.get("character_limit", 0) - data.get("character_count", 0),
                    "tier": data.get("tier", "free"),
                }
    except Exception as e:
        print(f"Error fetching usage: {e}")
    
    return {"error": "Failed to fetch usage"}


# Quick test function
async def test_tts():
    """Test TTS with a simple phrase."""
    result = await text_to_speech("Hello! Welcome to SpeakFit. Let's learn to read together!")
    if result:
        print(f"✅ TTS working! Audio saved to: {result}")
        return True
    else:
        print("❌ TTS failed")
        return False


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_tts())
