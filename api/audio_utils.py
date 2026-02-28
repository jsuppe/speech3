"""
Audio utilities for SpeakFit API
- Transcoding to Opus for efficient delivery
- Audio format detection and conversion
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import hashlib

# Cache directory for transcoded audio
AUDIO_CACHE_DIR = Path("/home/melchior/speech3/audio_cache")
AUDIO_CACHE_DIR.mkdir(exist_ok=True)

# Default Opus settings for speech
OPUS_SPEECH_BITRATE = "24k"
OPUS_MUSIC_BITRATE = "64k"


def get_cache_path(source_path: str, target_format: str = "opus") -> Path:
    """Generate a cache path for transcoded audio."""
    # Hash the source path for cache key
    path_hash = hashlib.md5(source_path.encode()).hexdigest()[:12]
    source_name = Path(source_path).stem[:20]
    return AUDIO_CACHE_DIR / f"{source_name}_{path_hash}.{target_format}"


def transcode_to_opus(
    source_path: str,
    target_path: Optional[str] = None,
    bitrate: str = OPUS_SPEECH_BITRATE,
    application: str = "voip",  # voip=speech, audio=music
) -> Optional[str]:
    """
    Transcode audio file to Opus format.
    
    Args:
        source_path: Path to source audio file
        target_path: Optional output path (uses cache if not provided)
        bitrate: Target bitrate (e.g., "24k", "64k")
        application: Opus application type ("voip" for speech, "audio" for music)
    
    Returns:
        Path to transcoded Opus file, or None if failed
    """
    source = Path(source_path)
    if not source.exists():
        return None
    
    # Already Opus?
    if source.suffix.lower() == ".opus":
        return str(source)
    
    # Use cache path if no target specified
    if target_path is None:
        target = get_cache_path(source_path, "opus")
    else:
        target = Path(target_path)
    
    # Return cached version if exists
    if target.exists():
        return str(target)
    
    # Transcode with ffmpeg
    try:
        cmd = [
            "ffmpeg",
            "-i", str(source),
            "-c:a", "libopus",
            "-b:a", bitrate,
            "-vbr", "on",
            "-application", application,
            "-y",  # Overwrite
            "-loglevel", "error",
            str(target)
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        
        if result.returncode == 0 and target.exists():
            return str(target)
        else:
            print(f"Transcode failed: {result.stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"Transcode error: {e}")
        return None


def get_audio_format(file_path: str) -> Optional[str]:
    """Detect audio format from file."""
    ext = Path(file_path).suffix.lower()
    format_map = {
        ".mp3": "audio/mpeg",
        ".opus": "audio/opus",
        ".ogg": "audio/ogg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".webm": "audio/webm",
    }
    return format_map.get(ext)


def ensure_opus(audio_path: str, for_speech: bool = True) -> tuple[str, str]:
    """
    Ensure audio is in Opus format, transcoding if necessary.
    
    Args:
        audio_path: Path to audio file
        for_speech: If True, uses speech-optimized settings
    
    Returns:
        Tuple of (file_path, content_type)
    """
    path = Path(audio_path)
    
    # Already Opus
    if path.suffix.lower() == ".opus":
        return str(path), "audio/opus"
    
    # Transcode to Opus
    bitrate = OPUS_SPEECH_BITRATE if for_speech else OPUS_MUSIC_BITRATE
    application = "voip" if for_speech else "audio"
    
    opus_path = transcode_to_opus(
        str(path),
        bitrate=bitrate,
        application=application
    )
    
    if opus_path:
        return opus_path, "audio/opus"
    else:
        # Fallback to original
        content_type = get_audio_format(str(path)) or "audio/mpeg"
        return str(path), content_type


def clear_cache(max_age_days: int = 7) -> int:
    """Clear old cached audio files."""
    import time
    
    cleared = 0
    now = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    for f in AUDIO_CACHE_DIR.glob("*"):
        if f.is_file():
            age = now - f.stat().st_mtime
            if age > max_age_seconds:
                f.unlink()
                cleared += 1
    
    return cleared


# Batch transcode utility
def batch_transcode_directory(
    source_dir: str,
    target_dir: Optional[str] = None,
    source_formats: tuple = (".mp3", ".wav", ".m4a"),
    delete_originals: bool = False,
) -> dict:
    """
    Batch transcode all audio files in a directory to Opus.
    
    Returns dict with counts: {"converted": N, "skipped": N, "failed": N}
    """
    source = Path(source_dir)
    target = Path(target_dir) if target_dir else source
    
    results = {"converted": 0, "skipped": 0, "failed": 0}
    
    for fmt in source_formats:
        for audio_file in source.rglob(f"*{fmt}"):
            # Target path
            rel_path = audio_file.relative_to(source)
            opus_path = target / rel_path.with_suffix(".opus")
            opus_path.parent.mkdir(parents=True, exist_ok=True)
            
            if opus_path.exists():
                results["skipped"] += 1
                continue
            
            result = transcode_to_opus(str(audio_file), str(opus_path))
            
            if result:
                results["converted"] += 1
                if delete_originals:
                    audio_file.unlink()
            else:
                results["failed"] += 1
    
    return results
