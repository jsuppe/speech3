"""
Audio Storage Utility

Encodes raw PCM audio to Opus for efficient storage and replay.
Uses ffmpeg for encoding.

Storage: ~11 MB/hour (vs ~112 MB/hour for raw PCM)
"""

import os
import subprocess
import tempfile
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("speechscore.audio_storage")

# Storage directory for session recordings
AUDIO_STORAGE_DIR = Path("/home/melchior/speech3/session_audio")
AUDIO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Opus encoding settings optimized for speech
OPUS_BITRATE = "24k"  # 24 kbps - excellent quality for speech, ~11 MB/hour
SAMPLE_RATE = 16000


def encode_pcm_to_opus(
    pcm_data: bytes,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
    output_path: Optional[str] = None,
) -> Tuple[Optional[str], int]:
    """
    Encode raw PCM16 audio to Opus using ffmpeg.
    
    Args:
        pcm_data: Raw PCM16 (16-bit signed little-endian) audio bytes
        sample_rate: Sample rate of input audio (default 16000)
        channels: Number of channels (default 1 = mono)
        output_path: Optional output path. If None, creates temp file.
        
    Returns:
        Tuple of (output_path, file_size_bytes) or (None, 0) on error
    """
    if not pcm_data:
        return None, 0
    
    try:
        # Create temp file for input PCM
        with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as pcm_file:
            pcm_file.write(pcm_data)
            pcm_path = pcm_file.name
        
        # Generate output path if not provided
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.opus')
        
        # ffmpeg command to encode PCM to Opus
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-f', 's16le',   # Input format: signed 16-bit little-endian
            '-ar', str(sample_rate),  # Input sample rate
            '-ac', str(channels),     # Input channels
            '-i', pcm_path,  # Input file
            '-c:a', 'libopus',  # Opus codec
            '-b:a', OPUS_BITRATE,  # Bitrate
            '-vbr', 'on',    # Variable bitrate for efficiency
            '-application', 'voip',  # Optimized for speech
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,  # 1 minute timeout
        )
        
        # Clean up temp PCM file
        os.unlink(pcm_path)
        
        if result.returncode != 0:
            logger.error(f"ffmpeg encode failed: {result.stderr.decode()}")
            return None, 0
        
        file_size = os.path.getsize(output_path)
        logger.info(f"Encoded {len(pcm_data)} bytes PCM to {file_size} bytes Opus ({file_size/len(pcm_data)*100:.1f}%)")
        
        return output_path, file_size
        
    except Exception as e:
        logger.error(f"Opus encoding error: {e}")
        return None, 0


def decode_opus_to_pcm(opus_path: str, sample_rate: int = SAMPLE_RATE) -> Optional[bytes]:
    """
    Decode Opus file to raw PCM16 bytes.
    
    Args:
        opus_path: Path to Opus file
        sample_rate: Output sample rate (default 16000)
        
    Returns:
        Raw PCM16 bytes or None on error
    """
    try:
        # Create temp file for output PCM
        with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as pcm_file:
            pcm_path = pcm_file.name
        
        cmd = [
            'ffmpeg', '-y',
            '-i', opus_path,
            '-f', 's16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            pcm_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        
        if result.returncode != 0:
            logger.error(f"ffmpeg decode failed: {result.stderr.decode()}")
            os.unlink(pcm_path)
            return None
        
        with open(pcm_path, 'rb') as f:
            pcm_data = f.read()
        
        os.unlink(pcm_path)
        return pcm_data
        
    except Exception as e:
        logger.error(f"Opus decoding error: {e}")
        return None


def save_session_audio(
    pcm_data: bytes,
    user_id: Optional[int] = None,
    session_type: str = "live",
    metadata: Optional[dict] = None,
) -> Optional[str]:
    """
    Save session audio as Opus with metadata.
    
    Args:
        pcm_data: Raw PCM16 audio bytes
        user_id: Optional user ID for organization
        session_type: Type of session (live, dementia, etc.)
        metadata: Optional metadata dict to save alongside
        
    Returns:
        Session ID (filename without extension) or None on error
    """
    if not pcm_data:
        return None
    
    # Generate session ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{session_type}_{timestamp}_{uuid.uuid4().hex[:8]}"
    
    # Create user directory if specified
    if user_id:
        user_dir = AUDIO_STORAGE_DIR / f"user_{user_id}"
        user_dir.mkdir(exist_ok=True)
        opus_path = user_dir / f"{session_id}.opus"
    else:
        opus_path = AUDIO_STORAGE_DIR / f"{session_id}.opus"
    
    # Encode and save
    result_path, file_size = encode_pcm_to_opus(pcm_data, output_path=str(opus_path))
    
    if result_path is None:
        return None
    
    # Save metadata if provided
    if metadata:
        import json
        meta_path = opus_path.with_suffix('.json')
        metadata['session_id'] = session_id
        metadata['file_size_bytes'] = file_size
        metadata['pcm_size_bytes'] = len(pcm_data)
        metadata['compression_ratio'] = len(pcm_data) / file_size if file_size > 0 else 0
        metadata['created_at'] = datetime.now().isoformat()
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved session {session_id}: {file_size} bytes Opus")
    return session_id


def get_session_audio(session_id: str, user_id: Optional[int] = None) -> Tuple[Optional[bytes], Optional[dict]]:
    """
    Retrieve session audio as PCM bytes.
    
    Returns:
        Tuple of (pcm_bytes, metadata) or (None, None) on error
    """
    if user_id:
        base_path = AUDIO_STORAGE_DIR / f"user_{user_id}" / session_id
    else:
        base_path = AUDIO_STORAGE_DIR / session_id
    
    opus_path = Path(str(base_path) + '.opus')
    meta_path = Path(str(base_path) + '.json')
    
    if not opus_path.exists():
        logger.warning(f"Session not found: {session_id}")
        return None, None
    
    pcm_data = decode_opus_to_pcm(str(opus_path))
    
    metadata = None
    if meta_path.exists():
        import json
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    
    return pcm_data, metadata


def list_sessions(user_id: Optional[int] = None, session_type: Optional[str] = None) -> list:
    """
    List available sessions.
    
    Returns:
        List of session IDs
    """
    if user_id:
        search_dir = AUDIO_STORAGE_DIR / f"user_{user_id}"
    else:
        search_dir = AUDIO_STORAGE_DIR
    
    if not search_dir.exists():
        return []
    
    sessions = []
    for opus_file in search_dir.glob("*.opus"):
        session_id = opus_file.stem
        if session_type is None or session_id.startswith(session_type):
            sessions.append(session_id)
    
    return sorted(sessions, reverse=True)  # Most recent first


def get_storage_stats() -> dict:
    """Get storage usage statistics."""
    total_files = 0
    total_bytes = 0
    
    for opus_file in AUDIO_STORAGE_DIR.rglob("*.opus"):
        total_files += 1
        total_bytes += opus_file.stat().st_size
    
    return {
        "total_sessions": total_files,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "storage_path": str(AUDIO_STORAGE_DIR),
    }
