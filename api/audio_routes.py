"""
Audio Storage REST API Endpoints

Provides endpoints for listing, retrieving, and managing stored session audio.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import io

from .audio_storage import (
    get_session_audio,
    list_sessions,
    get_storage_stats,
    decode_opus_to_pcm,
    AUDIO_STORAGE_DIR,
)

logger = logging.getLogger("speechscore.audio_routes")

router = APIRouter(prefix="/v1/audio", tags=["Audio Storage"])


@router.get("/sessions")
async def list_audio_sessions(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    session_type: Optional[str] = Query(None, description="Filter by session type (live, dementia, etc.)"),
    limit: int = Query(50, description="Maximum sessions to return"),
):
    """
    List stored audio sessions.
    
    Returns session IDs sorted by most recent first.
    """
    sessions = list_sessions(user_id=user_id, session_type=session_type)
    return {
        "sessions": sessions[:limit],
        "total": len(sessions),
    }


@router.get("/sessions/{session_id}")
async def get_session_metadata(
    session_id: str,
    user_id: Optional[int] = Query(None),
):
    """
    Get session metadata without audio.
    """
    import json
    from pathlib import Path
    
    if user_id:
        meta_path = AUDIO_STORAGE_DIR / f"user_{user_id}" / f"{session_id}.json"
    else:
        meta_path = AUDIO_STORAGE_DIR / f"{session_id}.json"
    
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


@router.get("/sessions/{session_id}/audio")
async def get_session_audio_endpoint(
    session_id: str,
    user_id: Optional[int] = Query(None),
    format: str = Query("opus", description="Output format: opus or wav"),
):
    """
    Retrieve session audio file.
    
    Returns Opus file by default (efficient). Use format=wav for playback compatibility.
    """
    from pathlib import Path
    
    if user_id:
        opus_path = AUDIO_STORAGE_DIR / f"user_{user_id}" / f"{session_id}.opus"
    else:
        opus_path = AUDIO_STORAGE_DIR / f"{session_id}.opus"
    
    if not opus_path.exists():
        raise HTTPException(status_code=404, detail="Session audio not found")
    
    if format == "opus":
        # Return raw Opus file
        with open(opus_path, 'rb') as f:
            content = f.read()
        return StreamingResponse(
            io.BytesIO(content),
            media_type="audio/opus",
            headers={"Content-Disposition": f'attachment; filename="{session_id}.opus"'},
        )
    
    elif format == "wav":
        # Decode to WAV for universal playback
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
            wav_path = wav_file.name
        
        try:
            cmd = ['ffmpeg', '-y', '-i', str(opus_path), '-ar', '16000', '-ac', '1', wav_path]
            subprocess.run(cmd, capture_output=True, check=True)
            
            with open(wav_path, 'rb') as f:
                content = f.read()
            
            import os
            os.unlink(wav_path)
            
            return StreamingResponse(
                io.BytesIO(content),
                media_type="audio/wav",
                headers={"Content-Disposition": f'attachment; filename="{session_id}.wav"'},
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e}")
    
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'opus' or 'wav'")


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user_id: Optional[int] = Query(None),
):
    """
    Delete a stored session and its metadata.
    """
    from pathlib import Path
    import os
    
    if user_id:
        base_path = AUDIO_STORAGE_DIR / f"user_{user_id}" / session_id
    else:
        base_path = AUDIO_STORAGE_DIR / session_id
    
    opus_path = Path(str(base_path) + '.opus')
    meta_path = Path(str(base_path) + '.json')
    
    deleted = False
    
    if opus_path.exists():
        os.unlink(opus_path)
        deleted = True
    
    if meta_path.exists():
        os.unlink(meta_path)
        deleted = True
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"deleted": session_id}


@router.get("/storage/stats")
async def get_storage_statistics():
    """
    Get storage usage statistics.
    """
    return get_storage_stats()
