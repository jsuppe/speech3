"""
Quick transcription endpoint for Live Coach feature.
Just runs Whisper STT on uploaded audio, returns text.
"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from .user_auth import flexible_auth

router = APIRouter()


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    user_id: int = Depends(flexible_auth)
):
    """
    Quick transcription of audio file.
    Returns just the text, optimized for low-latency live feedback.
    """
    from .pipeline_runner import run_whisper_stt
    
    # Save uploaded file to temp location
    suffix = os.path.splitext(audio.filename or "audio.m4a")[1] or ".m4a"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Run Whisper STT
        result = run_whisper_stt(tmp_path)
        
        if result and "text" in result:
            return {"text": result["text"].strip()}
        else:
            return {"text": ""}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
