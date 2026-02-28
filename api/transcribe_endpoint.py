"""
Quick transcription endpoint for Live Coach feature.
Runs Whisper STT on uploaded audio and optionally analyzes confidence.
"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from .user_auth import flexible_auth

router = APIRouter()


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    analyze_confidence: bool = Query(default=True, description="Include confidence/nervousness analysis"),
    user_id: int = Depends(flexible_auth)
):
    """
    Quick transcription of audio file with optional confidence analysis.
    Returns text and optionally confidence metrics for nervousness detection.
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
        
        text = ""
        if result and "text" in result:
            text = result["text"].strip()
        
        response = {"text": text}
        
        # Optionally analyze confidence/nervousness from audio + text
        if analyze_confidence and text:
            try:
                from .confidence_analysis import analyze_confidence_quick
                confidence_result = analyze_confidence_quick(tmp_path, text)
                response["confidence"] = confidence_result
            except Exception as e:
                # Don't fail the whole request if confidence fails
                response["confidence"] = {
                    "confidence_score": 50,
                    "confidence_level": "neutral",
                    "indicators": {"error": str(e)}
                }
        
        return response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
