"""
Presentation Readiness API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import sqlite3
import logging
import json

from .user_auth import flexible_auth
from .config import DB_PATH
from .presentation_analysis import get_presentation_analyzer
from .content_analysis import get_content_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/speeches", tags=["presentation"])


def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@router.post("/{speech_id}/presentation-readiness")
async def analyze_presentation_readiness(
    speech_id: int,
    user = Depends(flexible_auth)
):
    """
    Analyze a speech for presentation readiness.
    Returns actionable feedback for presentation preparation.
    """
    db = get_db()
    
    # Get speech
    speech = db.execute(
        "SELECT * FROM speeches WHERE id = ?", (speech_id,)
    ).fetchone()
    
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    
    # Get transcription
    trans = db.execute(
        "SELECT text, segments FROM transcriptions WHERE speech_id = ?",
        (speech_id,)
    ).fetchone()
    
    if not trans:
        raise HTTPException(status_code=400, detail="No transcription available")
    
    # Get analysis
    analysis = db.execute(
        "SELECT * FROM analyses WHERE speech_id = ?",
        (speech_id,)
    ).fetchone()
    
    if not analysis:
        raise HTTPException(status_code=400, detail="No analysis available")
    
    # Get advanced audio from result JSON if available
    import json
    advanced_audio = None
    if analysis['result']:
        try:
            result_json = json.loads(analysis['result'])
            advanced_audio = result_json.get('advanced_audio')
        except:
            pass
    
    # Parse segments
    segments = []
    if trans['segments']:
        try:
            segments = json.loads(trans['segments'])
        except:
            pass
    
    # Build analysis data dict with defaults for null values
    analysis_data = {
        'wpm': analysis['wpm'] or 150,
        'pitch_mean_hz': analysis['pitch_mean_hz'] or 150,
        'pitch_std_hz': analysis['pitch_std_hz'] or 30,
        'jitter_pct': analysis['jitter_pct'] or 1.0,
        'shimmer_pct': analysis['shimmer_pct'] or 3.0,
        'hnr_db': analysis['hnr_db'] or 15,
        'vocal_fry_ratio': analysis['vocal_fry_ratio'] or 0.1,
        'lexical_diversity': analysis['lexical_diversity'] or 0.7,
    }
    
    # Run presentation analysis
    analyzer = get_presentation_analyzer()
    report = analyzer.analyze(
        transcript_text=trans['text'],
        segments=segments,
        analysis_data=analysis_data,
        advanced_audio=advanced_audio
    )
    
    return report.to_dict()


@router.get("/{speech_id}/presentation-readiness")
async def get_presentation_readiness(
    speech_id: int,
    user = Depends(flexible_auth)
):
    """
    Get cached presentation readiness or run new analysis.
    """
    # For now, just run the analysis
    return await analyze_presentation_readiness(speech_id, user)


@router.post("/{speech_id}/content-analysis")
async def analyze_content(
    speech_id: int,
    user = Depends(flexible_auth)
):
    """
    Analyze speech content and tone alignment.
    Uses LLM to understand what's being said and compares to how it's delivered.
    """
    db = get_db()
    
    # Get speech info
    speech = db.execute(
        "SELECT * FROM speeches WHERE id = ?", (speech_id,)
    ).fetchone()
    
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    
    # Get transcription
    trans = db.execute(
        "SELECT text, segments FROM transcriptions WHERE speech_id = ?",
        (speech_id,)
    ).fetchone()
    
    if not trans:
        raise HTTPException(status_code=400, detail="No transcription available")
    
    # Parse segments
    segments = []
    if trans['segments']:
        try:
            segments = json.loads(trans['segments'])
            logger.info(f"Parsed {len(segments)} transcript segments from DB")
        except Exception as e:
            logger.error(f"Failed to parse segments: {e}")
    else:
        logger.warning("No segments data in transcription")
    
    # Get emotion data from oratory analysis
    emotion_data = None
    audio_path = speech['source_file'] if speech else None
    
    if audio_path:
        try:
            from .oratory_analysis import get_oratory_analyzer
            oratory = get_oratory_analyzer()
            
            # Run quick emotion analysis only
            oratory_result = oratory.analyze(audio_path, word_timestamps=segments)
            
            # Convert to emotion data format expected by content analyzer
            emotion_data = []
            for seg in oratory_result.segments:
                emotion_data.append({
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "primary_emotion": seg.primary_emotion,
                    "confidence": seg.emotion_confidence,
                    "all_emotions": seg.emotion_scores
                })
            logger.info(f"Got emotion data for {len(emotion_data)} segments")
        except Exception as e:
            logger.warning(f"Could not get emotion data: {e}")
    
    # Run content analysis
    logger.info(f"Calling content analyzer with {len(segments)} segments and {len(emotion_data) if emotion_data else 0} emotion data")
    analyzer = get_content_analyzer()
    result = analyzer.analyze(
        transcript=trans['text'],
        segments=segments,
        emotion_data=emotion_data
    )
    
    return result.to_dict()
