"""
Oratory Analysis API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import sqlite3
from pathlib import Path
import logging

from .oratory_analysis import get_oratory_analyzer, OratoryAnalysisResult
from .user_auth import flexible_auth
from .config import DB_PATH

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/speeches", tags=["Oratory Analysis"])


# Response models
class SegmentResponse(BaseModel):
    start_time: float
    end_time: float
    primary_emotion: str
    emotion_confidence: float
    emotion_scores: Dict[str, float]
    mean_energy: float
    energy_variation: float
    peak_energy: float
    mean_pitch: float
    pitch_range: float
    pitch_variation: float
    speaking_rate: float
    pause_ratio: float
    emphasized_words: List[str]
    emphasis_score: float

class EnergyArcResponse(BaseModel):
    arc_type: str
    peak_position: float
    dynamic_range: float
    energy_curve: List[float]
    description: str

class RhetoricalMetricsResponse(BaseModel):
    vocal_variety_score: float
    rhythm_score: float
    authority_index: float
    engagement_prediction: float
    pitch_dynamism: float
    volume_dynamism: float
    pacing_variation: float
    pause_effectiveness: float
    strengths: List[str]
    improvements: List[str]

class StyleResponse(BaseModel):
    primary_style: str
    style_scores: Dict[str, float]
    similar_to: List[str]
    description: str

class OratoryAnalysisResponse(BaseModel):
    speech_id: int
    segments: List[SegmentResponse]
    energy_arc: EnergyArcResponse
    rhetorical_metrics: RhetoricalMetricsResponse
    style: StyleResponse
    dominant_emotion: str
    emotion_range: float
    emotion_transitions: int
    overall_score: float


def get_audio_path(source_file: str) -> Optional[Path]:
    """Find the audio file from various possible locations."""
    if not source_file:
        return None
    
    # Check common locations
    candidates = [
        Path(source_file),
        Path("/home/melchior/speech3/audio") / source_file,
        Path("/home/melchior/speech3/uploads") / source_file,
        Path("/tmp") / source_file,
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    return None


def extract_audio_from_db(speech_id: int) -> Optional[Path]:
    """Extract audio from database to temp file."""
    import tempfile
    import subprocess
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT data, format FROM audio WHERE speech_id = ?", (speech_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    audio_data, fmt = row
    
    # Write to temp file
    suffix = f".{fmt}" if fmt else ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_data)
        temp_path = Path(f.name)
    
    # Convert to wav if needed
    if suffix != ".wav":
        wav_path = temp_path.with_suffix(".wav")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(temp_path),
                "-ar", "16000", "-ac", "1",
                str(wav_path)
            ], check=True, capture_output=True)
            temp_path.unlink()
            return wav_path
        except Exception as e:
            logger.warning(f"Failed to convert audio: {e}")
            return temp_path
    
    return temp_path


@router.post("/{speech_id}/oratory", response_model=OratoryAnalysisResponse)
async def analyze_oratory(
    speech_id: int,
    segment_duration: float = 5.0,
    auth: dict = Depends(flexible_auth)
):
    """
    Run comprehensive oratory analysis on a speech.
    
    Returns:
    - Per-segment emotion detection
    - Energy arc analysis
    - Rhetorical effectiveness metrics
    - Delivery style classification
    """
    # Get speech info
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT source_file FROM speeches WHERE id = ?", (speech_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Speech not found")
    
    # Find audio file
    audio_path = get_audio_path(row['source_file'])
    temp_audio = None
    
    if not audio_path:
        # Try to extract from database
        audio_path = extract_audio_from_db(speech_id)
        if audio_path:
            temp_audio = audio_path
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Audio file not found: '{row['source_file']}'"
            )
    
    try:
        # Run oratory analysis
        analyzer = get_oratory_analyzer()
        result = analyzer.analyze(
            str(audio_path),
            segment_duration=segment_duration
        )
        
        # Convert to response
        response = OratoryAnalysisResponse(
            speech_id=speech_id,
            segments=[SegmentResponse(**s.__dict__) for s in result.segments],
            energy_arc=EnergyArcResponse(**result.energy_arc.__dict__),
            rhetorical_metrics=RhetoricalMetricsResponse(**result.rhetorical_metrics.__dict__),
            style=StyleResponse(**result.style.__dict__),
            dominant_emotion=result.dominant_emotion,
            emotion_range=result.emotion_range,
            emotion_transitions=result.emotion_transitions,
            overall_score=result.overall_score
        )
        
        return response
        
    except Exception as e:
        logger.exception(f"Oratory analysis failed for speech {speech_id}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
    finally:
        # Cleanup temp file
        if temp_audio and temp_audio.exists():
            try:
                temp_audio.unlink()
            except:
                pass


@router.get("/{speech_id}/oratory/summary")
async def get_oratory_summary(
    speech_id: int,
    auth: dict = Depends(flexible_auth)
):
    """
    Get a brief oratory summary for display in the app.
    Faster than full analysis - returns cached data if available.
    """
    # Check for cached analysis
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT oratory_analysis FROM analysis 
        WHERE speech_id = ? AND oratory_analysis IS NOT NULL
    """, (speech_id,))
    row = cursor.fetchone()
    
    if row and row['oratory_analysis']:
        import json
        try:
            cached = json.loads(row['oratory_analysis'])
            return {
                "speech_id": speech_id,
                "cached": True,
                "overall_score": cached.get('overall_score'),
                "dominant_emotion": cached.get('dominant_emotion'),
                "style": cached.get('style', {}).get('primary_style'),
                "energy_arc": cached.get('energy_arc', {}).get('arc_type'),
            }
        except:
            pass
    
    conn.close()
    
    # No cache - run quick analysis
    return await analyze_oratory(speech_id, segment_duration=10.0, auth=auth)
