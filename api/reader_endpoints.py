"""
Reader Analysis API Endpoints.
Specialized analysis for reading practice: hesitations, self-corrections,
expression, and reference text comparison.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sqlite3
import json
import logging

from .config import DB_PATH
from .user_auth import flexible_auth

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from reader_analysis import (
    analyze_reading,
    detect_hesitations,
    detect_self_corrections,
    analyze_expression,
    compare_to_reference,
)

logger = logging.getLogger("speechscore.reader")
router = APIRouter(prefix="/v1/reader", tags=["Reader Analysis"])


class ReaderAnalysisRequest(BaseModel):
    """Request for reader analysis."""
    reference_text: Optional[str] = None  # Text that should have been read


class ReaderAnalysisResponse(BaseModel):
    """Response from reader analysis."""
    speech_id: int
    overall_score: Dict[str, Any]
    hesitations: Dict[str, Any]
    self_corrections: Dict[str, Any]
    expression: Dict[str, Any]
    accuracy: Optional[Dict[str, Any]] = None
    reading_speed: Optional[Dict[str, Any]] = None


@router.post("/analyze/{speech_id}")
async def analyze_reader_speech(
    speech_id: int,
    request: ReaderAnalysisRequest = None,
    auth: dict = Depends(flexible_auth)
) -> Dict[str, Any]:
    """
    Run comprehensive reader analysis on a speech.
    
    Analyzes:
    - Hesitations (pauses before words)
    - Self-corrections (repeats, restarts)
    - Expression (pitch variation, appropriate pauses)
    - Accuracy (if reference text provided)
    - Reading speed (WPM)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Get speech data
        cursor.execute("""
            SELECT s.id, s.profile, t.text, t.segments,
                   a.pitch_mean_hz, a.pitch_std_hz, 
                   a.pitch_range_hz,
                   a.wpm
            FROM speeches s
            LEFT JOIN transcriptions t ON t.speech_id = s.id
            LEFT JOIN analyses a ON a.speech_id = s.id
            WHERE s.id = ?
        """, (speech_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Speech not found")
        
        transcript = row['text'] or ''
        if not transcript:
            raise HTTPException(status_code=400, detail="No transcript available")
        
        # Parse word timings from segments
        word_timings = []
        if row['segments']:
            try:
                segments = json.loads(row['segments'])
                for seg in segments:
                    if 'words' in seg:
                        for w in seg['words']:
                            word_timings.append({
                                'word': w.get('word', ''),
                                'start': w.get('start', 0),
                                'end': w.get('end', 0),
                                'confidence': w.get('probability', 0),
                            })
            except:
                pass
        
        # Build pitch data dict
        pitch_mean = row['pitch_mean_hz'] or 0
        pitch_std = row['pitch_std_hz'] or 0
        pitch_cv = (pitch_std / pitch_mean * 100) if pitch_mean > 0 else 0
        pitch_range_hz = row['pitch_range_hz'] or 0
        # Convert Hz range to approximate semitones (12 semitones = 1 octave = 2x frequency)
        pitch_range_semitones = (pitch_range_hz / pitch_mean * 12) if pitch_mean > 0 else 0
        
        pitch_data = {
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_cv': pitch_cv,
            'pitch_range_semitones': pitch_range_semitones,
        }
        
        # Estimate pause data from word timings if available
        pause_count = 0
        total_pause_time = 0
        for i in range(1, len(word_timings)):
            gap = word_timings[i].get('start', 0) - word_timings[i-1].get('end', 0)
            if gap > 0.3:  # Pause > 300ms
                pause_count += 1
                total_pause_time += gap
        
        pause_data = {
            'pause_count': pause_count,
            'pause_ratio': total_pause_time / (word_timings[-1].get('end', 1) if word_timings else 1),
        }
        
        wpm = row['wpm']
        reference_text = request.reference_text if request else None
        
        # Run comprehensive analysis
        results = analyze_reading(
            transcript=transcript,
            word_timings=word_timings,
            pitch_data=pitch_data,
            pause_data=pause_data,
            reference_text=reference_text,
            wpm=wpm,
        )
        
        results['speech_id'] = speech_id
        
        # Cache the results
        cursor.execute("""
            UPDATE speeches 
            SET cached_score_json = ?, cached_score_profile = 'reader'
            WHERE id = ?
        """, (json.dumps(results), speech_id))
        conn.commit()
        
        return results
        
    finally:
        conn.close()


@router.get("/hesitations/{speech_id}")
async def get_hesitations(
    speech_id: int,
    threshold_ms: float = 500,
    auth: dict = Depends(flexible_auth)
) -> Dict[str, Any]:
    """
    Get hesitation analysis for a speech.
    
    Hesitations are pauses before words that indicate reading difficulty.
    Returns difficult words and pause patterns.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT segments FROM transcriptions WHERE speech_id = ?
        """, (speech_id,))
        row = cursor.fetchone()
        
        if not row or not row['segments']:
            raise HTTPException(status_code=400, detail="No word timing data available")
        
        # Extract word timings from segments
        word_timings = []
        segments = json.loads(row['segments'])
        for seg in segments:
            if 'words' in seg:
                for w in seg['words']:
                    word_timings.append({
                        'word': w.get('word', ''),
                        'start': w.get('start', 0),
                        'end': w.get('end', 0),
                        'confidence': w.get('probability', 0),
                    })
        results = detect_hesitations(word_timings, threshold_ms=threshold_ms)
        results['speech_id'] = speech_id
        
        return results
        
    finally:
        conn.close()


@router.get("/self-corrections/{speech_id}")
async def get_self_corrections(
    speech_id: int,
    auth: dict = Depends(flexible_auth)
) -> Dict[str, Any]:
    """
    Detect self-corrections in a reading.
    
    Self-corrections include:
    - Repeated words ("the the cat")
    - False starts ("I went to the— to the store")
    - Restarts of phrases
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT text FROM transcriptions WHERE speech_id = ?
        """, (speech_id,))
        row = cursor.fetchone()
        
        if not row or not row['text']:
            raise HTTPException(status_code=400, detail="No transcript available")
        
        results = detect_self_corrections(row['text'])
        results['speech_id'] = speech_id
        
        return results
        
    finally:
        conn.close()


@router.post("/compare/{speech_id}")
async def compare_to_reference_text(
    speech_id: int,
    reference_text: str,
    auth: dict = Depends(flexible_auth)
) -> Dict[str, Any]:
    """
    Compare actual reading to reference text.
    
    Returns word-by-word comparison showing:
    - Correct words
    - Substitutions (wrong word)
    - Insertions (extra words)
    - Deletions (missed words)
    - Overall accuracy percentage
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT text FROM transcriptions WHERE speech_id = ?
        """, (speech_id,))
        row = cursor.fetchone()
        
        if not row or not row['text']:
            raise HTTPException(status_code=400, detail="No transcript available")
        
        results = compare_to_reference(row['text'], reference_text)
        results['speech_id'] = speech_id
        
        return results
        
    finally:
        conn.close()


@router.get("/expression/{speech_id}")
async def get_expression_analysis(
    speech_id: int,
    auth: dict = Depends(flexible_auth)
) -> Dict[str, Any]:
    """
    Analyze reading expression and intonation.
    
    Measures:
    - Pitch variation (monotone vs expressive)
    - Pause appropriateness (pausing at punctuation)
    - Question intonation (rising pitch for questions)
    - Emphasis patterns
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT t.text, t.segments, a.pitch_mean_hz, a.pitch_std_hz, 
                   a.pitch_range_hz
            FROM speeches s
            LEFT JOIN transcriptions t ON t.speech_id = s.id
            LEFT JOIN analyses a ON a.speech_id = s.id
            WHERE s.id = ?
        """, (speech_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Speech not found")
        
        # Calculate pitch metrics
        pitch_mean = row['pitch_mean_hz'] or 0
        pitch_std = row['pitch_std_hz'] or 0
        pitch_cv = (pitch_std / pitch_mean * 100) if pitch_mean > 0 else 0
        pitch_range_hz = row['pitch_range_hz'] or 0
        pitch_range_semitones = (pitch_range_hz / pitch_mean * 12) if pitch_mean > 0 else 0
        
        pitch_data = {
            'pitch_cv': pitch_cv,
            'pitch_range_semitones': pitch_range_semitones,
        }
        
        # Extract word timings for pause analysis
        word_timings = []
        if row['segments']:
            try:
                segments = json.loads(row['segments'])
                for seg in segments:
                    if 'words' in seg:
                        for w in seg['words']:
                            word_timings.append({
                                'start': w.get('start', 0),
                                'end': w.get('end', 0),
                            })
            except:
                pass
        
        # Calculate pause data
        pause_count = 0
        total_pause_time = 0
        for i in range(1, len(word_timings)):
            gap = word_timings[i].get('start', 0) - word_timings[i-1].get('end', 0)
            if gap > 0.3:
                pause_count += 1
                total_pause_time += gap
        
        total_duration = word_timings[-1].get('end', 1) if word_timings else 1
        pause_data = {
            'pause_count': pause_count,
            'pause_ratio': total_pause_time / total_duration if total_duration > 0 else 0,
        }
        
        results = analyze_expression(pitch_data, pause_data, row['text'] or '')
        results['speech_id'] = speech_id
        
        return results
        
    finally:
        conn.close()
