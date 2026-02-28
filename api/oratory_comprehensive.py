"""
Comprehensive Oratory Analysis Endpoint
Returns all data needed for the 6-section Oratory results screen:
1. Persuasion Score
2. Sentiment Tracking
3. Delivery Analysis
4. Device Detection
5. Great Speech Comparison
6. Argument Mapping
"""
import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends

from .user_auth import flexible_auth
from .config import DB_PATH

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from rhetorical_analysis import analyze_repetition, analyze_all_devices
from sentiment_analysis import analyze_sentiment
from scoring import score_speech, benchmark_speech, SCORING_PROFILES
from speech_db import SpeechDB
from argument_analysis import analyze_arguments, format_for_display

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/oratory", tags=["Comprehensive Oratory"])


# Famous speech benchmarks for comparison
GREAT_SPEECHES = {
    "mlk_dream": {"name": "MLK - I Have a Dream", "wpm": 107, "pitch_variation": 32, "repetition": 24},
    "jfk_inaugural": {"name": "JFK - Inaugural", "wpm": 127, "pitch_variation": 28, "repetition": 18},
    "churchill_beaches": {"name": "Churchill - We Shall Fight", "wpm": 128, "pitch_variation": 35, "repetition": 31},
    "lincoln_gettysburg": {"name": "Lincoln - Gettysburg", "wpm": 112, "pitch_variation": 22, "repetition": 15},
    "fdr_infamy": {"name": "FDR - Day of Infamy", "wpm": 118, "pitch_variation": 25, "repetition": 12},
}


@router.get("/{speech_id}")
async def get_comprehensive_oratory(
    speech_id: int,
    auth: dict = Depends(flexible_auth)
) -> Dict[str, Any]:
    """
    Get comprehensive oratory analysis for a speech.
    
    Returns all 6 sections:
    1. persuasion_score - Combined delivery + argument quality
    2. sentiment - Emotional arc and sentiment breakdown
    3. delivery - Pitch, pacing, voice quality metrics
    4. devices - Rhetorical devices (anaphora, repetition, etc.)
    5. comparison - How you compare to great orators
    6. arguments - Claim-evidence-impact structure
    """
    db = SpeechDB(DB_PATH)
    
    # Get speech and analysis
    speech = db.get_speech(speech_id)
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    
    analysis = db.get_analysis(speech_id)
    if not analysis:
        raise HTTPException(status_code=400, detail="No analysis available")
    
    # Get transcript
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM transcriptions WHERE speech_id = ?", (speech_id,))
    row = cursor.fetchone()
    transcript = row['text'] if row else ""
    conn.close()
    
    # 1. Get oratory score
    score_result = score_speech(db, speech_id, profile="oratory")
    oratory_score = score_result.get("overall_score", 0) if score_result else 0
    
    # 2. Sentiment Analysis
    sentiment_data = _get_sentiment(transcript, analysis)
    
    # 3. Delivery Analysis
    delivery_data = _get_delivery(analysis)
    
    # 4. Device Detection (rhetorical)
    devices_data = _get_devices(transcript)
    
    # 5. Great Speech Comparison
    comparison_data = _get_comparison(analysis, db, speech_id)
    
    # 6. Argument Mapping (async call to LLM - may be slow)
    # We'll return a placeholder and let the app fetch separately
    argument_summary = {
        "available": bool(transcript and len(transcript) > 100),
        "endpoint": f"/v1/speeches/{speech_id}/arguments",
        "note": "Call endpoint separately for full argument analysis"
    }
    
    # Calculate Persuasion Score (will be updated when arguments are fetched)
    # For now: 70% oratory, 30% device usage
    device_score = min(100, devices_data.get("total_devices", 0) * 15)
    persuasion_score = int(oratory_score * 0.7 + device_score * 0.3)
    
    return {
        "speech_id": speech_id,
        "title": speech.get("title", "Untitled"),
        
        # 1. Persuasion Score
        "persuasion_score": {
            "overall": persuasion_score,
            "oratory_component": oratory_score,
            "device_component": device_score,
            "grade": _score_to_grade(persuasion_score),
            "label": _score_to_label(persuasion_score),
        },
        
        # 2. Sentiment Tracking
        "sentiment": sentiment_data,
        
        # 3. Delivery Analysis
        "delivery": delivery_data,
        
        # 4. Device Detection
        "devices": devices_data,
        
        # 5. Great Speech Comparison
        "comparison": comparison_data,
        
        # 6. Argument Mapping (placeholder)
        "arguments": argument_summary,
    }


def _get_sentiment(transcript: str, analysis: dict) -> dict:
    """Extract sentiment data."""
    # Try to get from analysis first
    sentiment_val = analysis.get("sentiment")
    
    # Run sentiment analysis if transcript available
    if transcript and len(transcript) > 50:
        try:
            result = analyze_sentiment(transcript)
            return {
                "overall": result.get("overall", {}).get("label", "NEUTRAL"),
                "overall_score": result.get("overall", {}).get("score", 0.5),
                "arc": result.get("emotional_arc", "flat"),
                "arc_description": _arc_description(result.get("emotional_arc", "flat")),
                "journey": _sentiment_journey(result),
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
    
    # Fallback
    return {
        "overall": "NEUTRAL",
        "overall_score": 0.5,
        "arc": "flat",
        "arc_description": "Consistent emotional tone throughout.",
        "journey": [],
    }


def _arc_description(arc: str) -> str:
    """Human-readable arc description."""
    descriptions = {
        "rising": "Builds emotional intensity toward the end.",
        "falling": "Starts strong and settles to a calmer conclusion.",
        "peak": "Rises to an emotional peak, then resolves.",
        "valley": "Dips emotionally in the middle before recovering.",
        "flat": "Maintains consistent emotional tone throughout.",
        "volatile": "Frequent emotional shifts — dynamic and engaging.",
    }
    return descriptions.get(arc, "Varied emotional expression.")


def _sentiment_journey(result: dict) -> List[dict]:
    """Create sentiment journey for visualization."""
    per_segment = result.get("per_segment", [])
    if not per_segment:
        return []
    
    journey = []
    for i, seg in enumerate(per_segment[:10]):  # Limit to 10 points
        journey.append({
            "position": i / max(len(per_segment) - 1, 1),
            "sentiment": seg.get("label", "NEUTRAL"),
            "score": seg.get("score", 0.5),
        })
    return journey


def _get_delivery(analysis: dict) -> dict:
    """Extract delivery metrics."""
    wpm = analysis.get("wpm", 0)
    pitch_mean = analysis.get("pitch_mean_hz", 0)
    pitch_std = analysis.get("pitch_std_hz", 0)
    pitch_range = analysis.get("pitch_range_hz", 0)
    hnr = analysis.get("hnr_db", 0)
    jitter = analysis.get("jitter_pct", 0)
    shimmer = analysis.get("shimmer_pct", 0)
    vocal_fry = analysis.get("vocal_fry_ratio", 0)
    
    # Calculate pitch variation as CV
    pitch_cv = (pitch_std / pitch_mean * 100) if pitch_mean > 0 else 0
    
    return {
        "pacing": {
            "wpm": round(wpm, 1),
            "rating": _rate_wpm(wpm),
            "target": "110-140 WPM for oratory",
        },
        "pitch": {
            "mean_hz": round(pitch_mean, 1),
            "variation_cv": round(pitch_cv, 1),
            "range_hz": round(pitch_range, 1),
            "expressiveness": _rate_pitch_variation(pitch_cv),
        },
        "voice_quality": {
            "hnr_db": round(hnr, 1),
            "rating": _rate_hnr(hnr),
            "clarity": "Clear" if hnr > 15 else ("Good" if hnr > 10 else "Needs work"),
        },
        "stability": {
            "jitter_pct": round(jitter, 2),
            "shimmer_pct": round(shimmer, 2),
            "vocal_fry_pct": round(vocal_fry * 100, 1),
            "rating": "Stable" if jitter < 1.5 and shimmer < 5 else "Some instability",
        },
    }


def _get_devices(transcript: str) -> dict:
    """Detect all rhetorical devices: anaphora, epistrophe, tricolon, antithesis, rhetorical questions, climax."""
    if not transcript or len(transcript) < 50:
        return {
            "total_devices": 0,
            "anaphora": [],
            "epistrophe": [],
            "tricolon": [],
            "rhetorical_questions": [],
            "antithesis": [],
            "climax": [],
            "repetition_score": 0,
            "device_density": 0,
            "device_score": 0,
        }
    
    try:
        result = analyze_all_devices(transcript)
        
        word_count = len(transcript.split())
        device_count = result.get("device_count", 0)
        density = (device_count / word_count * 100) if word_count > 0 else 0
        
        return {
            "total_devices": device_count,
            "device_types_used": result.get("device_types_used", 0),
            "device_score": result.get("device_score", 0),
            "anaphora": result.get("anaphora", [])[:5],
            "epistrophe": result.get("epistrophe", [])[:5],
            "tricolon": result.get("tricolon", [])[:5],
            "rhetorical_questions": result.get("rhetorical_questions", [])[:5],
            "antithesis": result.get("antithesis", [])[:5],
            "climax": result.get("climax", [])[:5],
            "repetition_score": result.get("repetition_score", 0),
            "device_density": round(density, 2),
            "summary": result.get("summary", {}),
        }
    except Exception as e:
        logger.warning(f"Device detection failed: {e}")
        return {
            "total_devices": 0,
            "anaphora": [],
            "epistrophe": [],
            "tricolon": [],
            "rhetorical_questions": [],
            "antithesis": [],
            "climax": [],
            "repetition_score": 0,
            "device_density": 0,
            "device_score": 0,
        }


def _get_comparison(analysis: dict, db: SpeechDB, speech_id: int) -> dict:
    """Compare to great speeches."""
    wpm = analysis.get("wpm", 0)
    pitch_std = analysis.get("pitch_std_hz", 0)
    pitch_mean = analysis.get("pitch_mean_hz", 1)
    pitch_cv = (pitch_std / pitch_mean * 100) if pitch_mean > 0 else 0
    repetition = analysis.get("repetition_score", 0)
    
    # Find closest match
    closest = None
    closest_distance = float('inf')
    
    comparisons = []
    for key, speech in GREAT_SPEECHES.items():
        # Simple Euclidean distance on normalized metrics
        wpm_diff = abs(wpm - speech["wpm"]) / 50
        pitch_diff = abs(pitch_cv - speech["pitch_variation"]) / 20
        rep_diff = abs(repetition - speech["repetition"]) / 15
        
        distance = (wpm_diff + pitch_diff + rep_diff) / 3
        similarity = max(0, 100 - distance * 100)
        
        comparisons.append({
            "name": speech["name"],
            "similarity": round(similarity, 1),
            "their_wpm": speech["wpm"],
            "their_pitch_var": speech["pitch_variation"],
            "their_repetition": speech["repetition"],
        })
        
        if distance < closest_distance:
            closest_distance = distance
            closest = speech["name"]
    
    # Sort by similarity
    comparisons.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Get percentile rankings from benchmark
    try:
        benchmark = benchmark_speech(db, speech_id)
        percentiles = {
            "wpm": benchmark.get("wpm", {}).get("percentile", 50) if benchmark else 50,
            "pitch": benchmark.get("pitch_variation", {}).get("percentile", 50) if benchmark else 50,
            "repetition": benchmark.get("repetition", {}).get("percentile", 50) if benchmark else 50,
        }
    except:
        percentiles = {"wpm": 50, "pitch": 50, "repetition": 50}
    
    return {
        "most_similar": closest,
        "comparisons": comparisons[:3],  # Top 3
        "your_metrics": {
            "wpm": round(wpm, 1),
            "pitch_variation": round(pitch_cv, 1),
            "repetition": round(repetition, 1),
        },
        "percentiles": percentiles,
    }


def _rate_wpm(wpm: float) -> str:
    if wpm < 100:
        return "Deliberate - Good for emphasis"
    elif wpm < 130:
        return "Ideal oratory pace"
    elif wpm < 160:
        return "Conversational"
    else:
        return "Fast - May lose impact"


def _rate_pitch_variation(cv: float) -> str:
    if cv < 10:
        return "Monotone - Add variation"
    elif cv < 20:
        return "Moderate expression"
    elif cv < 35:
        return "Expressive - Ideal"
    else:
        return "Highly dynamic"


def _rate_hnr(hnr: float) -> str:
    if hnr > 20:
        return "Excellent clarity"
    elif hnr > 15:
        return "Good voice quality"
    elif hnr > 10:
        return "Average"
    else:
        return "Needs improvement"


def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B+"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C+"
    elif score >= 50:
        return "C"
    else:
        return "D"


def _score_to_label(score: float) -> str:
    if score >= 85:
        return "Master Orator"
    elif score >= 70:
        return "Skilled Speaker"
    elif score >= 55:
        return "Developing"
    else:
        return "Beginner"
