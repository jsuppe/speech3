"""
Pronunciation analysis API endpoints.

GET /v1/speeches/{id}/pronunciation - Get stored pronunciation features
POST /v1/speeches/{id}/pronunciation/analyze - Run on-demand analysis
POST /v1/speeches/{id}/pronunciation/deep - Run MFA deep analysis (phoneme-level)
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pronunciation import PronunciationAnalyzer


router = APIRouter(prefix="/v1/speeches", tags=["pronunciation"])

DB_PATH = Path(__file__).parent.parent / 'speechscore.db'
AUDIO_BASE = Path(__file__).parent.parent / 'audio'

# Lazy-loaded analyzer
_analyzer: Optional[PronunciationAnalyzer] = None


def get_analyzer() -> PronunciationAnalyzer:
    """Get or create the pronunciation analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = PronunciationAnalyzer()
    return _analyzer


class PronunciationResponse(BaseModel):
    """Pronunciation analysis response."""
    speech_id: int
    analyzed_at: Optional[str]
    mean_confidence: Optional[float]
    std_confidence: Optional[float]
    problem_ratio: Optional[float]
    words_analyzed: Optional[int]
    features: Optional[dict]
    word_analysis: Optional[list] = None  # Saved word-level analysis
    phoneme_analysis_at: Optional[str] = None  # When phoneme analysis was done


class WordAnalysis(BaseModel):
    """Per-word pronunciation analysis."""
    word: str
    confidence: float
    start_time: float
    end_time: float
    is_problem: bool
    expected_phonemes: Optional[list[str]] = None
    pronunciation_score: Optional[float] = None
    issues: Optional[list[str]] = None


class DeepAnalysisResponse(BaseModel):
    """Deep (MFA) pronunciation analysis response."""
    speech_id: int
    words: list[WordAnalysis]
    summary: dict
    phonemes: Optional[dict] = None  # Phoneme-level when MFA is used (contains 'words' list)


def get_audio_path(source_file: str) -> Optional[Path]:
    """Find the audio file path."""
    if not source_file:
        return None
    
    candidates = [
        AUDIO_BASE / source_file,
        Path(source_file),
        Path(__file__).parent.parent / source_file,
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    return None


def extract_audio_from_db(speech_id: int) -> Optional[Path]:
    """Extract audio from database to temp file for processing."""
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
    suffix = '.opus' if fmt == 'opus' else f'.{fmt}'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_data)
        temp_opus = f.name
    
    # If opus, convert to wav for analysis
    if fmt == 'opus':
        temp_wav = temp_opus.replace('.opus', '.wav')
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', temp_opus, '-ar', '16000', '-ac', '1', temp_wav],
                check=True, capture_output=True
            )
            Path(temp_opus).unlink()  # Remove opus temp
            return Path(temp_wav)
        except subprocess.CalledProcessError:
            return Path(temp_opus)  # Return opus if conversion fails
    
    return Path(temp_opus)


@router.get("/{speech_id}/pronunciation", response_model=PronunciationResponse)
async def get_pronunciation(speech_id: int):
    """Get stored pronunciation analysis for a speech."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            id, 
            pronunciation_mean_confidence,
            pronunciation_std_confidence,
            pronunciation_problem_ratio,
            pronunciation_words_analyzed,
            pronunciation_analyzed_at,
            pronunciation_features_json,
            word_analysis_json,
            phoneme_analysis_at
        FROM speeches 
        WHERE id = ?
    """, (speech_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Speech not found")
    
    features = None
    if row['pronunciation_features_json']:
        try:
            features = json.loads(row['pronunciation_features_json'])
        except:
            pass
    
    import json
    word_analysis = None
    if row['word_analysis_json']:
        try:
            word_analysis = json.loads(row['word_analysis_json'])
        except:
            pass
    
    return PronunciationResponse(
        speech_id=speech_id,
        analyzed_at=row['pronunciation_analyzed_at'],
        mean_confidence=row['pronunciation_mean_confidence'],
        std_confidence=row['pronunciation_std_confidence'],
        problem_ratio=row['pronunciation_problem_ratio'],
        words_analyzed=row['pronunciation_words_analyzed'],
        features=features,
        word_analysis=word_analysis,
        phoneme_analysis_at=row['phoneme_analysis_at']
    )


@router.post("/{speech_id}/pronunciation/analyze", response_model=PronunciationResponse)
async def analyze_pronunciation(speech_id: int, force: bool = False):
    """
    Run pronunciation analysis on a speech.
    
    Args:
        speech_id: ID of the speech to analyze
        force: If True, re-analyze even if already analyzed
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get speech info
    cursor.execute("SELECT source_file, pronunciation_analyzed_at FROM speeches WHERE id = ?", (speech_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Speech not found")
    
    # Check if already analyzed
    if row['pronunciation_analyzed_at'] and not force:
        conn.close()
        return await get_pronunciation(speech_id)
    
    # Find audio file
    audio_path = get_audio_path(row['source_file'])
    if not audio_path:
        conn.close()
        raise HTTPException(status_code=400, detail="Audio file not found")
    
    # Run analysis
    try:
        analyzer = get_analyzer()
        analysis = analyzer.analyze(str(audio_path))
        
        # Convert WordPronunciation objects to features dict
        word_data = [
            {
                'word': w.word,
                'confidence': w.whisper_confidence,
                'start_time': w.start,
                'end_time': w.end,
            }
            for w in analysis.words
        ]
        
        import numpy as np
        if word_data:
            confidences = [w['confidence'] for w in word_data]
            durations = [(w['end_time'] - w['start_time']) * 1000 for w in word_data]
            threshold = 0.6
            problem_count = sum(1 for c in confidences if c < threshold)
            duration_seconds = word_data[-1]['end_time'] if word_data else 0
            
            features = {
                'total_words': len(word_data),
                'problem_word_count': problem_count,
                'problem_word_ratio': problem_count / len(word_data) if word_data else 0,
                'mean_word_confidence': float(np.mean(confidences)),
                'std_word_confidence': float(np.std(confidences)),
                'words_per_minute': len(word_data) / (duration_seconds / 60) if duration_seconds > 0 else 0,
            }
        else:
            features = {}
        
        # Save to database
        cursor.execute("""
            UPDATE speeches SET
                pronunciation_mean_confidence = ?,
                pronunciation_std_confidence = ?,
                pronunciation_problem_ratio = ?,
                pronunciation_words_analyzed = ?,
                pronunciation_analyzed_at = ?,
                pronunciation_features_json = ?
            WHERE id = ?
        """, (
            features.get('mean_word_confidence'),
            features.get('std_word_confidence'),
            features.get('problem_word_ratio'),
            features.get('total_words'),
            datetime.utcnow().isoformat(),
            json.dumps(features),
            speech_id
        ))
        conn.commit()
        conn.close()
        
        return await get_pronunciation(speech_id)
        
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/{speech_id}/pronunciation/deep", response_model=DeepAnalysisResponse)
async def deep_pronunciation_analysis(
    speech_id: int, 
    use_mfa: bool = False,
    segment_start: Optional[float] = None,
    segment_end: Optional[float] = None
):
    """
    Run deep pronunciation analysis with word-level details.
    
    Args:
        speech_id: ID of the speech to analyze
        use_mfa: If True, use MFA for phoneme-level analysis (slower, ~40s)
        segment_start: Optional start time in seconds for segment analysis
        segment_end: Optional end time in seconds for segment analysis
    
    Returns:
        Word-by-word pronunciation analysis with confidence scores
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT source_file FROM speeches WHERE id = ?", (speech_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Speech not found")
    
    audio_path = get_audio_path(row['source_file'])
    temp_audio = None
    
    if not audio_path:
        # Try to extract from database
        audio_path = extract_audio_from_db(speech_id)
        if audio_path:
            temp_audio = audio_path  # Mark for cleanup
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Audio file not found: '{row['source_file']}'. The original recording may have been deleted."
            )
    
    try:
        analyzer = get_analyzer()
        
        # Run wav2vec2 analysis
        analysis = analyzer.analyze(str(audio_path))
        
        # Convert to response format
        words = []
        problem_threshold = 0.6
        
        for w in analysis.words:
            words.append(WordAnalysis(
                word=w.word,
                confidence=w.whisper_confidence,
                start_time=w.start,
                end_time=w.end,
                is_problem=w.pronunciation_score < problem_threshold,
                expected_phonemes=w.expected_phonemes,
                pronunciation_score=w.pronunciation_score,
                issues=w.issues
            ))
        
        # Filter by segment if specified
        if segment_start is not None or segment_end is not None:
            start = segment_start or 0
            end = segment_end or float('inf')
            words = [w for w in words if w.start_time >= start and w.end_time <= end]
        
        # Calculate summary
        confidences = [w.confidence for w in words]
        summary = {
            'total_words': len(words),
            'problem_words': sum(1 for w in words if w.is_problem),
            'mean_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'lowest_confidence_words': sorted(
                [{'word': w.word, 'confidence': w.confidence, 'time': w.start_time} 
                 for w in words if w.is_problem],
                key=lambda x: x['confidence']
            )[:10]
        }
        
        # MFA phoneme analysis (optional, slower)
        phonemes = None
        if use_mfa:
            try:
                from mfa_alignment import align_audio
                # Get transcript from words
                transcript = ' '.join(w.word for w in words)
                mfa_result = align_audio(str(audio_path), transcript)
                if mfa_result and 'words' in mfa_result:
                    phonemes = mfa_result
            except Exception as e:
                # MFA failed, continue without phonemes
                summary['mfa_error'] = str(e)
        
        # Save word analysis to database
        try:
            import json
            from datetime import datetime
            word_data = [{'word': w.word, 'confidence': w.confidence, 'start_time': w.start_time, 'end_time': w.end_time, 'is_problem': w.is_problem} for w in words]
            conn2 = sqlite3.connect(DB_PATH)
            conn2.execute(
                "UPDATE speeches SET word_analysis_json = ?, phoneme_analysis_at = ? WHERE id = ?",
                (json.dumps(word_data), datetime.now().isoformat(), speech_id)
            )
            conn2.commit()
            conn2.close()
        except Exception as save_err:
            # Don't fail the request if save fails
            pass
        
        return DeepAnalysisResponse(
            speech_id=speech_id,
            words=words,
            summary=summary,
            phonemes=phonemes
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deep analysis failed: {str(e)}")
    
    finally:
        # Clean up temp audio file if we extracted from DB
        if temp_audio and temp_audio.exists():
            try:
                temp_audio.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Accent Settings
# ---------------------------------------------------------------------------

SUPPORTED_ACCENTS = [
    {"code": "en-US", "name": "American English", "model": "wav2vec2-base-960h"},
    {"code": "en-GB", "name": "British English", "model": "wav2vec2-base-960h"},  # Future: UK model
    {"code": "en-AU", "name": "Australian English", "model": "wav2vec2-base-960h"},
    {"code": "en-IN", "name": "Indian English", "model": "wav2vec2-base-960h"},
    {"code": "en-ZA", "name": "South African English", "model": "wav2vec2-base-960h"},
]


class AccentSettings(BaseModel):
    """User's accent preference."""
    preferred_accent: str = "en-US"


@router.get("/pronunciation/accents")
async def list_supported_accents():
    """List all supported target accents for pronunciation scoring."""
    return {
        "accents": SUPPORTED_ACCENTS,
        "default": "en-US",
        "note": "Currently all accents use the same model. Accent-specific models coming soon."
    }


@router.get("/pronunciation/settings")
async def get_pronunciation_settings(user_id: int):
    """Get user's pronunciation settings including preferred accent."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT preferred_accent FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "preferred_accent": row[0] or "en-US",
        "supported_accents": SUPPORTED_ACCENTS
    }


@router.put("/pronunciation/settings")
async def update_pronunciation_settings(user_id: int, settings: AccentSettings):
    """Update user's pronunciation settings."""
    # Validate accent
    valid_codes = [a["code"] for a in SUPPORTED_ACCENTS]
    if settings.preferred_accent not in valid_codes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid accent. Supported: {valid_codes}"
        )
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE users SET preferred_accent = ? WHERE id = ?",
        (settings.preferred_accent, user_id)
    )
    conn.commit()
    
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    conn.close()
    return {"status": "updated", "preferred_accent": settings.preferred_accent}


@router.get("/pronunciation/samples")
async def get_pronunciation_samples():
    """Get reference audio samples demonstrating pronunciation quality grades (A+ through F).
    
    Returns a manifest with sample audio files for each grade level across categories:
    - clarity: General sentence clarity
    - consonants: Consonant-heavy tongue twisters
    - vowels: Vowel-focused phrases
    
    Audio files are served at /static/pronunciation_samples/<filename>
    """
    import json
    
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "frontend", "static", "pronunciation_samples", "manifest.json"
    )
    
    if not os.path.exists(manifest_path):
        raise HTTPException(
            status_code=404, 
            detail="Pronunciation samples not generated. Run tools/generate_pronunciation_samples.py"
        )
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    return manifest
