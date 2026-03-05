"""
Pronunciation Comparison Analysis

Compares expected text (reference) with spoken audio to provide
word-by-word pronunciation accuracy feedback.

POST /v1/pronunciation/compare - Compare reference text with audio
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
import numpy as np

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

logger = logging.getLogger("speechscore.pronunciation_compare")

router = APIRouter(prefix="/v1/pronunciation", tags=["pronunciation"])


class WordComparison(BaseModel):
    """Single word comparison result"""
    expected: str
    spoken: Optional[str]  # What was actually said (None if omitted)
    status: str  # "correct", "mispronounced", "omitted", "inserted"
    confidence: float  # 0-1, how confident in the match
    position: int  # Position in reference text
    phoneme_issues: Optional[List[str]] = None  # Specific sound issues
    tip: Optional[str] = None  # Improvement tip


class ComparisonResult(BaseModel):
    """Full comparison result"""
    accuracy: float  # Overall accuracy 0-100
    grade: str  # A, B, C, etc.
    words: List[WordComparison]
    focus_areas: List[Dict]  # Patterns that need work
    summary: str  # Human-readable summary


def normalize_word(word: str) -> str:
    """Normalize a word for comparison"""
    return re.sub(r"[^a-z']", "", word.lower())


def get_phoneme_pattern(word: str) -> List[str]:
    """Get common pronunciation patterns in a word"""
    patterns = []
    word_lower = word.lower()
    
    # Common pronunciation challenges
    if 'th' in word_lower:
        patterns.append('th')
    if 'r' in word_lower:
        patterns.append('r')
    if word_lower.endswith('ed'):
        patterns.append('-ed ending')
    if word_lower.endswith('ly'):
        patterns.append('-ly ending')
    if word_lower.endswith('ing'):
        patterns.append('-ing ending')
    if word_lower.endswith('tion'):
        patterns.append('-tion ending')
    if 'ough' in word_lower:
        patterns.append('ough')
    if 'ch' in word_lower:
        patterns.append('ch')
    if 'sh' in word_lower:
        patterns.append('sh')
    if word_lower.startswith('wh'):
        patterns.append('wh-')
        
    return patterns


def calculate_word_similarity(expected: str, spoken: str) -> Tuple[float, List[str]]:
    """Calculate similarity between expected and spoken word, return score and issues"""
    if not expected or not spoken:
        return 0.0, ["word missing"]
    
    exp_norm = normalize_word(expected)
    spk_norm = normalize_word(spoken)
    
    if exp_norm == spk_norm:
        return 1.0, []
    
    # Use sequence matcher for fuzzy comparison
    similarity = SequenceMatcher(None, exp_norm, spk_norm).ratio()
    
    issues = []
    
    # Detect specific issues
    if len(exp_norm) != len(spk_norm):
        if len(spk_norm) < len(exp_norm):
            issues.append("syllable dropped")
        else:
            issues.append("extra syllable")
    
    # Check for common substitutions
    if 'th' in exp_norm and ('d' in spk_norm or 't' in spk_norm or 'f' in spk_norm):
        issues.append("th sound substitution")
    
    if exp_norm.endswith('ing') and not spk_norm.endswith('ing'):
        issues.append("-ing ending unclear")
        
    if exp_norm.endswith('ed') and not spk_norm.endswith('ed'):
        issues.append("-ed ending dropped")
    
    return similarity, issues


def generate_tip(word: str, issues: List[str]) -> Optional[str]:
    """Generate a pronunciation tip based on issues"""
    if not issues:
        return None
    
    tips = {
        "th sound substitution": f"Try placing your tongue between your teeth for the 'th' in '{word}'",
        "-ing ending unclear": f"Make sure to pronounce the final 'ng' sound in '{word}'",
        "-ed ending dropped": f"Don't forget the '-ed' ending - it may sound like 'd' or 't'",
        "syllable dropped": f"Speak '{word}' more slowly to catch all syllables",
        "extra syllable": f"'{word}' may have fewer syllables than you're saying",
    }
    
    for issue in issues:
        if issue in tips:
            return tips[issue]
    
    return f"Practice '{word}' slowly, focusing on each sound"


def align_words(reference_words: List[str], spoken_words: List[str]) -> List[WordComparison]:
    """
    Align reference words with spoken words using sequence matching.
    Handles insertions, deletions, and substitutions.
    """
    comparisons = []
    
    # Normalize for matching
    ref_norm = [normalize_word(w) for w in reference_words]
    spk_norm = [normalize_word(w) for w in spoken_words]
    
    # Use SequenceMatcher to find the best alignment
    matcher = SequenceMatcher(None, ref_norm, spk_norm)
    
    ref_idx = 0
    spk_idx = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Words match
            for k in range(i2 - i1):
                comparisons.append(WordComparison(
                    expected=reference_words[i1 + k],
                    spoken=spoken_words[j1 + k],
                    status="correct",
                    confidence=1.0,
                    position=i1 + k,
                ))
        elif tag == 'replace':
            # Words are different (mispronounced or different words)
            for k in range(max(i2 - i1, j2 - j1)):
                ref_word = reference_words[i1 + k] if i1 + k < i2 else None
                spk_word = spoken_words[j1 + k] if j1 + k < j2 else None
                
                if ref_word and spk_word:
                    similarity, issues = calculate_word_similarity(ref_word, spk_word)
                    tip = generate_tip(ref_word, issues)
                    
                    # Determine if it's mispronounced (similar) or wrong word (different)
                    if similarity > 0.5:
                        status = "mispronounced"
                    else:
                        status = "wrong_word"
                    
                    comparisons.append(WordComparison(
                        expected=ref_word,
                        spoken=spk_word,
                        status=status,
                        confidence=similarity,
                        position=i1 + k,
                        phoneme_issues=issues if issues else None,
                        tip=tip,
                    ))
                elif ref_word:
                    # Word was omitted
                    comparisons.append(WordComparison(
                        expected=ref_word,
                        spoken=None,
                        status="omitted",
                        confidence=0.0,
                        position=i1 + k,
                        tip=f"You skipped '{ref_word}'",
                    ))
                elif spk_word:
                    # Extra word was inserted
                    comparisons.append(WordComparison(
                        expected="",
                        spoken=spk_word,
                        status="inserted",
                        confidence=0.0,
                        position=-1,  # Not in reference
                    ))
        elif tag == 'delete':
            # Words in reference but not in spoken (omitted)
            for k in range(i2 - i1):
                comparisons.append(WordComparison(
                    expected=reference_words[i1 + k],
                    spoken=None,
                    status="omitted",
                    confidence=0.0,
                    position=i1 + k,
                    tip=f"You skipped '{reference_words[i1 + k]}'",
                ))
        elif tag == 'insert':
            # Words in spoken but not in reference (inserted)
            for k in range(j2 - j1):
                comparisons.append(WordComparison(
                    expected="",
                    spoken=spoken_words[j1 + k],
                    status="inserted",
                    confidence=0.0,
                    position=-1,
                ))
    
    return comparisons


def identify_focus_areas(comparisons: List[WordComparison]) -> List[Dict]:
    """Identify patterns that need practice"""
    pattern_counts = {}
    
    for comp in comparisons:
        if comp.status in ["mispronounced", "omitted"] and comp.expected:
            patterns = get_phoneme_pattern(comp.expected)
            for pattern in patterns:
                if pattern not in pattern_counts:
                    pattern_counts[pattern] = {"pattern": pattern, "count": 0, "examples": []}
                pattern_counts[pattern]["count"] += 1
                if len(pattern_counts[pattern]["examples"]) < 3:
                    pattern_counts[pattern]["examples"].append(comp.expected)
    
    # Sort by frequency and return top patterns
    focus_areas = sorted(pattern_counts.values(), key=lambda x: x["count"], reverse=True)
    return focus_areas[:5]  # Top 5 focus areas


def calculate_grade(accuracy: float) -> str:
    """Convert accuracy to letter grade"""
    if accuracy >= 95:
        return "A+"
    elif accuracy >= 90:
        return "A"
    elif accuracy >= 85:
        return "A-"
    elif accuracy >= 80:
        return "B+"
    elif accuracy >= 75:
        return "B"
    elif accuracy >= 70:
        return "B-"
    elif accuracy >= 65:
        return "C+"
    elif accuracy >= 60:
        return "C"
    elif accuracy >= 55:
        return "C-"
    elif accuracy >= 50:
        return "D"
    else:
        return "F"


def generate_summary(accuracy: float, focus_areas: List[Dict], total_words: int) -> str:
    """Generate a human-readable summary"""
    if accuracy >= 90:
        intro = "Excellent pronunciation!"
    elif accuracy >= 75:
        intro = "Good job!"
    elif accuracy >= 60:
        intro = "Nice effort!"
    else:
        intro = "Keep practicing!"
    
    if focus_areas:
        focus_text = f" Focus on: {', '.join(f['pattern'] for f in focus_areas[:2])}."
    else:
        focus_text = ""
    
    return f"{intro} You pronounced {int(accuracy)}% of {total_words} words correctly.{focus_text}"


def compare_pronunciation(reference_text: str, transcript: str) -> ComparisonResult:
    """
    Compare reference text with actual transcript.
    
    Args:
        reference_text: The text the user was supposed to read
        transcript: What the user actually said (from STT)
    
    Returns:
        ComparisonResult with word-by-word analysis
    """
    # Split into words
    reference_words = re.findall(r"[\w']+", reference_text)
    spoken_words = re.findall(r"[\w']+", transcript)
    
    if not reference_words:
        raise ValueError("Reference text is empty")
    
    # Align and compare
    comparisons = align_words(reference_words, spoken_words)
    
    # Calculate accuracy (correct words / reference words)
    correct_count = sum(1 for c in comparisons if c.status == "correct")
    total_reference = len(reference_words)
    accuracy = (correct_count / total_reference * 100) if total_reference > 0 else 0
    
    # Identify focus areas
    focus_areas = identify_focus_areas(comparisons)
    
    # Generate grade and summary
    grade = calculate_grade(accuracy)
    summary = generate_summary(accuracy, focus_areas, total_reference)
    
    return ComparisonResult(
        accuracy=round(accuracy, 1),
        grade=grade,
        words=comparisons,
        focus_areas=focus_areas,
        summary=summary,
    )


@router.post("/compare", response_model=ComparisonResult)
async def compare_pronunciation_endpoint(
    reference_text: str = Form(..., description="The text user was supposed to read"),
    transcript: str = Form(..., description="What the user actually said"),
):
    """
    Compare reference text with spoken transcript.
    
    This is a text-only comparison - the client should have already
    transcribed the audio using Whisper or similar.
    """
    try:
        result = compare_pronunciation(reference_text, transcript)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Comparison failed")


@router.post("/compare-with-audio", response_model=ComparisonResult)
async def compare_with_audio_endpoint(
    reference_text: str = Form(..., description="The text user was supposed to read"),
    audio: UploadFile = File(..., description="Audio recording"),
):
    """
    Compare reference text with audio recording.
    
    Transcribes the audio server-side then performs comparison.
    """
    import tempfile
    import os
    from pathlib import Path
    
    # Import transcription (lazy load to avoid startup cost)
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(status_code=503, detail="Whisper not available")
    
    # Save audio to temp file
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            content = await audio.read()
            f.write(content)
            temp_path = f.name
        
        # Transcribe
        model = WhisperModel("base", device="cuda", compute_type="float16")
        segments, info = model.transcribe(temp_path, language="en")
        transcript = " ".join(seg.text for seg in segments).strip()
        
        if not transcript:
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        # Compare
        result = compare_pronunciation(reference_text, transcript)
        return result
        
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
