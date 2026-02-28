"""
Reader Analysis Module for SpeakFit Reader Mode.
Specialized analysis for reading practice: hesitations, self-corrections,
expression/intonation, and reference text comparison.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger("speechscore.reader")


@dataclass
class WordTiming:
    """Word with timing information from Whisper."""
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class HesitationResult:
    """A detected hesitation before a word."""
    word: str
    pause_duration: float
    position: int  # word index
    timestamp: float


@dataclass 
class SelfCorrectionResult:
    """A detected self-correction."""
    original: str
    correction: str
    correction_type: str  # 'repeat', 'restart', 'substitution'
    position: int
    timestamp: float


@dataclass
class ExpressionMetrics:
    """Expression/intonation analysis results."""
    expression_score: float  # 0-100
    pitch_variation_score: float
    pause_appropriateness: float
    question_intonation: float
    emphasis_score: float
    rating: str  # 'monotone', 'developing', 'natural', 'expressive'


# ============================================================================
# HESITATION DETECTION
# ============================================================================

def detect_hesitations(
    word_timings: List[Dict],
    threshold_ms: float = 500,
    min_pause_ms: float = 300
) -> Dict:
    """
    Detect hesitations (pauses before words) that indicate reading difficulty.
    
    Args:
        word_timings: List of word objects with 'word', 'start', 'end' keys
        threshold_ms: Pause duration (ms) to flag as significant hesitation
        min_pause_ms: Minimum pause to consider (filters out natural pauses)
    
    Returns:
        Dict with hesitations list, difficult words, and summary stats
    """
    if not word_timings or len(word_timings) < 2:
        return {
            "hesitations": [],
            "difficult_words": [],
            "hesitation_count": 0,
            "avg_hesitation_ms": 0,
            "hesitation_rate": 0,
        }
    
    hesitations = []
    
    for i in range(1, len(word_timings)):
        prev_word = word_timings[i - 1]
        curr_word = word_timings[i]
        
        # Calculate pause between words
        pause_ms = (curr_word.get('start', 0) - prev_word.get('end', 0)) * 1000
        
        # Filter: must be above minimum and below absurd (10 seconds)
        if min_pause_ms <= pause_ms <= 10000:
            word_text = curr_word.get('word', '').strip().lower()
            word_text = re.sub(r'[^\w]', '', word_text)  # Remove punctuation
            
            if pause_ms >= threshold_ms and word_text:
                hesitations.append({
                    "word": word_text,
                    "pause_ms": round(pause_ms),
                    "position": i,
                    "timestamp": round(curr_word.get('start', 0), 2),
                    "severity": _classify_hesitation_severity(pause_ms),
                })
    
    # Find most difficult words (repeated hesitations)
    word_counts = {}
    for h in hesitations:
        w = h["word"]
        if w not in word_counts:
            word_counts[w] = {"word": w, "count": 0, "total_pause_ms": 0}
        word_counts[w]["count"] += 1
        word_counts[w]["total_pause_ms"] += h["pause_ms"]
    
    difficult_words = sorted(
        word_counts.values(),
        key=lambda x: x["count"],
        reverse=True
    )[:10]
    
    total_words = len(word_timings)
    hesitation_rate = (len(hesitations) / total_words * 100) if total_words > 0 else 0
    avg_pause = np.mean([h["pause_ms"] for h in hesitations]) if hesitations else 0
    
    return {
        "hesitations": hesitations,
        "difficult_words": difficult_words,
        "hesitation_count": len(hesitations),
        "avg_hesitation_ms": round(avg_pause),
        "hesitation_rate": round(hesitation_rate, 1),
        "total_words": total_words,
    }


def _classify_hesitation_severity(pause_ms: float) -> str:
    """Classify hesitation severity based on pause duration."""
    if pause_ms < 500:
        return "mild"
    elif pause_ms < 1000:
        return "moderate"
    elif pause_ms < 2000:
        return "significant"
    else:
        return "extended"


# ============================================================================
# SELF-CORRECTION DETECTION
# ============================================================================

def detect_self_corrections(transcript: str, word_timings: List[Dict] = None) -> Dict:
    """
    Detect self-corrections in reading (repeats, restarts, substitutions).
    
    Patterns detected:
    - Repetitions: "the the cat" → repeated word
    - False starts: "I went to the— to the store"
    - Restarts: "The big... The big brown dog"
    
    Args:
        transcript: Full transcript text
        word_timings: Optional word timings for position info
    
    Returns:
        Dict with corrections list and summary
    """
    if not transcript:
        return {
            "corrections": [],
            "correction_count": 0,
            "correction_types": {},
            "self_monitoring_score": 0,
        }
    
    corrections = []
    words = transcript.lower().split()
    
    # Pattern 1: Immediate word repetitions ("the the")
    for i in range(len(words) - 1):
        w1 = re.sub(r'[^\w]', '', words[i])
        w2 = re.sub(r'[^\w]', '', words[i + 1])
        
        if w1 and w1 == w2 and len(w1) > 1:
            corrections.append({
                "type": "repeat",
                "text": f"{words[i]} {words[i+1]}",
                "position": i,
                "description": f"Repeated word: '{w1}'"
            })
    
    # Pattern 2: Phrase repetitions ("to the... to the")
    for window in [2, 3, 4]:
        for i in range(len(words) - window * 2):
            phrase1 = ' '.join(words[i:i+window])
            phrase2 = ' '.join(words[i+window:i+window*2])
            
            # Clean for comparison
            p1_clean = re.sub(r'[^\w\s]', '', phrase1)
            p2_clean = re.sub(r'[^\w\s]', '', phrase2)
            
            if p1_clean == p2_clean and len(p1_clean) > 3:
                corrections.append({
                    "type": "restart",
                    "text": f"{phrase1}... {phrase2}",
                    "position": i,
                    "description": f"Restarted phrase: '{p1_clean}'"
                })
    
    # Pattern 3: False starts with dashes/ellipses (from transcript punctuation)
    false_start_pattern = r'(\w+)[\-–—]+\s*(\w+)'
    for match in re.finditer(false_start_pattern, transcript.lower()):
        corrections.append({
            "type": "false_start",
            "text": match.group(0),
            "position": transcript[:match.start()].count(' '),
            "description": f"False start: '{match.group(1)}' → '{match.group(2)}'"
        })
    
    # Deduplicate corrections at same position
    seen_positions = set()
    unique_corrections = []
    for c in corrections:
        if c["position"] not in seen_positions:
            seen_positions.add(c["position"])
            unique_corrections.append(c)
    
    # Count by type
    type_counts = {}
    for c in unique_corrections:
        t = c["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    
    # Self-monitoring score (0-100)
    # More corrections = better self-monitoring (they're catching errors)
    # But too many suggests struggle
    word_count = len(words)
    correction_rate = len(unique_corrections) / word_count if word_count > 0 else 0
    
    if correction_rate < 0.01:
        monitoring_score = 70  # Few corrections - either fluent or not catching
    elif correction_rate < 0.03:
        monitoring_score = 90  # Healthy self-monitoring
    elif correction_rate < 0.06:
        monitoring_score = 75  # Active monitoring but some struggle
    else:
        monitoring_score = 50  # High correction rate suggests difficulty
    
    return {
        "corrections": unique_corrections,
        "correction_count": len(unique_corrections),
        "correction_types": type_counts,
        "self_monitoring_score": monitoring_score,
        "interpretation": _interpret_corrections(len(unique_corrections), word_count),
    }


def _interpret_corrections(count: int, word_count: int) -> str:
    """Provide interpretation of self-correction patterns."""
    if word_count == 0:
        return "No text to analyze"
    
    rate = count / word_count * 100
    
    if rate < 1:
        return "Minimal self-corrections — fluent reading or reading without self-monitoring"
    elif rate < 3:
        return "Healthy self-correction rate — actively monitoring comprehension"
    elif rate < 6:
        return "Moderate self-corrections — some challenging words or passages"
    else:
        return "Frequent self-corrections — text may be above comfort level"


# ============================================================================
# EXPRESSION / INTONATION ANALYSIS
# ============================================================================

def analyze_expression(
    pitch_data: Dict,
    pause_data: Dict,
    transcript: str,
    word_timings: List[Dict] = None
) -> Dict:
    """
    Analyze reading expression and natural intonation.
    
    Measures:
    - Pitch variation within sentences
    - Appropriate pauses at punctuation
    - Rising intonation for questions
    - Emphasis patterns
    
    Args:
        pitch_data: Pitch analysis from audio_analysis (pitch_mean, pitch_std, etc.)
        pause_data: Pause analysis (pause_count, pause_ratio, etc.)
        transcript: Full transcript
        word_timings: Optional word-level timing data
    
    Returns:
        Dict with expression metrics and rating
    """
    scores = {}
    
    # 1. Pitch Variation Score (0-100)
    pitch_cv = pitch_data.get('pitch_cv', 0)  # Coefficient of variation
    if pitch_cv < 5:
        scores['pitch_variation'] = 20  # Very monotone
    elif pitch_cv < 10:
        scores['pitch_variation'] = 50  # Somewhat flat
    elif pitch_cv < 20:
        scores['pitch_variation'] = 80  # Good variation
    else:
        scores['pitch_variation'] = 95  # Expressive
    
    # 2. Pause Appropriateness (0-100)
    # Check if pauses align with punctuation
    pause_ratio = pause_data.get('pause_ratio', 0)
    sentence_count = transcript.count('.') + transcript.count('!') + transcript.count('?')
    pause_count = pause_data.get('pause_count', 0)
    
    if sentence_count > 0:
        pauses_per_sentence = pause_count / sentence_count
        if 0.8 <= pauses_per_sentence <= 2.0:
            scores['pause_appropriateness'] = 90
        elif 0.5 <= pauses_per_sentence <= 3.0:
            scores['pause_appropriateness'] = 70
        else:
            scores['pause_appropriateness'] = 50
    else:
        scores['pause_appropriateness'] = 70  # Default
    
    # 3. Question Intonation (0-100)
    # For now, simplified check based on pitch patterns
    question_count = transcript.count('?')
    if question_count > 0:
        # If we have pitch data showing rising patterns, score higher
        pitch_range = pitch_data.get('pitch_range_semitones', 0)
        if pitch_range > 8:
            scores['question_intonation'] = 85
        elif pitch_range > 5:
            scores['question_intonation'] = 70
        else:
            scores['question_intonation'] = 55
    else:
        scores['question_intonation'] = 75  # No questions to evaluate
    
    # 4. Emphasis Score (0-100)
    # Based on intensity variation
    intensity_cv = pitch_data.get('intensity_cv', 0) if 'intensity_cv' in pitch_data else 15
    if intensity_cv < 5:
        scores['emphasis'] = 30  # Very flat
    elif intensity_cv < 12:
        scores['emphasis'] = 60
    elif intensity_cv < 20:
        scores['emphasis'] = 80
    else:
        scores['emphasis'] = 90
    
    # Calculate overall expression score
    overall = (
        scores['pitch_variation'] * 0.35 +
        scores['pause_appropriateness'] * 0.25 +
        scores['question_intonation'] * 0.15 +
        scores['emphasis'] * 0.25
    )
    
    # Determine rating
    if overall >= 85:
        rating = "expressive"
        description = "Reading with natural expression and appropriate intonation"
    elif overall >= 70:
        rating = "natural"
        description = "Good expression with room for more variation"
    elif overall >= 50:
        rating = "developing"
        description = "Some expression present; work on pitch and pause variation"
    else:
        rating = "monotone"
        description = "Reading is flat; focus on varying pitch and using pauses"
    
    return {
        "expression_score": round(overall),
        "pitch_variation_score": scores['pitch_variation'],
        "pause_appropriateness_score": scores['pause_appropriateness'],
        "question_intonation_score": scores['question_intonation'],
        "emphasis_score": scores['emphasis'],
        "rating": rating,
        "description": description,
        "tips": _get_expression_tips(scores),
    }


def _get_expression_tips(scores: Dict) -> List[str]:
    """Generate actionable tips based on expression scores."""
    tips = []
    
    if scores.get('pitch_variation', 100) < 60:
        tips.append("Try varying your voice pitch — go higher for excitement, lower for serious parts")
    
    if scores.get('pause_appropriateness', 100) < 70:
        tips.append("Pause at periods and commas — it helps listeners follow along")
    
    if scores.get('emphasis', 100) < 60:
        tips.append("Emphasize important words by saying them slightly louder or slower")
    
    if not tips:
        tips.append("Great expression! Keep reading with feeling")
    
    return tips


# ============================================================================
# REFERENCE TEXT COMPARISON
# ============================================================================

def compare_to_reference(
    transcript: str,
    reference_text: str,
    word_timings: List[Dict] = None
) -> Dict:
    """
    Compare actual reading to reference text.
    
    Calculates:
    - Overall accuracy percentage
    - Word-by-word comparison
    - Insertions, deletions, substitutions
    - Correctly read words
    
    Args:
        transcript: What was actually read (from STT)
        reference_text: What should have been read
        word_timings: Optional timing data for word positions
    
    Returns:
        Dict with accuracy metrics and word-by-word analysis
    """
    if not transcript or not reference_text:
        return {
            "accuracy_percent": 0,
            "word_comparison": [],
            "insertions": 0,
            "deletions": 0,
            "substitutions": 0,
            "correct": 0,
            "total_reference_words": 0,
        }
    
    # Normalize texts
    def normalize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.split()
    
    ref_words = normalize(reference_text)
    actual_words = normalize(transcript)
    
    # Use dynamic programming for alignment (Levenshtein-style)
    alignment = _align_words(ref_words, actual_words)
    
    # Count error types
    correct = sum(1 for a in alignment if a['status'] == 'correct')
    substitutions = sum(1 for a in alignment if a['status'] == 'substitution')
    insertions = sum(1 for a in alignment if a['status'] == 'insertion')
    deletions = sum(1 for a in alignment if a['status'] == 'deletion')
    
    total_ref = len(ref_words)
    accuracy = (correct / total_ref * 100) if total_ref > 0 else 0
    
    return {
        "accuracy_percent": round(accuracy, 1),
        "word_comparison": alignment,
        "correct": correct,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "total_reference_words": total_ref,
        "total_actual_words": len(actual_words),
        "accuracy_rating": _rate_accuracy(accuracy),
    }


def _align_words(reference: List[str], actual: List[str]) -> List[Dict]:
    """
    Align reference and actual words using edit distance.
    Returns list of alignment results.
    """
    m, n = len(reference), len(actual)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i-1] == actual[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    # Backtrack to get alignment
    alignment = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i-1] == actual[j-1]:
            alignment.append({
                "reference": reference[i-1],
                "actual": actual[j-1],
                "status": "correct",
            })
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append({
                "reference": reference[i-1],
                "actual": actual[j-1],
                "status": "substitution",
            })
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            alignment.append({
                "reference": None,
                "actual": actual[j-1],
                "status": "insertion",
            })
            j -= 1
        else:
            alignment.append({
                "reference": reference[i-1],
                "actual": None,
                "status": "deletion",
            })
            i -= 1
    
    alignment.reverse()
    return alignment


def _rate_accuracy(accuracy: float) -> str:
    """Rate accuracy percentage."""
    if accuracy >= 98:
        return "excellent"
    elif accuracy >= 95:
        return "very_good"
    elif accuracy >= 90:
        return "good"
    elif accuracy >= 80:
        return "developing"
    elif accuracy >= 70:
        return "needs_practice"
    else:
        return "challenging"


# ============================================================================
# COMPREHENSIVE READER ANALYSIS
# ============================================================================

def analyze_reading(
    transcript: str,
    word_timings: List[Dict],
    pitch_data: Dict,
    pause_data: Dict,
    reference_text: str = None,
    wpm: float = None
) -> Dict:
    """
    Comprehensive reading analysis combining all metrics.
    
    Args:
        transcript: Transcribed text
        word_timings: Word-level timing data from Whisper
        pitch_data: Pitch analysis results
        pause_data: Pause analysis results
        reference_text: Optional reference text for comparison
        wpm: Words per minute (if already calculated)
    
    Returns:
        Complete reader analysis with all metrics
    """
    results = {
        "profile": "reader",
        "analysis_version": "1.0",
    }
    
    # Hesitation analysis
    results["hesitations"] = detect_hesitations(word_timings)
    
    # Self-correction analysis
    results["self_corrections"] = detect_self_corrections(transcript, word_timings)
    
    # Expression analysis
    results["expression"] = analyze_expression(pitch_data, pause_data, transcript, word_timings)
    
    # Reference comparison (if provided)
    if reference_text:
        results["accuracy"] = compare_to_reference(transcript, reference_text, word_timings)
    
    # Reading speed
    if wpm is not None:
        results["reading_speed"] = {
            "wpm": round(wpm),
            "rating": _rate_reading_speed(wpm),
            "target_range": "150-200 WPM for fluent adult reading",
        }
    
    # Overall reading score
    results["overall_score"] = _calculate_reading_score(results)
    
    return results


def _rate_reading_speed(wpm: float) -> str:
    """Rate reading speed for adult readers."""
    if wpm < 100:
        return "slow"
    elif wpm < 150:
        return "developing"
    elif wpm < 200:
        return "fluent"
    elif wpm < 250:
        return "proficient"
    else:
        return "rapid"


def _calculate_reading_score(results: Dict) -> Dict:
    """Calculate overall reading score from component metrics."""
    scores = []
    weights = []
    
    # Expression (25%)
    if "expression" in results:
        scores.append(results["expression"]["expression_score"])
        weights.append(0.25)
    
    # Hesitation rate inverted (25%)
    if "hesitations" in results:
        hes_rate = results["hesitations"]["hesitation_rate"]
        # Lower hesitation = higher score
        hes_score = max(0, 100 - hes_rate * 10)
        scores.append(hes_score)
        weights.append(0.25)
    
    # Self-monitoring (15%)
    if "self_corrections" in results:
        scores.append(results["self_corrections"]["self_monitoring_score"])
        weights.append(0.15)
    
    # Accuracy (35%) - if available
    if "accuracy" in results:
        scores.append(results["accuracy"]["accuracy_percent"])
        weights.append(0.35)
    else:
        # Redistribute weights if no reference text
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
    
    if scores and weights:
        overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    else:
        overall = 0
    
    return {
        "score": round(overall),
        "grade": _score_to_grade(overall),
        "summary": _generate_reading_summary(results),
    }


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def _generate_reading_summary(results: Dict) -> str:
    """Generate human-readable summary of reading performance."""
    parts = []
    
    if "expression" in results:
        rating = results["expression"]["rating"]
        if rating in ["expressive", "natural"]:
            parts.append("Good vocal expression")
        elif rating == "developing":
            parts.append("Expression developing")
        else:
            parts.append("Work on vocal variety")
    
    if "hesitations" in results:
        count = results["hesitations"]["hesitation_count"]
        if count == 0:
            parts.append("smooth reading")
        elif count <= 3:
            parts.append("few hesitations")
        else:
            parts.append(f"{count} hesitations noted")
    
    if "accuracy" in results:
        acc = results["accuracy"]["accuracy_percent"]
        if acc >= 95:
            parts.append(f"{acc:.0f}% accurate")
        else:
            parts.append(f"{acc:.0f}% accuracy — keep practicing")
    
    return "; ".join(parts) if parts else "Analysis complete"


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test with sample data
    test_transcript = "The the cat sat on the mat. I went to the— to the store."
    test_reference = "The cat sat on the mat. I went to the store."
    
    test_timings = [
        {"word": "The", "start": 0.0, "end": 0.2},
        {"word": "the", "start": 0.8, "end": 1.0},  # Hesitation before repeat
        {"word": "cat", "start": 1.1, "end": 1.3},
        {"word": "sat", "start": 1.4, "end": 1.6},
        {"word": "on", "start": 1.7, "end": 1.8},
        {"word": "the", "start": 1.9, "end": 2.0},
        {"word": "mat", "start": 2.1, "end": 2.3},
    ]
    
    print("=== Hesitation Detection ===")
    hes = detect_hesitations(test_timings)
    print(f"Hesitations: {hes['hesitation_count']}")
    
    print("\n=== Self-Correction Detection ===")
    corr = detect_self_corrections(test_transcript)
    print(f"Corrections: {corr['correction_count']}")
    for c in corr['corrections']:
        print(f"  - {c['type']}: {c['description']}")
    
    print("\n=== Reference Comparison ===")
    acc = compare_to_reference(test_transcript, test_reference)
    print(f"Accuracy: {acc['accuracy_percent']}%")
    print(f"Substitutions: {acc['substitutions']}, Insertions: {acc['insertions']}")
