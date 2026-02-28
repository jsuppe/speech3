"""
Presentation Analysis Module

Specialized analysis for presentations including:
- Opening/hook strength assessment
- Closing/conclusion impact
- Pacing consistency score
- Energy/engagement metrics
- Pause quality assessment
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


# Strong opening patterns (hooks)
STRONG_HOOKS = [
    r"^(imagine|picture this|what if|consider|think about)",  # Visualization
    r"^(did you know|here's a fact|surprisingly|remarkably)",  # Surprising fact
    r"^(let me (tell|share|ask) you)",  # Story/question lead
    r"^(today (i'll|we'll|i'm going|we're going))",  # Clear agenda
    r"^(the (biggest|most important|key|critical))",  # Significance
    r"^(have you ever|when was the last time)",  # Rhetorical question
    r"(once upon|there was a time|last year|recently)",  # Story hook
]

# Weak opening patterns
WEAK_OPENINGS = [
    r"^(so|okay|um|uh|well|alright)",  # Hesitant starts
    r"^(hi|hello|hey).*my name is",  # Bland intro (unless specific context)
    r"^(i'm (going to|gonna) talk about)",  # Too generic
    r"^(today's (topic|presentation|talk) is)",  # Boring statement
    r"^(thank you for (having|inviting))",  # Too formal/delayed
]

# Strong closing patterns
STRONG_CLOSINGS = [
    r"(in (conclusion|summary|closing))",  # Clear signal
    r"(the (key|main) takeaway)",  # Memorable point
    r"(i challenge you to|i urge you to|i invite you to)",  # Call to action
    r"(remember|don't forget|keep in mind)",  # Memorable anchor
    r"(imagine a (world|future) where)",  # Vision
    r"(thank you|questions\??)",  # Proper conclusion
    r"(let's make it happen|together we can)",  # Inspiring close
]

# Weak closing patterns
WEAK_CLOSINGS = [
    r"(that's (it|all|about it)|i guess that's)",  # Abrupt/uncertain
    r"(so (yeah|yep)|and yeah)",  # Trailing off
    r"(i think i covered|did i miss)",  # Uncertain
    r"(any questions\??)$",  # Only Q&A, no summary
]


def analyze_presentation(
    transcript: str,
    word_timings: Optional[List[Dict]] = None,
    wpm_segments: Optional[List[Dict]] = None,
    pitch_data: Optional[Dict] = None,
    pause_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Comprehensive presentation analysis.
    
    Args:
        transcript: Full text of the presentation
        word_timings: List of {word, start_time, end_time} dicts
        wpm_segments: List of {segment, wpm} for pacing analysis
        pitch_data: Pitch analysis results {mean, std, range, etc.}
        pause_data: Pause analysis results {count, total_duration, etc.}
    
    Returns:
        Dict with opening_analysis, closing_analysis, pacing_score,
        energy_score, pause_quality, and overall_presentation_score
    """
    result = {
        "opening_analysis": analyze_opening(transcript),
        "closing_analysis": analyze_closing(transcript),
        "pacing_score": calculate_pacing_score(wpm_segments),
        "energy_score": calculate_energy_score(pitch_data),
        "pause_quality": calculate_pause_quality(pause_data),
    }
    
    # Calculate overall presentation score (weighted average)
    scores = [
        (result["opening_analysis"]["score"], 0.20),
        (result["closing_analysis"]["score"], 0.20),
        (result["pacing_score"]["score"], 0.25),
        (result["energy_score"]["score"], 0.20),
        (result["pause_quality"]["score"], 0.15),
    ]
    
    result["overall_score"] = round(sum(s * w for s, w in scores), 1)
    
    return result


def analyze_opening(transcript: str, first_n_words: int = 50) -> Dict[str, Any]:
    """
    Analyze the opening/hook of a presentation.
    
    Returns:
        score (0-100), has_strong_hook, issues, suggestions
    """
    if not transcript:
        return {"score": 0, "has_strong_hook": False, "issues": [], "suggestions": []}
    
    # Get first N words
    words = transcript.split()
    opening_text = " ".join(words[:first_n_words]).lower()
    
    score = 70  # Start neutral
    has_strong_hook = False
    issues = []
    suggestions = []
    
    # Check for strong hooks (+15-20 points)
    for pattern in STRONG_HOOKS:
        if re.search(pattern, opening_text, re.IGNORECASE):
            has_strong_hook = True
            score += 20
            break
    
    # Check for weak openings (-15-25 points)
    for pattern in WEAK_OPENINGS:
        if re.search(pattern, opening_text, re.IGNORECASE):
            score -= 20
            issues.append("Opening may not grab attention immediately")
            suggestions.append("Consider starting with a question, story, or surprising fact")
            break
    
    # Check for engagement elements
    if "?" in opening_text:
        score += 5  # Questions engage audience
    
    if re.search(r"\byou\b", opening_text):
        score += 5  # Addressing audience directly
    
    # Cap score
    score = max(0, min(100, score))
    
    if not has_strong_hook and not issues:
        suggestions.append("Try opening with a hook: question, story, statistic, or bold statement")
    
    return {
        "score": score,
        "has_strong_hook": has_strong_hook,
        "opening_text": " ".join(words[:min(20, len(words))]) + "...",
        "issues": issues,
        "suggestions": suggestions,
    }


def analyze_closing(transcript: str, last_n_words: int = 50) -> Dict[str, Any]:
    """
    Analyze the closing/conclusion of a presentation.
    
    Returns:
        score (0-100), has_strong_close, has_call_to_action, issues, suggestions
    """
    if not transcript:
        return {"score": 0, "has_strong_close": False, "has_call_to_action": False, "issues": [], "suggestions": []}
    
    # Get last N words
    words = transcript.split()
    closing_text = " ".join(words[-last_n_words:]).lower()
    
    score = 70  # Start neutral
    has_strong_close = False
    has_call_to_action = False
    issues = []
    suggestions = []
    
    # Check for strong closings (+15-20 points)
    for pattern in STRONG_CLOSINGS:
        if re.search(pattern, closing_text, re.IGNORECASE):
            has_strong_close = True
            score += 15
            break
    
    # Check for call to action (+10 points)
    cta_patterns = [r"(i challenge|i urge|i invite|let's|we should|we can|we must)"]
    for pattern in cta_patterns:
        if re.search(pattern, closing_text, re.IGNORECASE):
            has_call_to_action = True
            score += 10
            break
    
    # Check for weak closings (-15-25 points)
    for pattern in WEAK_CLOSINGS:
        if re.search(pattern, closing_text, re.IGNORECASE):
            score -= 20
            issues.append("Closing may feel abrupt or uncertain")
            suggestions.append("End with a memorable takeaway, call to action, or inspiring statement")
            break
    
    # Cap score
    score = max(0, min(100, score))
    
    if not has_strong_close and not issues:
        suggestions.append("Consider summarizing key points and ending with a clear call to action")
    
    return {
        "score": score,
        "has_strong_close": has_strong_close,
        "has_call_to_action": has_call_to_action,
        "closing_text": "..." + " ".join(words[-min(20, len(words)):]),
        "issues": issues,
        "suggestions": suggestions,
    }


def calculate_pacing_score(wpm_segments: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Calculate pacing consistency score.
    
    Target: 130-150 WPM with low variation (CV < 20%)
    """
    if not wpm_segments:
        return {"score": 75, "avg_wpm": None, "variation": None, "issues": [], "suggestions": []}
    
    wpms = [seg.get("wpm", 0) for seg in wpm_segments if seg.get("wpm")]
    if not wpms:
        return {"score": 75, "avg_wpm": None, "variation": None, "issues": [], "suggestions": []}
    
    avg_wpm = sum(wpms) / len(wpms)
    
    # Calculate coefficient of variation
    if avg_wpm > 0:
        std_wpm = (sum((w - avg_wpm) ** 2 for w in wpms) / len(wpms)) ** 0.5
        cv = (std_wpm / avg_wpm) * 100
    else:
        cv = 0
    
    score = 100
    issues = []
    suggestions = []
    
    # Penalize for being outside 130-150 range
    if avg_wpm < 100:
        score -= 25
        issues.append(f"Speaking too slowly ({avg_wpm:.0f} WPM)")
        suggestions.append("Try to increase your pace slightly for better engagement")
    elif avg_wpm < 130:
        score -= 10
        issues.append(f"Pace slightly slow ({avg_wpm:.0f} WPM)")
    elif avg_wpm > 180:
        score -= 25
        issues.append(f"Speaking too fast ({avg_wpm:.0f} WPM)")
        suggestions.append("Slow down to let your audience absorb key points")
    elif avg_wpm > 150:
        score -= 10
        issues.append(f"Pace slightly fast ({avg_wpm:.0f} WPM)")
    
    # Penalize for high variation
    if cv > 30:
        score -= 20
        issues.append(f"Inconsistent pacing (CV: {cv:.0f}%)")
        suggestions.append("Practice maintaining a more consistent speaking rate")
    elif cv > 20:
        score -= 10
    
    score = max(0, min(100, score))
    
    return {
        "score": score,
        "avg_wpm": round(avg_wpm, 1),
        "variation_cv": round(cv, 1),
        "ideal_range": "130-150 WPM",
        "issues": issues,
        "suggestions": suggestions,
    }


def calculate_energy_score(pitch_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Calculate vocal energy score based on pitch variation and range.
    
    Target: 15-30% pitch coefficient of variation
    """
    if not pitch_data:
        return {"score": 75, "pitch_cv": None, "issues": [], "suggestions": []}
    
    pitch_mean = pitch_data.get("pitch_mean_hz", 0)
    pitch_std = pitch_data.get("pitch_std_hz", 0)
    pitch_range = pitch_data.get("pitch_range_hz", 0)
    
    if pitch_mean <= 0:
        return {"score": 75, "pitch_cv": None, "issues": [], "suggestions": []}
    
    pitch_cv = (pitch_std / pitch_mean) * 100
    
    score = 100
    issues = []
    suggestions = []
    
    # Check pitch variation
    if pitch_cv < 10:
        score -= 30
        issues.append("Monotone delivery detected")
        suggestions.append("Vary your pitch: go higher for excitement, lower for emphasis")
    elif pitch_cv < 15:
        score -= 15
        issues.append("Limited pitch variation")
        suggestions.append("Add more vocal variety to maintain audience engagement")
    elif pitch_cv > 35:
        score -= 10
        issues.append("Pitch may be too variable")
        suggestions.append("Slightly moderate pitch swings for more professional delivery")
    
    # Check pitch range
    if pitch_range < 50:
        score -= 10
        suggestions.append("Expand your vocal range for more expressive delivery")
    
    score = max(0, min(100, score))
    
    return {
        "score": score,
        "pitch_cv": round(pitch_cv, 1),
        "target_cv": "15-30%",
        "pitch_range_hz": round(pitch_range, 1) if pitch_range else None,
        "issues": issues,
        "suggestions": suggestions,
    }


def calculate_pause_quality(pause_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Calculate strategic pause usage score.
    
    Target: 15-25% of speaking time as pauses
    """
    if not pause_data:
        return {"score": 75, "pause_ratio": None, "issues": [], "suggestions": []}
    
    pause_ratio = pause_data.get("pause_ratio", 0)
    pause_count = pause_data.get("pause_count", 0)
    avg_pause_duration = pause_data.get("avg_pause_duration", 0)
    
    score = 100
    issues = []
    suggestions = []
    
    # Check pause ratio
    if pause_ratio < 0.10:
        score -= 25
        issues.append("Too few pauses")
        suggestions.append("Use strategic pauses before key points and after important statements")
    elif pause_ratio < 0.15:
        score -= 10
        issues.append("Could use more strategic pauses")
    elif pause_ratio > 0.35:
        score -= 25
        issues.append("Excessive pausing")
        suggestions.append("Long pauses can lose audience attention - maintain momentum")
    elif pause_ratio > 0.25:
        score -= 10
        issues.append("Slightly too many pauses")
    
    score = max(0, min(100, score))
    
    return {
        "score": score,
        "pause_ratio": round(pause_ratio * 100, 1) if pause_ratio else None,
        "target_ratio": "15-25%",
        "pause_count": pause_count,
        "avg_pause_ms": round(avg_pause_duration * 1000, 0) if avg_pause_duration else None,
        "issues": issues,
        "suggestions": suggestions,
    }


# Convenience function for quick analysis without all data
def quick_presentation_analysis(transcript: str) -> Dict[str, Any]:
    """
    Quick analysis using just the transcript.
    For full analysis, use analyze_presentation() with timing data.
    """
    return {
        "opening_analysis": analyze_opening(transcript),
        "closing_analysis": analyze_closing(transcript),
        "note": "For pacing, energy, and pause analysis, audio timing data is required.",
    }
