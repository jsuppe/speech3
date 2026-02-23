"""
SpeechScore Scoring Engine

Assigns letter grades (A-F) and numeric scores (0-100) to speeches
based on analysis metrics compared against the full database distribution.

Scoring approach:
- Percentile-based: each metric is scored by where the speech falls
  in the distribution of all analyzed speeches
- Profile-specific weights determine how much each metric contributes
- Composite score = weighted sum of individual metric scores
- Letter grades: A (90-100), B (80-89), C (70-79), D (60-69), F (<60)
"""

import logging
from typing import Optional

logger = logging.getLogger("speechscore.scoring")


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
# Each metric has:
#   - column: analysis table column name
#   - direction: "higher_better", "lower_better", or "optimal"
#   - optimal: target value for "optimal" direction metrics
#   - optimal_range: acceptable range around optimal (for bell-curve scoring)
#   - label: human-readable name
#   - unit: display unit

METRIC_DEFS = {
    "wpm": {
        "column": "wpm",
        "direction": "optimal",
        "optimal": 150.0,
        "optimal_range": 50.0,  # 100-200 is good range
        "label": "Speaking Rate",
        "unit": "WPM",
    },
    "pitch_variation": {
        "column": "pitch_std_hz",
        "direction": "higher_better",
        "label": "Pitch Variation",
        "unit": "Hz std",
    },
    "pitch_mean": {
        "column": "pitch_mean_hz",
        "direction": "optimal",
        "optimal": 170.0,
        "optimal_range": 80.0,
        "label": "Pitch Mean",
        "unit": "Hz",
    },
    "voice_quality": {
        "column": "hnr_db",
        "direction": "higher_better",
        "label": "Voice Quality (HNR)",
        "unit": "dB",
    },
    "jitter": {
        "column": "jitter_pct",
        "direction": "lower_better",
        "label": "Jitter",
        "unit": "%",
    },
    "shimmer": {
        "column": "shimmer_pct",
        "direction": "lower_better",
        "label": "Shimmer",
        "unit": "%",
    },
    "vocal_fry": {
        "column": "vocal_fry_ratio",
        "direction": "lower_better",
        "label": "Vocal Fry",
        "unit": "ratio",
    },
    "lexical_diversity": {
        "column": "lexical_diversity",
        "direction": "higher_better",
        "label": "Lexical Diversity",
        "unit": "",
    },
    "syntactic_complexity": {
        "column": "syntactic_complexity",
        "direction": "optimal",
        "optimal": 18.0,
        "optimal_range": 10.0,
        "label": "Syntactic Complexity",
        "unit": "depth",
    },
    "grade_level": {
        "column": "fk_grade_level",
        "direction": "optimal",
        "optimal": 10.0,
        "optimal_range": 5.0,
        "label": "Grade Level (FK)",
        "unit": "",
    },
    "repetition": {
        "column": "repetition_score",
        "direction": "context",  # higher can be good for oratory, bad for general
        "label": "Repetition Score",
        "unit": "",
    },
    "snr": {
        "column": "snr_db",
        "direction": "higher_better",
        "label": "Signal-to-Noise",
        "unit": "dB",
    },
    "articulation_rate": {
        "column": "articulation_rate",
        "direction": "optimal",
        "optimal": 150.0,
        "optimal_range": 40.0,
        "label": "Articulation Rate",
        "unit": "WPM",
    },
    "connectedness": {
        "column": "connectedness",
        "direction": "higher_better",
        "label": "Discourse Connectedness",
        "unit": "",
    },
    "pitch_range": {
        "column": "pitch_range_hz",
        "direction": "higher_better",
        "label": "Pitch Range",
        "unit": "Hz",
    },
    "vowel_space": {
        "column": "vowel_space_index",
        "direction": "higher_better",
        "label": "Vowel Space Index",
        "unit": "",
    },
    # Pronunciation metrics (from wav2vec2 analysis)
    "pronunciation_clarity": {
        "column": "pronunciation_mean_confidence",
        "direction": "higher_better",
        "label": "Pronunciation Clarity",
        "unit": "%",
        "scale": 100,  # Convert 0-1 to 0-100
    },
    "pronunciation_consistency": {
        "column": "pronunciation_std_confidence",
        "direction": "lower_better",
        "label": "Pronunciation Consistency",
        "unit": "std",
    },
    "pronunciation_accuracy": {
        "column": "pronunciation_problem_ratio",
        "direction": "lower_better",
        "label": "Pronunciation Accuracy",
        "unit": "% problems",
        "scale": 100,  # Convert ratio to percentage
    },
}


# ---------------------------------------------------------------------------
# Scoring profiles — weights per metric (must sum to ~1.0)
# ---------------------------------------------------------------------------

SCORING_PROFILES = {
    "general": {
        "description": "Balanced assessment across all speech dimensions",
        "weights": {
            "wpm": 0.12,
            "pitch_variation": 0.12,
            "voice_quality": 0.14,
            "jitter": 0.08,
            "shimmer": 0.05,
            "vocal_fry": 0.05,
            "lexical_diversity": 0.15,
            "syntactic_complexity": 0.10,
            "grade_level": 0.05,
            "repetition": 0.05,
            "connectedness": 0.0,  # DISABLED: 90% of recordings have 0, metric broken
            "pitch_range": 0.09,
        },
        "repetition_direction": "lower_better",  # for general, less repetition is cleaner
    },
    "oratory": {
        "description": "Rhetorical skill, delivery, and expressiveness — repetition is rewarded",
        "weights": {
            # FOCUSED: 6 metrics that matter for persuasive speaking
            "pitch_variation": 0.22,    # Expressiveness
            "pitch_range": 0.18,        # Dynamic range
            "wpm": 0.15,                # Pacing for impact
            "repetition": 0.20,         # Anaphora is a feature!
            "voice_quality": 0.15,      # Authority
            "connectedness": 0.10,      # Logical flow
            # EXCLUDED: lexical_diversity, syntactic_complexity, grade_level, jitter, shimmer, vocal_fry
        },
        "repetition_direction": "higher_better",  # orators use repetition deliberately
    },
    "casual": {
        "description": "Natural conversational speech — relaxed standards for filler and fry",
        "weights": {
            # FOCUSED: 5 metrics for natural conversation
            "wpm": 0.25,                # Natural pace
            "pitch_variation": 0.25,    # Engagement
            "voice_quality": 0.25,      # Clarity
            "vocal_fry": 0.15,          # Common in casual, lower weight
            "pitch_range": 0.10,        # Some expression
            # EXCLUDED: grade_level, syntactic, repetition, connectedness, lexical, jitter, shimmer
        },
        "repetition_direction": "lower_better",
    },
    "reading_fluency": {
        "description": "Audiobook and narration quality — clarity and consistent pacing",
        "weights": {
            # FOCUSED: 6 metrics for reading/narration
            "wpm": 0.25,                # Consistent pace
            "pitch_variation": 0.20,    # Expression/character voices
            "voice_quality": 0.20,      # Clarity
            "jitter": 0.15,             # Smooth voice
            "shimmer": 0.10,            # Consistent volume
            "pitch_range": 0.10,        # Expressive range
            # EXCLUDED: lexical, syntactic, repetition, grade_level (not your words)
        },
        "repetition_direction": "lower_better",
    },
    "presentation": {
        "description": "Professional business and educational presenting",
        "weights": {
            # FOCUSED: 7 metrics for professional delivery
            "wpm": 0.15,                # Clear pacing
            "pitch_variation": 0.18,    # Engagement
            "voice_quality": 0.15,      # Professionalism
            "grade_level": 0.12,        # Audience-appropriate
            "connectedness": 0.15,      # Logical structure
            "lexical_diversity": 0.15,  # Vocabulary
            "vocal_fry": 0.10,          # Professional sound
            # EXCLUDED: repetition, syntactic_complexity, extreme pitch_range, jitter, shimmer
        },
        "repetition_direction": "lower_better",
    },
    "clinical": {
        "description": "Voice health and pathology screening — focuses on acoustic markers",
        "weights": {
            # FOCUSED: 6 voice health metrics only
            "jitter": 0.22,             # Vocal cord stability
            "shimmer": 0.20,            # Amplitude consistency
            "voice_quality": 0.25,      # HNR - breathiness/hoarseness
            "vocal_fry": 0.15,          # Can indicate pathology
            "wpm": 0.10,                # Motor control
            "pitch_variation": 0.08,    # Basic expressiveness check
            # EXCLUDED: all language metrics (lexical, syntactic, grade_level, repetition, connectedness)
        },
        "repetition_direction": "lower_better",
    },
    "group": {
        "description": "Multi-party meeting analysis with speaker tracking",
        "weights": {},  # No scoring — focus on transcription, diarization, and LLM analysis
        "repetition_direction": "lower_better",
        "skip_scoring": True,  # Flag to skip numeric scoring
    },
    "pronunciation": {
        "description": "Pronunciation clarity and accent training — compare against target accent",
        "weights": {
            # PRIMARY: Pronunciation metrics (70%)
            "pronunciation_clarity": 0.35,      # Mean word confidence
            "pronunciation_consistency": 0.20,  # Consistent pronunciation
            "pronunciation_accuracy": 0.15,     # Low problem word ratio
            # SUPPORTING: Related delivery metrics (30%)
            "wpm": 0.10,                        # Speaking rate affects clarity
            "voice_quality": 0.10,              # HNR - clear voice
            "articulation_rate": 0.10,          # Clear articulation
        },
        "repetition_direction": "lower_better",
        "accent_aware": True,  # Flag: requires accent selection
        "supported_accents": [
            "en-US",    # American English (default wav2vec2 model)
            "en-GB",    # British English
            "en-AU",    # Australian English
            "en-IN",    # Indian English
            "en-ZA",    # South African English
            # Future: non-English accents with appropriate models
        ],
        "default_accent": "en-US",
    },
}


# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------

def _percentile_rank(value: float, distribution: list) -> float:
    """
    Compute the percentile rank of a value within a distribution.
    Returns 0.0-100.0.
    """
    if not distribution:
        return 50.0
    n = len(distribution)
    below = sum(1 for v in distribution if v < value)
    equal = sum(1 for v in distribution if v == value)
    return ((below + 0.5 * equal) / n) * 100.0


def _score_metric(
    value: float,
    distribution: list,
    direction: str,
    optimal: float = None,
    optimal_range: float = None,
) -> float:
    """
    Score a single metric value (0-100) based on direction and distribution.

    - higher_better: percentile rank IS the score
    - lower_better: 100 - percentile rank
    - optimal: bell-curve around optimal value, mapped via distribution width
    """
    if value is None or not distribution:
        return None

    if direction == "higher_better":
        return _percentile_rank(value, distribution)

    elif direction == "lower_better":
        return 100.0 - _percentile_rank(value, distribution)

    elif direction in ("optimal", "context"):
        if optimal is None:
            # Fallback: use median as optimal
            sorted_dist = sorted(distribution)
            optimal = sorted_dist[len(sorted_dist) // 2]
            optimal_range = (sorted_dist[-1] - sorted_dist[0]) * 0.25 if len(sorted_dist) > 1 else 1.0

        # Distance from optimal, normalized by range
        distance = abs(value - optimal)
        if optimal_range and optimal_range > 0:
            # Score drops as distance increases
            # At distance = 0: score = 95 (near perfect)
            # At distance = optimal_range: score ~= 60
            # At distance = 2*optimal_range: score ~= 25
            normalized = distance / optimal_range
            score = max(0.0, 100.0 * (1.0 - 0.4 * normalized ** 1.3))
            return min(100.0, score)
        else:
            return 50.0

    return 50.0


def _letter_grade(score: float) -> str:
    """
    Convert numeric score (0-100) to letter grade.
    
    Thresholds calibrated for bell-curve distribution:
    - A: top 10% (>= 68.5)
    - B: next 25% (>= 60)
    - C: middle 30% (>= 52)
    - D: next 25% (>= 41)
    - F: bottom 10% (< 41)
    """
    if score >= 75:
        return "A+"
    elif score >= 71:
        return "A"
    elif score >= 68.5:
        return "A-"
    elif score >= 65:
        return "B+"
    elif score >= 62:
        return "B"
    elif score >= 60:
        return "B-"
    elif score >= 57:
        return "C+"
    elif score >= 54:
        return "C"
    elif score >= 52:
        return "C-"
    elif score >= 48:
        return "D+"
    elif score >= 44:
        return "D"
    elif score >= 41:
        return "D-"
    else:
        return "F"


def _grade_color(grade: str) -> str:
    """Return a CSS-friendly color for a letter grade."""
    if grade.startswith("A"):
        return "#22c55e"  # green
    elif grade.startswith("B"):
        return "#3b82f6"  # blue
    elif grade.startswith("C"):
        return "#f59e0b"  # amber
    elif grade.startswith("D"):
        return "#f97316"  # orange
    else:
        return "#ef4444"  # red


# ---------------------------------------------------------------------------
# Distribution cache
# ---------------------------------------------------------------------------

_distribution_cache = {}
_cache_timestamp = 0
CACHE_TTL_SEC = 300  # refresh every 5 minutes


def _get_distributions(db) -> dict:
    """
    Load metric distributions from the database.
    Uses the latest analysis per speech (avoids counting duplicates).
    Cached for CACHE_TTL_SEC.
    """
    import time
    global _distribution_cache, _cache_timestamp

    now = time.time()
    if _distribution_cache and (now - _cache_timestamp) < CACHE_TTL_SEC:
        return _distribution_cache

    # Collect columns, separating analyses columns from speeches columns
    analysis_columns = set()
    speech_columns = set()  # pronunciation metrics are in speeches table
    for mdef in METRIC_DEFS.values():
        col = mdef["column"]
        if col.startswith("pronunciation_"):
            speech_columns.add(col)
        else:
            analysis_columns.add(col)
    
    analysis_col_list = sorted(analysis_columns)
    speech_col_list = sorted(speech_columns)

    # Get distributions from analyses table
    distributions = {}
    if analysis_col_list:
        col_str = ", ".join(f"a.{c}" for c in analysis_col_list)
        rows = db.conn.execute(f"""
            SELECT {col_str}
            FROM analyses a
            INNER JOIN (
                SELECT speech_id, MAX(id) as max_id
                FROM analyses
                GROUP BY speech_id
            ) latest ON a.id = latest.max_id
            WHERE a.wpm IS NOT NULL
        """).fetchall()
        
        for col in analysis_col_list:
            vals = []
            for row in rows:
                v = row[col]
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            distributions[col] = sorted(vals)
    
    # Get distributions from speeches table for pronunciation metrics
    if speech_col_list:
        col_str = ", ".join(speech_col_list)
        pron_rows = db.conn.execute(f"""
            SELECT {col_str}
            FROM speeches
            WHERE pronunciation_mean_confidence IS NOT NULL
        """).fetchall()
        
        for col in speech_col_list:
            vals = []
            for row in pron_rows:
                v = row[col]
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            distributions[col] = sorted(vals)

    _distribution_cache = distributions
    _cache_timestamp = now
    logger.info(f"Loaded scoring distributions: {len(distributions)} metrics")
    return distributions


# ---------------------------------------------------------------------------
# Summary Generation
# ---------------------------------------------------------------------------

def generate_summary(
    speech_title: str,
    speaker: str,
    profile: str,
    profile_description: str,
    composite_score: float,
    letter_grade: str,
    metric_scores: dict,
    category_scores: dict,
    strengths: list,
    improvements: list,
    duration_sec: float = None,
) -> str:
    """
    Generate a narrative summary paragraph explaining the speech analysis results.
    
    Returns a clear, readable summary suitable for display at the top of the results page.
    """
    # Determine overall assessment based on grade
    if letter_grade.startswith('A'):
        overall = "excellent"
        assessment = "demonstrates outstanding communication skills"
    elif letter_grade.startswith('B'):
        overall = "good"
        assessment = "shows solid communication abilities"
    elif letter_grade.startswith('C'):
        overall = "adequate"
        assessment = "demonstrates competent communication"
    elif letter_grade.startswith('D'):
        overall = "below average"
        assessment = "shows room for improvement in communication"
    else:
        overall = "needs significant work"
        assessment = "has significant areas requiring development"
    
    # Build the summary parts
    parts = []
    
    # Opening with title/speaker and overall score
    speaker_text = f" by {speaker}" if speaker and speaker.lower() != "unknown speaker" else ""
    title_text = speech_title if speech_title and speech_title.lower() != "untitled speech" else "This speech"
    
    duration_text = ""
    if duration_sec:
        mins = int(duration_sec // 60)
        secs = int(duration_sec % 60)
        if mins > 0:
            duration_text = f" ({mins}m {secs}s)"
        else:
            duration_text = f" ({secs}s)"
    
    parts.append(
        f"{title_text}{speaker_text}{duration_text} received an overall grade of **{letter_grade}** "
        f"({composite_score:.1f}/100), which {assessment}."
    )
    
    # Profile context
    parts.append(
        f"This analysis used the **{profile.replace('_', ' ').title()}** profile, "
        f"which emphasizes {profile_description.lower()}."
    )
    
    # Category highlights (top and bottom)
    if category_scores:
        sorted_cats = sorted(
            [(k, v) for k, v in category_scores.items() if v.get('score') is not None],
            key=lambda x: x[1]['score'],
            reverse=True
        )
        if sorted_cats:
            top_cat = sorted_cats[0]
            top_name = top_cat[1].get('label', top_cat[0].replace('_', ' ').title())
            top_score = top_cat[1]['score']
            top_grade = top_cat[1].get('grade', '')
            
            if len(sorted_cats) > 1:
                bottom_cat = sorted_cats[-1]
                bottom_name = bottom_cat[1].get('label', bottom_cat[0].replace('_', ' ').title())
                bottom_score = bottom_cat[1]['score']
                
                if top_score >= 75 and bottom_score < 65:
                    parts.append(
                        f"The strongest area is **{top_name}** ({top_grade}, {top_score:.0f}), "
                        f"while **{bottom_name}** ({bottom_score:.0f}) offers the most opportunity for growth."
                    )
                elif top_score >= 75:
                    parts.append(f"The analysis shows particular strength in **{top_name}** ({top_grade}, {top_score:.0f}).")
            elif top_score >= 75:
                parts.append(f"The analysis shows strength in **{top_name}** ({top_grade}, {top_score:.0f}).")
    
    # Specific strengths
    if strengths:
        strength_names = [s.split(':')[0].strip() for s in strengths[:2]]
        if len(strength_names) == 1:
            parts.append(f"Key strength: {strength_names[0]}.")
        elif len(strength_names) >= 2:
            parts.append(f"Key strengths include {strength_names[0]} and {strength_names[1]}.")
    
    # Improvement areas
    if improvements:
        improvement_names = [s.split(':')[0].strip() for s in improvements[:2]]
        if len(improvement_names) == 1:
            parts.append(f"For improvement, focus on {improvement_names[0]}.")
        elif len(improvement_names) >= 2:
            parts.append(f"For improvement, focus on {improvement_names[0]} and {improvement_names[1]}.")
    
    # Closing encouragement based on grade
    if letter_grade.startswith('A'):
        parts.append("This is a high-quality recording that demonstrates strong communication skills.")
    elif letter_grade.startswith('B'):
        parts.append("With some refinements in the noted areas, this could reach an excellent level.")
    elif letter_grade.startswith('C'):
        parts.append("Review the detailed metrics below and tap any metric for specific improvement tips.")
    elif letter_grade.startswith('D') or letter_grade.startswith('F'):
        parts.append("Consider the improvement tips for each metric and practice regularly to build skills.")
    
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Profiles hidden from user selection (still usable internally)
HIDDEN_PROFILES = {"general", "clinical"}


def get_available_profiles() -> dict:
    """Return user-visible scoring profiles with their descriptions."""
    return {
        name: {
            "description": profile["description"],
            "metrics": list(profile["weights"].keys()),
            "metric_count": len(profile["weights"]),
        }
        for name, profile in SCORING_PROFILES.items()
        if name not in HIDDEN_PROFILES
    }


def score_speech(
    db,
    speech_id: int,
    profile: str = "general",
) -> Optional[dict]:
    """
    Score a speech using the specified profile.

    Returns:
        {
            "speech_id": int,
            "profile": str,
            "composite_score": float,
            "letter_grade": str,
            "grade_color": str,
            "metric_scores": {metric_name: {score, value, percentile, weight, grade, ...}},
            "category_scores": {category: {score, grade}},
            "strengths": [str],
            "improvements": [str],
        }
    """
    if profile not in SCORING_PROFILES:
        return None

    profile_def = SCORING_PROFILES[profile]
    weights = profile_def["weights"]
    rep_direction = profile_def.get("repetition_direction", "lower_better")

    # Get speech analysis
    analysis = db.get_analysis(speech_id)
    if not analysis:
        return None
    
    # For pronunciation profile, also fetch pronunciation metrics from speeches table
    if profile == "pronunciation":
        pron_row = db.conn.execute("""
            SELECT pronunciation_mean_confidence, pronunciation_std_confidence,
                   pronunciation_problem_ratio, pronunciation_words_analyzed
            FROM speeches WHERE id = ?
        """, (speech_id,)).fetchone()
        if pron_row:
            analysis["pronunciation_mean_confidence"] = pron_row["pronunciation_mean_confidence"]
            analysis["pronunciation_std_confidence"] = pron_row["pronunciation_std_confidence"]
            analysis["pronunciation_problem_ratio"] = pron_row["pronunciation_problem_ratio"]
            analysis["pronunciation_words_analyzed"] = pron_row["pronunciation_words_analyzed"]

    # Get distributions
    distributions = _get_distributions(db)

    # Score each metric - iterate ALL metrics, not just those in profile
    # This allows the app to show greyed-out metrics that aren't scored
    metric_scores = {}
    weighted_sum = 0.0
    total_weight = 0.0

    # Core metrics to always show (subset of METRIC_DEFS that users care about)
    core_metrics = [
        "wpm", "pitch_variation", "pitch_range", "voice_quality", 
        "jitter", "shimmer", "vocal_fry", "lexical_diversity",
        "syntactic_complexity", "grade_level", "repetition", "connectedness"
    ]
    
    # Add pronunciation metrics for pronunciation profile
    if profile == "pronunciation":
        core_metrics.extend([
            "pronunciation_clarity", "pronunciation_consistency", "pronunciation_accuracy"
        ])

    for metric_name in core_metrics:
        mdef = METRIC_DEFS.get(metric_name)
        if not mdef:
            continue

        # Weight is 0 if metric is not in this profile
        weight = weights.get(metric_name, 0)

        col = mdef["column"]
        value = analysis.get(col)
        dist = distributions.get(col, [])

        if value is None or not dist:
            # No data available for this metric
            metric_scores[metric_name] = {
                "label": mdef["label"],
                "value": None,
                "unit": mdef["unit"],
                "score": None,
                "percentile": None,
                "grade": "N/A",
                "grade_color": "#6b7280",
                "weight": weight,
                "available": False,
            }
            continue

        value = float(value)

        # Determine direction (repetition is context-dependent)
        direction = mdef["direction"]
        if metric_name == "repetition":
            direction = rep_direction

        # Calculate raw percentile rank
        raw_pct = _percentile_rank(value, dist)

        # Calculate score (already direction-aware)
        score = _score_metric(
            value, dist, direction,
            optimal=mdef.get("optimal"),
            optimal_range=mdef.get("optimal_range"),
        )

        if score is not None:
            score = round(min(100.0, max(0.0, score)), 1)
            grade = _letter_grade(score)
            # Only add to weighted sum if this metric is in the profile
            if weight > 0:
                weighted_sum += score * weight
                total_weight += weight
        else:
            grade = "N/A"
        
        # Adjust percentile for display: should always mean "better than X%"
        # For lower_better, being in raw percentile 11.5 means 88.5% have worse (higher) values
        if direction == "lower_better":
            display_pct = 100.0 - raw_pct
        else:
            display_pct = raw_pct

        # Apply display scale if defined (e.g., 0-1 to 0-100 for percentages)
        display_value = value * mdef.get("scale", 1)
        
        metric_scores[metric_name] = {
            "label": mdef["label"],
            "value": round(display_value, 3) if display_value >= 1 else round(display_value, 3),
            "unit": mdef["unit"],
            "score": score,
            "percentile": round(display_pct, 1),  # Now direction-adjusted for UI
            "grade": grade,
            "grade_color": _grade_color(grade),
            "weight": weight,
            "available": True,
            "direction": direction,
        }

    # Composite score
    composite = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0
    overall_grade = _letter_grade(composite)

    # Category groupings
    categories = {
        "delivery": {
            "metrics": ["wpm", "pitch_variation", "pitch_range", "articulation_rate"],
            "label": "Delivery & Pace",
        },
        "voice": {
            "metrics": ["voice_quality", "jitter", "shimmer", "vocal_fry"],
            "label": "Voice Quality",
        },
        "language": {
            "metrics": ["lexical_diversity", "syntactic_complexity", "grade_level", "connectedness"],
            "label": "Language & Content",
        },
        "expression": {
            "metrics": ["repetition", "pitch_range", "pitch_variation"],
            "label": "Expressiveness",
        },
    }

    category_scores = {}
    for cat_key, cat_def in categories.items():
        cat_total = 0.0
        cat_weight = 0.0
        for m in cat_def["metrics"]:
            ms = metric_scores.get(m)
            if ms and ms.get("available") and ms.get("score") is not None:
                w = weights.get(m, 0)
                if w > 0:
                    cat_total += ms["score"] * w
                    cat_weight += w
        if cat_weight > 0:
            cat_score = round(cat_total / cat_weight, 1)
            category_scores[cat_key] = {
                "label": cat_def["label"],
                "score": cat_score,
                "grade": _letter_grade(cat_score),
                "grade_color": _grade_color(_letter_grade(cat_score)),
            }

    # Identify strengths and areas for improvement
    scored_metrics = [
        (name, data)
        for name, data in metric_scores.items()
        if data.get("available") and data.get("score") is not None
    ]
    scored_metrics.sort(key=lambda x: x[1]["score"], reverse=True)

    strengths = []
    improvements = []
    for name, data in scored_metrics[:3]:
        if data["score"] >= 75:
            strengths.append(f"{data['label']}: {data['grade']} ({data['score']})")
    for name, data in scored_metrics[-3:]:
        if data["score"] < 70:
            improvements.append(f"{data['label']}: {data['grade']} ({data['score']})")

    # Get speech metadata for summary
    speech = db.get_speech(speech_id)
    speech_title = speech.get("title", "Untitled Speech") if speech else "Untitled Speech"
    speaker = speech.get("speaker", "Unknown Speaker") if speech else "Unknown Speaker"
    duration_sec = speech.get("duration_sec") if speech else None
    
    # Generate narrative summary
    summary = generate_summary(
        speech_title=speech_title,
        speaker=speaker,
        profile=profile,
        profile_description=profile_def["description"],
        composite_score=composite,
        letter_grade=overall_grade,
        metric_scores=metric_scores,
        category_scores=category_scores,
        strengths=strengths,
        improvements=improvements,
        duration_sec=duration_sec,
    )

    # Include weights for ALL profiles so app can switch without API calls
    all_profile_weights = {
        name: {
            "description": p["description"],
            "weights": p["weights"],
            "repetition_direction": p.get("repetition_direction", "lower_better"),
        }
        for name, p in SCORING_PROFILES.items()
        if not p.get("skip_scoring")  # Exclude "group" profile
    }

    # Grade thresholds for client-side calculation
    grade_thresholds = [
        {"min": 75, "grade": "A+", "color": "#22c55e"},
        {"min": 71, "grade": "A", "color": "#22c55e"},
        {"min": 68.5, "grade": "A-", "color": "#22c55e"},
        {"min": 65, "grade": "B+", "color": "#3b82f6"},
        {"min": 62, "grade": "B", "color": "#3b82f6"},
        {"min": 60, "grade": "B-", "color": "#3b82f6"},
        {"min": 57, "grade": "C+", "color": "#f59e0b"},
        {"min": 54, "grade": "C", "color": "#f59e0b"},
        {"min": 52, "grade": "C-", "color": "#f59e0b"},
        {"min": 48, "grade": "D+", "color": "#f97316"},
        {"min": 44, "grade": "D", "color": "#f97316"},
        {"min": 41, "grade": "D-", "color": "#f97316"},
        {"min": 0, "grade": "F", "color": "#ef4444"},
    ]

    return {
        "speech_id": speech_id,
        "profile": profile,
        "profile_description": profile_def["description"],
        "composite_score": composite,
        "letter_grade": overall_grade,
        "grade_color": _grade_color(overall_grade),
        "metric_scores": metric_scores,
        "category_scores": category_scores,
        "strengths": strengths,
        "improvements": improvements,
        "metrics_scored": sum(1 for m in metric_scores.values() if m.get("available")),
        "metrics_total": len(weights),
        "summary": summary,
        "all_profile_weights": all_profile_weights,
        "grade_thresholds": grade_thresholds,
    }


def score_multiple(
    db,
    speech_ids: list,
    profile: str = "general",
) -> list:
    """Score multiple speeches for comparison. Returns list of score results."""
    return [
        score_speech(db, sid, profile)
        for sid in speech_ids
    ]


def benchmark_speech(
    db,
    speech_id: int,
) -> Optional[dict]:
    """
    Benchmark a speech against the entire database.
    Returns percentile rankings for all available metrics.
    """
    analysis = db.get_analysis(speech_id)
    if not analysis:
        return None

    distributions = _get_distributions(db)
    speech = db.get_speech(speech_id)

    rankings = {}
    for metric_name, mdef in METRIC_DEFS.items():
        col = mdef["column"]
        value = analysis.get(col)
        dist = distributions.get(col, [])

        if value is None or not dist:
            rankings[metric_name] = {
                "label": mdef["label"],
                "value": None,
                "unit": mdef["unit"],
                "percentile": None,
                "rank": None,
                "total": len(dist),
                "available": False,
            }
            continue

        value = float(value)
        pct = _percentile_rank(value, dist)

        # Rank: 1 = best
        sorted_dist = sorted(dist, reverse=True)
        rank = 1
        for v in sorted_dist:
            if v > value:
                rank += 1
            else:
                break

        rankings[metric_name] = {
            "label": mdef["label"],
            "value": round(value, 3),
            "unit": mdef["unit"],
            "percentile": round(pct, 1),
            "rank": rank,
            "total": len(dist),
            "direction": mdef["direction"],
            "available": True,
        }

    # Overall ranking by general profile
    general_score = score_speech(db, speech_id, "general")

    return {
        "speech_id": speech_id,
        "speech_title": speech.get("title", "") if speech else "",
        "speaker": speech.get("speaker", "") if speech else "",
        "rankings": rankings,
        "general_score": general_score["composite_score"] if general_score else None,
        "general_grade": general_score["letter_grade"] if general_score else None,
        "all_profiles": {
            p: score_speech(db, speech_id, p) for p in SCORING_PROFILES
        },
        "database_size": len(distributions.get("wpm", [])),
    }
