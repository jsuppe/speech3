"""
Pipeline stage definitions for all SpeakFit apps.
Generic enough to work across Present, Memory, Pronounce, etc.
"""

# Base stages (all profiles)
BASE_STAGES = [
    {"id": "upload", "name": "Uploading", "icon": "cloud_upload", "weight": 5},
    {"id": "convert", "name": "Processing Audio", "icon": "audiotrack", "weight": 5},
    {"id": "transcribe", "name": "Transcribing", "icon": "mic", "weight": 40},
    {"id": "text_analysis", "name": "Analyzing Text", "icon": "text_fields", "weight": 15},
    {"id": "audio_analysis", "name": "Analyzing Audio", "icon": "graphic_eq", "weight": 15},
    {"id": "scoring", "name": "Scoring", "icon": "score", "weight": 10},
    {"id": "persist", "name": "Saving Results", "icon": "save", "weight": 10},
]

# Profile-specific additional stages
PROFILE_STAGES = {
    "dementia": [
        {"id": "preposition_analysis", "name": "Analyzing Prepositions", "icon": "abc", "weight": 5},
        {"id": "repetition_detection", "name": "Detecting Repetitions", "icon": "repeat", "weight": 10},
        {"id": "word_finding", "name": "Checking Word-Finding", "icon": "search", "weight": 5},
        {"id": "cross_session", "name": "Cross-Session Tracking", "icon": "timeline", "weight": 5},
    ],
    "presentation": [
        {"id": "filler_detection", "name": "Detecting Fillers", "icon": "error_outline", "weight": 5},
        {"id": "pace_analysis", "name": "Analyzing Pace", "icon": "speed", "weight": 5},
        {"id": "confidence_scoring", "name": "Scoring Confidence", "icon": "trending_up", "weight": 5},
    ],
    "pronunciation": [
        {"id": "phoneme_analysis", "name": "Analyzing Phonemes", "icon": "record_voice_over", "weight": 10},
        {"id": "comparison", "name": "Comparing to Reference", "icon": "compare", "weight": 10},
    ],
    "reading_fluency": [
        {"id": "wpm_calculation", "name": "Calculating WPM", "icon": "timer", "weight": 5},
        {"id": "accuracy_check", "name": "Checking Accuracy", "icon": "check_circle", "weight": 5},
    ],
}


def get_stages_for_profile(profile: str) -> list:
    """Get all pipeline stages for a given profile."""
    stages = BASE_STAGES.copy()
    if profile in PROFILE_STAGES:
        # Insert profile-specific stages before scoring
        scoring_idx = next(i for i, s in enumerate(stages) if s["id"] == "scoring")
        for stage in reversed(PROFILE_STAGES[profile]):
            stages.insert(scoring_idx, stage)
    
    # Normalize weights to sum to 100
    total_weight = sum(s["weight"] for s in stages)
    for stage in stages:
        stage["percent"] = round(stage["weight"] / total_weight * 100)
    
    return stages


def get_progress_percent(profile: str, current_stage: str) -> int:
    """Get cumulative progress percentage for a given stage."""
    stages = get_stages_for_profile(profile)
    cumulative = 0
    for stage in stages:
        if stage["id"] == current_stage:
            return cumulative + stage["percent"] // 2  # Mid-stage
        cumulative += stage["percent"]
    return 100


# Export for API
PIPELINE_METADATA = {
    "profiles": ["general", "dementia", "presentation", "pronunciation", "reading_fluency", "clinical"],
    "base_stages": BASE_STAGES,
    "profile_stages": PROFILE_STAGES,
}
