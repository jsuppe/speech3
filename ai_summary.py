"""
AI Summary Generation for Speech Analysis
Generates profile-specific summaries using local Ollama LLM.
"""

import sqlite3
import json
import httpx
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

# Profile-specific metric groups
PROFILE_METRICS = {
    "general": {
        "name": "General Speech",
        "metrics": ["wpm", "pitch_mean", "pitch_variation", "hnr", "lexical_diversity", 
                   "fk_grade_level", "filler_ratio", "pause_ratio"],
        "focus": "overall communication effectiveness, clarity, and engagement"
    },
    "oratory": {
        "name": "Oratory/Public Speaking",
        "metrics": ["wpm", "pitch_variation", "emphasis_score", "repetition_score",
                   "rhetorical_devices", "pause_effectiveness", "vocal_variety"],
        "focus": "persuasiveness, rhetorical impact, and audience engagement"
    },
    "pronunciation": {
        "name": "Pronunciation Practice",
        "metrics": ["pronunciation_clarity", "pronunciation_consistency", "pronunciation_accuracy",
                   "problem_words", "phoneme_analysis", "word_confidence_scores"],
        "focus": "pronunciation accuracy, clarity, and specific problem areas"
    },
    "presentation": {
        "name": "Presentation/Business",
        "metrics": ["wpm", "filler_ratio", "pause_ratio", "clarity_score",
                   "professional_vocabulary", "structure_coherence"],
        "focus": "professional delivery, clarity, and business communication"
    },
    "reading_fluency": {
        "name": "Reading Fluency",
        "metrics": ["wpm", "pause_patterns", "prosody", "accuracy", "expression"],
        "focus": "reading speed, accuracy, and natural expression"
    },
    "clinical": {
        "name": "Clinical Assessment",
        "metrics": ["jitter", "shimmer", "hnr", "vocal_fry_ratio", "pitch_stability",
                   "articulation_rate", "speech_rate_variability"],
        "focus": "voice health indicators and clinical speech patterns"
    }
}


def build_summary_prompt(profile: str, metrics: Dict[str, Any], transcript: str = None,
                         grammar_analysis: Dict = None, word_analysis: list = None) -> str:
    """Build a prompt for generating a profile-specific summary."""
    
    profile_info = PROFILE_METRICS.get(profile, PROFILE_METRICS["general"])
    
    prompt = f"""You are an expert speech coach providing feedback on a {profile_info['name']} recording.

Focus on: {profile_info['focus']}

Here are the analysis metrics:
"""
    
    # Add relevant metrics
    for key, value in metrics.items():
        if value is not None:
            if isinstance(value, float):
                prompt += f"- {key.replace('_', ' ').title()}: {value:.2f}\n"
            else:
                prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
    
    # Add pronunciation-specific data
    if profile == "pronunciation" and word_analysis:
        problem_words = [w for w in word_analysis if w.get('is_problem') or w.get('confidence', 1) < 0.7]
        if problem_words:
            prompt += f"\nProblem words detected ({len(problem_words)}):\n"
            for w in problem_words[:10]:  # Limit to 10
                prompt += f"- '{w.get('word')}' (confidence: {w.get('confidence', 0):.0%})\n"
    
    # Add grammar analysis
    if grammar_analysis:
        filler_count = grammar_analysis.get('filler_count', 0)
        filler_ratio = grammar_analysis.get('filler_ratio', 0)
        if filler_count > 0:
            prompt += f"\nFiller words: {filler_count} ({filler_ratio:.1%} of speech)\n"
        
        issues = grammar_analysis.get('issues', [])
        grammar_issues = [i for i in issues if i.get('type') not in ['filler', 'weak_word']]
        if grammar_issues:
            prompt += f"Grammar issues: {len(grammar_issues)}\n"
    
    # Add transcript snippet
    if transcript:
        snippet = transcript[:500] + "..." if len(transcript) > 500 else transcript
        prompt += f"\nTranscript excerpt:\n\"{snippet}\"\n"
    
    prompt += f"""
Write a concise 2-3 paragraph summary that:
1. Highlights the key strengths observed
2. Identifies specific areas for improvement
3. Gives 1-2 actionable tips for practice

Be specific and reference the actual metrics. Keep it encouraging but honest.
Do not use markdown formatting. Write in plain text."""
    
    return prompt


def generate_summary(speech_id: int, profile: str, db_path: str) -> Optional[str]:
    """Generate an AI summary for a speech recording with a specific profile."""
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get speech data
        cursor.execute("""
            SELECT s.*, t.text as transcript, t.word_count
            FROM speeches s
            LEFT JOIN transcriptions t ON t.speech_id = s.id
            WHERE s.id = ?
        """, (speech_id,))
        
        speech = cursor.fetchone()
        if not speech:
            conn.close()
            return None
        
        # Get analysis data
        cursor.execute("SELECT * FROM analyses WHERE speech_id = ?", (speech_id,))
        analysis = cursor.fetchone()
        
        # Build metrics dict
        metrics = {}
        if analysis:
            for key in analysis.keys():
                if key not in ['id', 'speech_id', 'created_at', 'updated_at']:
                    metrics[key] = analysis[key]
        
        # Add speech-level metrics
        for key in ['pronunciation_mean_confidence', 'pronunciation_std_confidence',
                   'pronunciation_problem_ratio', 'pronunciation_words_analyzed']:
            if speech[key] is not None:
                metrics[key] = speech[key]
        
        # Get grammar analysis
        grammar_analysis = None
        if speech['grammar_analysis_json']:
            try:
                grammar_analysis = json.loads(speech['grammar_analysis_json'])
            except:
                pass
        
        # Get word analysis for pronunciation
        word_analysis = None
        if speech['word_analysis_json']:
            try:
                word_analysis = json.loads(speech['word_analysis_json'])
            except:
                pass
        
        conn.close()
        
        # Build prompt
        prompt = build_summary_prompt(
            profile=profile,
            metrics=metrics,
            transcript=speech['transcript'],
            grammar_analysis=grammar_analysis,
            word_analysis=word_analysis
        )
        
        # Call Ollama
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500,
                }
            },
            timeout=60.0
        )
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get("response", "").strip()
            return summary
        else:
            logger.error(f"Ollama returned {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.exception(f"Failed to generate summary: {e}")
        return None


def save_summary(speech_id: int, profile: str, summary: str, db_path: str) -> bool:
    """Save an AI summary to the database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT OR REPLACE INTO speech_summaries (speech_id, profile, summary, created_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (speech_id, profile, summary))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.exception(f"Failed to save summary: {e}")
        return False


def get_summary(speech_id: int, profile: str, db_path: str) -> Optional[str]:
    """Get a saved summary from the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT summary FROM speech_summaries
            WHERE speech_id = ? AND profile = ?
        """, (speech_id, profile))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        logger.exception(f"Failed to get summary: {e}")
        return None


def generate_and_save_summary(speech_id: int, profile: str, db_path: str) -> Optional[str]:
    """Generate and save a summary, returning the result."""
    summary = generate_summary(speech_id, profile, db_path)
    if summary:
        save_summary(speech_id, profile, summary, db_path)
    return summary


def generate_all_profile_summaries(speech_id: int, db_path: str) -> Dict[str, str]:
    """Generate summaries for all profiles for a speech."""
    summaries = {}
    for profile in PROFILE_METRICS.keys():
        summary = generate_and_save_summary(speech_id, profile, db_path)
        if summary:
            summaries[profile] = summary
    return summaries


if __name__ == "__main__":
    # Test with a speech
    import sys
    db_path = "/home/melchior/speech3/speechscore.db"
    speech_id = int(sys.argv[1]) if len(sys.argv) > 1 else 99935
    profile = sys.argv[2] if len(sys.argv) > 2 else "pronunciation"
    
    print(f"Generating {profile} summary for speech {speech_id}...")
    summary = generate_and_save_summary(speech_id, profile, db_path)
    print(f"\n{summary}")
