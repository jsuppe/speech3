"""Frontend page routes — serves Jinja2 templates."""

import os
import sys
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Request, Query, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

SPEECH3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SPEECH3_DIR not in sys.path:
    sys.path.insert(0, SPEECH3_DIR)

from speech_db import SpeechDB
from scoring import score_speech, benchmark_speech, SCORING_PROFILES
from .auth import (
    get_current_user, require_auth, create_session, destroy_session,
    verify_google_token, is_email_allowed, SESSION_COOKIE
)

logger = logging.getLogger("speechscore.frontend")

# Google OAuth client ID (same as API)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "819724788424-r7p1tr8l7k2b2glsdn1ktj0o6k7cspb0.apps.googleusercontent.com")

# Human-readable metric explanations used by tooltips across all templates
METRIC_EXPLANATIONS = {
    # Key metrics grid
    "WPM": "Words Per Minute — how fast the speaker talks. Deliberate orators: 100–130. Conversational: 140–170. Fast readers/debaters: 180+.",
    "Pitch": "Average fundamental frequency (F0) of the voice in Hertz. Typical male: 85–180 Hz. Typical female: 165–255 Hz. Higher pitch often signals energy or stress.",
    "Jitter": "Cycle-to-cycle variation in vocal pitch. Low (<1%) = steady, healthy voice. High (>2%) may indicate vocal strain, aging, or neurological conditions.",
    "Shimmer": "Cycle-to-cycle variation in vocal loudness. Low (<5%) = smooth voice. High values suggest breathiness, hoarseness, or reduced vocal control.",
    "HNR": "Harmonics-to-Noise Ratio — how clear and resonant the voice sounds. Higher = cleaner. 20+ dB is excellent; below 7 dB suggests significant hoarseness.",
    "Vocal Fry": "Ratio of creaky, low-frequency vocal pulses. Common in casual modern speech (20–50%). Rare in trained speakers (<5%). Not inherently bad — it's a style marker.",
    "Lex Diversity": "Lexical Diversity — ratio of unique words to total words (type-token ratio). Higher = richer vocabulary. Gehrig's farewell: 0.84. Typical speech: 0.40–0.65.",
    "Complexity": "Syntactic Complexity — average parse-tree depth of sentences. Higher = more embedded clauses and complex grammar. Academic: 15+. Casual: 8–12.",
    "Grade Level": "Flesch-Kincaid Grade Level — the U.S. school grade needed to understand the text. Grade 6 = easy for most. Grade 12+ = college-level complexity.",
    "Repetition": "Rhetorical repetition score — measures deliberate reuse of phrases for emphasis. MLK's 'I Have a Dream': 24.5. Most speakers: 0–3. High scores signal oratorical craft.",
    "Sentiment": "Overall emotional tone of the speech — positive, negative, or neutral, determined by a RoBERTa language model trained on social media text.",
    "Quality": "Audio signal quality level (HIGH / MEDIUM / LOW / UNUSABLE) based on Signal-to-Noise Ratio. HIGH means clean recording suitable for reliable analysis.",

    # Scored metrics (may overlap with above)
    "pitch_std_hz": "Pitch Standard Deviation — how much the voice pitch varies. High variation = expressive, dynamic delivery. Low = monotone. Great speakers: 30–60 Hz std.",
    "pitch_range_hz": "The span between lowest and highest pitch used. Wider range = more vocal expression. Narrow range can sound flat or robotic.",
    "uptalk_ratio": "Fraction of phrases ending with rising pitch (like questions). Common in uncertain speech. Low in authoritative delivery.",
    "rate_variability_cv": "Coefficient of variation in speaking rate — how much speed changes between segments. Some variation is natural; too much may indicate disfluency.",
    "fluency_score": "Overall fluency rating combining pause patterns, hesitations, and flow. Higher = smoother, more connected speech.",
    "flesch_reading_ease": "Flesch Reading Ease — how easy the text is to read. 90–100 = very easy (5th grade). 30–50 = college level. 0–30 = very difficult.",
    "anaphora_count": "Count of anaphoric repetitions — phrases repeated at the start of consecutive sentences/clauses. A classic rhetorical device (e.g., 'We shall fight…').",
    "rhythm_regularity": "How evenly spaced the stressed syllables are. Higher = more metered, speech-like cadence. Lower = choppy or irregular pacing.",
    "snr_db": "Signal-to-Noise Ratio in decibels — how much louder the speech is versus background noise. 30+ dB = studio quality. <10 dB = very noisy.",
    "lexical_diversity": "Lexical Diversity — ratio of unique words to total words (type-token ratio). Higher = richer vocabulary.",
    "syntactic_complexity": "Average parse-tree depth of sentences. Higher = more embedded clauses and complex grammar.",
    "fk_grade_level": "Flesch-Kincaid Grade Level — the U.S. school grade needed to understand the text.",
    "repetition_score": "Rhetorical repetition score — measures deliberate reuse of phrases for emphasis.",
    "wpm": "Words Per Minute — how fast the speaker talks.",
    "pitch_mean_hz": "Average fundamental frequency (F0) of the voice in Hertz.",
    "jitter_pct": "Cycle-to-cycle variation in vocal pitch percentage.",
    "shimmer_pct": "Cycle-to-cycle variation in vocal loudness percentage.",
    "hnr_db": "Harmonics-to-Noise Ratio — how clear and resonant the voice sounds.",
    "vocal_fry_ratio": "Ratio of creaky, low-frequency vocal pulses in the speech.",
    "quality_level": "Audio signal quality level based on Signal-to-Noise Ratio.",

    # Scoring system display labels (matched by m.label in metric breakdown)
    "Speaking Rate": "Words Per Minute — how fast the speaker talks. Deliberate orators: 100–130. Conversational: 140–170. Fast readers/debaters: 180+.",
    "Pitch Variation": "Pitch Standard Deviation — how much the voice pitch varies. High variation = expressive, dynamic delivery. Low = monotone.",
    "Pitch Mean": "Average fundamental frequency (F0) of the voice in Hertz. Typical male: 85–180 Hz. Typical female: 165–255 Hz.",
    "Voice Quality (HNR)": "Harmonics-to-Noise Ratio — how clear and resonant the voice sounds. Higher = cleaner. 20+ dB is excellent; below 7 dB suggests significant hoarseness.",
    "Jitter": "Cycle-to-cycle variation in vocal pitch. Low (<1%) = steady, healthy voice. High (>2%) may indicate vocal strain, aging, or neurological conditions.",
    "Shimmer": "Cycle-to-cycle variation in vocal loudness. Low (<5%) = smooth voice. High values suggest breathiness, hoarseness, or reduced vocal control.",
    "Vocal Fry": "Ratio of creaky, low-frequency vocal pulses. Common in casual modern speech (20–50%). Rare in trained speakers (<5%). Not inherently bad — it's a style marker.",
    "Lexical Diversity": "Ratio of unique words to total words (type-token ratio). Higher = richer vocabulary. Gehrig's farewell: 0.84. Typical speech: 0.40–0.65.",
    "Syntactic Complexity": "Average parse-tree depth of sentences. Higher = more embedded clauses and complex grammar. Academic: 15+. Casual: 8–12.",
    "Grade Level (FK)": "Flesch-Kincaid Grade Level — the U.S. school grade needed to understand the text. Grade 6 = easy for most. Grade 12+ = college-level.",
    "Repetition Score": "Rhetorical repetition score — measures deliberate reuse of phrases for emphasis. MLK's 'I Have a Dream': 24.5. Most speakers: 0–3.",
    "Signal-to-Noise": "Signal-to-Noise Ratio in decibels — how much louder the speech is versus background noise. 30+ dB = studio quality. <10 dB = very noisy.",
    "Articulation Rate": "Speech rate excluding pauses — measures how quickly words are actually produced during active speaking segments.",
    "Discourse Connectedness": "How well ideas flow from one sentence to the next, measured by semantic similarity between adjacent sentences.",
    "Pitch Range": "The span between lowest and highest pitch used. Wider range = more vocal expression. Narrow range can sound flat or robotic.",
    "Vowel Space Index": "How distinctly vowels are articulated, based on formant frequencies. Larger vowel space = clearer pronunciation.",
    "Fluency Score": "Overall fluency rating combining pause patterns, hesitations, and flow. Higher = smoother, more connected speech.",
    "Reading Ease": "Flesch Reading Ease — how easy the text is to read. 90–100 = very easy (5th grade). 30–50 = college level. 0–30 = very difficult.",
}

TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

router = APIRouter()

_db = None


def get_db() -> SpeechDB:
    global _db
    if _db is None:
        _db = SpeechDB()
    return _db


# ---------------------------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------------------------

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = Query(None)):
    """Login page — redirects to landing page with login section."""
    # If already logged in, redirect to dashboard
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    
    # Redirect to landing page login section
    if error:
        return RedirectResponse(url=f"/?error={error}#login", status_code=302)
    return RedirectResponse(url="/#login", status_code=302)


@router.post("/auth/google/callback")
async def google_callback(request: Request, credential: str = Form(...)):
    """Handle Google Sign-In callback."""
    # Verify the Google token
    user_info = await verify_google_token(credential)
    
    if not user_info or not user_info.get("email"):
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid Google credential"}
        )
    
    email = user_info["email"]
    
    # Check if email is allowed
    if not is_email_allowed(email):
        logger.warning(f"Unauthorized login attempt from: {email}")
        return RedirectResponse(
            url="/?error=Access+denied.+Your+email+is+not+authorized.#login",
            status_code=302
        )
    
    # Create session
    session_token = create_session(email)
    
    # Redirect to dashboard with session cookie
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_token,
        httponly=True,
        secure=True,  # Only send over HTTPS
        samesite="lax",
        max_age=60 * 60 * 24 * 7,  # 1 week
    )
    
    logger.info(f"User logged in: {email}")
    return response


@router.get("/logout")
async def logout(request: Request):
    """Log out and clear session."""
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        destroy_session(token)
    
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(SESSION_COOKIE)
    return response


# ---------------------------------------------------------------------------
# Protected Dashboard Routes
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request, error: str = Query(None)):
    """Landing page — public info about SpeechScore with login option."""
    # If already logged in, redirect to dashboard
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    
    db = get_db()
    stats = db.stats()
    
    # Full metrics catalog with explanations, categories, and scoring info
    metrics_catalog = [
        # Voice Quality Metrics
        {
            "name": "Speaking Rate (WPM)",
            "category": "Voice",
            "description": "Words Per Minute — how fast the speaker talks.",
            "details": "Measures the overall pace of speech including pauses. Deliberate orators like MLK spoke at 100–130 WPM. Conversational speech is typically 140–170 WPM. Fast readers and debaters can exceed 180 WPM.",
            "scoring": "Scored on a bell curve. Moderate rates (120–160 WPM) score highest for general audiences. Very slow or very fast rates score lower.",
            "range": "60–220 WPM typical",
        },
        {
            "name": "Pitch Mean",
            "category": "Voice",
            "description": "Average fundamental frequency (F0) of the voice.",
            "details": "The base pitch of the speaker's voice measured in Hertz. Typical male voices range 85–180 Hz. Female voices typically range 165–255 Hz. Children have higher pitches.",
            "scoring": "Not directly scored — used as a baseline for other pitch metrics.",
            "range": "85–300 Hz typical",
        },
        {
            "name": "Pitch Variation",
            "category": "Voice",
            "description": "Standard deviation of pitch — how much the voice pitch varies.",
            "details": "High variation indicates expressive, dynamic delivery that keeps listeners engaged. Low variation suggests monotone delivery. Great speakers use pitch strategically for emphasis.",
            "scoring": "Higher variation scores better (up to a point). Monotone speech scores poorly.",
            "range": "10–80 Hz std typical",
        },
        {
            "name": "Pitch Range",
            "category": "Voice",
            "description": "The span between lowest and highest pitch used.",
            "details": "Wider pitch range indicates more vocal expression and emotional range. Narrow range can sound flat, robotic, or disengaged. Skilled speakers use their full range.",
            "scoring": "Wider ranges score higher, indicating vocal expressiveness.",
            "range": "50–300 Hz span typical",
        },
        {
            "name": "Jitter",
            "category": "Voice",
            "description": "Cycle-to-cycle variation in vocal pitch timing.",
            "details": "Measures micro-instabilities in vocal fold vibration. Low jitter (<1%) indicates a steady, healthy voice. High jitter (>2%) may indicate vocal strain, fatigue, aging, or neurological conditions.",
            "scoring": "Lower is better. Very high jitter indicates voice quality issues.",
            "range": "0.2–3% typical",
        },
        {
            "name": "Shimmer",
            "category": "Voice",
            "description": "Cycle-to-cycle variation in vocal loudness.",
            "details": "Measures amplitude stability of the voice. Low shimmer (<5%) indicates smooth voice production. High values suggest breathiness, hoarseness, or reduced vocal control.",
            "scoring": "Lower is better. High shimmer indicates voice quality problems.",
            "range": "1–10% typical",
        },
        {
            "name": "HNR (Harmonics-to-Noise)",
            "category": "Voice",
            "description": "How clear and resonant the voice sounds.",
            "details": "Measures the ratio of periodic (harmonic) to aperiodic (noise) components in the voice. Higher HNR means a cleaner, more resonant voice. 20+ dB is excellent. Below 7 dB suggests significant hoarseness.",
            "scoring": "Higher is better. Low HNR indicates hoarse or breathy voice.",
            "range": "5–25 dB typical",
        },
        {
            "name": "Vocal Fry",
            "category": "Voice",
            "description": "Ratio of creaky, low-frequency vocal pulses.",
            "details": "A vocal register characterized by low, creaky sounds. Common in casual modern speech (20–50% of utterances). Rare in classically trained speakers (<5%). Not inherently bad — it's a style marker that signals casualness or intimacy.",
            "scoring": "Context-dependent. High fry scores lower for formal presentations but may be neutral for casual speech.",
            "range": "0–60% typical",
        },
        {
            "name": "Uptalk Ratio",
            "category": "Voice",
            "description": "Fraction of phrases ending with rising pitch.",
            "details": "Rising intonation at the end of statements (making them sound like questions). Common in uncertain or seeking-approval speech. Less common in authoritative, confident delivery.",
            "scoring": "Lower scores better for authoritative speech. High uptalk can undermine perceived confidence.",
            "range": "0–40% typical",
        },
        {
            "name": "Articulation Rate",
            "category": "Voice",
            "description": "Speech rate excluding pauses.",
            "details": "Measures how quickly words are produced during active speaking segments (not counting silent pauses). Differs from WPM by isolating pure articulation speed.",
            "scoring": "Moderate rates score highest. Very fast articulation can reduce clarity.",
            "range": "3–7 syllables/sec typical",
        },
        {
            "name": "Rate Variability",
            "category": "Voice",
            "description": "How much speaking speed changes throughout.",
            "details": "Coefficient of variation in speaking rate across segments. Some variation is natural and engaging. Too much may indicate disfluency or poor planning. Too little sounds robotic.",
            "scoring": "Moderate variability scores best. Extremes score lower.",
            "range": "0.1–0.5 CV typical",
        },
        {
            "name": "Fluency Score",
            "category": "Voice",
            "description": "Overall smoothness combining pause patterns and flow.",
            "details": "Composite metric evaluating pause duration, frequency, filled pauses (um, uh), false starts, and overall speech flow. Higher scores indicate smoother, more connected speech.",
            "scoring": "Higher is better. Frequent hesitations and long pauses score lower.",
            "range": "0–100 scale",
        },
        # Text & Language Metrics
        {
            "name": "Lexical Diversity",
            "category": "Text",
            "description": "Ratio of unique words to total words.",
            "details": "Type-token ratio measuring vocabulary richness. Higher values indicate more varied word choice. Lou Gehrig's farewell speech scored 0.84 (exceptional). Typical speech ranges 0.40–0.65.",
            "scoring": "Higher is better, indicating richer vocabulary and less repetition of common words.",
            "range": "0.3–0.9 typical",
        },
        {
            "name": "Syntactic Complexity",
            "category": "Text",
            "description": "Average parse-tree depth of sentences.",
            "details": "Measures grammatical complexity via sentence structure analysis. Higher values indicate more embedded clauses, subordination, and complex grammar. Academic writing: 15+. Casual speech: 8–12.",
            "scoring": "Context-dependent. Higher complexity isn't always better — clarity matters.",
            "range": "5–20 depth typical",
        },
        {
            "name": "Grade Level (Flesch-Kincaid)",
            "category": "Text",
            "description": "U.S. school grade needed to understand the text.",
            "details": "Based on sentence length and syllable count. Grade 6 means readable by most adults. Grade 12+ indicates college-level complexity. Most successful speeches aim for grade 8–10.",
            "scoring": "Lower is often better for broad audiences. Very high levels may lose listeners.",
            "range": "4–16 grade level",
        },
        {
            "name": "Reading Ease",
            "category": "Text",
            "description": "Flesch Reading Ease score — how easy to understand.",
            "details": "Scale from 0–100. 90–100 is very easy (5th grade). 60–70 is standard. 30–50 is college level. 0–30 is very difficult (academic/legal). Inverse of grade level.",
            "scoring": "Higher is easier to understand. Best speeches often score 60–80.",
            "range": "0–100 scale",
        },
        {
            "name": "Mean Length of Utterance",
            "category": "Text",
            "description": "Average words per sentence or clause.",
            "details": "Simple measure of sentence length. Shorter MLU is easier to follow. Very long sentences can lose listeners. Great speakers vary their sentence length strategically.",
            "scoring": "Moderate lengths score best. Very long or very short sentences score lower.",
            "range": "8–25 words typical",
        },
        {
            "name": "Discourse Connectedness",
            "category": "Text",
            "description": "How well ideas flow from sentence to sentence.",
            "details": "Measured by semantic similarity between adjacent sentences. Higher scores indicate logical flow and coherent progression of ideas. Low scores suggest disjointed or random topic shifts.",
            "scoring": "Higher is better, indicating smooth logical flow.",
            "range": "0.2–0.8 similarity",
        },
        # Rhetorical Metrics
        {
            "name": "Repetition Score",
            "category": "Rhetoric",
            "description": "Deliberate reuse of phrases for emphasis.",
            "details": "Measures rhetorical repetition — intentional restatement for persuasive effect. MLK's 'I Have a Dream' scored 24.5. Most speakers score 0–3. High scores indicate oratorical craft.",
            "scoring": "Higher indicates more sophisticated rhetorical technique.",
            "range": "0–30 typical",
        },
        {
            "name": "Anaphora Count",
            "category": "Rhetoric",
            "description": "Phrases repeated at the start of consecutive clauses.",
            "details": "Classic rhetorical device where the same words begin multiple sentences or clauses. Churchill's 'We shall fight...' is a famous example. Creates rhythm and emphasis.",
            "scoring": "Presence of anaphora indicates rhetorical sophistication.",
            "range": "0–20 instances",
        },
        {
            "name": "Rhythm Regularity",
            "category": "Rhetoric",
            "description": "How evenly spaced stressed syllables are.",
            "details": "Measures the metered quality of speech. Higher regularity creates a more speech-like, almost poetic cadence. Lower values indicate choppy or irregular pacing.",
            "scoring": "Higher regularity can enhance memorability and impact.",
            "range": "0–1 scale",
        },
        # Sentiment Metrics
        {
            "name": "Sentiment",
            "category": "Sentiment",
            "description": "Overall emotional tone — positive, negative, or neutral.",
            "details": "Classified using a RoBERTa language model trained on diverse text. Indicates the predominant emotional valence of the speech content.",
            "scoring": "Not scored — descriptive classification.",
            "range": "Positive / Neutral / Negative",
        },
        {
            "name": "Sentiment Positive",
            "category": "Sentiment",
            "description": "Confidence score for positive sentiment.",
            "details": "Probability that the overall speech is positive in tone. Used with negative and neutral scores to determine final classification.",
            "scoring": "Higher indicates more confident positive classification.",
            "range": "0–1 probability",
        },
        {
            "name": "Sentiment Negative",
            "category": "Sentiment",
            "description": "Confidence score for negative sentiment.",
            "details": "Probability that the overall speech is negative in tone. High values indicate critical, sad, angry, or pessimistic content.",
            "scoring": "Higher indicates more confident negative classification.",
            "range": "0–1 probability",
        },
        {
            "name": "Sentiment Neutral",
            "category": "Sentiment",
            "description": "Confidence score for neutral sentiment.",
            "details": "Probability that the speech is emotionally neutral — factual, informational, or balanced without strong positive/negative lean.",
            "scoring": "Higher indicates more confident neutral classification.",
            "range": "0–1 probability",
        },
        # Audio Quality Metrics
        {
            "name": "Signal-to-Noise Ratio",
            "category": "Quality",
            "description": "How much louder speech is versus background noise.",
            "details": "Measured in decibels (dB). 30+ dB is studio quality. 20–30 dB is good. 10–20 dB is acceptable. Below 10 dB is very noisy and may affect analysis accuracy.",
            "scoring": "Higher is better. Low SNR may cause analysis to be flagged as unreliable.",
            "range": "-10 to 50 dB",
        },
        {
            "name": "Audio Quality Level",
            "category": "Quality",
            "description": "Overall recording quality classification.",
            "details": "Categorical assessment based on SNR and other factors. HIGH = clean studio recording. MEDIUM = acceptable with some noise. LOW = noisy but usable. UNUSABLE = too noisy for reliable analysis.",
            "scoring": "HIGH/MEDIUM preferred. LOW/UNUSABLE may affect metric reliability.",
            "range": "HIGH / MEDIUM / LOW / UNUSABLE",
        },
        # Formant Metrics
        {
            "name": "F1 Mean",
            "category": "Formants",
            "description": "First formant frequency — related to tongue height.",
            "details": "Acoustic resonance related to vowel quality and mouth openness. Higher F1 indicates more open vowels (like 'ah'). Lower F1 indicates closed vowels (like 'ee').",
            "scoring": "Not directly scored — used for vowel space analysis.",
            "range": "200–900 Hz typical",
        },
        {
            "name": "F2 Mean",
            "category": "Formants",
            "description": "Second formant frequency — related to tongue position.",
            "details": "Acoustic resonance related to front-back tongue position. Higher F2 indicates front vowels (like 'ee'). Lower F2 indicates back vowels (like 'oo').",
            "scoring": "Not directly scored — used for vowel space analysis.",
            "range": "800–2500 Hz typical",
        },
        {
            "name": "Vowel Space Index",
            "category": "Formants",
            "description": "How distinctly vowels are articulated.",
            "details": "Calculated from the spread of F1/F2 formant values. Larger vowel space indicates clearer pronunciation and better articulation. Reduced vowel space may indicate mumbling or slurred speech.",
            "scoring": "Larger vowel space scores better, indicating clearer articulation.",
            "range": "Relative scale",
        },
    ]
    
    return templates.TemplateResponse("landing.html", {
        "request": request,
        "stats": stats,
        "metrics_catalog": metrics_catalog,
        "google_client_id": GOOGLE_CLIENT_ID,
        "error": error,
    })


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard with stats overview (requires auth)."""
    # Require authentication
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
    db = get_db()
    stats = db.stats()
    recent = db.list_speeches(limit=10)

    # Get database-wide metric averages
    avg_metrics = db.conn.execute("""
        SELECT 
            COUNT(*) as total_analyzed,
            ROUND(AVG(wpm), 1) as avg_wpm,
            ROUND(AVG(articulation_rate), 1) as avg_articulation_rate,
            ROUND(AVG(pitch_mean_hz), 1) as avg_pitch_hz,
            ROUND(AVG(pitch_std_hz), 1) as avg_pitch_std,
            ROUND(AVG(pitch_range_hz), 1) as avg_pitch_range,
            ROUND(AVG(jitter_pct), 2) as avg_jitter_pct,
            ROUND(AVG(shimmer_pct), 2) as avg_shimmer_pct,
            ROUND(AVG(hnr_db), 1) as avg_hnr_db,
            ROUND(AVG(vocal_fry_ratio) * 100, 1) as avg_vocal_fry_pct,
            ROUND(AVG(uptalk_ratio) * 100, 1) as avg_uptalk_pct,
            ROUND(AVG(rate_variability_cv), 2) as avg_rate_var,
            ROUND(AVG(lexical_diversity), 2) as avg_lex_diversity,
            ROUND(AVG(fk_grade_level), 1) as avg_fk_grade,
            ROUND(AVG(flesch_reading_ease), 1) as avg_flesch_ease,
            ROUND(AVG(snr_db), 1) as avg_snr_db
        FROM analyses
        WHERE quality_level NOT IN ('UNUSABLE', '')
    """).fetchone()
    
    # Quality distribution
    quality_dist = db.conn.execute("""
        SELECT quality_level, COUNT(*) as count
        FROM analyses
        WHERE quality_level IS NOT NULL AND quality_level != ''
        GROUP BY quality_level
        ORDER BY count DESC
    """).fetchall()

    # Get metric distributions for charts
    rows = db.conn.execute("""
        SELECT s.id, s.title, s.speaker, s.category, s.products,
               a.wpm, a.pitch_mean_hz, a.vocal_fry_ratio,
               a.lexical_diversity, a.repetition_score, a.quality_level
        FROM speeches s
        LEFT JOIN analyses a ON a.speech_id = s.id
        ORDER BY s.id
    """).fetchall()

    speeches_data = []
    for r in rows:
        speeches_data.append({
            "id": r["id"],
            "title": r["title"],
            "speaker": r["speaker"] or "Unknown",
            "category": r["category"] or "",
            "wpm": r["wpm"],
            "pitch": r["pitch_mean_hz"],
            "fry": r["vocal_fry_ratio"],
            "lexdiv": r["lexical_diversity"],
            "rep": r["repetition_score"],
            "quality": r["quality_level"],
        })

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "recent": recent,
        "speeches_data": json.dumps(speeches_data),
        "metric_info": METRIC_EXPLANATIONS,
        "avg_metrics": dict(avg_metrics) if avg_metrics else {},
        "quality_dist": [{"level": q["quality_level"], "count": q["count"]} for q in quality_dist],
    })


@router.get("/speeches", response_class=HTMLResponse)
async def speech_list(
    request: Request,
    category: str = Query(None),
    product: str = Query(None),
    q: str = Query(None),
):
    """Browse all speeches with filters."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
    db = get_db()
    speeches = db.list_speeches(limit=500, category=category, product=product)

    # Text search filter
    if q:
        q_lower = q.lower()
        speeches = [
            s for s in speeches
            if q_lower in (s["title"] or "").lower()
            or q_lower in (s["speaker"] or "").lower()
            or q_lower in (s.get("description") or "").lower()
        ]

    # Get metrics for each speech
    for s in speeches:
        analysis = db.get_analysis(s["id"])
        s["metrics"] = {}
        if analysis:
            s["metrics"] = {
                k: analysis[k] for k in [
                    "wpm", "pitch_mean_hz", "vocal_fry_ratio",
                    "lexical_diversity", "quality_level", "sentiment",
                ] if analysis.get(k) is not None
            }

    # Get all categories and products for filter dropdowns
    all_categories = sorted(set(
        s["category"] for s in db.list_speeches(limit=500) if s.get("category")
    ))
    all_products = sorted(db.stats().get("products", {}).keys())

    return templates.TemplateResponse("speeches.html", {
        "request": request,
        "speeches": speeches,
        "categories": all_categories,
        "products": all_products,
        "current_category": category,
        "current_product": product,
        "current_q": q or "",
        "total": len(speeches),
    })


@router.get("/speeches/{speech_id}", response_class=HTMLResponse)
async def speech_detail(
    request: Request,
    speech_id: int,
    profile: str = Query("general"),
):
    """Detailed view of a single speech analysis."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
    db = get_db()
    speech = db.get_speech(speech_id)
    if not speech:
        return HTMLResponse("<h1>Speech not found</h1>", status_code=404)

    analysis = db.get_analysis(speech_id)
    transcription = db.get_transcription(speech_id)
    audio = db.get_audio(speech_id)
    spectrogram = db.get_spectrogram(speech_id, "combined")
    full_result = db.get_full_analysis(speech_id)

    # Get scoring data
    if profile not in SCORING_PROFILES:
        profile = "general"
    score_data = score_speech(db, speech_id, profile) if analysis else None
    available_profiles = list(SCORING_PROFILES.keys())

    return templates.TemplateResponse("speech_detail.html", {
        "request": request,
        "speech": speech,
        "analysis": analysis,
        "transcription": transcription,
        "has_audio": audio is not None,
        "audio_size": audio["size_bytes"] if audio else 0,
        "has_spectrogram": spectrogram is not None,
        "full_result": json.dumps(full_result, indent=2) if full_result else "{}",
        "score": score_data,
        "score_json": json.dumps(score_data) if score_data else "null",
        "current_profile": profile,
        "available_profiles": available_profiles,
        "metric_info": METRIC_EXPLANATIONS,
    })


@router.get("/speeches/{speech_id}/spectrogram.png")
async def frontend_spectrogram(speech_id: int):
    """Serve spectrogram image for frontend (no auth required)."""
    from fastapi.responses import Response
    db = get_db()
    spec = db.get_spectrogram(speech_id, spec_type="combined")
    if not spec:
        # Try any type
        for t in ("spectrogram", "mel", "loudness"):
            spec = db.get_spectrogram(speech_id, spec_type=t)
            if spec:
                break
    if not spec:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("No spectrogram", status_code=404)
    fmt = spec.get("format", "png")
    mime = {"png": "image/png", "jpg": "image/jpeg"}.get(fmt, "image/png")
    return Response(content=spec["data"], media_type=mime,
                    headers={"Cache-Control": "public, max-age=3600"})


@router.get("/speeches/{speech_id}/audio.opus")
async def frontend_audio(speech_id: int):
    """Serve audio for frontend player (no auth required)."""
    from fastapi.responses import Response
    db = get_db()
    audio = db.get_audio(speech_id)
    if not audio:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("No audio", status_code=404)
    return Response(content=audio["data"], media_type="audio/opus",
                    headers={"Cache-Control": "public, max-age=3600"})


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload page for new audio analysis."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
    return templates.TemplateResponse("upload.html", {
        "request": request,
    })


@router.get("/methodology", response_class=HTMLResponse)
async def methodology_page(request: Request):
    """Scoring methodology and metrics documentation."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
    # Check if user is admin (for pipeline internals section)
    is_admin = user.get("email", "").lower() == "jon.suppe@gmail.com"
    
    from scoring import METRIC_DEFS, SCORING_PROFILES
    db = get_db()
    stats = db.stats()
    return templates.TemplateResponse("methodology.html", {
        "request": request,
        "metric_defs": METRIC_DEFS,
        "metric_info": METRIC_EXPLANATIONS,
        "profiles": SCORING_PROFILES,
        "total_speeches": stats["speeches"],
        "total_analyses": stats["analyses"],
        "is_admin": is_admin,
    })


@router.get("/compare", response_class=HTMLResponse)
async def compare_page(
    request: Request,
    ids: str = Query(None),
    profile: str = Query("general"),
):
    """Side-by-side speech comparison with scoring."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
    db = get_db()
    speeches = db.list_speeches(limit=500)

    if profile not in SCORING_PROFILES:
        profile = "general"

    compared = []
    if ids:
        id_list = [int(x.strip()) for x in ids.split(",") if x.strip().isdigit()]
        for sid in id_list[:4]:  # max 4 comparisons
            speech = db.get_speech(sid)
            if speech:
                analysis = db.get_analysis(sid)
                transcription = db.get_transcription(sid)
                score_data = score_speech(db, sid, profile) if analysis else None
                compared.append({
                    "speech": speech,
                    "analysis": analysis,
                    "transcription": transcription,
                    "score": score_data,
                })

    return templates.TemplateResponse("compare.html", {
        "request": request,
        "speeches": speeches,
        "compared": compared,
        "metric_info": METRIC_EXPLANATIONS,
        "compared_json": json.dumps([
            {
                "id": c["speech"]["id"],
                "title": c["speech"]["title"],
                "score": c.get("score"),
                "metrics": {
                    k: c["analysis"][k] for k in [
                        "wpm", "pitch_mean_hz", "pitch_std_hz", "jitter_pct", "shimmer_pct",
                        "hnr_db", "vocal_fry_ratio", "lexical_diversity", "syntactic_complexity",
                        "fk_grade_level", "repetition_score",
                    ] if c.get("analysis") and c["analysis"].get(k) is not None
                } if c.get("analysis") else {},
            }
            for c in compared
        ]) if compared else "[]",
        "current_ids": ids or "",
        "current_profile": profile,
        "available_profiles": list(SCORING_PROFILES.keys()),
    })


@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Admin dashboard with platform statistics. Restricted to jon.suppe@gmail.com."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
    # Only allow Jon
    if user.get("email", "").lower() != "jon.suppe@gmail.com":
        return HTMLResponse("<h1>Access Denied</h1><p>Admin access is restricted.</p>", status_code=403)
    
    db = get_db()
    from datetime import datetime, timedelta
    
    # Basic stats
    total_users = db.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_recordings = db.conn.execute("SELECT COUNT(*) FROM speeches WHERE user_id IS NOT NULL").fetchone()[0]
    
    # Active users (7 and 30 days)
    now = datetime.utcnow()
    week_ago = (now - timedelta(days=7)).isoformat()
    month_ago = (now - timedelta(days=30)).isoformat()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    week_start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    
    active_7d = db.conn.execute("""
        SELECT COUNT(DISTINCT user_id) FROM speeches 
        WHERE user_id IS NOT NULL AND created_at >= ?
    """, (week_ago,)).fetchone()[0]
    
    active_30d = db.conn.execute("""
        SELECT COUNT(DISTINCT user_id) FROM speeches 
        WHERE user_id IS NOT NULL AND created_at >= ?
    """, (month_ago,)).fetchone()[0]
    
    # Recordings today and this week
    recordings_today = db.conn.execute("""
        SELECT COUNT(*) FROM speeches WHERE user_id IS NOT NULL AND created_at >= ?
    """, (today_start,)).fetchone()[0]
    
    recordings_week = db.conn.execute("""
        SELECT COUNT(*) FROM speeches WHERE user_id IS NOT NULL AND created_at >= ?
    """, (week_start,)).fetchone()[0]
    
    # Storage (audio files)
    audio_dir = Path("/home/jsuppe/speech3/audio")
    try:
        storage_bytes = sum(f.stat().st_size for f in audio_dir.rglob("*") if f.is_file()) if audio_dir.exists() else 0
    except:
        storage_bytes = 0
    storage_mb = storage_bytes / (1024 * 1024)
    
    # Average score (from analyses with scores)
    avg_score_row = db.conn.execute("""
        SELECT AVG(
            (COALESCE(lexical_diversity, 0.5) * 20) +
            (COALESCE(hnr_db, 15) / 25 * 20) +
            (MIN(MAX(wpm, 100), 160) - 100) / 60 * 20 +
            (20 - MIN(COALESCE(fk_grade_level, 10), 20)) +
            (COALESCE(pitch_std_hz, 30) / 60 * 20)
        ) as avg_score
        FROM analyses WHERE wpm IS NOT NULL
    """).fetchone()
    avg_score = avg_score_row[0] if avg_score_row and avg_score_row[0] else None
    
    # Top users by recordings
    top_users_rows = db.conn.execute("""
        SELECT u.*, COUNT(s.id) as speech_count,
               AVG(CASE WHEN a.wpm IS NOT NULL THEN
                   (COALESCE(a.lexical_diversity, 0.5) * 20) +
                   (COALESCE(a.hnr_db, 15) / 25 * 20) +
                   (MIN(MAX(a.wpm, 100), 160) - 100) / 60 * 20
               END) as avg_score
        FROM users u
        LEFT JOIN speeches s ON s.user_id = u.id
        LEFT JOIN analyses a ON a.speech_id = s.id
        GROUP BY u.id
        ORDER BY speech_count DESC
        LIMIT 10
    """).fetchall()
    top_users = [dict(r) for r in top_users_rows]
    
    # Recent recordings with user info
    recent_rows = db.conn.execute("""
        SELECT s.*, u.name as user_name,
               (COALESCE(a.lexical_diversity, 0.5) * 20) +
               (COALESCE(a.hnr_db, 15) / 25 * 20) +
               (MIN(MAX(a.wpm, 100), 160) - 100) / 60 * 20 as score
        FROM speeches s
        LEFT JOIN users u ON u.id = s.user_id
        LEFT JOIN analyses a ON a.speech_id = s.id
        WHERE s.user_id IS NOT NULL
        ORDER BY s.created_at DESC
        LIMIT 15
    """).fetchall()
    recent_recordings = [dict(r) for r in recent_rows]
    
    # Chart data: recordings by day (last 30 days)
    recordings_by_day = db.conn.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM speeches
        WHERE user_id IS NOT NULL AND created_at >= ?
        GROUP BY DATE(created_at)
        ORDER BY date
    """, (month_ago,)).fetchall()
    
    # Chart data: user signups by day (last 30 days)
    users_by_day = db.conn.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM users
        WHERE created_at >= ?
        GROUP BY DATE(created_at)
        ORDER BY date
    """, (month_ago,)).fetchall()
    
    # Score distribution
    score_dist = [0, 0, 0, 0, 0]  # 0-20, 20-40, 40-60, 60-80, 80-100
    score_rows = db.conn.execute("""
        SELECT (COALESCE(lexical_diversity, 0.5) * 20) +
               (COALESCE(hnr_db, 15) / 25 * 20) +
               (MIN(MAX(wpm, 100), 160) - 100) / 60 * 20 as score
        FROM analyses WHERE wpm IS NOT NULL
    """).fetchall()
    for row in score_rows:
        if row[0] is not None:
            idx = min(int(row[0] / 20), 4)
            score_dist[idx] += 1
    
    # Category distribution
    categories = db.conn.execute("""
        SELECT COALESCE(category, 'uncategorized') as category, COUNT(*) as count
        FROM speeches
        WHERE user_id IS NOT NULL
        GROUP BY category
        ORDER BY count DESC
    """).fetchall()
    
    chart_data = json.dumps({
        "recordings_by_day": [{"date": r[0], "count": r[1]} for r in recordings_by_day],
        "users_by_day": [{"date": r[0], "count": r[1]} for r in users_by_day],
        "score_distribution": score_dist,
        "categories": [{"category": c[0], "count": c[1]} for c in categories],
    })
    
    # Get all users for the full users table
    all_users_rows = db.conn.execute("""
        SELECT u.*, COUNT(s.id) as speech_count
        FROM users u
        LEFT JOIN speeches s ON s.user_id = u.id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """).fetchall()
    all_users = [dict(r) for r in all_users_rows]
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "stats": {
            "total_users": total_users,
            "total_recordings": total_recordings,
            "active_7d": active_7d,
            "active_30d": active_30d,
            "recordings_today": recordings_today,
            "recordings_week": recordings_week,
            "storage_mb": storage_mb,
            "avg_score": avg_score,
        },
        "top_users": top_users,
        "all_users": all_users,
        "recent_recordings": recent_recordings,
        "chart_data": chart_data,
    })


@router.get("/users", response_class=HTMLResponse)
async def users_page(request: Request):
    """Redirect to admin page."""
    return RedirectResponse(url="/admin", status_code=302)


@router.get("/admin/users/{user_id}", response_class=HTMLResponse)
async def admin_user_detail(request: Request, user_id: int):
    """View details for a specific user (admin only)."""
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/", status_code=302)
    
    # Only allow Jon
    if current_user.get("email", "").lower() != "jon.suppe@gmail.com":
        return HTMLResponse("<h1>Access Denied</h1><p>Admin access is restricted.</p>", status_code=403)
    
    db = get_db()
    
    # Get user info
    row = db.conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        return HTMLResponse("<h1>User not found</h1>", status_code=404)
    
    user = dict(row)
    
    # Get user's speeches with analysis data
    speech_rows = db.conn.execute("""
        SELECT s.*, a.quality_level, a.wpm
        FROM speeches s
        LEFT JOIN analyses a ON a.speech_id = s.id
        WHERE s.user_id = ?
        ORDER BY s.created_at DESC
    """, (user_id,)).fetchall()
    speeches = [dict(r) for r in speech_rows]
    
    # Get coach messages
    coach_messages = db.get_coach_messages(user_id, limit=50)
    coach_stats = db.get_user_coach_stats(user_id)
    
    return templates.TemplateResponse("user_detail.html", {
        "request": request,
        "user": user,
        "speeches": speeches,
        "coach_messages": coach_messages,
        "coach_stats": coach_stats,
        "is_admin": True,
    })


@router.get("/users/{user_id}", response_class=HTMLResponse)
async def users_detail_redirect(request: Request, user_id: int):
    """Redirect old user detail URLs to admin section."""
    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=302)


# ---------------------------------------------------------------------------
# Frontend API Endpoints (use session auth instead of API auth)
# ---------------------------------------------------------------------------

@router.patch("/api/speeches/{speech_id}")
async def frontend_update_speech(
    request: Request,
    speech_id: int,
    title: str = Form(None),
    speaker: str = Form(None),
):
    """Update speech metadata from frontend (uses session auth)."""
    user = get_current_user(request)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": {"code": "UNAUTHORIZED", "message": "Not logged in"}}
        )
    
    # Parse JSON body if Content-Type is application/json
    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            title = body.get("title", title)
            speaker = body.get("speaker", speaker)
        except:
            pass
    
    if title is None and speaker is None:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "NO_CHANGES", "message": "No fields to update"}}
        )
    
    db = get_db()
    
    # Get the speech
    speech = db.get_speech(speech_id)
    if not speech:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "NOT_FOUND", "message": "Speech not found"}}
        )
    
    # Update fields
    updates = []
    params = []
    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if speaker is not None:
        updates.append("speaker = ?")
        params.append(speaker)
    
    params.append(speech_id)
    
    db.conn.execute(
        f"UPDATE speeches SET {', '.join(updates)} WHERE id = ?",
        params
    )
    db.conn.commit()
    
    logger.info(f"Updated speech {speech_id}: title={title}, speaker={speaker}")
    
    return JSONResponse(content={
        "success": True,
        "speech_id": speech_id,
        "title": title,
        "speaker": speaker,
    })
